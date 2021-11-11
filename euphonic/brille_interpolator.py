import spglib as spg
import numpy as np
import brille as br
from multiprocessing import cpu_count

from euphonic import ureg, QpointPhononModes

class BrilleInterpolator(object):
    """
    A class to perform linear interpolation of eigenvectors and
    frequencies at arbitrary q-points using the Brille library

    Attributes
    ----------
    crystal : Crystal
        Lattice and atom information
    """
    def __init__(self, force_constants, grid_type='trellis',
                 n_grid_points=1000, grid_kwargs=None,
                 interpolation_kwargs=None):

        crystal = force_constants.crystal
        cell_vectors = crystal._cell_vectors
        cell = crystal.to_spglib_cell()

        dataset = spg.get_symmetry_dataset(cell)
        rotations = dataset['rotations']  # in fractional
        translations = dataset['translations']  # in fractional

        symmetry = br.Symmetry(rotations, translations)
        direct = br.Direct(*cell)
        direct.spacegroup = symmetry
        bz = br.BrillouinZone(direct.star)

        print(f'Generating grid...')
        if grid_type == 'trellis':
            if grid_kwargs is None:
                vol = bz.ir_polyhedron.volume
                grid_kwargs = {
                    'node_volume_fraction': vol/n_grid_points}
            grid = br.BZTrellisQdc(bz, **grid_kwargs)
        elif grid_type == 'mesh':
            if grid_kwargs is None:
                grid_kwargs = {
                    'max_size': bz.ir_polyhedron.volume/n_grid_points,
                    'max_points': n_grid_points}
            grid = br.BZMeshQdc(bz, **grid_kwargs)
        elif grid_type == 'nest':
            if grid_kwargs is None:
                grid_kwargs = {'number_density': n_grid_points}
            grid = br.BZNestQdc(bz, **grid_kwargs)
        else:
            raise ValueError(f'Grid type "{grid_type}" not recognised')

        print(f'Grid generated with {len(grid.rlu)} q-points. '
               'Calculating frequencies/eigenvectors...')
        if interpolation_kwargs is None:
            interpolation_kwargs = {}
        interpolation_kwargs['insert_gamma'] = False
        interpolation_kwargs['reduce_qpts'] = False
        phonons = force_constants.calculate_qpoint_phonon_modes(
            grid.rlu, **interpolation_kwargs)
        # Convert eigenvectors from Cartesian to cell vectors basis
        # for storage in grid 
        evecs_basis = np.einsum('ba,ijkb->ijka', np.linalg.inv(cell_vectors),
                                phonons.eigenvectors)
        n_atoms = crystal.n_atoms
        frequencies = np.reshape(phonons._frequencies,
                                 phonons._frequencies.shape + (1,))
        freq_el = (1,)
        freq_weight = (1., 0., 0.)
        evecs = np.reshape(evecs_basis,
                           (evecs_basis.shape[0], 3*n_atoms, 3*n_atoms))
        cost_function = 0
        evecs_el = (0, 3*n_atoms, 0, 3, 0, cost_function)
        evecs_weight = (0., 1., 0.)
        print(f'Filling grid...')
        grid.fill(frequencies, freq_el, freq_weight, evecs, evecs_el,
                  evecs_weight)
        self._grid = grid
        self.crystal = crystal

    def calculate_qpoint_phonon_modes(self, qpts, **kwargs):
        if not kwargs:
            kwargs = {'useparallel': True, 'threads': cpu_count()} 
        vals, vecs = self._grid.ir_interpolate_at(qpts, **kwargs)
        # Eigenvectors in grid are stored in cell vectors basis,
        # convert to Cartesian
        vecs_cart = self._br_evec_to_eu(
            vecs, cell_vectors=self.crystal._cell_vectors)
        frequencies = vals.squeeze()*ureg('hartree').to('meV')
        return QpointPhononModes(
            self.crystal, qpts, frequencies, vecs_cart)

    @staticmethod
    def _br_evec_to_eu(br_evecs, cell_vectors=None):
        n_branches = len(br_evecs[1])
        eu_evecs = br_evecs.view().reshape(-1, n_branches, n_branches//3, 3)
        if cell_vectors is not None:
            # Convert Brille evecs (stored in basis coordinates) to
            # Cartesian coordinates
            eu_evecs = np.einsum('ab,ijka->ijkb', cell_vectors, eu_evecs)
        return eu_evecs
