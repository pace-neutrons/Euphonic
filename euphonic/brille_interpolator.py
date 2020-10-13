import spglib as spg
import numpy as np
import brille as br

from euphonic import ureg, QpointPhononModes

class BrilleInterpolator(object):
    """
    Brillebrillebrille
    """
    def __init__(self, force_constants, grid_type='mesh', **grid_kwargs):

        crystal = force_constants.crystal
        cell_vectors = crystal._cell_vectors
        cell = crystal.to_spglib_cell()

        dataset = spg.get_symmetry_dataset(cell)
        rotations = dataset['rotations'] # in fractional
        translations = dataset['translations'] # in fractional

        symmetry = br.Symmetry(rotations, translations)
        direct = br.Direct(*cell)
        direct.spacegroup = symmetry
        bz = br.BrillouinZone(direct.star)

        if grid_type == 'mesh':
            grid = br.BZMeshQdc(bz, **grid_kwargs)
        elif grid_type == 'nest':
            grid = br.BZMeshQdc(bz, **grid_kwargs)
        elif grid_type == 'trellis':
            grid = br.BZMeshQdc(bz, **grid_kwargs)
        else:
            raise ValueError(f'Grid type "{grid_type}" not recognised')

        print(f'Grid generated with {len(grid.rlu)} q-points')
        phonons = force_constants.calculate_qpoint_phonon_modes(
            grid.rlu, insert_gamma=False, reduce_qpts=False,
            use_c=True, n_threads=2)
        print(f'Frequencies/eigenvectors calculated for {len(grid.rlu)} q-points')
        evecs_basis = np.einsum('ba,ijkb->ijka', np.linalg.inv(cell_vectors),
                                phonons.eigenvectors)
        # Degenerate modes check - don't bother with this for now?
        n_atoms = crystal.n_atoms
        frequencies = np.reshape(phonons._frequencies, phonons._frequencies.shape + (1,))
        freq_el = (1,)
        #freq_weight = (13605.693, 0., 0.) # Rydberg/meV - could this just be 1.0?
        freq_weight = (1., 0., 0.)
        evecs = np.reshape(evecs_basis, (evecs_basis.shape[0], 3*n_atoms, 3*n_atoms))
        cost_function = 0 # Don't know what this does
        evecs_el = (0, 3*n_atoms, 0, 3, 0, cost_function)
        evecs_weight = (0., 1., 0.)
        grid.fill(frequencies, freq_el, freq_weight, evecs, evecs_el, evecs_weight)
        self._grid = grid
        self.crystal = crystal

    def calculate_qpoint_phonon_modes(self, qpts, **kwargs):
        vals, vecs = self._grid.ir_interpolate_at(qpts, **kwargs)
        n_atoms = self.crystal.n_atoms
        vecs = vecs.reshape(len(qpts), 3*n_atoms, n_atoms, 3)
        vecs_cart = np.einsum('ab,ijka->ijkb', self.crystal._cell_vectors, vecs)
        frequencies = vals.squeeze()*ureg('INTERNAL_ENERGY_UNIT').to(
            'mDEFAULT_ENERGY_UNIT')
        return QpointPhononModes(
            self.crystal, qpts, frequencies, vecs_cart)
