from multiprocessing import cpu_count
from typing import Union, Optional, Dict, Any, Type, TypeVar

import spglib as spg
import numpy as np
try:
    import brille as br
except ModuleNotFoundError as err:
    raise ModuleNotFoundError(
        'Cannot import Brille for use with BrilleInterpolator '
        '(maybe Brille is not installed?). To install Euphonic\'s '
        'optional Brille dependency, try:\n\npip install '
        'euphonic[brille]\n') from err

from euphonic.validate import _check_constructor_inputs
from euphonic import (ureg, QpointPhononModes, QpointFrequencies,
                      ForceConstants, Crystal)

class BrilleInterpolator:
    """
    A class to perform linear interpolation of eigenvectors and
    frequencies at arbitrary q-points using the Brille library

    Attributes
    ----------
    crystal : Crystal
        Lattice and atom information
    """
    T = TypeVar('T', bound='BrilleInterpolator')

    def __init__(self, crystal: Crystal,
                 grid: Union[br.BZTrellisQdc, br.BZMeshQdc,
                             br.BZNestQdc]) -> None:

        _check_constructor_inputs(
            [crystal, grid],
            [Crystal, [br.BZTrellisQdc, br.BZMeshQdc, br.BZNestQdc]],
            [(), ()],
            ['crystal', 'grid'])
        # Check grid has been filled and vals/vecs are the correct shape
        n_atoms = crystal.n_atoms
        n_qpts = len(grid.rlu)
        _check_constructor_inputs(
            [grid.values, grid.vectors],
            [np.ndarray, np.ndarray],
            [(n_qpts, 3*n_atoms, 1), (n_qpts, 3*n_atoms, 3*n_atoms)],
            ['grid.values', 'grid.vectors'])

        self._grid = grid
        self.crystal = crystal

    def calculate_qpoint_phonon_modes(self, qpts: np.ndarray, **kwargs
                                      ) -> QpointPhononModes:
        """
        Calculate phonon frequencies and eigenvectors at specified
        q-points via linear interpolation

        Parameters
        ----------
        qpts
            Shape (n_qpts, 3) float ndarray. The q-points to
            interpolate onto in reciprocal cell vector units
        **kwargs
            Will be passed to the
            brille.BZTrellis/Mesh/Nest.ir_interpolate_at
            method. By default useparallel=True and
            threads=multiprocessing.cpu_count() are passed

        Returns
        -------
        qpoint_phonon_modes
            An object containing frequencies and eigenvectors
            linearly interpolated at each q-point
        """
        vals, vecs = self._br_grid_calculate_phonons(
            self._grid, qpts, **kwargs)
        frequencies = vals.squeeze(axis=-1)*ureg('hartree').to('meV')
        # Eigenvectors in grid are stored in cell vectors basis,
        # convert to Cartesian
        vecs_cart = self._br_evec_to_eu(
            vecs, cell_vectors=self.crystal._cell_vectors)
        return QpointPhononModes(
            self.crystal, qpts, frequencies, vecs_cart)

    def calculate_qpoint_frequencies(self, qpts: np.ndarray, **kwargs
                                      ) -> QpointFrequencies:
        """
        Calculate phonon frequencies at specified
        q-points via linear interpolation

        Parameters
        ----------
        qpts
            Shape (n_qpts, 3) float ndarray. The q-points to
            interpolate onto in reciprocal cell vector units
        **kwargs
            Will be passed to the
            brille.BZTrellis/Mesh/Nest.ir_interpolate_at
            method. By default useparallel=True and
            threads=multiprocessing.cpu_count() are passed

        Returns
        -------
        qpoint_frequencies
            An object containing frequencies linearly interpolated
            at each q-point
        """
        vals, _ = self._br_grid_calculate_phonons(self._grid, qpts, **kwargs)
        frequencies = vals.squeeze(axis=-1)*ureg('hartree').to('meV')
        return QpointFrequencies(
            self.crystal, qpts, frequencies)

    @staticmethod
    def _br_grid_calculate_phonons(grid, qpts, **kwargs):
        qpts = np.ascontiguousarray(qpts)
        if not kwargs:
            kwargs = {'useparallel': True, 'threads': cpu_count()}
        return grid.ir_interpolate_at(qpts, **kwargs)

    @staticmethod
    def _br_evec_to_eu(br_evecs, cell_vectors=None):
        n_branches = br_evecs.shape[1]
        eu_evecs = br_evecs.view().reshape(-1, n_branches, n_branches//3, 3)
        if cell_vectors is not None:
            # Convert Brille evecs (stored in basis coordinates) to
            # Cartesian coordinates
            eu_evecs = np.einsum('ab,ijka->ijkb', cell_vectors, eu_evecs)
        return eu_evecs

    @classmethod
    def from_force_constants(
            cls: Type[T], force_constants: ForceConstants,
            grid_type: str = 'trellis', grid_npts: int = 1000,
            grid_density: Optional[int] = None,
            grid_kwargs: Optional[Dict[str, Any]] = None,
            interpolation_kwargs: Optional[Dict[str, Any]] = None) -> T:
        """
        Generates a grid over the irreducible Brillouin Zone to be
        used for linear interpolation with Brille, with properties
        determined by the grid_type, grid_npts and grid_kwargs
        arguments. Then uses ForceConstants to fill the grid points
        with phonon frequencies and eigenvectors via Fourier
        interpolation. This returns a BrilleInterpolator object that
        can then be used for linear interpolation.

        Parameters
        ----------
        grid_type
            The Brille grid type to be used, one of {'trellis',
            'mesh', 'nest'}, creating a brille.BZTrellisQdc,
            brille.BZMeshQdc or brille.BZNestQdc grid
            respectively
        grid_npts
            The approximate number of q-points in the Brille grid.
            This is used to set the kwargs for the grid creation so
            that a grid with approximately the desired number of
            points is created, note this number is approximate
            as the number of grid points generated depends on
            Brille's internal algorithm. If this number is higher,
            the linear interpolation is likely to give values
            closer to the ForceConstants Fourier interpolation,
            but the initialisation and memory costs will be higher.
            This does nothing if grid_kwargs is set
        grid_density
            The approximate density of q-points per 1/angstrom^3
            volume
        grid_kwargs
            Kwargs to be passed to the grid constructor (e.g.
            brille.BZTrellisQdc). If set grid_npts does
            nothing
        interpolation_kwargs
            Kwargs to be passed to
            ForceConstants.calculate_qpoint_phonon_modes. Note
            that insert_gamma and reduce_qpoints are incompatible
            and will be ignored

        Returns
        -------
        brille_interpolator
        """
        grid_type_opts = ('trellis', 'mesh', 'nest')
        if grid_type not in grid_type_opts:
            raise ValueError(f'Grid type "{grid_type}" not recognised')

        crystal = force_constants.crystal
        cell_vectors = crystal._cell_vectors
        cell = crystal.to_spglib_cell()

        dataset = spg.get_symmetry_dataset(cell, symprec=1e-8)
        rotations = dataset['rotations']  # in fractional
        translations = dataset['translations']  # in fractional

        symmetry = br.Symmetry(rotations, translations)
        basis = br.Basis(cell[1], cell[2])
        lattice = br.Lattice(cell[0], symmetry, basis, snap_to_symmetry=True)
        # snap_to_symmetry will take care of slightly off-symmetry atom
        # basis positions, but not basis vectors, so include tolerance
        # for lattice vectors when creating the BZ
        ac = br.ApproxConfig()
        ac.real_space_tolerance = 1e-8
        ac.reciprocal_space_tolerance = 1e-8
        bz = br.BrillouinZone(lattice)#, approx_config=ac)

        print('Generating grid...')
        vol = bz.ir_polyhedron.volume
        if grid_type == 'trellis':
            if grid_kwargs is None:
                if grid_density is not None:
                    grid_kwargs = {
                        'node_volume_fraction': 1/grid_density}
                else:
                    # node_volume_fraction actually describes cube
                    # volume used to generate tetrahedra
                    grid_kwargs = {
                        'node_volume_fraction': vol/grid_npts}
                grid_kwargs = {**grid_kwargs,
                               'always_triangulate': False,
                               'approx_config': ac}
            grid = br.BZTrellisQdc(bz, **grid_kwargs)
        elif grid_type == 'mesh':
            if grid_kwargs is None:
                if grid_density is not None:
                    grid_kwargs = {
                        'max_size': 1/grid_density,
                        'max_points': int(grid_density*vol)}
                else:
                    grid_kwargs = {
                        'max_size': vol/grid_npts,
                        'max_points': grid_npts}
            grid = br.BZMeshQdc(bz, **grid_kwargs)
        elif grid_type == 'nest':
            if grid_kwargs is None:
                if grid_density is not None:
                    grid_kwargs = {'number_density': int(grid_density*vol)}
                else:
                    grid_kwargs = {'number_density': grid_npts}
            grid = br.BZNestQdc(bz, **grid_kwargs)

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
        print('Filling grid...')
        grid.fill(frequencies, freq_el, freq_weight, evecs, evecs_el,
                  evecs_weight)
        return cls(force_constants.crystal, grid)
