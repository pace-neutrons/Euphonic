import math
from typing import Dict, Optional, Union, TypeVar, Any

import numpy as np

from euphonic.validate import _check_constructor_inputs
from euphonic.io import _obj_from_json_file, _obj_to_dict, _process_dict
from euphonic.readers import castep, phonopy
from euphonic.util import (direction_changed, is_gamma, get_reference_data)
from euphonic import (ureg, Quantity, Crystal, DebyeWaller, QpointFrequencies,
                      StructureFactor)


T = TypeVar('T', bound='QpointPhononModes')


class QpointPhononModes(QpointFrequencies):
    """
    A class to read and store vibrational data from model (e.g. CASTEP)
    output files

    Attributes
    ----------
    crystal
        Lattice and atom information
    n_qpts
        Number of q-points in the object
    qpts
        Shape (n_qpts, 3) float ndarray. Q-point coordinates, in
        fractional coordinates of the reciprocal lattice
    weights
        Shape (n_qpts,) float ndarray. The weight for each q-point
    frequencies
        Shape (n_qpts, 3*crystal.n_atoms) float Quantity. Phonon
        frequencies per q-point and mode
    eigenvectors
        Shape (n_qpts, 3*crystal.n_atoms, crystal.n_atoms, 3) complex
        ndarray. The dynamical matrix eigenvectors
    """

    def __init__(self, crystal: Crystal, qpts: np.ndarray,
                 frequencies: Quantity, eigenvectors: Quantity,
                 weights: Optional[np.ndarray] = None):
        """
        Parameters
        ----------
        crystal
            Lattice and atom information
        qpts
            Shape (n_qpts, 3) float ndarray. The Q-point coordinates
        frequencies
            Shape (n_qpts, 3*crystal.n_atoms) float Quantity. Phonon
            frequencies per q-point and mode
        eigenvectors
            Shape (n_qpts, 3*crystal.n_atoms, crystal.n_atoms, 3)
            complex ndarray. The dynamical matrix eigenvectors
        weights
            Shape (n_qpts,) float ndarray. The weight for each q-point.
            If None, equal weights are assumed
        """
        super().__init__(crystal, qpts, frequencies, weights)
        n_qpts = len(qpts)
        n_at = crystal.n_atoms
        # Check freqs axis 1 shape here - QpointFrequencies doesn't
        # enforce that the number of modes = 3*(number of atoms)
        _check_constructor_inputs(
            [frequencies, eigenvectors],
            [Quantity, np.ndarray],
            [(n_qpts, 3*n_at), (n_qpts, 3*n_at, n_at, 3)],
            ['frequencies', 'eigenvectors'])
        self.eigenvectors = eigenvectors

    def reorder_frequencies(self,
                            reorder_gamma: bool = True) -> None:
        """
        By doing a dot product of eigenvectors at adjacent q-points,
        determines which modes are most similar and reorders the
        frequencies at each q-point

        Parameters
        ----------
        reorder_gamma
            Whether to reorder frequencies at gamma-equivalent points.
            If an analytical correction has been applied at the gamma
            points (i.e LO-TO splitting) mode assignments can be
            incorrect at adjacent q-points where the correction hasn't
            been applied. So you might not want to reorder at gamma for
            some materials
        """
        n_qpts = self.n_qpts
        n_branches = 3*self.crystal.n_atoms
        qpts = self.qpts
        eigenvecs = self.eigenvectors

        # Initialise map, don't reorder first q-point
        mode_map = np.zeros((n_qpts, n_branches), dtype=np.int32)
        mode_map[0] = np.arange(n_branches)

        # Only calculate reordering if the direction hasn't
        # changed
        calc_reorder = np.concatenate(([True], np.logical_not(
            direction_changed(qpts))))

        if not reorder_gamma:
            gamma_i = np.where(is_gamma(qpts))[0]
            # Don't reorder at gamma
            calc_reorder[gamma_i[gamma_i > 0] - 1] = False
            # Or at the point after
            calc_reorder[gamma_i[gamma_i < n_qpts - 1]] = False

        for i in range(1, n_qpts):
            # Compare eigenvectors for each mode for this q-point with
            # every mode for the previous q-point
            # Explicitly broadcast arrays with repeat and tile to ensure
            # correct multiplication of modes
            curr_evecs = eigenvecs[i, :, :, :]
            prev_evecs = eigenvecs[i - 1, :, :, :]
            current_eigenvecs = np.repeat(curr_evecs, n_branches, axis=0)
            prev_eigenvecs = np.tile(prev_evecs, (n_branches, 1, 1))

            if calc_reorder[i-1]:
                # Compute complex conjugated dot product of every mode
                # of this q-point with every mode of previous q-point,
                # and sum the dot products over atoms (i.e. multiply
                # eigenvectors elementwise, then sum over the last 2
                # dimensions)
                dots = np.absolute(np.einsum('ijk,ijk->i',
                                             np.conj(prev_eigenvecs),
                                             current_eigenvecs))

                # Create matrix of dot products for each mode of this
                # q-point with each mode of the previous q-point
                dot_mat = np.reshape(dots, (n_branches, n_branches))

                # Find greatest dot product
                for j in range(n_branches):
                    max_i = (np.argmax(dot_mat))
                    # Modes are dot_mat rows
                    mode = int(max_i/n_branches)
                    # Prev q-pt modes are columns
                    prev_mode = max_i % n_branches
                    # Ensure modes aren't mapped more than once
                    dot_mat[mode, :] = 0
                    dot_mat[:, prev_mode] = 0

                    prev_mode_idx = np.where(mode_map[i-1] == prev_mode)[0][0]
                    mode_map[i, prev_mode_idx] = mode
            else:
                mode_map[i] = mode_map[i-1]

        # Actually rearrange frequencies/eigenvectors
        for i in range(n_qpts):
            self._frequencies[i] = self._frequencies[i, mode_map[i]]
            evec_tmp = np.copy(self.eigenvectors[i, mode_map[i]])
            self.eigenvectors[i] = evec_tmp

    def calculate_structure_factor(self,
        scattering_lengths: Union[str, Dict[str, Quantity]] = 'Sears1992',
        dw: Optional[DebyeWaller] = None,
        ) -> StructureFactor:
        """
        Calculate the one phonon inelastic scattering for neutrons at
        each q-point

        Parameters
        ----------
        scattering_lengths
            Dataset of coherent scattering length for each
            element in the structure. This may be provided in 3 ways:

            - A string naming an appropriate data collection packaged with
              Euphonic (including the default value 'Sears1992'). This will be
              passed to the ``collection`` argument of
              :obj:`euphonic.util.get_reference_data()`.

            - A string filename for a user's customised data file in the same
              format as those packaged with Euphonic.

            - An explicit dictionary of float Quantity, giving spin- and
              isotope-averaged coherent scattering length for each element in
              the structure, e.g.::

                {'O': 5.803*ureg('fm'), 'Zn': 5.680*ureg('fm')}

        dw
            Data for thermal motion effects. Typically this is computed over a
            converged Monkhort-Pack grid, which need not correspond to the
            q-points of this QpointPhononModes object.

        Returns
        -------
        sf
            An object containing the structure factor for each q-point
            and phonon mode

        Notes
        -----

        This function calculates :math:`|F(Q, \\nu)|^2` per unit cell, where
        :math:`F(Q, \\nu)` is defined as [1]_:

        .. math::

          F(Q, \\nu) = \\frac{b_\\kappa}{M_{\\kappa}^{1/2}\\omega_{q\\nu}^{1/2}} \\
          [Q\\cdot\\epsilon_{q\\nu\\kappa\\alpha}]e^{iQ{\\cdot}r_\\kappa}e^{-W}

        Where :math:`\\nu` runs over phonon modes, :math:`\\kappa` runs
        over atoms, :math:`\\alpha` runs over the Cartesian directions,
        :math:`b_\\kappa` is the coherent neutron scattering length,
        :math:`M_{\\kappa}` is the atom mass, :math:`r_{\\kappa}` is the
        vector to atom :math:`\\kappa` in the unit cell,
        :math:`\\epsilon_{q\\nu\\kappa\\alpha}` are the eigevectors,
        :math:`\\omega_{q\\nu}` are the frequencies and :math:`e^{-W}`
        is the Debye-Waller factor. Note that a factor N for the
        number of unit cells in the sample hasn't been included, so the
        returned structure factor is per unit cell.

        .. [1] M.T. Dove, Structure and Dynamics, Oxford University Press, Oxford, 2003, 225-226

        """
        if isinstance(scattering_lengths, str):
            scattering_length_data = get_reference_data(
                collection=scattering_lengths,
                physical_property='coherent_scattering_length')
        elif isinstance(scattering_lengths, dict):
            scattering_length_data = scattering_lengths

        sl = [scattering_length_data[x].to('bohr').magnitude
              for x in self.crystal.atom_type]

        # Calculate normalisation factor
        norm_factor = sl/np.sqrt(self.crystal._atom_mass)

        # Calculate the exp factor for all atoms and qpts. atom_r is in
        # fractional coords, so Qdotr = 2pi*qh*rx + 2pi*qk*ry...
        exp_factor = np.exp(1J*2*math.pi*np.einsum(
            'ij,kj->ik', self.qpts, self.crystal.atom_r))

        # Eigenvectors are in Cartesian so need to convert hkl to
        # Cartesian by computing dot with hkl and reciprocal lattice
        recip = self.crystal.reciprocal_cell().to('1/bohr').magnitude
        Q = np.einsum('ij,jk->ik', self.qpts, recip)

        # Calculate dot product of Q and eigenvectors for all branches
        # atoms and q-points
        eigenv_dot_q = np.einsum('ijkl,il->ijk', np.conj(self.eigenvectors), Q)

        # Calculate Debye-Waller factors
        temperature = None
        if dw:
            temperature = dw.temperature
            if dw.crystal.n_atoms != self.crystal.n_atoms:
                raise ValueError((
                    'The DebyeWaller object used as dw is not '
                    'compatible with the QPointPhononModes object (they'
                    ' have a different number of atoms)'))
            dw_factor = np.exp(-np.einsum('jkl,ik,il->ij',
                                          dw._debye_waller, Q, Q))
            exp_factor *= dw_factor

        # Multiply Q.eigenvector, exp factor and normalisation factor
        term = np.einsum('ijk,ik,k->ij', eigenv_dot_q, exp_factor, norm_factor)

        # Take mod squared and divide by frequency to get intensity
        sf = np.real(
            np.absolute(term*np.conj(term))/np.absolute(self._frequencies))

        return StructureFactor(
            self.crystal, self.qpts, self.frequencies,
            sf*ureg('bohr**2').to(self.crystal.cell_vectors.units**2),
            temperature=temperature)

    def calculate_debye_waller(self, temperature: Quantity) -> DebyeWaller:
        """
        Calculate the 3 x 3 Debye-Waller exponent for each atom over the
        q-points contained in this object

        Parameters
        ----------
        temperature
            Scalar float Quantity. The temperature to use in the
            Debye-Waller calculation.

        Returns
        -------
        dw
            An object containing the 3x3 Debye-Waller exponent for each
            atom

        Notes
        -----

        As part of the structure factor calculation, the anisotropic
        Debye-Waller factor is defined as:

        .. math::

          e^{-W} = e^{-\\sum_{\\alpha\\beta}\\frac{W^{\\kappa}_{\\alpha\\beta}Q_{\\alpha}Q_{\\beta}}{2}}

        The Debye-Waller exponent is defined as
        :math:`W^{\\kappa}_{\\alpha\\beta}` and is independent of Q, so
        for efficiency can be precalculated to be used in the structure
        factor calculation. The Debye-Waller exponent is calculated by [2]_

        .. math::

          W^{\\kappa}_{\\alpha\\beta} =
          \\frac{1}{2N_{q}M_{\\kappa}}
          \\sum_{BZ}\\frac{\\epsilon_{q\\nu\\kappa\\alpha}\\epsilon^{*}_{q\\nu\\kappa\\beta}}
          {\\omega_{q\\nu}}
          coth(\\frac{\\omega_{q\\nu}}{2k_BT})

        Where :math:`\\nu` runs over phonon modes, :math:`\\kappa` runs
        over atoms, :math:`\\alpha,\\beta` run over the Cartesian
        directions, :math:`M_{\\kappa}` is the atom mass,
        :math:`\\epsilon_{q\\nu\\kappa\\alpha}` are the eigenvectors,
        :math:`\\omega_{q\\nu}` are the frequencies, :math:`\\sum_{BZ}`
        is a sum over the 1st Brillouin Zone, and :math:`N_q` is the
        number of q-point samples in the BZ.

        .. [2] G.L. Squires, Introduction to the Theory of Thermal Neutron Scattering, Dover Publications, New York, 1996, 34-37
        """

        # Convert units
        kB = (1*ureg.k).to('hartree/K').magnitude
        n_atoms = self.crystal.n_atoms
        atom_mass = self.crystal._atom_mass
        freqs = self._frequencies
        qpts = self.qpts
        evecs = self.eigenvectors
        weights = self.weights
        temp = temperature.to('K').magnitude

        mass_term = 1/(4*atom_mass)

        # Determine q-points near the gamma point and mask out their
        # acoustic modes due to the potentially large 1/frequency factor
        TOL = 1e-8
        is_small_q = np.sum(np.square(qpts), axis=1) < TOL
        freq_mask = np.ones(freqs.shape)
        freq_mask[is_small_q, :3] = 0

        if temp > 0:
            x = freqs/(2*kB*temp)
            freq_term = 1/(freqs*np.tanh(x))
        else:
            freq_term = 1/(freqs)
        dw = np.zeros((n_atoms, 3, 3))
        # Calculating the e.e* term is expensive, do in chunks
        chunk = 1000
        for i in range(int((len(qpts) - 1)/chunk) + 1):
            qi = i*chunk
            qf = min((i + 1)*chunk, len(qpts))

            evec_term = np.real(
                np.einsum('ijkl,ijkm->ijklm',
                          evecs[qi:qf],
                          np.conj(evecs[qi:qf])))

            dw += (np.einsum('i,k,ij,ij,ijklm->klm',
                             weights[qi:qf], mass_term, freq_term[qi:qf],
                             freq_mask[qi:qf], evec_term))

        dw = dw/np.sum(weights)
        dw = dw*ureg('bohr**2').to(self.crystal.cell_vectors_unit + '**2')

        return DebyeWaller(self.crystal, dw, temperature)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a dictionary. See QpointPhononModes.from_dict for
        details on keys/values
        """
        dout = _obj_to_dict(self, ['crystal', 'n_qpts', 'qpts', 'frequencies',
                                   'eigenvectors', 'weights'])
        return dout

    def to_qpoint_frequencies(self) -> QpointFrequencies:
        """
        Create a QpointFrequencies object
        """
        return QpointFrequencies(
            self.crystal, self.qpts, self.frequencies, self.weights)

    @classmethod
    def from_dict(cls: T, d: Dict[str, Any]) -> T:
        """
        Convert a dictionary to a QpointPhononModes object

        Parameters
        ----------
        d : dict
            A dictionary with the following keys/values:

            - 'crystal': dict, see Crystal.from_dict
            - 'qpts': (n_qpts, 3) float ndarray
            - 'frequencies': (n_qpts, 3*crystal.n_atoms) float ndarray
            - 'frequencies_unit': str
            - 'eigenvectors': (n_qpts, 3*crystal.n_atoms, crystal.n_atoms, 3) complex ndarray

            There are also the following optional keys:

            - 'weights': (n_qpts,) float ndarray
        """
        crystal = Crystal.from_dict(d['crystal'])
        d = _process_dict(d, quantities=['frequencies'], optional=['weights'])
        return cls(crystal, d['qpts'], d['frequencies'],
                   d['eigenvectors'], d['weights'])

    @classmethod
    def from_json_file(cls: T, filename: str) -> T:
        """
        Read from a JSON file. See QpointPhononModes.from_dict for
        required fields

        Parameters
        ----------
        filename
            The file to read from
        """
        return _obj_from_json_file(cls, filename,
                                   type_dict={'eigenvectors': np.complex128})

    @classmethod
    def from_castep(cls: T, filename: str) -> T:
        """
        Reads precalculated phonon mode data from a CASTEP .phonon file

        Parameters
        ----------
        filename
            The path and name of the .phonon file to read
        """
        data = castep.read_phonon_data(filename)
        return cls.from_dict(data)

    @classmethod
    def from_phonopy(cls: T, path: str = '.',
                     phonon_name: str = 'band.yaml',
                     phonon_format: Optional[str] = None,
                     summary_name: str = 'phonopy.yaml') -> T:
        """
        Reads precalculated phonon mode data from a Phonopy
        mesh/band/qpoints.yaml/hdf5 file. May also read from
        phonopy.yaml for structure information.

        Parameters
        ----------
        path
            Path to directory containing the file(s)
        phonon_name
            Name of Phonopy file including the frequencies
        phonon_format
            Format of the phonon_name file if it isn't obvious from the
            phonon_name extension, one of {'yaml', 'hdf5'}
        summary_name
            Name of Phonopy summary file to read the crystal information
            from. Crystal information in the phonon_name file takes
            priority, but if it isn't present, crystal information is
            read from summary_name instead
        """
        data = phonopy.read_phonon_data(
            path=path, phonon_name=phonon_name, phonon_format=phonon_format,
            summary_name=summary_name)
        return cls.from_dict(data)
