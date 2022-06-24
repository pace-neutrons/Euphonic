import math
from typing import Dict, Optional, Union, TypeVar, Any, Type
from collections.abc import Mapping

import numpy as np

from euphonic.validate import _check_constructor_inputs
from euphonic.io import _obj_from_json_file, _obj_to_dict, _process_dict
from euphonic.readers import castep, phonopy
from euphonic.util import (direction_changed, is_gamma, get_reference_data)
from euphonic import (ureg, Quantity, Crystal, DebyeWaller, QpointFrequencies,
                      StructureFactor, Spectrum1DCollection)


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
        Shape (n_qpts,) float ndarray. The weight for each q-point,
        for Brillouin Zone integration over symmetry-reduced q-points
    frequencies
        Shape (n_qpts, 3*crystal.n_atoms) float Quantity. Phonon
        frequencies per q-point and mode
    eigenvectors
        Shape (n_qpts, 3*crystal.n_atoms, crystal.n_atoms, 3) complex
        ndarray. The dynamical matrix eigenvectors in Cartesian
        coordinates, using the same Cartesian basis as the
        cell_vectors in the crystal object
    """
    T = TypeVar('T', bound='QpointPhononModes')

    def __init__(self, crystal: Crystal, qpts: np.ndarray,
                 frequencies: Quantity, eigenvectors: np.ndarray,
                 weights: Optional[np.ndarray] = None) -> None:
        """
        Parameters
        ----------
        crystal
            Lattice and atom information
        qpts
            Shape (n_qpts, 3) float ndarray. Q-point coordinates, in
            fractional coordinates of the reciprocal lattice
        frequencies
            Shape (n_qpts, 3*crystal.n_atoms) float Quantity. Phonon
            frequencies per q-point and mode
        eigenvectors
            Shape (n_qpts, 3*crystal.n_atoms, crystal.n_atoms, 3)
            complax ndarray. The dynamical matrix eigenvectors in
            Cartesian coordinates, using the same Cartesian basis
            as the cell_vectors in the crystal object
        weights
            Shape (n_qpts,) float ndarray. The weight for each q-point,
            for Brillouin Zone integration over symmetry-reduced
            q-points. If None, equal weights are assumed
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
        frequencies at each q-point. This means that the same mode
        will have the same index across different q-points, so will
        be plotted as the same colour in a dispersion plot, and can
        be followed across q-space.

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

    def calculate_structure_factor(
        self,
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

        This method calculates the mode-resolved (not binned in energy)
        coherent one-phonon neutron scattering function
        :math:`S(Q, \\omega_{\\mathbf{q}\\nu})` **per atom**, as defined in
        [1]_:

        .. math::

          S(Q, \\omega_{\\mathbf{q}\\nu}) =
              \\frac{1}{2N_{atom}} \\
              \\left\\lvert
              {\\sum_{\\kappa}{\\frac{b_\\kappa}{M_{\\kappa}^{1/2}{\\omega_{\\mathbf{q}\\nu}^{1/2}}}
              (\\mathbf{Q}\\cdot \\mathbf{e}_{\\mathbf{q}\\nu\\kappa})\\exp\\left(i\\mathbf{Q}{\\cdot}\\mathbf{r}_{\\kappa}\\right)
              \\exp\\left(-W_{\\kappa}\\right)}}
              \\right\\rvert^2

        Where :math:`\\nu` runs over phonon modes, :math:`\\kappa` runs
        over atoms, :math:`b_\\kappa` is the coherent neutron
        scattering length, :math:`M_{\\kappa}` is the atom mass,
        :math:`r_{\\kappa}` is the vector from the origin to atom
        :math:`\\kappa` in the unit cell,
        :math:`\\mathbf{e}_{\\mathbf{q}\\nu\\kappa}` are the normalised
        Cartesian eigenvectors, :math:`\\omega_{\\mathbf{q}\\nu}` are
        the frequencies and :math:`\\exp\\left(-W_\\kappa\\right)` is the
        Debye-Waller factor. :math:`N_{atom}` is the number of atoms in
        the unit cell, so the returned structure factor is **per atom**
        of sample.

        .. [1] M.T. Dove, Structure and Dynamics, Oxford University Press, Oxford, 2003, 225-226

        """
        if isinstance(scattering_lengths, str):
            scattering_length_data = get_reference_data(
                collection=scattering_lengths,
                physical_property='coherent_scattering_length')
        elif isinstance(scattering_lengths, dict):
            scattering_length_data = scattering_lengths
        else:
            raise TypeError((
                f'Unexpected type for scattering_lengths, should be str '
                f'or dict, got {type(scattering_lengths)}'))

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
        sf /= 2*self.crystal.n_atoms

        return StructureFactor(
            self.crystal, self.qpts, self.frequencies,
            sf*ureg('bohr**2').to('mbarn'),
            temperature=temperature)

    def calculate_debye_waller(
        self, temperature: Quantity,
        frequency_min: Quantity = Quantity(0.01, 'meV'),
        symmetrise: bool = True) -> DebyeWaller:
        """
        Calculate the 3 x 3 Debye-Waller exponent for each atom over the
        q-points contained in this object

        Parameters
        ----------
        temperature
            Scalar float Quantity. The temperature to use in the
            Debye-Waller calculation
        frequency_min
            Scalar float Quantity in energy units. Excludes frequencies below
            this limit from the calculation, as the calculation contains a
            1/frequency factor which would result in infinite values. This
            also allows negative frequencies to be excluded
        symmetrise
            Whether to symmetrise the Debye-Waller factor based on the
            crystal symmetry operations. Note that if the Debye-Waller
            exponent is not symmetrised, the results may not be the
            same for unfolded and symmetry-reduced q-point grids

        Returns
        -------
        dw
            An object containing the 3x3 Debye-Waller exponent matrix
            for each atom

        Notes
        -----

        As part of the structure factor calculation,
        :math:`\\exp\\left(W_\\kappa\\right)` is the anisotropic
        Debye-Waller factor for atom :math:`\\kappa`, and the exponent can
        be written as:

        .. math::

          W_\\kappa =
            \\sum\\limits_{\\alpha\\beta}W^{\\alpha\\beta}_{\\kappa}
              Q_{\\alpha}Q_{\\beta}

        This function calculates the :math:`W^{\\alpha\\beta}_{\\kappa}`
        part of the exponent, which is  a
        :math:`N_\\mathrm{atom}\\times3\\times3` matrix that is
        independent of Q, so for efficiency can be precalculated to be
        used in the structure factor calculation. The Debye-Waller
        exponent matrix is calculated by [2]_:

        .. math::

          W^{\\alpha\\beta}_{\\kappa} =
          \\frac{\\hbar}{4M_{\\kappa}\\sum\\limits_{\mathbf{q}}{\\mathrm{weight}_\mathbf{q}}}
          \\sum\\limits_{\mathbf{q}\\nu \in{BZ}}\\mathrm{weight}_\mathbf{q}\\frac{e_{\mathbf{q}\\nu\\kappa\\alpha}e^{*}_{\mathbf{q}\\nu\\kappa\\beta}}
          {\\omega_{\mathbf{q}\\nu}}
          \\mathrm{coth}\\left(\\frac{\\hbar\\omega_{\mathbf{q}\\nu}}{2k_BT}\\right)

        Where the sum is over q-points and modes :math:`\\nu` in the
        first Brillouin Zone (BZ), :math:`\\kappa` runs over atoms,
        :math:`\\alpha,\\beta` run over the Cartesian directions,
        :math:`M_{\\kappa}` is the atom mass,
        :math:`e_{q\\nu\\kappa\\alpha}` are the scalar components of the
        normalised Cartesian eigenvectors, :math:`\\omega_{\\mathbf{q}\\nu}`
        are the frequencies, and :math:`\\mathrm{weight}_\\mathbf{q}`
        is the per q-point symmetry weight (if the q-points are not
        symmetry-reduced, all weights will be equal).

        .. [2] G.L. Squires, Introduction to the Theory of Thermal Neutron Scattering, Dover Publications, New York, 1996, 34-37
        """

        # Convert units
        kB = (1*ureg.k).to('hartree/K').magnitude
        n_atoms = self.crystal.n_atoms
        atom_mass = self.crystal._atom_mass
        freqs = self._frequencies
        freq_min = frequency_min.to('hartree').magnitude
        qpts = self.qpts
        evecs = self.eigenvectors
        weights = self.weights
        temp = temperature.to('K').magnitude

        mass_term = 1/(4*atom_mass)

        # Mask out frequencies below frequency_min
        freq_mask = np.ones(freqs.shape)
        freq_mask[freqs < freq_min] = 0

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
        if symmetrise:
            dw_tmp = np.zeros(dw.shape)
            (rot, trans,
             eq_atoms) = self.crystal.get_symmetry_equivalent_atoms()
            cell_vec = self.crystal._cell_vectors
            recip_vec = self.crystal.reciprocal_cell().to('1/bohr').magnitude
            rot_cart = np.einsum('ijk,jl,km->ilm',
                                 rot, cell_vec, recip_vec)/(2*np.pi)
            for s in range(len(rot)):
                dw_tmp[eq_atoms[s]] += np.einsum('ij,kjl,ml->kim',
                                                 rot_cart[s], dw, rot_cart[s])
            dw = dw_tmp/len(rot)

        dw = dw*ureg('bohr**2').to(self.crystal.cell_vectors_unit + '**2')
        return DebyeWaller(self.crystal, dw, temperature)

    def calculate_pdos(
            self, dos_bins: Quantity,
            mode_widths: Optional[Quantity] = None,
            mode_widths_min: Quantity = Quantity(0.01, 'meV'),
            adaptive_method: Optional[str] = 'reference',
            adaptive_error: Optional[float] = 0.01,
            weighting: Optional[str] = None,
            cross_sections: Union[str, Dict[str, Quantity]] = 'BlueBook',
            ) -> Spectrum1DCollection:
        """
        Calculates partial density of states for each atom in the unit
        cell.

        Parameters
        ----------
        dos_bins
            Shape (n_e_bins + 1,) float Quantity. The energy bin edges
            to use for calculating the DOS.
        mode_widths
            Shape (n_qpts, n_branches) float Quantity in energy units.
            The broadening width for each mode at each q-point, for
            adaptive broadening.
        mode_widths_min
            Scalar float Quantity in energy units. Sets a lower limit on
            the mode widths, as mode widths of zero will result in
            infinitely sharp peaks.
        adaptive_method
            String. Specifies whether to use slow, reference adaptive method or
            faster, approximate method. Allowed options are 'reference'
            or 'fast', default is 'reference'.
        adaptive_error
            Scalar float. Acceptable error for gaussian approximations
            when using the fast adaptive method, defined as the absolute
            difference between the areas of the true and approximate gaussians
        weighting
            One of {'coherent', 'incoherent', 'coherent-plus-incoherent'}.
            If provided, produces a neutron-weighted DOS, weighted by
            either the coherent, incoherent, or sum of coherent and
            incoherent neutron scattering cross-sections.
        cross_sections
            A dataset of cross-sections for each element in the structure,
            it can be a string specifying a dataset, or a dictionary
            explicitly giving the cross-sections for each element.

            If cross_sections is a string, it is passed to the ``collection``
            argument of :obj:`euphonic.util.get_reference_data()`. This
            collection must contain the 'coherent_cross_section' or
            'incoherent_cross_section' physical property, depending on
            the ``weighting`` argument. If ``weighting`` is None, this
            string argument is not used.

            If cross sections is a dictionary, the ``weighting`` argument is
            ignored, and these cross-sections are used directly to calculate
            the neutron-weighted DOS. It must contain a key for each element
            in the structure, and each value must be a Quantity in the
            appropriate units, e.g::

                {'La': 8.0*ureg('barn'), 'Zr': 9.5*ureg('barn')}

        Returns
        -------
        dos
            A collection of spectra, with the energy bins on the x-axis and
            PDOS for each atom in the unit cell on the y-axis. If weighting
            is None, the y-axis is in 1/energy units. If weighting is
            specified or cross_sections are supplied, the y-axis
            is in area/energy units per average atom.
        """
        weighting_opts = [None, 'coherent', 'incoherent',
                          'coherent-plus-incoherent']
        if weighting not in weighting_opts:
            raise ValueError(f'Invalid value for weighting, got '
                             f'{weighting}, should be one of '
                             f'{weighting_opts}')

        cross_sections_data = None
        if isinstance(cross_sections, str):
            if weighting is not None:
                if weighting == 'coherent-plus-incoherent':
                    weights = ['coherent', 'incoherent']
                else:
                    weights = [weighting]
                cross_sections_data = [get_reference_data(
                    collection=cross_sections,
                    physical_property=f'{weight}_cross_section')
                                       for weight in weights]
        elif isinstance(cross_sections, Mapping):
            cross_sections_data = [cross_sections]
        else:
            raise TypeError(f'Unexpected type for cross_sections, expected '
                            f'str or dict, got {type(cross_sections)}')

        if cross_sections_data is not None:
            cs = [cross_sections_data[0][x] for x in self.crystal.atom_type]
            if len(cross_sections_data) == 2:
                cs2 = [cross_sections_data[1][x]
                        for x in self.crystal.atom_type]
                cs = [sum(x) for x in zip(cs, cs2)]
            # Account for cross sections in different, or invalid, units
            ex_units = '[length]**2'
            if not cs[0].check(ex_units):
                raise ValueError((
                    f'Unexpected dimensionality in cross_sections units, '
                    f'expected {ex_units}, got {str(cs[0].dimensionality)}'))
            cs = [x.to('mbarn').magnitude for x in cs]*ureg('mbarn')
        else:
            cs = None

        evec_weights = np.real(np.einsum('ijkl,ijkl->ijk',
                                         self.eigenvectors,
                                         np.conj(self.eigenvectors)))
        crystal = self.crystal
        for i in range(crystal.n_atoms):
            dos = self._calculate_dos(dos_bins, mode_widths=mode_widths,
                                      mode_widths_min=mode_widths_min,
                                      adaptive_method=adaptive_method,
                                      adaptive_error=adaptive_error,
                                      mode_weights=evec_weights[:, :, i])
            if cs is not None:
                # Neutron weighted DOS is per atom of sample
                avg_atom_mass = np.mean(crystal._atom_mass)
                dos *= cs[i]*avg_atom_mass/crystal._atom_mass[i]
            if i == 0:
                all_dos_y_data = np.zeros((
                    crystal.n_atoms, len(dos_bins) - 1))*dos.units
            all_dos_y_data[i] = dos
        metadata = {'line_data':
            [{'species': symbol, 'index': idx}
             for idx, symbol in enumerate(crystal.atom_type)]}
        if weighting is not None:
            metadata['weighting'] = weighting

        return Spectrum1DCollection(
            dos_bins, all_dos_y_data, metadata=metadata)

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
    def from_dict(cls: Type[T], d: Dict[str, Any]) -> T:
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
    def from_json_file(cls: Type[T], filename: str) -> T:
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
    def from_castep(cls: Type[T], filename: str,
                    average_repeat_points: bool = False) -> T:
        """
        Reads precalculated phonon mode data from a CASTEP .phonon file

        Parameters
        ----------
        filename
            The path and name of the .phonon file to read
        average_repeat_points
            If multiple frequency/eigenvectors blocks are included with the
            same q-point index (i.e. for Gamma-point with LO-TO splitting),
            scale the weights such that these sum to the given weight
        """
        data = castep.read_phonon_data(
            filename,
            average_repeat_points=average_repeat_points)

        return cls.from_dict(data)

    @classmethod
    def from_phonopy(cls: Type[T], path: str = '.',
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
