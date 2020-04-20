import math
import numpy as np
from euphonic import ureg, Crystal
from euphonic.util import (direction_changed, bose_factor, is_gamma, lorentzian,
                           gaussian)
from euphonic._readers import _castep
from euphonic._readers import _phonopy


class PhononData(object):
    """
    A class to read and store vibrational data from model (e.g. CASTEP) output
    files

    Attributes
    ----------
    crystal : Crystal
        Lattice and atom information
    n_qpts : int
        Number of q-points in the .phonon file
    qpts : (n_qpts, 3) float ndarray
        Q-point coordinates
    weights : (n_qpts,) float ndarray
        The weight for each q-point
    freqs: (n_qpts, 3*n_atoms) float ndarray
        Phonon frequencies, ordered according to increasing q-point
        number. Default units meV
    eigenvecs: (n_qpts, 3*n_atoms, n_atoms, 3) complex ndarray
        Dynamical matrix eigenvectors. Empty if read_eigenvecs is False
    """

    def __init__(self, crystal, qpts, freqs, eigenvecs, weights=None):
        """
        Parameters
        ----------
        crystal : Crystal
            Lattice and atom information
        qpts : (n_qpts, 3) float ndarray
            Q-point coordinates
        freqs: (n_qpts, 3*n_atoms) float Quantity
            Phonon frequencies, ordered according to increasing q-point
            number. Default units meV
        eigenvecs: (n_qpts, 3*n_atoms, n_atoms, 3) complex ndarray
            Dynamical matrix eigenvectors. Empty if read_eigenvecs is False
        weights : (n_qpts,) float ndarray, optional, default None
            The weight for each q-point. If None, equal weights are assumed
        """
        self.crystal = crystal
        self.qpts = qpts
        self.n_qpts = len(qpts)
        self._freqs = freqs.to(
            ureg.INTERNAL_ENERGY_UNIT).magnitude
        self.freqs_unit = str(freqs.units)
        self.eigenvecs = eigenvecs

        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.full(self.n_qpts, 1/self.n_qpts)

    @property
    def freqs(self):
        return self._freqs*ureg(
            'INTERNAL_ENERGY_UNIT').to(self.freqs_unit, 'spectroscopy')

    @property
    def sqw_ebins(self):
        return self._sqw_ebins*ureg(
            'INTERNAL_ENERGY_UNIT').to(self.freqs_unit, 'spectroscopy')

    @property
    def dos_bins(self):
        return self._dos_bins*ureg('E_h').to(self.freqs_unit, 'spectroscopy')

    def reorder_freqs(self, reorder_gamma=True):
        """
        By doing a dot product of eigenvectors at adjacent q-points,
        determines which modes are most similar and creates a _mode_map
        attribute in the Data object, which specifies which order the
        frequencies should be in at each q-point. The branch ordering can be
        seen when plotting dispersion

        Parameters
        ----------
        reorder_gamma : bool, default True
            Whether to reorder frequencies at gamma-equivalent points. If
            an analytical correction has been applied at the gamma points
            (i.e LO-TO splitting) mode assignments can be incorrect at
            adjacent q-points where the correction hasn't been applied.
            So you might not want to reorder at gamma for some materials
        """
        n_qpts = self.n_qpts
        n_branches = 3*self.crystal.n_atoms
        qpts = self.qpts
        eigenvecs = self.eigenvecs

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
            # Compare eigenvectors for each mode for this q-point with every
            # mode for the previous q-point
            # Explicitly broadcast arrays with repeat and tile to ensure
            # correct multiplication of modes
            curr_evecs = eigenvecs[i, :, :, :]
            prev_evecs = eigenvecs[i - 1, :, :, :]
            current_eigenvecs = np.repeat(curr_evecs, n_branches, axis=0)
            prev_eigenvecs = np.tile(prev_evecs, (n_branches, 1, 1))

            if calc_reorder[i-1]:
                # Compute complex conjugated dot product of every mode of this
                # q-point with every mode of previous q-point, and sum the dot
                # products over atoms (i.e. multiply eigenvectors elementwise,
                # then sum over the last 2 dimensions)
                dots = np.absolute(np.einsum('ijk,ijk->i',
                                             np.conj(prev_eigenvecs),
                                             current_eigenvecs))

                # Create matrix of dot products for each mode of this q-point
                # with each mode of the previous q-point
                dot_mat = np.reshape(dots, (n_branches, n_branches))

                # Find greatest dot product
                for j in range(n_branches):
                    max_i = (np.argmax(dot_mat))
                    mode = int(max_i/n_branches)  # Modes are dot_mat rows
                    prev_mode = max_i % n_branches  # Prev q-pt modes are cols
                    # Ensure modes aren't mapped more than once
                    dot_mat[mode, :] = 0
                    dot_mat[:, prev_mode] = 0

                    prev_mode_idx = np.where(mode_map[i-1] == prev_mode)[0][0]
                    mode_map[i, prev_mode_idx] = mode
            else:
                mode_map[i] = mode_map[i-1]

        self._mode_map = mode_map

    def calculate_structure_factor(self, scattering_lengths, T=5.0, scale=1.0,
                                   calc_bose=True, dw_data=None):
        """
        Calculate the one phonon inelastic scattering at each q-point
        See M. Dove Structure and Dynamics Pg. 226

        Parameters
        ----------
        scattering_lengths : dictionary
            Dictionary of spin and isotope averaged coherent scattering legnths
            for each element in the structure in fm e.g.
            {'O': 5.803, 'Zn': 5.680}
        T : float, optional, default 5.0
            The temperature in Kelvin to use when calculating the Bose and
            Debye-Waller factors
        scale : float, optional, default 1.0
            Apply a multiplicative factor to the final structure factor.
        calc_bose : boolean, optional, default True
            Whether to calculate and apply the Bose factor
        dw_data : InterpolationData or PhononData object
            A PhononData or InterpolationData object with
            frequencies/eigenvectors calculated on a q-grid over which the
            Debye-Waller factor will be calculated

        Returns
        -------
        sf : (n_qpts, 3*n_atoms) float ndarray
            The structure factor for each q-point and phonon branch
        """
        sl = [scattering_lengths[x] for x in self.crystal.atom_type]
        sl = (sl*ureg('fm').to('bohr')).magnitude

        # Calculate normalisation factor
        norm_factor = sl/np.sqrt(self.crystal._atom_mass)

        # Calculate the exponential factor for all atoms and q-points
        # atom_r in fractional coords, so Qdotr = 2pi*qh*rx + 2pi*qk*ry...
        exp_factor = np.exp(1J*2*math.pi*np.einsum(
            'ij,kj->ik', self.qpts, self.crystal.atom_r))

        # Eigenvectors are in Cartesian so need to convert hkl to Cartesian by
        # computing the dot product with hkl and reciprocal lattice
        recip = self.crystal.reciprocal_cell().to('1/bohr').magnitude
        Q = np.einsum('ij,jk->ik', self.qpts, recip)

        # Calculate dot product of Q and eigenvectors for all branches, atoms
        # and q-points
        eigenv_dot_q = np.einsum('ijkl,il->ijk', np.conj(self.eigenvecs), Q)

        # Calculate Debye-Waller factors
        if dw_data:
            if dw_data.crystal.n_atoms != self.crystal.n_atoms:
                raise Exception((
                    'The Data object used as dw_data is not compatible with the'
                    ' object that calculate_structure_factor has been called on'
                    ' (they have a different number of atoms). Is dw_data '
                    'correct?'))
            dw = dw_data._dw_coeff(T)
            dw_factor = np.exp(-np.einsum('jkl,ik,il->ij', dw, Q, Q)/2)
            exp_factor *= dw_factor

        # Multiply Q.eigenvector, exp factor and normalisation factor
        term = np.einsum('ijk,ik,k->ij', eigenv_dot_q, exp_factor, norm_factor)

        # Take mod squared and divide by frequency to get intensity
        sf = np.absolute(term*np.conj(term))/np.absolute(self._freqs)

        # Multiply by Bose factor
        if calc_bose:
            sf = sf*bose_factor(self._freqs, T)

        sf = np.real(sf*scale)

        return sf

    def _dw_coeff(self, T):
        """
        Calculate the 3 x 3 Debye-Waller coefficients for each atom over the
        q-points contained in this object

        Parameters
        ----------
        T : float
            Temperature in Kelvin

        Returns
        -------
        dw : (n_atoms, 3, 3) float ndarray
            The DW coefficients for each atom
        """

        # Convert units
        kB = (1*ureg.k).to('E_h/K').magnitude
        n_atoms = self.crystal.n_atoms
        atom_mass = self.crystal._atom_mass
        freqs = self._freqs
        qpts = self.qpts
        evecs = self.eigenvecs
        weights = self.weights

        mass_term = 1/(2*atom_mass)

        # Determine q-points near the gamma point and mask out their acoustic
        # modes due to the potentially large 1/frequency factor
        TOL = 1e-8
        is_small_q = np.sum(np.square(qpts), axis=1) < TOL
        freq_mask = np.ones(freqs.shape)
        freq_mask[is_small_q, :3] = 0

        if T > 0:
            x = freqs/(2*kB*T)
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

        return dw

    def calculate_sqw_map(self, scattering_lengths, ebins, calc_bose=True,
                          **kwargs):
        """
        Calculate the structure factor for each q-point contained in data, and
        bin according to ebins to create a S(Q,w) map

        Parameters
        ----------
        scattering_lengths : dictionary
            Dictionary of spin and isotope averaged coherent scattering legnths
            for each element in the structure in fm e.g.
            {'O': 5.803, 'Zn': 5.680}
        ebins : (n_ebins + 1,) float ndarray
            The energy bin edges in the same units as PhononData.freqs
        calc_bose : boolean, optional, default True
            Whether to calculate and apply the Bose factor
        **kwargs
            Passes keyword arguments on to
            PhononData.calculate_structure_factor

        Returns
        -------
        sqw_map : (n_qpts, n_ebins) float ndarray
            The intensity for each q-point and energy bin
        """

        # Convert units
        freqs = self._freqs
        ebins = (ebins*ureg(self.freqs_unit).to('E_h')).magnitude

        # Create initial sqw_map with an extra an energy bin either side, for
        # any branches that fall outside the energy bin range
        sqw_map = np.zeros((self.n_qpts, len(ebins) + 1))
        sf = self.calculate_structure_factor(
            scattering_lengths, calc_bose=False, **kwargs)
        if calc_bose:
            if 'T' in kwargs:
                T = kwargs['T']
            else:
                T = 5.0
            p_intensity = sf*bose_factor(freqs, T)
            n_intensity = sf*bose_factor(-freqs, T)
        else:
            p_intensity = sf
            n_intensity = sf

        p_bin = np.digitize(freqs, ebins)
        n_bin = np.digitize(-freqs, ebins)

        # Sum intensities into bins
        first_index = np.transpose(
            np.tile(range(self.n_qpts), (3*self.crystal.n_atoms, 1)))
        np.add.at(sqw_map, (first_index, p_bin), p_intensity)
        np.add.at(sqw_map, (first_index, n_bin), n_intensity)
        sqw_map = sqw_map[:, 1:-1]  # Exclude values outside ebin range

        self._sqw_ebins = ebins
        self.sqw_map = sqw_map

        return sqw_map


    def calculate_dos(self, dos_bins, gwidth=0, lorentz=False, weights=None):
        """
        Calculates a density of states with fixed width Gaussian/Lorentzian
        broadening

        Parameters
        ----------
        dos_bins : (n_ebins + 1,) float ndarray
            The energy bin edges to use for calculating the DOS, in the same
            units as freqs
        gwidth : float, optional, default 0
            FWHM of Gaussian/Lorentzian for broadening the DOS bins, in the
            same units as freqs
        lorentz : boolean, optional
            Whether to use a Lorentzian or Gaussian broadening function.
            Default: False
        weights : (n_qpts, 3*n_atoms) float ndarray, optional
            The weights to use for each q-points and mode. If unspecified,
            uses the q-point weights stored in the Data object

        Returns
        -------
        dos : (n_ebins,) float ndarray
            The density of states for each bin
        """

        freqs = self._freqs
        # Convert dos_bins to Hartree. If no units are specified, assume
        # dos_bins is in same units as freqs
        try:
            dos_bins = dos_bins.to('E_h', 'spectroscopy').magnitude
        except AttributeError:
            dos_bins = (dos_bins*ureg(self.freqs_unit).to(
                'E_h', 'spectroscopy')).magnitude
        try:
            gwidth = gwidth.to('E_h', 'spectroscopy').magnitude
        except AttributeError:
            gwidth = (gwidth*ureg(self.freqs_unit).to(
                'E_h', 'spectroscopy')).magnitude

        # Bin frequencies
        if weights is None:
            weights = np.repeat(self.weights[:, np.newaxis],
                                3*self.crystal.n_atoms,
                                axis=1)
        hist, bin_edges = np.histogram(freqs, dos_bins, weights=weights)

        bwidth = np.mean(np.diff(dos_bins))
        # Only broaden if broadening is more than bin width
        if gwidth > bwidth:
            # Calculate broadening for adjacent nbin_broaden bins
            if lorentz:
                # 25 * Lorentzian FWHM
                nbin_broaden = int(math.floor(25.0*gwidth/bwidth))
                broadening = lorentzian(
                    np.arange(-nbin_broaden, nbin_broaden)*bwidth, gwidth)
            else:
                # 3 * Gaussian FWHM
                nbin_broaden = int(math.floor(3.0*gwidth/bwidth))
                sigma = gwidth/(2*math.sqrt(2*math.log(2)))
                broadening = gaussian(
                    np.arange(-nbin_broaden, nbin_broaden)*bwidth, sigma)

            if hist.size > 0:
                # Allow broadening beyond edge of bins
                dos = np.zeros(len(hist) + 2*nbin_broaden)
                for i, h in enumerate(hist):
                    # Broaden each hist bin value to adjacent bins
                    bhist = h*broadening
                    dos[i:i+2*nbin_broaden] += bhist
                # Slice dos array to same size as bins
                dos = dos[nbin_broaden:-nbin_broaden]
        else:
            dos = hist

        self.dos = dos
        self._dos_bins = dos_bins

        return dos

    @classmethod
    def from_dict(cls, d):
        crystal = Crystal.from_dict(d)
        for key in ['weights']:
            if not key in d.keys():
                d[key] = None
        d['freqs'] = d['freqs']*ureg(d['freqs_unit'])
        return cls(crystal, d['qpts'], d['freqs'], d['eigenvecs'],
                   d['weights'])

    @classmethod
    def from_castep(cls, seedname, path=''):
        """
        Reads precalculated phonon mode data from a CASTEP .phonon file

        Parameters
        ----------
        seedname : str
            Seedname of file(s) to read e.g. if seedname = 'quartz' then
            the 'quartz.phonon' file will be read
        path : str
            Path to dir containing the file(s), if in another directory
        """
        data = _castep._read_phonon_data(seedname, path)
        return cls.from_dict(data)

    @classmethod
    def from_phonopy(cls, path='.', phonon_name='band.yaml',
                     phonon_format=None, summary_name='phonopy.yaml'):
        """
        Reads precalculated phonon mode data from a Phonopy
        mesh/band/qpoints.yaml/hdf5 file. May also read from phonopy.yaml for
        structure information.

        Parameters
        ----------
        path : str, optional, default '.'
            Path to directory containing the file(s)
        phonon_name : str, optional, default 'band.yaml'
            Name of Phonopy file including the frequencies and eigenvectors
        phonon_format : {'yaml', 'hdf5'} str, optional, default None
            Format of the phonon_name file if it isn't obvious from the
            phonon_name extension
        summary_name : str, optional, default 'phonopy.yaml'
            Name of Phonopy summary file to read the crystal information from.
            Crystal information in the phonon_name file takes priority, but if
            it isn't present, crystal information is read from summary_name
            instead
        """
        data = _phonopy._read_phonon_data(path=path, phonon_name=phonon_name,
                            phonon_format=phonon_format, summary_name=summary_name)
        return cls.from_dict(data)
