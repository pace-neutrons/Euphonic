import math
import numpy as np
from euphonic import ureg
from euphonic.util import direction_changed, bose_factor, is_gamma
from euphonic.data.data import Data
from euphonic._readers import _castep


class PhononData(Data):
    """
    A class to read and store vibrational data from model (e.g. CASTEP) output
    files

    Attributes
    ----------
    seedname : str
        Seedname specifying file(s) to read from
    model : str
        Records what model the data came from
    n_ions : int
        Number of ions in the unit cell
    n_branches : int
        Number of phonon dispersion branches
    n_qpts : int
        Number of q-points in the .phonon file
    cell_vec : (3, 3) float ndarray
        The unit cell vectors. Default units Angstroms
    recip_vec : (3, 3) float ndarray
        The reciprocal lattice vectors. Default units inverse Angstroms
    ion_r : (n_ions, 3) float ndarray
        The fractional position of each ion within the unit cell
    ion_type : (n_ions,) string ndarray
        The chemical symbols of each ion in the unit cell. Ions are in the
        same order as in ion_r
    ion_mass : (n_ions,) float ndarray
        The mass of each ion in the unit cell in atomic units
    qpts : (n_qpts, 3) float ndarray
        Q-point coordinates
    weights : (n_qpts,) float ndarray
        The weight for each q-point
    freqs: (n_qpts, 3*n_ions) float ndarray
        Phonon frequencies, ordered according to increasing q-point
        number. Default units meV
    eigenvecs: (n_qpts, 3*n_ions, n_ions, 3) complex ndarray
        Dynamical matrix eigenvectors. Empty if read_eigenvecs is False
    split_i : (n_splits,) int ndarray
        The q-point indices where there is LO-TO splitting, if applicable.
        Otherwise empty.
    split_freqs : (n_splits, 3*n_ions) float ndarray
        Holds the additional LO-TO split phonon frequencies for the q-points
        specified in split_i. Empty if no LO-TO splitting. Default units meV
    split_eigenvecs : (n_splits, 3*n_ions, n_ions, 3) complex ndarray
        Holds the additional LO-TO split dynamical matrix eigenvectors for the
        q-points specified in split_i. Empty if no LO-TO splitting
    """

    def __init__(self, data):
        """
        Calls functions to read the correct file(s) and sets PhononData
        attributes

        Parameters
        ----------
        data : dict
            A dict containing the following keys: n_ions, n_branches, n_qpts,
            cell_vec, recip_vec, ion_r, ion_type, ion_mass, qpts, weights,
            freqs, eigenvecs, split_i, split_freqs, split_eigenvecs.
            meta :
                model:{'CASTEP'}
                    Which model has been used
                path : str, default ''
                    Location of seed files on filesystem
            meta (CASTEP) :
                seedname : str
                    Seedname of file that is read
        """
        if type(data) is str:
            raise Exception('The old interface is now replaced by',
                            'PhononData.from_castep(seedname, path="<path>").',
                            '(Please see documentation for more information.)')

        self._set_data(data)

        self._l_units = 'angstrom'
        self._e_units = 'meV'


    @property
    def cell_vec(self):
        return self._cell_vec*ureg('bohr').to(self._l_units)

    @property
    def recip_vec(self):
        return self._recip_vec*ureg('1/bohr').to('1/' + self._l_units)

    @property
    def ion_mass(self):
        return self._ion_mass*ureg('e_mass').to('amu')

    @property
    def freqs(self):
        return self._freqs*ureg('E_h').to(self._e_units, 'spectroscopy')

    @property
    def split_freqs(self):
        return self._split_freqs*ureg('E_h').to(self._e_units, 'spectroscopy')

    @property
    def sqw_ebins(self):
        return self._sqw_ebins*ureg('E_h').to(self._e_units, 'spectroscopy')

    @classmethod
    def from_castep(self, seedname, path=''):
        """
        Calls the CASTEP phonon data reader and sets the PhononData attributes.

        Parameters
        ----------
        seedname : str
            Seedname of file(s) to read e.g. if seedname = 'quartz' then
            the 'quartz.phonon' file will be read
        path : str
            Path to dir containing the file(s), if in another directory
        """
        data = _castep._read_phonon_data(seedname, path)
        return self(data)

    def _set_data(self, data):
        self.n_ions = data['n_ions']
        self.n_branches = data['n_branches']
        self.n_qpts = data['n_qpts']
        self._cell_vec = data['cell_vec']
        self._recip_vec = data['recip_vec']
        self.ion_r = data['ion_r']
        self.ion_type = data['ion_type']
        self._ion_mass = data['ion_mass']
        self.qpts = data['qpts']
        self.weights = data['weights']
        self._freqs = data['freqs']
        self.eigenvecs = data['eigenvecs']
        self.split_i = data['split_i']
        self._split_freqs = data['split_freqs']
        self.split_eigenvecs = data['split_eigenvecs']

        try:
            self.model = data['model']
            if data['model'].lower() == 'castep':
                self.seedname = data['seedname']
                self.model = data['model']
                self.path = data['path']
        except KeyError:
            pass

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
        n_branches = self.n_branches
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
                # products over ions (i.e. multiply eigenvectors elementwise,
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
        sf : (n_qpts, n_branches) float ndarray
            The structure factor for each q-point and phonon branch
        """
        sl = [scattering_lengths[x] for x in self.ion_type]

        # Convert units
        recip = self._recip_vec
        freqs = self._freqs
        ion_mass = self._ion_mass
        sl = (sl*ureg('fm').to('bohr')).magnitude

        # Calculate normalisation factor
        norm_factor = sl/np.sqrt(ion_mass)

        # Calculate the exponential factor for all ions and q-points
        # ion_r in fractional coords, so Qdotr = 2pi*qh*rx + 2pi*qk*ry...
        exp_factor = np.exp(1J*2*math.pi*np.einsum('ij,kj->ik',
                                                   self.qpts, self.ion_r))

        # Eigenvectors are in Cartesian so need to convert hkl to Cartesian by
        # computing the dot product with hkl and reciprocal lattice
        Q = np.einsum('ij,jk->ik', self.qpts, recip)

        # Calculate dot product of Q and eigenvectors for all branches, ions
        # and q-points
        eigenv_dot_q = np.einsum('ijkl,il->ijk', np.conj(self.eigenvecs), Q)

        # Calculate Debye-Waller factors
        if dw_data:
            if dw_data.n_ions != self.n_ions:
                raise Exception((
                    'The Data object used as dw_data is not compatible with the'
                    ' object that calculate_structure_factor has been called on'
                    ' (they have a different number of ions). Is dw_data '
                    'correct?'))
            dw = dw_data._dw_coeff(T)
            dw_factor = np.exp(-np.einsum('jkl,ik,il->ij', dw, Q, Q)/2)
            exp_factor *= dw_factor

        # Multiply Q.eigenvector, exp factor and normalisation factor
        term = np.einsum('ijk,ik,k->ij', eigenv_dot_q, exp_factor, norm_factor)

        # Take mod squared and divide by frequency to get intensity
        sf = np.absolute(term*np.conj(term))/np.absolute(freqs)

        # Multiply by Bose factor
        if calc_bose:
            sf = sf*bose_factor(freqs, T)

        sf = np.real(sf*scale)

        return sf

    def _dw_coeff(self, T):
        """
        Calculate the 3 x 3 Debye-Waller coefficients for each ion over the
        q-points contained in this object

        Parameters
        ----------
        T : float
            Temperature in Kelvin

        Returns
        -------
        dw : (n_ions, 3, 3) float ndarray
            The DW coefficients for each ion
        """

        # Convert units
        kB = (1*ureg.k).to('E_h/K').magnitude
        n_ions = self.n_ions
        ion_mass = self._ion_mass
        freqs = self._freqs
        qpts = self.qpts
        evecs = self.eigenvecs
        weights = self.weights

        mass_term = 1/(2*ion_mass)

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
        dw = np.zeros((n_ions, 3, 3))
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
        ebins = (ebins*ureg(self._e_units).to('E_h')).magnitude

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
            np.tile(range(self.n_qpts), (self.n_branches, 1)))
        np.add.at(sqw_map, (first_index, p_bin), p_intensity)
        np.add.at(sqw_map, (first_index, n_bin), n_intensity)
        sqw_map = sqw_map[:, 1:-1]  # Exclude values outside ebin range

        self._sqw_ebins = ebins
        self.sqw_map = sqw_map

        return sqw_map
