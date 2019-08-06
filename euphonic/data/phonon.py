import math
import numpy as np
from euphonic import ureg
from euphonic.util import direction_changed, bose_factor
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

    def __init__(self, seedname, model='CASTEP', path=''):
        """
        Calls functions to read the correct file(s) and sets PhononData
        attributes

        Parameters
        ----------
        seedname : str
            Seedname of file(s) to read
        model : {'CASTEP'}, optional, default 'CASTEP'
            Which model has been used. e.g. if seedname = 'quartz' and
            model='CASTEP', the 'quartz.phonon' file will be read
        path : str, optional
            Path to dir containing the file(s), if in another directory
        """
        self._get_data(seedname, model, path)
        self.seedname = seedname
        self.model = model

    def _get_data(self, seedname, model, path):
        """"
        Calls the correct reader to get the required data, and sets the
        PhononData attributes

        Parameters
        ----------
        seedname : str
            Seedname of file(s) to read
        model : {'CASTEP'}, optional, default 'CASTEP'
            Which model has been used. e.g. if seedname = 'quartz' and
            model='CASTEP', the 'quartz.phonon' file will be read
        path : str
            Path to dir containing the file(s), if in another directory
        """
        if model.lower() == 'castep':
            data = _castep._read_phonon_data(seedname, path)
        else:
            raise ValueError(
                "{:s} is not a valid model, please use one of {{'CASTEP'}}"
                .format(model))

        self.n_ions = data['n_ions']
        self.n_branches = data['n_branches']
        self.n_qpts = data['n_qpts']
        self.cell_vec = data['cell_vec']
        self.recip_vec = data['recip_vec']
        self.ion_r = data['ion_r']
        self.ion_type = data['ion_type']
        self.ion_mass = data['ion_mass']
        self.qpts = data['qpts']
        self.weights = data['weights']
        self.freqs = data['freqs']
        self.eigenvecs = data['eigenvecs']
        self.split_i = data['split_i']
        self.split_freqs = data['split_freqs']
        self.split_eigenvecs = data['split_eigenvecs']

    def reorder_freqs(self):
        """
        Reorders frequencies across q-points in order to join branches, and
        sets the freqs and eigenvecs attributes to the newly ordered
        frequencies
        """
        n_qpts = self.n_qpts
        n_branches = self.n_branches
        qpts = self.qpts
        freqs = self.freqs.magnitude
        eigenvecs = self.eigenvecs

        ordered_freqs = np.zeros(freqs.shape)
        ordered_eigenvecs = np.zeros(eigenvecs.shape, dtype=np.complex128)
        qmap = np.arange(n_branches)

        # Only calculate qmap and reorder freqs if the direction hasn't changed
        # and there is no LO-TO splitting
        calculate_qmap = np.concatenate(([True], np.logical_not(
            direction_changed(qpts))))
        if hasattr(self, 'split_i'):
            split_freqs = self.split_freqs.magnitude
            split_eigenvecs = self.split_eigenvecs
            ordered_split_freqs = np.zeros(split_freqs.shape)
            ordered_split_eigenvecs = np.zeros(
                split_eigenvecs.shape, dtype=np.complex128)
            calculate_qmap[self.split_i + 1] = False

        # Don't reorder first q-point
        ordered_freqs[0, :] = freqs[0, :]
        ordered_eigenvecs[0, :] = eigenvecs[0, :]
        prev_evecs = eigenvecs[0, :, :, :]
        for i in range(1, n_qpts):
            # Initialise q-point mapping for this q-point
            qmap_tmp = np.arange(n_branches)
            # Compare eigenvectors for each mode for this q-point with every
            # mode for the previous q-point
            # Explicitly broadcast arrays with repeat and tile to ensure
            # correct multiplication of modes
            curr_evecs = eigenvecs[i, :, :, :]
            current_eigenvecs = np.repeat(curr_evecs, n_branches, axis=0)
            prev_eigenvecs = np.tile(prev_evecs, (n_branches, 1, 1))

            if calculate_qmap[i-1]:
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

                # Find greates exp(-iqr)-weighted dot product
                for j in range(n_branches):
                    max_i = (np.argmax(dot_mat))
                    mode = int(max_i/n_branches)  # Modes are dot_mat rows
                    prev_mode = max_i % n_branches  # Prev q-pt modes are cols
                    # Ensure modes aren't mapped more than once
                    dot_mat[mode, :] = 0
                    dot_mat[:, prev_mode] = 0
                    qmap_tmp[mode] = prev_mode
            # Map q-points according to previous q-point mapping
            qmap = qmap[qmap_tmp]

            prev_evecs = curr_evecs

            # Reorder frequencies and eigenvectors
            ordered_eigenvecs[i, qmap] = eigenvecs[i, :]
            ordered_freqs[i, qmap] = freqs[i, :]

            if hasattr(self, 'split_i') and i in self.split_i:
                idx = np.where(i == self.split_i)
                ordered_split_eigenvecs[idx, qmap] = split_eigenvecs[idx]
                ordered_split_freqs[idx, qmap] = split_freqs[idx]

        ordered_freqs = ordered_freqs*self.freqs.units
        self.eigenvecs = ordered_eigenvecs
        self.freqs = ordered_freqs
        if hasattr(self, 'split_i'):
            self.split_freqs = ordered_split_freqs*self.split_freqs.units
            self.split_eigenvecs = ordered_split_eigenvecs

    def convert_e_units(self, units):
        """
        Convert energy units of relevant attributes in place e.g. freqs,
        dos_bins

        Parameters
        ----------
        units : str
            The units to convert to e.g. '1/cm', 'hartree', 'eV'
        """
        super(PhononData, self).convert_e_units(units)
        self.split_freqs.ito(units, 'spectroscopy')
        if hasattr(self, 'sqw_ebins'):
            self.sqw_ebins.ito(units, 'spectroscopy')

    def calculate_structure_factor(self, scattering_lengths, T=5.0, scale=1.0,
                                   calc_bose=True, dw_arg=None, **kwargs):
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
        dw_arg : string, optional, default None
            If set, will calculate the Debye-Waller factor over the q-points in
            the .phonon file with this seedname.
        **kwargs
            If dw_arg has been set, passes keyword arguments to initialisation
            of the PhononData object for the Debye-Waller calculation

        Returns
        -------
        sf : (n_qpts, n_branches) float ndarray
            The structure factor for each q-point and phonon branch
        """
        sl = [scattering_lengths[x] for x in self.ion_type]

        # Convert units (use magnitudes for performance)
        recip = (self.recip_vec.to('1/bohr')).magnitude
        freqs = (self.freqs.to('E_h', 'spectroscopy')).magnitude
        ion_mass = (self.ion_mass.to('e_mass')).magnitude
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
        if dw_arg:
            dw_data = self._get_dw_data(dw_arg, **kwargs)
            dw = self._dw_coeff(dw_data, T)
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

    def _get_dw_data(self, dw_seedname, **kwargs):
        """
        Returns the PhononData object containing the q-points over which to
        calculate the Debye-Waller factor. This function exists so it can be
        overidden by the InterpolationData object when calculating the DW
        factor on a grid

        Parameters
        ----------
        dw_seedname : string
            The seedname of the PhononData object to create
        **kwargs
            Get passed to the PhononData initialisation
        Returns
        -------
        dw_data : PhononData
            The PhononData object with dw_seedname
        """
        return PhononData(dw_seedname, **kwargs)

    def _dw_coeff(self, data, T):
        """
        Calculate the 3 x 3 Debye-Waller coefficients for each ion

        Parameters
        ----------
        data : PhononData object
            PhononData object containing the q-point grid to calculate the DW
            factor over
        T : float
            Temperature in Kelvin

        Returns
        -------
        dw : (n_ions, 3, 3) float ndarray
            The DW coefficients for each ion
        """

        # Convert units (use magnitudes for performance)
        kB = (1*ureg.k).to('E_h/K').magnitude
        ion_mass = data.ion_mass.to('e_mass').magnitude
        freqs = data.freqs.to('E_h', 'spectroscopy').magnitude
        qpts = data.qpts
        evecs = data.eigenvecs
        weights = data.weights

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
        dw = np.zeros((data.n_ions, 3, 3))
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

    def calculate_sqw_map(self, scattering_lengths, ebins, set_attrs=True,
                          calc_bose=True, **kwargs):
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
        set_attrs : boolean, optional, default True
            Whether to set the sqw and sqw_ebins attributes of this object
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

        # Convert units (use magnitudes for performance)
        freqs = (self.freqs.to('E_h', 'spectroscopy')).magnitude
        ebins = ((ebins*self.freqs.units).to('E_h')).magnitude

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

        if set_attrs:
            self.sqw_ebins = ebins*ureg('E_h').to(
                self.freqs.units, 'spectroscopy')
            self.sqw_map = sqw_map

        return sqw_map
