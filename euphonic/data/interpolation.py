import math
import sys
import warnings
import numpy as np
import scipy
from scipy.linalg.lapack import zheev
from scipy.special import erfc
from euphonic import ureg
from euphonic.util import is_gamma, mp_grid
from euphonic.data.phonon import PhononData
from euphonic._readers import _castep


class InterpolationData(PhononData):
    """
    Extends PhononData. A class to read and store the data required for a
    phonon interpolation calculation from model (e.g. CASTEP) output, and
    calculate phonon frequencies/eigenvectors at arbitrary q-points via
    Fourier interpolation

    Attributes
    ----------
    seedname : str
        Seedname specifying castep_bin file to read from
    model : str
        Records what model the data came from
    n_ions : int
        Number of ions in the unit cell
    n_branches : int
        Number of phonon dispersion branches
    cell_vec : (3, 3) float ndarray
        The unit cell vectors. Default units Angstroms
    recip_vec : (3, 3) float ndarray
        The reciprocal lattice vectors. Default units inverse Angstroms
    ion_r : (n_ions,3) float ndarray
        The fractional position of each ion within the unit cell
    ion_type : (n_ions,) string ndarray
        The chemical symbols of each ion in the unit cell. Ions are in the
        same order as in ion_r
    ion_mass : (n_ions,) float ndarray
        The mass of each ion in the unit cell in atomic units
    n_cells_in_sc : int
        Number of cells in the supercell
    sc_matrix : (3, 3) int ndarray
        The supercell matrix
    cell_origins : (n_cells_in_sc, 3) int ndarray
        The locations of the unit cells within the supercell
    force_constants : (n_cells_in_sc, 3*n_ions, 3*n_ions) float ndarray
        Force constants matrix. Default units atomic units
    n_qpts : int
        Number of q-points used in the most recent interpolation calculation.
        Default value 0
    qpts : (n_qpts, 3) float ndarray
        Coordinates of the q-points used for the most recent interpolation
        calculation. Is empty by default
    weights : (n_qpts,) float ndarray
        The weight for each q-point
    freqs: (n_qpts, 3*n_ions) float ndarray
        Phonon frequencies from the most recent interpolation calculation.
        Default units meV. Is empty by default
    eigenvecs: (n_qpts, 3*n_ions, n_ions, 3) complex ndarray
        Dynamical matrix eigenvectors from the most recent interpolation
        calculation. Is empty by default
    asr : str
        Stores which the acoustic sum rule, if any, was used in the last phonon
        calculation. Ensures consistency of other calculations e.g. when
        calculating on a grid of phonons for the Debye-Waller factor
    dipole : boolean
        Stores whether the Ewald dipole tail correction was used in the last
        phonon calculation. Ensures consistency of other calculations e.g.
        when calculating on a grid of phonons for the Debye-Waller factor
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
        Calls functions to read the correct file(s) and sets InterpolationData
        attributes, additionally can calculate frequencies/eigenvectors at
        specified q-points

        Parameters
        ----------
        data : dict
            A dict containing the following keys: n_ions, n_branches, cell_vec,
            ion_r, ion_type, ion_mass, force_constants, sc_matrix,
            n_cells_in_sc, cell_origins, and optional : born, dielectric
            meta :
                model : {'CASTEP'}
                    Which model has been used
                path : str, default ''
                    Location of seed files on filesystem
            meta (CASTEP) :
                seedname : str
                    Seedname of file that is read
        """
        if type(data) is str:
            raise Exception((
                'The old interface now takes the form:'
                'InterpolationData.from_castep(seedname, path="<path>").'
                '(Please see documentation for more information.)'))

        self._set_data(data)

        self.n_qpts = 0
        self.qpts = np.empty((0, 3))

        self._reduced_freqs = np.empty((0, 3*self.n_ions))
        self._reduced_eigenvecs = np.empty((0, 3*self.n_ions, self.n_ions, 3),
                                           dtype=np.complex128)

        self.split_i = np.empty((0,), dtype=np.int32)
        self._split_freqs = np.empty((0, 3*self.n_ions))
        self.split_eigenvecs = np.empty((0, 3*self.n_ions, self.n_ions, 3),
                                        dtype=np.complex128)

        self._l_units = 'angstrom'
        self._e_units = 'meV'


    @property
    def _freqs(self):
        if len(self._reduced_freqs) > 0:
            return self._reduced_freqs[self._qpts_i]
        else:
            return self._reduced_freqs

    @property
    def eigenvecs(self):
        if len(self._reduced_eigenvecs) > 0:
            return self._reduced_eigenvecs[self._qpts_i]
        else:
            return self._reduced_eigenvecs

    @property
    def force_constants(self):
        return self._force_constants*ureg('hartree/bohr**2')

    @property
    def born(self):
        return self._born*ureg('e')

    @classmethod
    def from_castep(self, seedname, path=''):
        """
        Calls the CASTEP interpolation data reader and sets the InerpolationData attributes.

        Parameters
        ----------
        seedname : str
            Seedname of file(s) to read, e.g. if seedname = 'quartz'
            the 'quartz.castep_bin' file will be read
        path : str, optional
            Path to dir containing the file(s), if in another directory
        """
        data = _castep._read_interpolation_data(seedname, path)
        return self(data)

    def _set_data(self, data):
        self.n_ions = data['n_ions']
        self.n_branches = data['n_branches']
        self._cell_vec = data['cell_vec']
        self._recip_vec = data['recip_vec']
        self.ion_r = data['ion_r']
        self.ion_type = data['ion_type']
        self._ion_mass = data['ion_mass']
        self._force_constants = data['force_constants']
        self.sc_matrix = data['sc_matrix']
        self.n_cells_in_sc = data['n_cells_in_sc']
        self.cell_origins = data['cell_origins']

        try:
            self._born = data['born']
            self.dielectric = data['dielectric']
        except KeyError:
            pass

        try:
            self.model = data['model']
            if data['model'].lower() == 'castep':
                self.seedname = data['seedname']
                self.model = data['model']
                self.path = data['path']
        except KeyError:
            pass


    def calculate_fine_phonons(
        self, qpts, asr=None, precondition=False, dipole=True,
            eta_scale=1.0, splitting=True, reduce_qpts=True, use_c=False,
            n_threads=1):
        """
        Calculate phonon frequencies and eigenvectors at specified q-points
        from a supercell force constant matrix via interpolation. For more
        information on the method see section 2.5:
        http://www.tcm.phy.cam.ac.uk/castep/Phonons_Guide/Castep_Phonons.html

        Parameters
        ----------
        qpts : (n_qpts, 3) float ndarray
            The q-points to interpolate onto
        asr : {'realspace', 'reciprocal'}, optional, default None
            Which acoustic sum rule correction to apply. 'realspace' applies
            the correction to the force constant matrix in real space.
            'reciprocal' applies the correction to the dynamical matrix at
            every q-point
        dipole : boolean, optional, default True
            Calculates the dipole tail correction to the dynamical matrix at
            each q-point using the Ewald sum, if the Born charges and
            dielectric permitivitty tensor are present.
        eta_scale : float, optional, default 1.0
            Changes the cutoff in real/reciprocal space for the dipole Ewald
            sum. A higher value uses more reciprocal terms
        splitting : boolean, optional, default True
            Whether to calculate the LO-TO splitting at the gamma points. Only
            applied if dipole is True and the Born charges and dielectric
            permitivitty tensor are present.
        reduce_qpts : boolean, optional, default False
            Whether to use periodicity to reduce all q-points and only
            calculate for unique q-points within the 1st BZ. This won't change
            the output but could increase performance.
        use_c : boolean, optional, default False
            Whether to use C instead of Python to calculate and diagonalise
            the dynamical matrix
        n_threads : int, optional, default 1
            The number of threads to use when looping over q-points in C. Only
            applicable if use_c=True

        Returns
        -------
        freqs : (n_qpts, 3*n_ions) float ndarray
            The phonon frequencies (same as set to InterpolationData.freqs)
        eigenvecs : (n_qpts, 3*n_ions, n_ions, 3) complex ndarray
            The phonon eigenvectors (same as set to
            InterpolationData.eigenvecs)
        """

        # Reset obj freqs/eigenvecs to zero to reduce memory usage in case of
        # repeated calls to calculate_fine_phonons
        n_ions = self.n_ions
        self.qpts = np.array([])
        self._reduced_freqs = np.empty((0, 3*n_ions))
        self._reduced_eigenvecs = np.empty((0, 3*n_ions, n_ions, 3),
                                           dtype=np.complex128)

        self.split_i = np.empty((0,), dtype=np.int32)
        self._split_freqs = np.empty((0, 3*n_ions))
        self.split_eigenvecs = np.empty((0, 3*n_ions, n_ions, 3),
                                        dtype=np.complex128)

        # Set default splitting params
        if not hasattr(self, 'born') or not hasattr(self, 'dielectric'):
            dipole = False
        if not dipole:
            splitting = False

        if reduce_qpts:
            norm_qpts = qpts - np.rint(qpts)
            # Ensure gamma points are exactly zero, otherwise you may have a
            # case where small fp differences mean np.unique doesn't reduce
            # them, yet they're all classified as gamma points. This causes
            # indexing errors later when calculating q-directions as there are
            # then points in reduced_qpts whose index isn't in qpts_i
            gamma_i = np.where(is_gamma(qpts))[0]
            n_gamma = len(gamma_i)
            norm_qpts[gamma_i] = 0.

            reduced_qpts, qpts_i = np.unique(norm_qpts, return_inverse=True,
                                             axis=0)
            n_rqpts = len(reduced_qpts)
            # Special handling of gamma points - don't reduce gamma points if
            # LO-TO splitting
            if splitting and n_gamma > 1:
                # Replace any gamma points and their indices with new gamma
                # points appended onto the reduced q-point array, so each
                # gamma can have its own splitting
                qpts_i[gamma_i[1:]] = range(n_rqpts, n_rqpts + n_gamma - 1)
                reduced_qpts = np.append(reduced_qpts,
                                         np.tile(np.array([0., 0., 0.,]),
                                                 (n_gamma - 1, 1)),
                                         axis=0)
                n_rqpts = len(reduced_qpts)
        else:
            reduced_qpts = qpts
            qpts_i = np.arange(0, len(qpts), dtype=np.int32)
            n_rqpts = len(qpts)

        lim = 2  # Supercell image limit
        # Construct list of supercell ion images
        if not hasattr(self, 'sc_image_i'):
            self._calculate_supercell_images(lim)

        # Get a list of all the unique supercell image origins and cell origins
        # in x, y, z and how to rebuild them to minimise expensive phase
        # calculations later
        sc_image_r = self._get_all_origins(
            np.repeat(lim, 3) + 1, min_xyz=-np.repeat(lim, 3))
        sc_offsets = np.einsum('ji,kj->ki', self.sc_matrix,
                               sc_image_r).astype(np.int32)
        unique_sc_offsets = [[] for i in range(3)]
        unique_sc_i = np.zeros((len(sc_offsets), 3), dtype=np.int32)
        unique_cell_origins = [[] for i in range(3)]
        unique_cell_i = np.zeros((len(self.cell_origins), 3), dtype=np.int32)
        for i in range(3):
            unique_sc_offsets[i], unique_sc_i[:, i] = np.unique(
                sc_offsets[:, i], return_inverse=True)
            unique_cell_origins[i], unique_cell_i[:, i] = np.unique(
                self.cell_origins[:, i], return_inverse=True)

        # Precompute dynamical matrix mass weighting
        masses = np.tile(np.repeat(self._ion_mass, 3), (3*n_ions, 1))
        dyn_mat_weighting = 1/np.sqrt(masses*np.transpose(masses))

        # Initialise dipole correction calculation to FC matrix if required
        if dipole and (not hasattr(self, 'eta_scale') or
                       eta_scale != self._eta_scale):
            self._dipole_correction_init(eta_scale)

        if asr == 'realspace':
            if not hasattr(self, '_force_constants_asr'):
                self._force_constants_asr = self._enforce_realspace_asr()
            force_constants = self._force_constants_asr
        else:
            force_constants = self._force_constants
        # Precompute fc matrix weighted by number of supercell ion images
        # (for cumulant method)
        n_sc_images_repeat = (self._n_sc_images.
            repeat(3, axis=2).repeat(3, axis=1))
        fc_img_weighted = np.divide(
            force_constants, n_sc_images_repeat, where=n_sc_images_repeat != 0)

        recip_asr_correction = np.array([], dtype=np.complex128)
        if asr == 'reciprocal':
            # Calculate dyn mat at gamma for reciprocal ASR
            q_gamma = np.array([0., 0., 0.])
            dyn_mat_gamma = self._calculate_dyn_mat(
                q_gamma, fc_img_weighted, unique_sc_offsets,
                unique_sc_i, unique_cell_origins, unique_cell_i)
            if dipole:
                dyn_mat_gamma += self._calculate_dipole_correction(q_gamma)
            recip_asr_correction = self._enforce_reciprocal_asr(dyn_mat_gamma)
            if len(recip_asr_correction) == 0:
                # Finding acoustic modes failed
                asr = None

        split_i = np.empty((0,), dtype=np.int32)
        split_freqs = np.empty((0, 3*n_ions))
        split_eigenvecs = np.empty((0, 3*n_ions, n_ions, 3),
                                   dtype=np.complex128)
        rfreqs = np.zeros((n_rqpts, 3*n_ions))
        reigenvecs = np.zeros((n_rqpts, 3*n_ions, n_ions, 3),
                                  dtype=np.complex128)
        try:
            if use_c:
                try:
                    import euphonic._euphonic as euphonic_c
                    from euphonic.util import (_ensure_contiguous_args,
                                               _ensure_contiguous_attrs)
                except ImportError:
                    warnings.warn(('use_c=True is set, but the Euphonic\'s C '
                                   'extension couldn\'t be imported, it may '
                                   'not have been installed. Falling back to '
                                   'pure Python calculation'), stacklevel=2)
                    raise
            else:
                raise ImportError
            if splitting:
                gamma_i = np.where(is_gamma(qpts))[0]
                split_i = gamma_i[np.logical_and(gamma_i > 0,
                                                 gamma_i < len(qpts))]
                split_freqs = np.zeros((len(split_i), 3*n_ions))
                split_eigenvecs = np.zeros((len(split_i), 3*n_ions, n_ions, 3),
                                           dtype=np.complex128)
            # Make sure all arrays are contiguous before calling C
            (reduced_qpts, qpts_i, fc_img_weighted, sc_offsets, recip_asr_correction,
                dyn_mat_weighting, rfreqs, reigenvecs, split_freqs, split_eigenvecs,
                split_i) = _ensure_contiguous_args(
                    reduced_qpts, qpts_i, fc_img_weighted, sc_offsets,
                    recip_asr_correction, dyn_mat_weighting, rfreqs,
                    reigenvecs, split_freqs, split_eigenvecs, split_i)
            attrs = ['_n_sc_images', '_sc_image_i', 'cell_origins']
            dipole_attrs = ['_cell_vec', '_recip_vec', 'ion_r', '_born',
                            'dielectric', '_H_ab', '_cells', '_gvec_phases',
                            '_gvecs_cart', '_dipole_q0']
            _ensure_contiguous_attrs(self, attrs, opt_attrs=dipole_attrs)
            reciprocal_asr = 1 if asr == 'reciprocal' else 0
            euphonic_c.calculate_phonons(
                self, reduced_qpts, qpts_i, fc_img_weighted, sc_offsets,
                recip_asr_correction, dyn_mat_weighting, dipole,
                reciprocal_asr, splitting, rfreqs, reigenvecs, split_freqs,
                split_eigenvecs, split_i, n_threads, scipy.__path__[0])
        except ImportError:
            q_independent_args = (
                reduced_qpts, qpts_i, fc_img_weighted, unique_sc_offsets,
                unique_sc_i, unique_cell_origins, unique_cell_i,
                recip_asr_correction, dyn_mat_weighting, dipole, asr,
                splitting)
            for q in range(n_rqpts):
                rfreqs[q], reigenvecs[q], sfreqs, sevecs, si = \
                self._calculate_phonons_at_q(q, q_independent_args)
                if len(sfreqs) > 0:
                    split_i = np.concatenate((split_i, si))
                    split_freqs = np.concatenate((split_freqs, sfreqs))
                    split_eigenvecs = np.concatenate(
                        (split_eigenvecs, sevecs))

        self.asr = asr
        self.dipole = dipole
        self.qpts = qpts
        self.n_qpts = len(qpts)
        self.weights = np.full(len(qpts), 1.0/len(qpts))
        self._reduced_freqs = rfreqs
        self._reduced_eigenvecs = reigenvecs
        self._reduced_qpts = reduced_qpts
        self._qpts_i = qpts_i

        self.split_i = split_i
        self._split_freqs = split_freqs
        self.split_eigenvecs = split_eigenvecs

        return self.freqs, self.eigenvecs

    def _calculate_phonons_at_q(self, q, args):
        """
        Given a q-point and some precalculated q-independent values, calculate
        and diagonalise the dynamical matrix and return the frequencies and
        eigenvalues. Optionally also includes the Ewald dipole sum correction
        and LO-TO splitting
        """
        (reduced_qpts, qpts_i, fc_img_weighted, unique_sc_offsets,
         unique_sc_i, unique_cell_origins, unique_cell_i,
         recip_asr_correction, dyn_mat_weighting, dipole, asr,
         splitting) = args

        qpt = reduced_qpts[q]
        n_ions = self.n_ions

        dyn_mat = self._calculate_dyn_mat(
            qpt, fc_img_weighted, unique_sc_offsets, unique_sc_i,
            unique_cell_origins, unique_cell_i)

        if dipole:
            dipole_corr = self._calculate_dipole_correction(qpt)
            dyn_mat += dipole_corr

        if asr == 'reciprocal':
            dyn_mat += recip_asr_correction

        # Calculate LO-TO splitting by calculating non-analytic correction
        # to dynamical matrix
        if splitting and is_gamma(qpt):
            # If first q-point
            if qpts_i[0] == q:
                q_dirs = [reduced_qpts[qpts_i[1]]]
            # If last q-point
            elif qpts_i[-1] == q:
                q_dirs = [reduced_qpts[qpts_i[-2]]]
            else:
                # Find position in original qpts array (non reduced)
                qpos = np.where(qpts_i==q)[0][0]
                q_dirs = [-reduced_qpts[qpts_i[qpos - 1]],
                           reduced_qpts[qpts_i[qpos + 1]]]
            na_corrs = np.zeros((len(q_dirs), 3*n_ions, 3*n_ions),
                                dtype=np.complex128)
            for i, q_dir in enumerate(q_dirs):
                na_corrs[i] = self._calculate_gamma_correction(q_dir)
        else:
            # Correction is zero if not a gamma point or splitting = False
            na_corrs = np.array([0])

        split_i = np.empty((0,), dtype=np.int32)
        sfreqs = np.empty((0, 3*n_ions))
        sevecs = np.empty((0, 3*n_ions, n_ions, 3))
        for i, na_corr in enumerate(na_corrs):
            dyn_mat_corr = dyn_mat + na_corr

            # Mass weight dynamical matrix
            dyn_mat_corr *= dyn_mat_weighting

            try:
                evals, evecs = np.linalg.eigh(dyn_mat_corr)
            # Fall back to zheev if eigh fails (eigh calls zheevd)
            except np.linalg.LinAlgError:
                evals, evecs, info = zheev(dyn_mat_corr)
            evecs = np.reshape(np.transpose(evecs),
                               (3*n_ions, n_ions, 3))
            # Set imaginary frequencies to negative
            imag_freqs = np.where(evals < 0)
            evals = np.sqrt(np.abs(evals))
            evals[imag_freqs] *= -1

            if i == 0:
                freqs = evals
                eigenvecs = evecs
            else:
                split_i = np.concatenate((split_i, [np.where(qpts_i==q)[0][0]]))
                sfreqs = np.concatenate((sfreqs, [evals]))
                sevecs = np.concatenate((sevecs, [evecs]))

        return freqs, eigenvecs, sfreqs, sevecs, split_i

    def _calculate_dyn_mat(self, q, fc_img_weighted, unique_sc_offsets,
                           unique_sc_i, unique_cell_origins, unique_cell_i):
        """
        Calculate the non mass weighted dynamical matrix at a specified
        q-point from the image weighted force constants matrix and the indices
        specifying the periodic images. See eq. 1.5:
        http://www.tcm.phy.cam.ac.uk/castep/Phonons_Guide/Castep_Phonons.html

        Parameters
        ----------
        q : (3,) float ndarray
            The q-point to calculate the correction for
        fc_img_weighted : (n_cells_in_sc, 3*n_ions, 3*n_ions) float ndarray
            The force constants matrix weighted by the number of supercell ion
            images for each ij displacement
        unique_sc_offsets : list of lists of ints
            A list containing 3 lists of the unique supercell image offsets in
            each direction. The supercell offset is calculated by multiplying
            the supercell matrix by the supercell image indices (obtained by
            _get_all_origins()). A list of lists rather than a
            Numpy array is used as the 3 lists are independent and their size
            is not known beforehand
        unique_sc_i : ((2*lim + 1)**3, 3) int ndarray
            The indices needed to reconstruct sc_offsets from the unique
            values in unique_sc_offsets
        unique_cell_origins : list of lists of ints
            A list containing 3 lists of the unique cell origins in each
            direction. A list of lists rather than a Numpy array is used as
            the 3 lists are independent and their size is not known beforehand
        unique_sc_i : (cell_origins, 3) int ndarray
            The indices needed to reconstruct cell_origins from the unique
            values in unique_cell_origins

        Returns
        -------
        dyn_mat : (3*n_ions, 3*n_ions) complex ndarray
            The non mass weighted dynamical matrix at q
        """

        n_ions = self.n_ions
        sc_image_i = self._sc_image_i
        dyn_mat = np.zeros((n_ions*3, n_ions*3), dtype=np.complex128)

        # Cumulant method: for each ij ion-ion displacement sum phases for
        # all possible supercell images, then multiply by the cell phases
        # to account for j ions in different cells. Then multiply by the
        # image weighted fc matrix for each 3 x 3 ij displacement

        # Make sc_phases 1 longer than necessary, so when summing phases for
        # supercell images if there is no image, an index of -1 and hence
        # phase of zero can be used
        sc_phases = np.zeros(len(unique_sc_i) + 1, dtype=np.complex128)
        sc_phases[:-1], cell_phases = self._calculate_phases(
            q, unique_sc_offsets, unique_sc_i, unique_cell_origins,
            unique_cell_i)
        sc_phase_sum = np.sum(sc_phases[sc_image_i],
                              axis=3)
        ax = np.newaxis
        ij_phases = cell_phases[:, ax, ax]*sc_phase_sum
        full_dyn_mat = fc_img_weighted*(
            ij_phases.repeat(3, axis=2).repeat(3, axis=1))
        dyn_mat = np.sum(full_dyn_mat, axis=0)

        return dyn_mat

    def _dipole_correction_init(self, eta_scale=1.0):
        """
        Calculate the q-independent parts of the long range correction to the
        dynamical matrix for efficiency. The method used is based on the
        Ewald sum, see eqs 72-74 from Gonze and Lee PRB 55, 10355 (1997)

        Parameters
        ----------
        eta_scale : float, optional, default 1.0
            Changes the cutoff in real/reciprocal space for the dipole Ewald
            sum. A higher value uses more reciprocal terms
        """

        cell_vec = self._cell_vec
        recip = self._recip_vec
        n_ions = self.n_ions
        ion_r = self.ion_r
        born = self._born
        dielectric = self.dielectric
        inv_dielectric = np.linalg.inv(dielectric)
        sqrt_pi = math.sqrt(math.pi)

        # Calculate real/recip weighting
        abc_mag = np.linalg.norm(cell_vec, axis=1)
        mean_abc_mag = np.prod(abc_mag)**(1.0/3)
        eta = (sqrt_pi/mean_abc_mag)*n_ions**(1.0/6)
        # Use eta = lambda * |permittivity|**(1/6)
        eta = eta*np.power(np.linalg.det(dielectric), 1.0/6)*eta_scale
        eta_2 = eta**2

        # Set limits and tolerances
        max_shells = 50
        frac_tol = 1e-15

        # Calculate q=0 real space term
        real_q0 = np.zeros((n_ions, n_ions, 3, 3))
        # No. of independent i, j ion entries (to use i, j symmetry to
        # minimise size of stored H_ab)
        n_elems = np.sum(range(1, n_ions + 1))
        H_ab = np.zeros((0, n_elems, 3, 3))
        cells = np.zeros((0, 3))
        ion_r_cart = np.einsum('ij,jk->ik', ion_r, cell_vec)
        ion_r_e = np.einsum('ij,jk->ik', ion_r_cart, inv_dielectric)
        for n in range(max_shells):
            cells_tmp = self._get_shell_origins(n)
            cells_cart = np.einsum('ij,jk->ik', cells_tmp, cell_vec)
            cells_e = np.einsum(
                'ij,jk->ik', cells_cart, inv_dielectric)
            H_ab_tmp = np.zeros((len(cells_tmp), n_elems, 3, 3))
            for i in range(n_ions):
                idx = np.sum(range(n_ions - i, n_ions), dtype=np.int32)
                for j in range(i, n_ions):
                    if n == 0 and i == j:
                        continue
                    rij_cart = ion_r_cart[i] - ion_r_cart[j]
                    rij_e = ion_r_e[i] - ion_r_e[j]
                    diffs = rij_cart - cells_cart
                    deltas = rij_e - cells_e
                    norms_2 = np.einsum('ij,ij->i', deltas, diffs)*eta_2
                    norms = np.sqrt(norms_2)

                    # Calculate H_ab
                    exp_term = 2*np.exp(-norms_2)/(sqrt_pi*norms_2)
                    erfc_term = erfc(norms)/(norms*norms_2)
                    f1 = eta_2*(3*erfc_term/norms_2 + exp_term*(3/norms_2 + 2))
                    f2 = erfc_term + exp_term
                    deltas_ab = np.einsum('ij,ik->ijk', deltas, deltas)
                    H_ab_tmp[:, idx + j] = (
                        np.einsum('i,ijk->ijk', f1, deltas_ab)
                        - np.einsum('i,jk->ijk', f2, inv_dielectric))
            # End series when current terms are less than the fractional
            # tolerance multiplied by the term for the cell at R=0
            if n == 0:
                r0_max = np.amax(np.abs(H_ab_tmp))
            if np.amax(np.abs(H_ab_tmp)) > frac_tol*r0_max:
                H_ab = np.concatenate((H_ab, H_ab_tmp))
                cells = np.concatenate((cells, cells_tmp))
            else:
                break
        # Use compact H_ab to fill in upper triangular of the realspace term
        real_q0[np.triu_indices(n_ions)] = np.sum(H_ab, axis=0)
        real_q0 *= eta**3/math.sqrt(np.linalg.det(dielectric))

        # Calculate the q=0 reciprocal term
        recip_q0 = np.zeros((n_ions, n_ions, 3, 3), dtype=np.complex128)
        # Add G = 0 vectors to list, for later calculations when q !=0,
        # but don't calculate for q=0
        gvecs_cart = np.array([[0., 0., 0.]])
        gvec_phases = np.tile([1. + 0.j], (1, n_ions))
        for n in range(1, max_shells):
            gvecs = self._get_shell_origins(n)
            gvecs_cart_tmp = np.einsum('ij,jk->ik', gvecs, recip)
            gvec_dot_r = np.einsum('ij,kj->ik', gvecs, ion_r)
            gvec_phases_tmp = np.exp(2j*math.pi*gvec_dot_r)
            gvecs_ab = np.einsum('ij,ik->ijk', gvecs_cart_tmp, gvecs_cart_tmp)
            k_len_2 = np.einsum('ijk,jk->i', gvecs_ab, dielectric)/(4*eta_2)
            recip_exp = np.exp(-k_len_2)/k_len_2
            recip_q0_tmp = np.zeros((n_ions, n_ions, 3, 3),
                                    dtype=np.complex128)
            for i in range(n_ions):
                for j in range(i, n_ions):
                    phase_exp = gvec_phases_tmp[:, i]/gvec_phases_tmp[:, j]
                    recip_q0_tmp[i, j] = np.einsum(
                        'ijk,i,i->jk', gvecs_ab, phase_exp, recip_exp)
            # End series when current terms are less than the fractional
            # tolerance multiplied by the max term for the first shell
            if n == 1:
                first_shell_max = np.amax(np.abs(recip_q0_tmp))
            if np.amax(np.abs(recip_q0_tmp)) > frac_tol*first_shell_max:
                gvecs_cart = np.concatenate((gvecs_cart, gvecs_cart_tmp))
                gvec_phases = np.concatenate((gvec_phases, gvec_phases_tmp))
                recip_q0 += recip_q0_tmp
            else:
                break
        cell_volume = np.dot(cell_vec[0], np.cross(cell_vec[1], cell_vec[2]))
        recip_q0 *= math.pi/(cell_volume*eta_2)

        # Fill in remaining entries by symmetry
        for i in range(1, n_ions):
            for j in range(i):
                real_q0[i, j] = np.conj(real_q0[j, i])
                recip_q0[i, j] = np.conj(recip_q0[j, i])

        # Calculate the q=0 correction, to be subtracted from the corrected
        # diagonal at each q
        dipole_q0 = np.zeros((n_ions, 3, 3), dtype=np.complex128)
        for i in range(n_ions):
            for j in range(n_ions):
                for a in range(3):
                    for b in range(3):
                        dipole_q0[i, a, b] += np.sum(
                            np.einsum('i,j', born[i, a, :], born[j, b, :])
                            *(recip_q0[i, j] - real_q0[i, j]))
            # Symmetrise 3x3
            dipole_q0[i] = 0.5*(dipole_q0[i] + np.transpose(dipole_q0[i]))

        self._eta_scale = eta_scale
        self._eta = eta
        self._H_ab = H_ab
        self._cells = cells
        self._gvecs_cart = gvecs_cart
        self._gvec_phases = gvec_phases
        self._dipole_q0 = dipole_q0


    def _calculate_dipole_correction(self, q):
        """
        Calculate the long range correction to the dynamical matrix using the
        Ewald sum, see eqs 72-74 from Gonze and Lee PRB 55, 10355 (1997)

        Parameters
        ----------
        q : (3,) float ndarray
            The q-point to calculate the correction for

        Returns
        -------
        corr : (3*n_ions, 3*n_ions) complex ndarray
            The correction to the dynamical matrix
        """
        cell_vec = self._cell_vec
        recip = self._recip_vec
        n_ions = self.n_ions
        ion_r = self.ion_r
        born = self._born
        dielectric = self.dielectric
        eta = self._eta
        eta_2 = eta**2
        H_ab = self._H_ab
        cells = self._cells
        q_norm = q - np.rint(q)  # Normalised q-pt

        # Don't include G=0 vector if q=0
        if is_gamma(q_norm):
            gvec_phases = self._gvec_phases[1:]
            gvecs_cart = self._gvecs_cart[1:]
        else:
            gvec_phases = self._gvec_phases
            gvecs_cart = self._gvecs_cart

        # Calculate real space term
        real_dipole = np.zeros((n_ions, n_ions, 3, 3), dtype=np.complex128)
        # Calculate real space phase factor
        q_dot_ra = np.einsum('i,ji->j', q_norm, cells)
        real_phases = np.exp(2j*math.pi*q_dot_ra)
        real_dipole_tmp = np.einsum('i,ijkl->jkl', real_phases, H_ab)
        idx_u = np.triu_indices(n_ions)
        real_dipole[idx_u] = real_dipole_tmp
        real_dipole *= eta**3/math.sqrt(np.linalg.det(dielectric))

        # Calculate reciprocal term
        recip_dipole = np.zeros((n_ions, n_ions, 3, 3), dtype=np.complex128)
        # Calculate q-point phases
        q_dot_r = np.einsum('i,ji->j', q_norm, ion_r)
        q_phases = np.exp(2j*math.pi*q_dot_r)
        q_cart = np.dot(q_norm, recip)
        # Calculate k-vector symmetric matrix
        kvecs = gvecs_cart + q_cart
        kvecs_ab = np.einsum('ij,ik->ijk', kvecs, kvecs)
        k_len_2 = np.einsum('ijk,jk->i', kvecs_ab, dielectric)/(4*eta_2)
        recip_exp = np.einsum('ijk,i->ijk', kvecs_ab, np.exp(-k_len_2)/k_len_2)
        for i in range(n_ions):
                phase_exp = ((gvec_phases[:, i, None]*q_phases[i])
                             /(gvec_phases[:, i:]*q_phases[i:]))
                recip_dipole[i, i:] = np.einsum(
                    'ikl,ij->jkl', recip_exp, phase_exp)
        cell_volume = np.dot(cell_vec[0], np.cross(cell_vec[1], cell_vec[2]))
        recip_dipole *= math.pi/(cell_volume*eta_2)

        # Fill in remaining entries by symmetry
        # Mask so we don't count diagonal twice
        mask = np.tri(n_ions, k=-1)[:, :, np.newaxis, np.newaxis]
        real_dipole = real_dipole + mask*np.conj(
            np.transpose(real_dipole, axes=[1, 0, 2, 3]))
        recip_dipole = recip_dipole + mask*np.conj(
            np.transpose(recip_dipole, axes=[1, 0, 2, 3]))

        # Multiply by Born charges and subtract q=0 from diagonal
        dipole = np.zeros((n_ions, n_ions, 3, 3), dtype=np.complex128)
        dipole_tmp = recip_dipole - real_dipole
        for i in range(n_ions):
            dipole[i] = np.einsum('ij,klm,kjm->kil',
                                  born[i], born, dipole_tmp[i])
            dipole[i, i] -= self._dipole_q0[i]

        return np.reshape(np.transpose(dipole, axes=[0, 2, 1, 3]),
                          (3*n_ions, 3*n_ions))

    def _calculate_gamma_correction(self, q_dir):
        """
        Calculate non-analytic correction to the dynamical matrix at q=0 for
        a specified direction of approach. See Eq. 60 of X. Gonze and C. Lee,
        PRB (1997) 55, 10355-10368.

        Parameters
        ----------
        q_dir : (3,) float ndarray
            The direction along which q approaches 0, in reciprocal fractional
            coordinates

        Returns
        -------
        na_corr : (3*n_ions, 3*n_ions) complex ndarray
            The correction to the dynamical matrix
        """
        cell_vec = self._cell_vec
        n_ions = self.n_ions
        born = self._born
        dielectric = self.dielectric

        cell_volume = np.dot(cell_vec[0], np.cross(cell_vec[1], cell_vec[2]))
        denominator = np.einsum('ij,i,j', dielectric, q_dir, q_dir)
        factor = 4*math.pi/(cell_volume*denominator)

        q_born_sum = np.einsum('ijk,k->ij', born, q_dir)
        na_corr = np.zeros((3*n_ions, 3*n_ions), dtype=np.complex128)
        for i in range(n_ions):
            for j in range(n_ions):
                na_corr[3*i:3*(i+1), 3*j:3*(j+1)] = np.einsum(
                    'i,j->ij', q_born_sum[i], q_born_sum[j])
        na_corr *= factor

        return na_corr

    def _get_shell_origins(self, n):
        """
        Given the shell number, compute all the cell origins that lie in that
        shell

        Parameters
        ----------
        n : int
            The shell number

        Returns
        -------
        origins : ((2*n + 1)**3 - (2*n - 1)**3, 3) int ndarray
            The cell origins. Note: if n = 0, origins = [[0, 0, 0]]

        """

        if n == 0:
            return np.array([[0, 0, 0]], dtype=np.int32)

        # Coordinates of cells in xy plane where z=0, xz plane where y=0, yz
        # plane where x=0. Note: to avoid duplicating cells at the edges,
        # the xz plane has (2*n + 1) - 2 rows in z, rather than
        # (2*n + 1). The yz plane also has (2*n + 1) - 2 rows in z and
        # (2*n + 1) - 2 columns in y
        xy = self._get_all_origins([n+1, n+1, 1], min_xyz=[-n, -n, 0])
        xz = self._get_all_origins([n+1, 1, n], min_xyz=[-n, 0, -n+1])
        yz = self._get_all_origins([1, n, n], min_xyz=[0, -n+1, -n+1])

        # Offset each plane by n and -n to get the 6 planes that make up the
        # shell
        origins = np.zeros(((2*n+1)**3-(2*n-1)**3, 3), dtype=np.int32)
        for i, ni in enumerate([-n, n]):
            origins[i*len(xy):(i+1)*len(xy)] = xy
            origins[i*len(xy):(i+1)*len(xy), 2] = ni
            io = 2*len(xy)
            origins[io+i*len(xz):io+(i+1)*len(xz)] = xz
            origins[io+i*len(xz):io+(i+1)*len(xz), 1] = ni
            io = 2*(len(xy) + len(xz))
            origins[io+i*len(yz):io+(i+1)*len(yz)] = yz
            origins[io+i*len(yz):io+(i+1)*len(yz), 0] = ni

        return origins

    def _get_all_origins(self, max_xyz, min_xyz=[0, 0, 0], step=1):
        """
        Given the max/min number of cells in each direction, get a list of all
        possible cell origins

        Parameters
        ----------
        max_xyz : (3,) int ndarray
            The number of cells to count to in each direction
        min_xyz : (3,) int ndarray, optional, default [0,0,0]
            The cell number to count from in each direction
        step : integer, optional, default 1
            The step between cells

        Returns
        -------
        origins : (prod(max_xyz - min_xyz)/step, 3) int ndarray
            The cell origins
        """
        diff = np.absolute(np.subtract(max_xyz, min_xyz))
        nx = np.repeat(range(min_xyz[0], max_xyz[0], step), diff[1]*diff[2])
        ny = np.repeat(np.tile(range(min_xyz[1], max_xyz[1], step), diff[0]),
                       diff[2])
        nz = np.tile(range(min_xyz[2], max_xyz[2], step), diff[0]*diff[1])

        return np.column_stack((nx, ny, nz))

    def _enforce_realspace_asr(self):
        """
        Apply a transformation to the force constants matrix so that it
        satisfies the acousic sum rule. Diagonalise, shift the acoustic modes
        to almost zero then construct the correction to the force constants
        matrix using the eigenvectors. For more information see section 2.3.4:
        http://www.tcm.phy.cam.ac.uk/castep/Phonons_Guide/Castep_Phonons.html

        Returns
        -------
        force_constants : (n_cells_in_sc, 3*n_ions, 3*n_ions) float ndarray
            The corrected force constants matrix
        """
        cell_origins = self.cell_origins
        sc_matrix = self.sc_matrix
        n_cells_in_sc = self.n_cells_in_sc
        n_ions = self.n_ions
        force_constants = self._force_constants
        ax = np.newaxis

        # Compute square matrix giving relative index of cells in sc
        n_ions_in_sc = n_ions*n_cells_in_sc
        sq_fc = np.zeros((3*n_ions_in_sc, 3*n_ions_in_sc))
        inv_sc_matrix = np.linalg.inv(np.transpose(sc_matrix))
        cell_origins_sc = np.einsum('ij,kj->ik', cell_origins, inv_sc_matrix)
        for nc in range(n_cells_in_sc):
            # Get all possible cell-cell vector combinations
            inter_cell_vectors = cell_origins_sc - np.tile(cell_origins_sc[nc],
                                                           (n_cells_in_sc, 1))
            # Compare cell-cell vectors with origin-cell vectors and determine
            # which are equivalent
            # Do calculation in chunks, so loop can be broken if all
            # equivalent vectors have been found
            N = 100
            dist_min = np.full((n_cells_in_sc), sys.float_info.max)
            sc_relative_index = np.zeros(n_cells_in_sc, dtype=np.int32)
            for i in range(int((n_cells_in_sc - 1)/N) + 1):
                ci = i*N
                cf = min((i + 1)*N, n_cells_in_sc)
                dist = (inter_cell_vectors[:, ax, :] -
                        cell_origins_sc[ax, ci:cf, :])
                dist_frac = dist - np.rint(dist)
                dist_frac_sum = np.sum(np.abs(dist_frac), axis=2)
                scri_current = np.argmin(dist_frac_sum, axis=1)
                dist_min_current = dist_frac_sum[
                    range(n_cells_in_sc), scri_current]
                replace = dist_min_current < dist_min
                sc_relative_index[replace] = ci + scri_current[replace]
                dist_min[replace] = dist_min_current[replace]
                if np.all(dist_min <= 16*sys.float_info.epsilon):
                    break
            if (np.any(dist_min > 16*sys.float_info.epsilon)):
                warnings.warn(('Error correcting FC matrix for acoustic sum '
                               'rule, supercell relative index couldn\'t be '
                               'found. Returning uncorrected FC matrix'))
                return self.force_constants
            sq_fc[3*nc*n_ions:3*(nc+1)*n_ions, :] = np.transpose(
                np.reshape(force_constants[sc_relative_index],
                           (3*n_cells_in_sc*n_ions, 3*n_ions)))
        try:
            ac_i, evals, evecs = self._find_acoustic_modes(sq_fc)
        except Exception:
            warnings.warn(('\nError correcting for acoustic sum rule, could '
                           'not find 3 acoustic modes.\nReturning uncorrected '
                           'FC matrix'), stacklevel=2)
            return self.force_constants

        # Correct force constant matrix - set acoustic modes to almost zero
        fc_tol = 1e-8*np.min(np.abs(evals))
        for ac in ac_i:
            sq_fc -= (fc_tol + evals[ac])*np.einsum(
                'i,j->ij', evecs[:, ac], evecs[:, ac])

        fc = np.reshape(sq_fc[:, :3*n_ions],
                        (n_cells_in_sc, 3*n_ions, 3*n_ions))

        return fc

    def _enforce_reciprocal_asr(self, dyn_mat_gamma):
        """
        Calculate the correction to the dynamical matrix that would have to be
        applied to satisfy the acousic sum rule. Diagonalise the gamma-point
        dynamical matrix, shift the acoustic modes to almost zero then
        reconstruct the dynamical matrix using the eigenvectors. For more
        information see section 2.3.4:
        http://www.tcm.phy.cam.ac.uk/castep/Phonons_Guide/Castep_Phonons.html

        Parameters
        ----------
        dyn_mat_gamma : (3*n_ions, 3*n_ions) complex ndarray
            The non mass-weighted dynamical matrix at gamma

        Returns
        -------
        dyn_mat : (3*n_ions, 3*n_ions) complex ndarray or empty array
            The corrected, non mass-weighted dynamical matrix at q. Returns
            empty array (np.array([])) if finding the 3 acoustic modes fails
        """
#        tol = (ureg('amu').to('e_mass')
#               *0.1*ureg('1/cm').to('1/bohr')**2).magnitude
        tol = 5e-15

        try:
            ac_i, g_evals, g_evecs = self._find_acoustic_modes(dyn_mat_gamma)
        except Exception:
            warnings.warn(('\nError correcting for acoustic sum rule, '
                           'could not find 3 acoustic modes.\nNot '
                           'correcting dynamical matrix'), stacklevel=2)
            return np.array([], dtype=np.complex128)

        recip_asr_correction = np.zeros((3*self.n_ions, 3*self.n_ions),
                                        dtype=np.complex128)
        for i, ac in enumerate(ac_i):
            recip_asr_correction -= (tol*i + g_evals[ac])*np.einsum(
                'i,j->ij', g_evecs[:, ac], g_evecs[:, ac])

        return recip_asr_correction

    def _find_acoustic_modes(self, dyn_mat):
        """
        Find the acoustic modes from a dynamical matrix, they should have
        the sum of c of m amplitude squared = mass (note: have not actually
        included mass weighting here so assume mass = 1.0)

        Parameters
        ----------
        dyn_mat : (3*n_ions, 3*n_ions) complex ndarray
            A dynamical matrix

        Returns
        -------
        ac_i : (3,) int ndarray
            The indices of the acoustic modes
        evals : (3*n_ions) float ndarray
            Dynamical matrix eigenvalues
        evecs : (3*n_ions, n_ions, 3) complex ndarray
            Dynamical matrix eigenvectors
        """
        n_branches = dyn_mat.shape[0]
        n_ions = int(n_branches/3)

        evals, evecs = np.linalg.eigh(dyn_mat)
        evec_reshape = np.reshape(
            np.transpose(evecs), (n_branches, n_ions, 3))
        # Sum displacements for all ions in each branch
        c_of_m_disp = np.sum(evec_reshape, axis=1)
        c_of_m_disp_sq = np.sum(np.abs(c_of_m_disp)**2, axis=1)
        sensitivity = 0.5
        sc_mass = 1.0*n_ions
        # Check number of acoustic modes
        if np.sum(c_of_m_disp_sq > sensitivity*sc_mass) < 3:
            raise Exception('Could not find 3 acoustic modes')
        # Find indices of acoustic modes (3 largest c of m displacements)
        ac_i = np.argsort(c_of_m_disp_sq)[-3:]

        return ac_i, evals, evecs

    def _calculate_phases(self, q, unique_sc_offsets, unique_sc_i,
                          unique_cell_origins, unique_cell_i):
        """
        Calculate the phase factors for the supercell images and cells for a
        single q-point. The unique supercell and cell origins indices are
        required to minimise expensive exp and power operations

        Parameters
        ----------
        q : (3,) float ndarray
            The q-point to calculate the phase for
        unique_sc_offsets : list of lists of ints
            A list containing 3 lists of the unique supercell image offsets in
            each direction. The supercell offset is calculated by multiplying
            the supercell matrix by the supercell image indices (obtained by
            _get_all_origins()). A list of lists rather than a
            Numpy array is used as the 3 lists are independent and their size
            is not known beforehand
        unique_sc_i : ((2*lim + 1)**3, 3) int ndarray
            The indices needed to reconstruct sc_offsets from the unique
            values in unique_sc_offsets
        unique_cell_origins : list of lists of ints
            A list containing 3 lists of the unique cell origins in each
            direction. A list of lists rather than a Numpy array is used as
            the 3 lists are independent and their size is not known beforehand
        unique_cell_i : (cell_origins, 3) int ndarray
            The indices needed to reconstruct cell_origins from the unique
            values in unique_cell_origins

        Returns
        -------
        sc_phases : (unique_sc_i,) float ndarray
            Phase factors exp(iq.r) for each supercell image coordinate in
            sc_offsets
        cell_phases : (unique_cell_i,) float ndarray
            Phase factors exp(iq.r) for each cell coordinate in the supercell
        """

        # Only calculate exp(iq) once, then raise to power to get the phase at
        # different supercell/cell coordinates to minimise expensive exp
        # calculations
        # exp(iq.r) = exp(iqh.ra)*exp(iqk.rb)*exp(iql.rc)
        #           = (exp(iqh)^ra)*(exp(iqk)^rb)*(exp(iql)^rc)
        phase = np.exp(2j*math.pi*q)
        sc_phases = np.ones(len(unique_sc_i), dtype=np.complex128)
        cell_phases = np.ones(len(unique_cell_i), dtype=np.complex128)
        for i in range(3):
            unique_sc_phases = np.power(phase[i], unique_sc_offsets[i])
            sc_phases *= unique_sc_phases[unique_sc_i[:, i]]

            unique_cell_phases = np.power(phase[i], unique_cell_origins[i])
            cell_phases *= unique_cell_phases[unique_cell_i[:, i]]

        return sc_phases, cell_phases

    def _calculate_supercell_images(self, lim):
        """
        For each displacement of ion i in the unit cell and ion j in the
        supercell, calculate the number of supercell periodic images there are
        and which supercells they reside in, and sets the sc_image_i,
        and n_sc_images InterpolationData attributes

        Parameters
        ----------
        lim : int
            The supercell image limit
        """

        n_ions = self.n_ions
        cell_vec = self._cell_vec
        ion_r = self.ion_r
        cell_origins = self.cell_origins
        n_cells_in_sc = self.n_cells_in_sc
        sc_matrix = self.sc_matrix
        ax = np.newaxis

        # List of points defining Wigner-Seitz cell
        ws_frac = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                            [0, 1, -1], [1, 0, 0], [1, 0, 1], [1, 0, -1],
                            [1, 1, 0], [1, 1, 1], [1, 1, -1], [1, -1, 0],
                            [1, -1, 1], [1, -1, -1]])
        cutoff_scale = 1.0

        # Calculate points of WS cell for this supercell
        sc_vecs = np.dot(sc_matrix, cell_vec)
        ws_list = np.dot(ws_frac, sc_vecs)
        inv_ws_sq = 1.0/np.sum(np.square(ws_list[1:]), axis=1)
        ws_list_norm = ws_list[1:]*inv_ws_sq[:, ax]

        # Get Cartesian coords of supercell images and ions in supercell
        sc_image_r = self._get_all_origins(
            np.repeat(lim, 3) + 1, min_xyz=-np.repeat(lim, 3))
        sc_image_cart = np.einsum('ij,jk->ik', sc_image_r, sc_vecs)
        sc_ion_r = np.einsum('ijk,kl->ijl',
                             cell_origins[:, ax, :] + ion_r[ax, :, :],
                             np.linalg.inv(np.transpose(sc_matrix)))
        sc_ion_cart = np.einsum('ijk,kl->ijl', sc_ion_r, sc_vecs)

        sc_image_i = np.full((n_cells_in_sc, n_ions, n_ions, (2*lim + 1)**3),
                             -1, dtype=np.int32)
        n_sc_images = np.zeros((n_cells_in_sc, n_ions, n_ions), dtype=np.int32)

        # Ordering of loops here is for efficiency:
        # ions in unit cell -> periodic supercell images -> WS points
        # This is so the ion-ion vectors in each loop will be similar,
        # so they are more likely to pass/fail the WS check together
        # so the last loop can be broken early
        for i in range(n_ions):
            rij = sc_ion_cart[0, i] - sc_ion_cart
            for im, sc_r in enumerate(sc_image_cart):
                # Get vector between j in supercell image and i in unit cell
                dists = rij - sc_r
                # Only want to include images where ion < halfway to ALL ws
                # points, so compare vector to all ws points
                for n, wsp in enumerate(ws_list_norm):
                    dist_wsp = np.absolute(np.sum(dists*wsp, axis=-1))
                    if n == 0:
                        nc_idx, nj_idx = np.where(
                            dist_wsp <= ((0.5*cutoff_scale + 0.001)))
                        # Reindex dists to remove elements where the ion is
                        # > halfway to WS point, to avoid wasted computation
                        dists = dists[nc_idx, nj_idx]
                    else:
                        # After first reindex, dists is now 1D so need to
                        # reindex like this instead
                        idx = np.where(
                            dist_wsp <= ((0.5*cutoff_scale + 0.001)))[0]
                        nc_idx = nc_idx[idx]
                        nj_idx = nj_idx[idx]
                        dists = dists[idx]
                    if len(nc_idx) == 0:
                        break
                    # If ion-ion vector has been < halfway to all WS points,
                    # this is a valid image! Save it
                    if n == len(ws_list_norm) - 1:
                        n_im_idx = n_sc_images[nc_idx, i, nj_idx]
                        sc_image_i[nc_idx, i, nj_idx, n_im_idx] = im
                        n_sc_images[nc_idx, i, nj_idx] += 1

        self._n_sc_images = n_sc_images
        # Truncate sc_image_i to the maximum ACTUAL images rather than the
        # maximum possible images to avoid storing and summing over
        # nonexistent images
        self._sc_image_i = sc_image_i[:, :, :, :np.max(n_sc_images)]

    def reorder_freqs(self, **kwargs):
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
        if self.n_qpts == 0:
            raise Exception((
                'No frequencies in InterpolationData object, call '
                'calculate_fine_phonons before reordering frequencies'))
        super(InterpolationData, self).reorder_freqs(**kwargs)

    def calculate_structure_factor(self, scattering_lengths, **kwargs):
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
        if self.n_qpts == 0:
            raise Exception((
                'No frequencies in InterpolationData object, call '
                'calculate_fine_phonons before calling '
                'calculate_structure_factor'))
        sf = super(InterpolationData, self).calculate_structure_factor(
            scattering_lengths, **kwargs)

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
        if self.n_qpts == 0:
            raise Exception((
                'No frequencies in InterpolationData object, call '
                'calculate_fine_phonons before using object as a dw_data '
                'keyword argument to calculate_structure_factor'))
        dw = super(InterpolationData, self)._dw_coeff(T)

        return dw

    def calculate_sqw_map(self, scattering_lengths, ebins, **kwargs):
        """
        Calculate the structure factor for each q-point contained in data, and
        bin according to ebins to create a S(Q,w) map

        Parameters
        ----------
        scattering_lengths : dictionary
            Dictionary of spin and isotope averaged coherent scattering legnths
            for each element in the structure in fm e.g.
            {'O': 5.803, 'Zn': 5.680}
        ebins : (n_ebins + 1) float ndarray
            The energy bin edges in the same units as freqs
        **kwargs
            Passes keyword arguments on to
            PhononData.calculate_sqw_map

        Returns
        -------
        sqw_map : ndarray
            The intensity for each q-point and energy bin
        """
        if self.n_qpts == 0:
            raise Exception((
                'No frequencies in InterpolationData object, call '
                'calculate_fine_phonons before calling '
                'calculate_sqw_map'))
        sqw_map = super(InterpolationData, self).calculate_sqw_map(
            scattering_lengths, ebins, **kwargs)

        return sqw_map
