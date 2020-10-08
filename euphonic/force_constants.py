import math
import sys
import warnings

import numpy as np
from pint import Quantity
import scipy
from scipy.linalg.lapack import zheev
from scipy.special import erfc

from euphonic.validate import (
    _check_constructor_inputs, _check_unit_conversion,
    _ensure_contiguous_args, _ensure_contiguous_attrs)
from euphonic.io import (_obj_to_json_file, _obj_from_json_file,
                         _obj_to_dict, _process_dict)
from euphonic.readers import castep, phonopy
from euphonic.util import (is_gamma, get_all_origins,
                           _get_supercell_relative_idx)
from euphonic import ureg, Crystal, QpointPhononModes


class ImportCError(Exception):
    pass


class ForceConstants(object):
    """
    A class to read and store the data required for a phonon
    interpolation calculation from model (e.g. CASTEP) output,
    and calculate phonon frequencies/eigenvectors at arbitrary q-points
    via Fourier interpolation

    Attributes
    ----------
    crystal : Crystal
        Lattice and atom information
    force_constants : (n_cells_in_sc, 3*n_atoms, 3*n_atoms) float Quantity
        Force constants matrix
    sc_matrix : (3, 3) int ndarray
        The supercell matrix
    n_cells_in_sc : int
        Number of cells in the supercell
    cell_origins : (n_cells_in_sc, 3) int ndarray
        The locations of the unit cells within the supercell
    born : (n_atoms, 3, 3) float Quantity or None
        The Born charges for each atom
    dielectric : (3, 3) float Quantity or None
        The dielectric permittivity tensor
    """

    def __init__(self, crystal, force_constants, sc_matrix, cell_origins,
                 born=None, dielectric=None):
        """
        Parameters
        ----------
        crystal : Crystal
            Lattice and atom information
        force_constants : (n_cells_in_sc, 3*n_atoms, 3*n_atoms) float Quantity
            Force constants matrix
        sc_matrix : (3, 3) int ndarray
            The supercell matrix
        cell_origins : (n_cells_in_sc, 3) int ndarray
            The locations of the unit cells within the supercell
        born : (n_atoms, 3, 3) float Quantity, optional
            The Born charges for each atom
        dielectric : (3, 3) float Quantity, optional
            The dielectric permittivity tensor
        """
        # Check independent inputs first
        _check_constructor_inputs(
            [crystal, sc_matrix], [Crystal, np.ndarray], [(), (3, 3)],
            ['crystal', 'sc_matrix'])
        n_at = crystal.n_atoms
        n_sc = int(np.rint(np.absolute(np.linalg.det(sc_matrix))))
        # Now check other derived input shapes
        _check_constructor_inputs(
            [force_constants, cell_origins, born, dielectric],
            [Quantity, np.ndarray, [Quantity, type(None)],
                [Quantity, type(None)]],
            [(n_sc, 3*n_at, 3*n_at), (n_sc, 3), (n_at, 3, 3), (3, 3)],
            ['force_constants', 'cell_origins', 'born', 'dielectric'])
        self.crystal = crystal
        self._force_constants = force_constants.to(
            'INTERNAL_ENERGY_UNIT/INTERNAL_LENGTH_UNIT**2').magnitude
        self.force_constants_unit = str(force_constants.units)
        self.sc_matrix = sc_matrix
        self.cell_origins = cell_origins
        self.n_cells_in_sc = n_sc

        if born is not None:
            self._born = born.to(ureg.INTERNAL_CHARGE_UNIT).magnitude
            self.born_unit = str(born.units)
            self._dielectric = dielectric.to(ureg(
                '(INTERNAL_CHARGE_UNIT**2)/'
                '(INTERNAL_LENGTH_UNIT*INTERNAL_ENERGY_UNIT)')).magnitude
            self.dielectric_unit = str(dielectric.units)
        else:
            self._born = born
            self.born_unit = str(ureg.INTERNAL_CHARGE_UNIT)
            self._dielectric = dielectric
            self.dielectric_unit = str(ureg((
                '(INTERNAL_CHARGE_UNIT**2)/'
                '(INTERNAL_LENGTH_UNIT*INTERNAL_ENERGY_UNIT)')))

    @property
    def force_constants(self):
        return self._force_constants*ureg(
            'INTERNAL_ENERGY_UNIT/INTERNAL_LENGTH_UNIT**2').to(
                self.force_constants_unit)

    @property
    def born(self):
        if self._born is not None:
            return self._born*ureg('INTERNAL_CHARGE_UNIT').to(self.born_unit)
        else:
            return None

    @property
    def dielectric(self):
        if self._dielectric is not None:
            return self._dielectric*ureg((
                '(INTERNAL_CHARGE_UNIT**2)/'
                '(INTERNAL_LENGTH_UNIT*INTERNAL_ENERGY_UNIT)')).to(
                    self.dielectric_unit)
        else:
            return None

    def __setattr__(self, name, value):
        _check_unit_conversion(self, name, value,
                               ['force_constants_unit', 'born_unit',
                                'dielectric_unit'])
        super(ForceConstants, self).__setattr__(name, value)

    def calculate_qpoint_phonon_modes(
            self, qpts, weights=None, asr=None, dipole=True, eta_scale=1.0,
            splitting=True, insert_gamma=False, reduce_qpts=True, use_c=False,
            n_threads=1, fall_back_on_python=True):
        """
        Calculate phonon frequencies and eigenvectors at specified
        q-points from a force constants matrix via Fourier interpolation

        Parameters
        ----------
        qpts : (n_qpts, 3) float ndarray
            The q-points to interpolate onto
        weights : (n_qpts,) float ndarray, optional
            The weight for each q-point. If not given, equal weights are
            applied
        asr : {'realspace', 'reciprocal'}, optional
            Which acoustic sum rule correction to apply. 'realspace'
            applies the correction to the force constant matrix in real
            space. 'reciprocal' applies the correction to the dynamical
            matrix at every q-point
        dipole : boolean, optional
            Calculates the dipole tail correction to the dynamical
            matrix at each q-point using the Ewald sum, if the Born
            charges and dielectric permitivitty tensor are present.
        eta_scale : float, optional
            Changes the cutoff in real/reciprocal space for the dipole
            Ewald sum. A higher value uses more reciprocal terms. If tuned
            correctly this can result in performance improvements. See
            euphonic-optimise-eta program for help on choosing a good
            eta_scale.
        splitting : boolean, optional
            Whether to calculate the LO-TO splitting at the gamma
            points. Only applied if dipole is True and the Born charges
            and dielectric permitivitty tensor are present.
        insert_gamma : boolean, optional
            If splitting is True, this will insert gamma points into
            qpts to store the extra split frequencies. Note this means
            that the length of qpts in the output QpointPhononModes
            object will not necessarily be the same as the input qpts.
            If qpts already contains double gamma points where you want
            split frequencies, leave this as False.
        reduce_qpts : boolean, optional
            Whether to use periodicity to reduce all q-points and only
            calculate for unique q-points within the 1st BZ. This won't
            change the output but could increase performance.
        use_c : boolean, optional
            Whether to use C instead of Python to calculate and
            diagonalise the dynamical matrix
        n_threads : int, optional
            The number of OpenMP threads to use when looping over
            q-points in C. Only applicable if use_c=True
        fall_back_on_python : boolean, optional
            If we cannot use the C extension, fall back on using python
            if this is true, else raise an ImportCError.

        Returns
        -------
        QpointPhononModes
            A QpointPhononModes object containing the interpolated
            frequencies and eigenvectors at each q-point. Note that if
            there is LO-TO splitting, and insert_gamma=True, the number
            of input q-points may not be the same as in the output
            object

        Raises
        ------
        ImportCError
            If we have selected not to fall back on Python and cannot
            use the C extension

        Notes
        -----
        Phonon frequencies/eigenvectors are calculated at any q-point by
        Fourier interpolation of a force constants matrix. The force
        constants matrix is defined as [1]_:

        .. math::

          \\phi_{\\alpha, {\\alpha}'}^{\\kappa, {\\kappa}'} =
          \\frac{\\delta^{2}E}{{\\delta}u_{\\kappa,\\alpha}{\\delta}u_{{\\kappa}',{\\alpha}'}}

        Which gives the Dynamical matrix at q:

        .. math::

          D_{\\alpha, {\\alpha}'}^{\\kappa, {\\kappa}'}(q) =
          \\frac{1}{\\sqrt{M_\\kappa M_{\\kappa '}}}
          \\sum_{a}\\phi_{\\alpha, \\alpha '}^{\\kappa, \\kappa '}e^{-iq\\cdot r_a}

        The eigenvalue equation for the dynamical matrix is then:

        .. math::

          D_{\\alpha, {\\alpha}'}^{\\kappa, {\\kappa}'}(q) \\epsilon_{q\\nu\\kappa\\alpha} =
          \\omega_{q\\nu}^{2} \\epsilon_{q\\nu\\kappa\\alpha}

        Where :math:`\\nu` runs over phonon modes, :math:`\\kappa` runs
        over atoms, :math:`\\alpha` runs over the Cartesian directions,
        :math:`a` runs over unit cells in the supercell,
        :math:`u_{\\kappa, \\alpha}` is the displacement of atom
        :math:`\\kappa` in direction :math:`\\alpha`,
        :math:`M_{\\kappa}` is the mass of atom :math:`\\kappa`,
        :math:`r_{a}` is the vector to the origin of cell :math:`a` in
        the supercell, :math:`\\epsilon_{q\\nu\\kappa\\alpha}` are the
        eigevectors, and :math:`\\omega_{q\\nu}^{2}` are the frequencies
        squared.

        In polar materials, there is an additional long-ranged
        correction to the force constants matrix (applied if
        dipole=True) and a non-analytical correction at the gamma point
        [2]_ (applied if splitting=True).

        .. [1] M.T. Dove, Introduction to Lattice Dynamics, Cambridge University Press, Cambridge, 1993, 83-87
        .. [2] X. Gonze, K. C. Charlier, D. C. Allan, M. P. Teter, Phys. Rev. B, 1994, 50, 13035-13038
        """

        # Check weights is of appropriate type and shape, to avoid doing all
        # the interpolation only for it to fail creating QpointPhononModes
        _check_constructor_inputs(
            [weights], [[np.ndarray, type(None)]], [(len(qpts),)], ['weights'])

        # Set default splitting params
        if self.born is None:
            dipole = False
        if not dipole:
            splitting = False

        if splitting and insert_gamma:
            # Duplicate gamma points where there is splitting
            gamma_i = np.where(is_gamma(qpts))[0]
            split_gamma = gamma_i[np.where(
                np.logical_and(gamma_i > 0, gamma_i < len(qpts) - 1))]
            qpts = np.insert(qpts, split_gamma, np.array([0., 0., 0.]), axis=0)
            # It doesn't necessarily make sense to use both weights
            # (usually used for DOS) and splitting (usually used for
            # bandstructures) but we need to handle this case anyway
            # Where 1 q-point splits into 2, half the weight for each
            if weights is not None:
                # Don't change original array
                weights = np.copy(weights)
                weights[split_gamma] = weights[split_gamma]/2
                weights = np.insert(weights, split_gamma,
                                    weights[split_gamma])

        if reduce_qpts:
            norm_qpts = qpts - np.rint(qpts)
            # Ensure gamma points are exactly zero, otherwise you may
            # have a case where small fp differences mean np.unique
            # doesn't reduce them, yet they're all classified as gamma
            # points. This causes indexing errors later when calculating
            # q-directions as there are then points in reduced_qpts
            # whose index isn't in qpts_i
            gamma_i = np.where(is_gamma(qpts))[0]
            n_gamma = len(gamma_i)
            norm_qpts[gamma_i] = 0.

            reduced_qpts, qpts_i = np.unique(norm_qpts, return_inverse=True,
                                             axis=0)
            n_rqpts = len(reduced_qpts)
            # Special handling of gamma points - don't reduce gamma
            # points if LO-TO splitting
            if splitting and n_gamma > 1:
                # Replace any gamma points and their indices with new
                # gamma points appended onto the reduced q-point array,
                # so each gamma can have its own splitting
                qpts_i[gamma_i[1:]] = range(n_rqpts, n_rqpts + n_gamma - 1)
                reduced_qpts = np.append(reduced_qpts,
                                         np.tile(np.array([0., 0., 0., ]),
                                                 (n_gamma - 1, 1)),
                                         axis=0)
                n_rqpts = len(reduced_qpts)
        else:
            reduced_qpts = qpts
            qpts_i = np.arange(0, len(qpts), dtype=np.int32)
            n_rqpts = len(qpts)

        # Get q-directions for non-analytical corrections
        if splitting:
            split_idx = np.where(is_gamma(reduced_qpts))[0]
            q_dirs = self._get_q_dir(qpts, qpts_i, split_idx)
        else:
            split_idx = np.array([])
            q_dirs = np.array([])

        lim = 2  # Supercell image limit
        # Construct list of supercell ion images
        if not hasattr(self, '_sc_image_i'):
            self._calculate_supercell_images(lim)

        # Get a list of all the unique supercell image origins and cell
        # origins in x, y, z and how to rebuild them to minimise
        # expensive phase calculations later
        sc_image_r = get_all_origins(
            np.repeat(lim, 3) + 1, min_xyz=-np.repeat(lim, 3))
        sc_offsets = np.einsum('ij,jk->ik', sc_image_r,
                               self.sc_matrix).astype(np.int32)
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
        atom_mass = self.crystal._atom_mass
        n_atoms = self.crystal.n_atoms
        masses = np.tile(np.repeat(atom_mass, 3), (3*n_atoms, 1))
        dyn_mat_weighting = 1/np.sqrt(masses*np.transpose(masses))

        # Initialise dipole correction calc to FC matrix if required
        if dipole and (not hasattr(self, 'eta_scale') or
                       eta_scale != self._eta_scale):
            self._dipole_correction_init(eta_scale)

        if asr == 'realspace':
            if not hasattr(self, '_force_constants_asr'):
                self._force_constants_asr = self._enforce_realspace_asr()
            force_constants = self._force_constants_asr
        else:
            force_constants = self._force_constants
        # Precompute fc matrix weighted by number of supercell atom
        # images (for cumulant method)
        n_sc_images_repeat = (
            self._n_sc_images.repeat(3, axis=2).repeat(3, axis=1)
        )
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

        rfreqs = np.zeros((n_rqpts, 3*n_atoms))
        reigenvecs = np.zeros(
            (n_rqpts, 3*n_atoms, n_atoms, 3), dtype=np.complex128
        )
        try:
            if use_c:
                try:
                    import euphonic._euphonic as euphonic_c
                except ImportError:
                    warnings.warn((
                        'use_c=True is set, but the Euphonic\'s C '
                        'extension couldn\'t be imported, it may not '
                        'have been installed. Attempting to fall back '
                        'to pure Python calculation'), stacklevel=2)
                    raise
            else:
                raise ImportError
            # Make sure all arrays are contiguous before calling C
            cell_vectors = self.crystal._cell_vectors
            recip_vectors = self.crystal.reciprocal_cell().to(
                '1/bohr').magnitude
            (cell_vectors, recip_vectors, reduced_qpts, split_idx, q_dirs,
             fc_img_weighted, sc_offsets, recip_asr_correction,
             dyn_mat_weighting, rfreqs,
             reigenvecs) = _ensure_contiguous_args(
                 cell_vectors, recip_vectors, reduced_qpts, split_idx,
                 q_dirs, fc_img_weighted, sc_offsets, recip_asr_correction,
                 dyn_mat_weighting, rfreqs, reigenvecs)
            attrs = ['_n_sc_images', '_sc_image_i', 'cell_origins']
            dipole_attrs = ['atom_r', '_born', '_dielectric', '_H_ab',
                            '_cells', '_gvec_phases', '_gvecs_cart',
                            '_dipole_q0']
            _ensure_contiguous_attrs(self, attrs, opt_attrs=dipole_attrs)
            reciprocal_asr = 1 if asr == 'reciprocal' else 0
            euphonic_c.calculate_phonons(
                self, cell_vectors, recip_vectors, reduced_qpts, split_idx,
                q_dirs, fc_img_weighted, sc_offsets, recip_asr_correction,
                dyn_mat_weighting, dipole, reciprocal_asr, splitting, rfreqs,
                reigenvecs, n_threads, scipy.__path__[0])
        except ImportError:
            if not fall_back_on_python:
                raise ImportCError((
                    'use_c=True is set, but the Euphonic\'s C extension'
                    ' couldn\'t be imported, it may not have been '
                    'installed. You have selected not to fall back on '
                    'Python, therefore the calculation can\'t be '
                    'completed'))
            else:
                q_independent_args = (
                    reduced_qpts, split_idx, q_dirs, fc_img_weighted,
                    unique_sc_offsets, unique_sc_i, unique_cell_origins,
                    unique_cell_i, recip_asr_correction, dyn_mat_weighting,
                    dipole, asr, splitting)
                for q in range(n_rqpts):
                    rfreqs[q], reigenvecs[q] = self._calculate_phonons_at_q(
                        q, q_independent_args)

        freqs = rfreqs[qpts_i]*ureg('INTERNAL_ENERGY_UNIT').to(
            'mDEFAULT_ENERGY_UNIT')

        return QpointPhononModes(
            self.crystal, qpts, freqs, reigenvecs[qpts_i], weights=weights)

    def _calculate_phonons_at_q(self, q, args):
        """
        Given a q-point and some precalculated q-independent values,
        calculate and diagonalise the dynamical matrix and return the
        frequencies and eigenvalues. Optionally also includes the Ewald
        dipole sum correction and LO-TO splitting
        """
        (reduced_qpts, split_idx, q_dirs, fc_img_weighted, unique_sc_offsets,
         unique_sc_i, unique_cell_origins, unique_cell_i,
         recip_asr_correction, dyn_mat_weighting, dipole, asr,
         splitting) = args

        qpt = reduced_qpts[q]
        n_atoms = self.crystal.n_atoms

        dyn_mat = self._calculate_dyn_mat(
            qpt, fc_img_weighted, unique_sc_offsets, unique_sc_i,
            unique_cell_origins, unique_cell_i)

        if dipole:
            dipole_corr = self._calculate_dipole_correction(qpt)
            dyn_mat += dipole_corr

        if asr == 'reciprocal':
            dyn_mat += recip_asr_correction

        # Calculate LO-TO splitting by calculating non-analytic
        # correction to dynamical matrix
        if splitting and is_gamma(qpt):
            q_dir_idx = np.where(split_idx == q)[0][0]
            na_corr = self._calculate_gamma_correction(q_dirs[q_dir_idx])
        else:
            # Correction is zero if not a gamma point or splitting=False
            na_corr = np.array([0])

        dyn_mat += na_corr

        # Mass weight dynamical matrix
        dyn_mat *= dyn_mat_weighting

        evals, evecs = np.linalg.eigh(dyn_mat, UPLO='U')
        evecs = np.reshape(np.transpose(evecs),
                           (3*n_atoms, n_atoms, 3))
        # Set imaginary frequencies to negative
        imag_freqs = np.where(evals < 0)
        evals = np.sqrt(np.abs(evals))
        evals[imag_freqs] *= -1

        return evals, evecs

    def _calculate_dyn_mat(self, q, fc_img_weighted, unique_sc_offsets,
                           unique_sc_i, unique_cell_origins, unique_cell_i):
        """
        Calculate the non mass weighted dynamical matrix at a specified
        q-point from the image weighted force constants matrix and the
        indices specifying the periodic images. See eq. 1.5:
        http://www.tcm.phy.cam.ac.uk/castep/Phonons_Guide/Castep_Phonons.html

        Parameters
        ----------
        q : (3,) float ndarray
            The q-point to calculate the correction for
        fc_img_weighted : (n_cells_in_sc, 3*n_atoms, 3*n_atoms) float ndarray
            The force constants matrix weighted by the number of
            supercell atom images for each ij displacement
        unique_sc_offsets : list of lists of ints
            A list containing 3 lists of the unique supercell image
            offsets in each direction. The supercell offset is
            calculated by multiplying the supercell matrix by the
            supercell image indices (obtained by _get_all_origins()). A
            list of lists rather than a Numpy array is used as the 3
            lists are independent and their size is not known beforehand
        unique_sc_i : ((2*lim + 1)**3, 3) int ndarray
            The indices needed to reconstruct sc_offsets from the unique
            values in unique_sc_offsets
        unique_cell_origins : list of lists of ints
            A list containing 3 lists of the unique cell origins in each
            direction. A list of lists rather than a Numpy array is used
            as the 3 lists are independent and their size is not known
            beforehand
        unique_sc_i : (cell_origins, 3) int ndarray
            The indices needed to reconstruct cell_origins from the
            unique values in unique_cell_origins

        Returns
        -------
        dyn_mat : (3*n_atoms, 3*n_atoms) complex ndarray
            The non mass weighted dynamical matrix at q
        """

        sc_image_i = self._sc_image_i

        # Cumulant method: for each ij ion-ion displacement sum phases
        # for all possible supercell images, then multiply by the cell
        # phases to account for j ions in different cells. Then multiply
        # by the image weighted fc matrix for each 3 x 3 ij displacement

        # Make sc_phases 1 longer than necessary, so when summing phases
        # for supercell images if there is no image, an index of -1 and
        # hence phase of zero can be used
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
        Calculate the q-independent parts of the long range correction
        to the dynamical matrix for efficiency. The method used is based
        on the Ewald sum, see eqs 72-74 from Gonze and Lee PRB 55, 10355
        (1997)

        Parameters
        ----------
        eta_scale : float, optional
            Changes the cutoff in real/reciprocal space for the dipole
            Ewald sum. A higher value uses more reciprocal terms
        """

        cell_vectors = self.crystal._cell_vectors
        recip = self.crystal.reciprocal_cell().to('1/bohr').magnitude
        n_atoms = self.crystal.n_atoms
        atom_r = self.crystal.atom_r
        born = self._born
        dielectric = self._dielectric
        inv_dielectric = np.linalg.inv(dielectric)
        sqrt_pi = math.sqrt(math.pi)

        # Calculate real/recip weighting
        abc_mag = np.linalg.norm(cell_vectors, axis=1)
        mean_abc_mag = np.prod(abc_mag)**(1.0/3)
        eta = (sqrt_pi/mean_abc_mag)*n_atoms**(1.0/6)
        # Use eta = lambda * |permittivity|**(1/6)
        eta = eta*np.power(np.linalg.det(dielectric), 1.0/6)*eta_scale
        eta_2 = eta**2

        # Set limits and tolerances
        max_shells = 50
        frac_tol = 1e-15

        # Calculate q=0 real space term
        real_q0 = np.zeros((n_atoms, n_atoms, 3, 3))
        # No. of independent i, j ion entries (to use i, j symmetry to
        # minimise size of stored H_ab)
        n_elems = np.sum(range(1, n_atoms + 1))
        H_ab = np.zeros((0, n_elems, 3, 3))
        cells = np.zeros((0, 3))
        atom_r_cart = np.einsum('ij,jk->ik', atom_r, cell_vectors)
        atom_r_e = np.einsum('ij,jk->ik', atom_r_cart, inv_dielectric)
        for n in range(max_shells):
            cells_tmp = self._get_shell_origins(n)
            cells_cart = np.einsum('ij,jk->ik', cells_tmp, cell_vectors)
            cells_e = np.einsum(
                'ij,jk->ik', cells_cart, inv_dielectric)
            H_ab_tmp = np.zeros((len(cells_tmp), n_elems, 3, 3))
            for i in range(n_atoms):
                idx = np.sum(range(n_atoms - i, n_atoms), dtype=np.int32)
                for j in range(i, n_atoms):
                    if n == 0 and i == j:
                        continue
                    rij_cart = atom_r_cart[i] - atom_r_cart[j]
                    rij_e = atom_r_e[i] - atom_r_e[j]
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
        # Use compact H_ab to fill in upper triangular of realspace term
        real_q0[np.triu_indices(n_atoms)] = np.sum(H_ab, axis=0)
        real_q0 *= eta**3/math.sqrt(np.linalg.det(dielectric))

        # Calculate the q=0 reciprocal term
        recip_q0 = np.zeros((n_atoms, n_atoms, 3, 3), dtype=np.complex128)
        # Add G = 0 vectors to list, for later calculations when q !=0,
        # but don't calculate for q=0
        gvecs_cart = np.array([[0., 0., 0.]])
        gvec_phases = np.tile([1. + 0.j], (1, n_atoms))
        for n in range(1, max_shells):
            gvecs = self._get_shell_origins(n)
            gvecs_cart_tmp = np.einsum('ij,jk->ik', gvecs, recip)
            gvec_dot_r = np.einsum('ij,kj->ik', gvecs, atom_r)
            gvec_phases_tmp = np.exp(2j*math.pi*gvec_dot_r)
            gvecs_ab = np.einsum('ij,ik->ijk', gvecs_cart_tmp, gvecs_cart_tmp)
            k_len_2 = np.einsum('ijk,jk->i', gvecs_ab, dielectric)/(4*eta_2)
            recip_exp = np.exp(-k_len_2)/k_len_2
            recip_q0_tmp = np.zeros((n_atoms, n_atoms, 3, 3),
                                    dtype=np.complex128)
            for i in range(n_atoms):
                for j in range(i, n_atoms):
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
        vol = self.crystal._cell_volume()
        recip_q0 *= math.pi/(vol*eta_2)

        # Fill in remaining entries by symmetry
        for i in range(1, n_atoms):
            for j in range(i):
                real_q0[i, j] = np.conj(real_q0[j, i])
                recip_q0[i, j] = np.conj(recip_q0[j, i])

        # Calculate the q=0 correction, to be subtracted from the
        # corrected diagonal at each q
        dipole_q0 = np.zeros((n_atoms, 3, 3), dtype=np.complex128)
        for i in range(n_atoms):
            for j in range(n_atoms):
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
        Calculate the long range correction to the dynamical matrix
        using the Ewald sum, see eqs 72-74 from Gonze and Lee PRB 55,
        10355 (1997)

        Parameters
        ----------
        q : (3,) float ndarray
            The q-point to calculate the correction for

        Returns
        -------
        corr : (3*n_atoms, 3*n_atoms) complex ndarray
            The correction to the dynamical matrix
        """
        recip = self.crystal.reciprocal_cell().to('1/bohr').magnitude
        n_atoms = self.crystal.n_atoms
        atom_r = self.crystal.atom_r
        born = self._born
        dielectric = self._dielectric
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
        real_dipole = np.zeros((n_atoms, n_atoms, 3, 3), dtype=np.complex128)
        # Calculate real space phase factor
        q_dot_ra = np.einsum('i,ji->j', q_norm, cells)
        real_phases = np.exp(2j*math.pi*q_dot_ra)
        real_dipole_tmp = np.einsum('i,ijkl->jkl', real_phases, H_ab)
        idx_u = np.triu_indices(n_atoms)
        real_dipole[idx_u] = real_dipole_tmp
        real_dipole *= eta**3/math.sqrt(np.linalg.det(dielectric))

        # Calculate reciprocal term
        recip_dipole = np.zeros((n_atoms, n_atoms, 3, 3), dtype=np.complex128)
        # Calculate q-point phases
        q_dot_r = np.einsum('i,ji->j', q_norm, atom_r)
        q_phases = np.exp(2j*math.pi*q_dot_r)
        q_cart = np.dot(q_norm, recip)
        # Calculate k-vector symmetric matrix
        kvecs = gvecs_cart + q_cart
        kvecs_ab = np.einsum('ij,ik->ijk', kvecs, kvecs)
        k_len_2 = np.einsum('ijk,jk->i', kvecs_ab, dielectric)/(4*eta_2)
        recip_exp = np.einsum('ijk,i->ijk', kvecs_ab, np.exp(-k_len_2)/k_len_2)
        for i in range(n_atoms):
                phase_exp = ((gvec_phases[:, i, None]*q_phases[i])
                             /(gvec_phases[:, i:]*q_phases[i:]))
                recip_dipole[i, i:] = np.einsum(
                    'ikl,ij->jkl', recip_exp, phase_exp)
        cell_volume = self.crystal._cell_volume()
        recip_dipole *= math.pi/(cell_volume*eta_2)

        # Fill in remaining entries by symmetry
        # Mask so we don't count diagonal twice
        mask = np.tri(n_atoms, k=-1)[:, :, np.newaxis, np.newaxis]
        real_dipole = real_dipole + mask*np.conj(
            np.transpose(real_dipole, axes=[1, 0, 2, 3]))
        recip_dipole = recip_dipole + mask*np.conj(
            np.transpose(recip_dipole, axes=[1, 0, 2, 3]))

        # Multiply by Born charges and subtract q=0 from diagonal
        dipole = np.zeros((n_atoms, n_atoms, 3, 3), dtype=np.complex128)
        dipole_tmp = recip_dipole - real_dipole
        for i in range(n_atoms):
            dipole[i] = np.einsum('ij,klm,kjm->kil',
                                  born[i], born, dipole_tmp[i])
            dipole[i, i] -= self._dipole_q0[i]

        return np.reshape(np.transpose(dipole, axes=[0, 2, 1, 3]),
                          (3*n_atoms, 3*n_atoms))

    def _get_q_dir(self, qpts, qpts_i, gamma_idx):
        q_dirs = np.zeros((len(gamma_idx), 3))
        if len(qpts) > 1:
            for i, idx in enumerate(gamma_idx):
                idx_in_qpts = np.where(qpts_i == idx)[0]
                # If first q-point
                if idx_in_qpts == 0:
                    q_dirs[i] = qpts[1]
                # If last q-point
                elif idx_in_qpts == (len(qpts) - 1):
                    q_dirs[i] = qpts[-2]
                else:
                    # If splitting=True there should be an adjacent gamma
                    # point. Calculate splitting in whichever direction
                    # isn't gamma
                    q_dirs[i] = qpts[idx_in_qpts + 1]
                    if is_gamma(q_dirs[i]):
                        q_dirs[i] = -qpts[idx_in_qpts - 1]
        return q_dirs

    def _calculate_gamma_correction(self, q_dir):
        """
        Calculate non-analytic correction to the dynamical matrix at q=0
        for a specified direction of approach. See Eq. 60 of X. Gonze
        and C. Lee, PRB (1997) 55, 10355-10368.

        Parameters
        ----------
        q_dir : (3,) float ndarray
            The direction along which q approaches 0, in reciprocal
            fractional coordinates

        Returns
        -------
        na_corr : (3*n_atoms, 3*n_atoms) complex ndarray
            The correction to the dynamical matrix
        """
        n_atoms = self.crystal.n_atoms
        born = self._born
        dielectric = self._dielectric
        na_corr = np.zeros((3*n_atoms, 3*n_atoms), dtype=np.complex128)

        if is_gamma(q_dir):
            return na_corr

        cell_volume = self.crystal._cell_volume()
        denominator = np.einsum('ij,i,j', dielectric, q_dir, q_dir)
        factor = 4*math.pi/(cell_volume*denominator)

        q_born_sum = np.einsum('ijk,k->ij', born, q_dir)
        for i in range(n_atoms):
            for j in range(n_atoms):
                na_corr[3*i:3*(i+1), 3*j:3*(j+1)] = np.einsum(
                    'i,j->ij', q_born_sum[i], q_born_sum[j])
        na_corr *= factor

        return na_corr

    def _get_shell_origins(self, n):
        """
        Given the shell number, compute all the cell origins that lie in
        that shell

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

        # Coordinates of cells in xy plane where z=0, xz plane where
        # y=0, yz plane where x=0. Note: to avoid duplicating cells at
        # the edges, the xz plane has (2*n + 1) - 2 rows in z, rather
        # than (2*n + 1). The yz plane also has (2*n + 1) - 2 rows in z
        # and (2*n + 1) - 2 columns in y
        xy = get_all_origins([n+1, n+1, 1], min_xyz=[-n, -n, 0])
        xz = get_all_origins([n+1, 1, n], min_xyz=[-n, 0, -n+1])
        yz = get_all_origins([1, n, n], min_xyz=[0, -n+1, -n+1])

        # Offset each plane by n and -n to get the 6 planes that make up
        # the shell
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

    def _enforce_realspace_asr(self):
        """
        Apply a transformation to the force constants matrix so that it
        satisfies the acousic sum rule. Diagonalise, shift the acoustic
        modes to almost zero then construct the correction to the force
        constants matrix using the eigenvectors. For more information
        see section 2.3.4:
        http://www.tcm.phy.cam.ac.uk/castep/Phonons_Guide/Castep_Phonons.html

        Returns
        -------
        force_constants : (n_cells_in_sc, 3*n_atoms, 3*n_atoms) float ndarray
            The corrected force constants matrix
        """
        n_cells_in_sc = self.n_cells_in_sc
        n_atoms = self.crystal.n_atoms
        n_atoms_in_sc = n_cells_in_sc*n_atoms
        force_constants = self._force_constants

        sc_rel_idx = _get_supercell_relative_idx(self.cell_origins,
                                                 self.sc_matrix)

        # Compute square matrix from compact force constants
        sq_fc = np.zeros((3*n_atoms_in_sc, 3*n_atoms_in_sc))
        for nc in range(n_cells_in_sc):
            sq_fc[3*nc*n_atoms:3*(nc+1)*n_atoms, :] = np.transpose(
                np.reshape(force_constants[sc_rel_idx[nc]],
                           (3*n_cells_in_sc*n_atoms, 3*n_atoms)))

        try:
            ac_i, evals, evecs = self._find_acoustic_modes(sq_fc)
        except Exception:
            warnings.warn((
                '\nError correcting for acoustic sum rule, could not '
                'find 3 acoustic modes.\nReturning uncorrected FC '
                'matrix'), stacklevel=2)
            return self._force_constants

        # Correct fc matrix - set acoustic modes to almost zero
        fc_tol = 1e-8*np.min(np.abs(evals))
        for ac in ac_i:
            sq_fc -= (fc_tol + evals[ac])*np.einsum(
                'i,j->ij', evecs[:, ac], evecs[:, ac])

        fc = np.reshape(sq_fc[:, :3*n_atoms],
                        (n_cells_in_sc, 3*n_atoms, 3*n_atoms))

        return fc

    def _enforce_reciprocal_asr(self, dyn_mat_gamma):
        """
        Calculate the correction to the dynamical matrix that would have
        to be applied to satisfy the acousic sum rule. Diagonalise the
        gamma-point dynamical matrix, shift the acoustic modes to almost
        zero then reconstruct the dynamical matrix using the
        eigenvectors. For more information see section 2.3.4:
        http://www.tcm.phy.cam.ac.uk/castep/Phonons_Guide/Castep_Phonons.html

        Parameters
        ----------
        dyn_mat_gamma : (3*n_atoms, 3*n_atoms) complex ndarray
            The non mass-weighted dynamical matrix at gamma

        Returns
        -------
        dyn_mat : (3*n_atoms, 3*n_atoms) complex ndarray or empty array
            The corrected, non mass-weighted dynamical matrix at q.
            Returns empty array (np.array([])) if finding the 3 acoustic
            modes fails
        """
        tol = 5e-15

        try:
            ac_i, g_evals, g_evecs = self._find_acoustic_modes(
                dyn_mat_gamma)
        except Exception:
            warnings.warn(('\nError correcting for acoustic sum rule, '
                           'could not find 3 acoustic modes.\nNot '
                           'correcting dynamical matrix'), stacklevel=2)
            return np.array([], dtype=np.complex128)

        n_atoms = self.crystal.n_atoms
        recip_asr_correction = np.zeros((3*n_atoms, 3*n_atoms),
                                        dtype=np.complex128)
        for i, ac in enumerate(ac_i):
            recip_asr_correction -= (tol*i + g_evals[ac])*np.einsum(
                'i,j->ij', g_evecs[:, ac], g_evecs[:, ac])

        return recip_asr_correction

    def _find_acoustic_modes(self, dyn_mat):
        """
        Find the acoustic modes from a dynamical matrix, they should
        have the sum of c of m amplitude squared = mass (note: have not
        actually included mass weighting here so assume mass = 1.0)

        Parameters
        ----------
        dyn_mat : (3*n_atoms, 3*n_atoms) complex ndarray
            A dynamical matrix

        Returns
        -------
        ac_i : (3,) int ndarray
            The indices of the acoustic modes
        evals : (3*n_atoms) float ndarray
            Dynamical matrix eigenvalues
        evecs : (3*n_atoms, n_atoms, 3) complex ndarray
            Dynamical matrix eigenvectors
        """
        n_branches = dyn_mat.shape[0]
        n_atoms = int(n_branches/3)

        evals, evecs = np.linalg.eigh(dyn_mat, UPLO='U')
        evec_reshape = np.reshape(
            np.transpose(evecs), (n_branches, n_atoms, 3))
        # Sum displacements for all ions in each branch
        c_of_m_disp = np.sum(evec_reshape, axis=1)
        c_of_m_disp_sq = np.sum(np.abs(c_of_m_disp)**2, axis=1)
        sensitivity = 0.5
        sc_mass = 1.0*n_atoms
        # Check number of acoustic modes
        if np.sum(c_of_m_disp_sq > sensitivity*sc_mass) < 3:
            raise Exception('Could not find 3 acoustic modes')
        # Find idx of acoustic modes (3 largest c of m displacements)
        ac_i = np.argsort(c_of_m_disp_sq)[-3:]

        return ac_i, evals, evecs

    def _calculate_phases(self, q, unique_sc_offsets, unique_sc_i,
                          unique_cell_origins, unique_cell_i):
        """
        Calculate the phase factors for the supercell images and cells
        for a single q-point. The unique supercell and cell origins
        indices are required to minimise expensive exp and power
        operations

        Parameters
        ----------
        q : (3,) float ndarray
            The q-point to calculate the phase for
        unique_sc_offsets : list of lists of ints
            A list containing 3 lists of the unique supercell image
            offsets in each direction. The supercell offset is
            calculated by multiplying the supercell matrix by the
            supercell image indices (obtained by _get_all_origins()). A
            list of lists rather than a Numpy array is used as the 3
            lists are independent and their size is not known beforehand
        unique_sc_i : ((2*lim + 1)**3, 3) int ndarray
            The indices needed to reconstruct sc_offsets from the unique
            values in unique_sc_offsets
        unique_cell_origins : list of lists of ints
            A list containing 3 lists of the unique cell origins in each
            direction. A list of lists rather than a Numpy array is used
            as the 3 lists are independent and their size is not known
            beforehand
        unique_cell_i : (cell_origins, 3) int ndarray
            The indices needed to reconstruct cell_origins from the
            unique values in unique_cell_origins

        Returns
        -------
        sc_phases : (unique_sc_i,) float ndarray
            Phase factors exp(iq.r) for each supercell image coordinate
            in sc_offsets
        cell_phases : (unique_cell_i,) float ndarray
            Phase factors exp(iq.r) for each cell coordinate in the
            supercell
        """

        # Only calculate exp(iq) once, then raise to power to get the
        # phase at different supercell/cell coordinates to minimise
        # expensive exp calculations
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
        supercell, calculate the number of supercell periodic images
        there are and which supercells they reside in, and sets the
        sc_image_i, and n_sc_images ForceConstants attributes

        Parameters
        ----------
        lim : int
            The supercell image limit
        """

        n_atoms = self.crystal.n_atoms
        cell_vectors = self.crystal._cell_vectors
        atom_r = self.crystal.atom_r
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
        sc_vecs = np.einsum('ji,ik->jk', sc_matrix, cell_vectors)
        ws_list = np.einsum('ij,jk->ik', ws_frac, sc_vecs)
        inv_ws_sq = 1.0/np.sum(np.square(ws_list[1:]), axis=1)
        ws_list_norm = ws_list[1:]*inv_ws_sq[:, ax]

        # Get Cartesian coords of supercell images and ions in supercell
        sc_image_r = get_all_origins(
            np.repeat(lim, 3) + 1, min_xyz=-np.repeat(lim, 3))
        sc_image_cart = np.einsum('ij,jk->ik', sc_image_r, sc_vecs)
        sc_atom_cart = np.einsum('ijk,kl->ijl',
                                 cell_origins[:, ax, :] + atom_r[ax, :, :],
                                 cell_vectors)

        sc_image_i = np.full((n_cells_in_sc, n_atoms, n_atoms, (2*lim + 1)**3),
                             -1, dtype=np.int32)
        n_sc_images = np.zeros((n_cells_in_sc, n_atoms, n_atoms),
                               dtype=np.int32)

        # Ordering of loops here is for efficiency:
        # ions in unit cell -> periodic supercell images -> WS points
        # This is so the ion-ion vectors in each loop will be similar,
        # so they are more likely to pass/fail the WS check together
        # so the last loop can be broken early
        for i in range(n_atoms):
            rij = sc_atom_cart[0, i] - sc_atom_cart
            for im, sc_r in enumerate(sc_image_cart):
                # Get vector between j in sc image and i in unit cell
                dists = rij - sc_r
                # Only want to include images where ion < halfway to ALL
                # ws points, so compare vector to all ws points
                for n, wsp in enumerate(ws_list_norm):
                    dist_wsp = np.absolute(np.sum(dists*wsp, axis=-1))
                    if n == 0:
                        nc_idx, nj_idx = np.where(
                            dist_wsp <= (0.5*cutoff_scale + 0.001)
                        )
                        # Reindex dists to remove elements where the ion
                        # is > halfway to WS point for efficiency
                        dists = dists[nc_idx, nj_idx]
                    else:
                        # After first reindex, dists is now 1D so need
                        # to reindex like this instead
                        idx = np.where(
                            dist_wsp <= (0.5*cutoff_scale + 0.001)
                        )[0]
                        nc_idx = nc_idx[idx]
                        nj_idx = nj_idx[idx]
                        dists = dists[idx]
                    if len(nc_idx) == 0:
                        break
                    # If ion-ion vector has been < halfway to all WS
                    # points, this is a valid image! Save it
                    if n == len(ws_list_norm) - 1:
                        n_im_idx = n_sc_images[nc_idx, i, nj_idx]
                        sc_image_i[nc_idx, i, nj_idx, n_im_idx] = im
                        n_sc_images[nc_idx, i, nj_idx] += 1

        self._n_sc_images = n_sc_images
        # Truncate sc_image_i to the maximum ACTUAL images rather than
        # the maximum possible images to avoid storing and summing over
        # nonexistent images
        self._sc_image_i = sc_image_i[:, :, :, :np.max(n_sc_images)]

    def to_dict(self):
        """
        Convert to a dictionary. See ForceConstants.from_dict for
        details on keys/values

        Returns
        -------
        dict
        """
        dout = _obj_to_dict(self, ['crystal', 'force_constants',
                                   'n_cells_in_sc', 'sc_matrix',
                                   'cell_origins', 'born',
                                   'dielectric'])
        return dout

    def to_json_file(self, filename):
        """
        Write to a JSON file. JSON fields are equivalent to
        ForceConstants.from_dict keys

        Parameters
        ----------
        filename : str
            Name of the JSON file to write to
        """
        _obj_to_json_file(self, filename)

    @classmethod
    def from_dict(cls, d):
        """
        Convert a dictionary to a ForceConstants object

        Parameters
        ----------
        d : dict
            A dictionary with the following keys/values:

            - 'crystal': dict, see Crystal.from_dict
            - 'force_constants': (n_cells_in_sc, 3*crystal.n_atoms, 3*crystal.n_atoms) float ndarray
            - 'force_constants_unit': str
            - 'sc_matrix': (3,3) int ndarray
            - 'cell_origins': (n_cells_in_sc, 3) int ndarray

            There are also the following optional keys:

            - 'born': (3*crystal.n_atoms, 3, 3) float ndarray
            - 'born_unit': str
            - 'dielectric': (3, 3) float ndarray
            - 'dielectric_unit': str

        Returns
        -------
        ForceConstants
        """
        crystal = Crystal.from_dict(d['crystal'])
        d = _process_dict(
            d, quantities=['force_constants', 'born', 'dielectric'],
            optional=['born', 'dielectric'])
        return ForceConstants(crystal, d['force_constants'], d['sc_matrix'],
                              d['cell_origins'], d['born'], d['dielectric'])

    @classmethod
    def from_json_file(cls, filename):
        """
        Read from a JSON file. See ForceConstants.from_dict for required
        fields

        Parameters
        ----------
        filename : str
            The file to read from

        Returns
        -------
        ForceConstants
        """
        return _obj_from_json_file(cls, filename)

    @classmethod
    def from_castep(cls, filename):
        """
        Reads from a .castep_bin or .check file

        Parameters
        ----------
        filename : str
            The path and name of the file to read

        Returns
        -------
        ForceConstants
        """
        data = castep._read_interpolation_data(filename)
        return cls.from_dict(data)

    @classmethod
    def from_phonopy(cls, path='.', summary_name='phonopy.yaml',
                     born_name=None, fc_name='FORCE_CONSTANTS',
                     fc_format=None):
        """
        Reads data from the phonopy summary file (default phonopy.yaml)
        and optionally born and force constants files. Only attempts to
        read from born or force constants files if these can't be found
        in the summary file.

        Parameters
        ----------
        path : str, optional
            Path to directory containing the file(s)
        summary_name : str, optional
            Filename of phonopy summary file, default phonopy.yaml. By
            default any information (e.g. force constants) read from
            this file takes priority
        born_name : str, optional
            Name of the Phonopy file containing born charges and
            dielectric tensor (by convention in Phonopy this would be
            called BORN). Is only read if Born charges can't be found in
            the summary_name file
        fc_name : str, optional
            Name of file containing force constants. Is only read if
            force constants can't be found in summary_name
        fc_format : {'phonopy', 'hdf5'} str, optional
            Format of file containing force constants data.
            FORCE_CONSTANTS is type 'phonopy'

        Returns
        -------
        ForceConstants
        """
        data = phonopy._read_interpolation_data(
            path=path, summary_name=summary_name, born_name=born_name,
            fc_name=fc_name, fc_format=fc_format)
        return cls.from_dict(data)
