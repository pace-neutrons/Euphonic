import math
import struct
import sys
import os
import numpy as np
from scipy.linalg.lapack import zheev
from casteppy import ureg
from casteppy.data.data import Data

class InterpolationData(Data):
    """
    A class to read the data required for a supercell phonon interpolation
    calculation from a .castep_bin file, and store any calculated
    frequencies/eigenvectors

    Attributes
    ----------
    n_ions : int
        Number of ions in the unit cell
    n_branches : int
        Number of phonon dispersion branches
    cell_vec : ndarray
        The unit cell vectors. Default units Angstroms.
        dtype = 'float'
        shape = (3, 3)
    ion_r : ndarray
        The fractional position of each ion within the unit cell
        dtype = 'float'
        shape = (n_ions, 3)
    ion_type : ndarray
        The chemical symbols of each ion in the unit cell. Ions are in the
        same order as in ion_r
        dtype = 'string'
        shape = (n_ions,)
    ion_mass : ndarray
        The mass of each ion in the unit cell in atomic units
        dtype = 'float'
        shape = (n_ions,)
    n_cells_in_sc : int
        Number of cells in the supercell
    sc_matrix : ndarray
        The supercell matrix
        dtype = 'int'
        shape = (3, 3)
    cell_origins : ndarray
        The locations of the unit cells within the supercell
        dtype = 'int'
        shape = (n_cells_in_sc, 3)
    force_constants : ndarraylist of floats
        Force constants matrix. Default units atomic units
        dtype = 'float'
        shape = (n_ions, n_ions*n_cells_in_sc, 3, 3)
    n_qpts : int
        Number of q-points used in the most recent interpolation calculation.
        Default value 0
    qpts : ndarray
        Coordinates of the q-points used for the most recent interpolation
        calculation. Is empty by default
        dtype = 'float'
        shape = (n_qpts, 3)
    freqs: ndarray
        Phonon frequencies from the most recent interpolation calculation.
        Default units eV. Is empty by default
        dtype = 'float'
        shape = (n_qpts, 3*n_ions)
    eigenvecs: ndarray
        Atomic displacements (dynamical matrix eigenvectors) from the most
        recent interpolation calculation. Is empty by default
        dtype = 'complex'
        shape = (n_qpts, 3*n_ions, n_ions, 3)
    n_sc_images : ndarray
        The number or periodic supercell images for each displacement of ion i
        in the unit cell and ion j in the supercell. This attribute doesn't
        exist until calculate_fine_phonons has been called
        dtype = 'int'
        shape = (n_ions, n_ions*n_cells_in_sc)
    sc_image_i : ndarray
        The index describing the supercell each of the periodic images resides
        in. This is the index of the list of supercells as returned by
        _calculate_supercell_image_r. This attribute doesn't exist until
        calculate_fine_phonons has been called
        dtype = 'int'
        shape = (n_ions, n_ions*n_cells_in_sc, (2*lim + 1)**3)
    """

    def __init__(self, seedname, path='', qpts=np.array([])):
        """"
        Reads .castep_bin file, sets attributes, and calculates
        frequencies/eigenvectors at specific q-points if requested

        Parameters
        ----------
        seedname : str
            Name of .castep_bin file to read
        path : str, optional
            Path to dir containing the .castep_bin file, if it is in another 
            directory
        qpts : ndarray, optional
            Q-point coordinates to use for an initial interpolation calculation
            dtype = 'float'
            shape = (n_qpts, 3)
        """
        self._get_data(seedname, path)

        self.qpts = qpts
        self.n_qpts = len(qpts)
        self.eigenvecs = np.array([])
        energy_units = '{}'.format(self.force_constants.units).split('/')[0]
        self.freqs = np.array([])*ureg[energy_units]

        if self.n_qpts > 0:
            self.calculate_fine_phonons(qpts)


    def _get_data(self, seedname, path):
        """"
        Opens .castep_bin file for reading

        Parameters
        ----------
        seedname : str
            Name of .castep_bin file to read
        path : str
            Path to dir containing the .castep_bin file, if it is in another 
            directory
        """
        file = os.path.join(path, seedname + '.castep_bin')
        with open(file, 'rb') as f:
            self._read_interpolation_data(f)


    def _read_interpolation_data(self, file_obj):
        """
        Reads data from .castep_bin file and sets attributes

        Parameters
        ----------
        f : file object
            File object in read mode for the .castep_bin file containing the
            data
        """

        def read_entry(file_obj, dtype=''):
            """
            Read a record from a Fortran binary file, including the beginning
            and end record markers and the data inbetween
            """
            def record_mark_read(file_obj):
                # Read 4 byte Fortran record marker
                return struct.unpack('>i', file_obj.read(4))[0]

            begin = record_mark_read(file_obj)
            if dtype:
                n_bytes = int(dtype[-1])
                n_elems = int(begin/n_bytes)
                if n_elems > 1:
                    data = np.fromfile(file_obj, dtype=dtype, count=n_elems)
                else:
                    if 'i' in dtype:
                        data = struct.unpack('>i', file_obj.read(begin))[0]
                    elif 'f' in dtype:
                        data = struct.unpack('>d', file_obj.read(begin))[0]
                    else:
                        data = file_obj.read(begin)
            else:
                data = file_obj.read(begin)
            end = record_mark_read(file_obj)
            if begin != end:
                sys.exit("""Problem reading binary file: beginning and end
                            record markers do not match""")

            return data

        int_type = '>i4'
        float_type = '>f8'

        header = ''
        while header.strip() != b'END':
            header = read_entry(file_obj)
            if header.strip() == b'CELL%NUM_IONS':
                n_ions = read_entry(file_obj, int_type)
            elif header.strip() == b'CELL%REAL_LATTICE':
                cell_vec = np.transpose(np.reshape(
                    read_entry(file_obj, float_type), (3, 3)))
            elif header.strip() == b'CELL%NUM_SPECIES':
                n_species = read_entry(file_obj, int_type)
            elif header.strip() == b'CELL%NUM_IONS_IN_SPECIES':
                n_ions_in_species = read_entry(file_obj, int_type)
                if n_species == 1:
                    n_ions_in_species = np.array([n_ions_in_species])
            elif header.strip() == b'CELL%IONIC_POSITIONS':
                max_ions_in_species = max(n_ions_in_species)
                ion_r_tmp = np.reshape(read_entry(file_obj, float_type),
                                  (n_species, max_ions_in_species, 3))
            elif header.strip() == b'CELL%SPECIES_MASS':
                ion_mass_tmp = read_entry(file_obj, float_type)
                if n_species == 1:
                    ion_mass_tmp = np.array([ion_mass_tmp])
            elif header.strip() == b'CELL%SPECIES_SYMBOL':
                # Need to decode binary string for Python 3 compatibility
                if n_species == 1:
                    ion_type_tmp = [read_entry(file_obj, 'S8').strip().decode('utf-8')]
                else:
                    ion_type_tmp = [x.strip().decode('utf-8') for x in read_entry(file_obj, 'S8')]
            elif header.strip() == b'FORCE_CON':
                sc_matrix = np.transpose(np.reshape(
                    read_entry(file_obj, int_type), (3, 3)))
                n_cells_in_sc = int(np.rint(np.absolute(
                    np.linalg.det(sc_matrix))))
                fc_tmp = np.reshape(
                    read_entry(file_obj, float_type),
                    (n_cells_in_sc*n_ions, 3, n_ions, 3))
                fc_tmp = np.transpose(fc_tmp, axes=[2, 0, 1, 3])
                force_constants = np.zeros(
                    (n_ions, n_cells_in_sc*n_ions, 3, 3))
                for nc in range(n_cells_in_sc):
                    force_constants[:, nc*n_ions:(nc + 1)*n_ions] = np.transpose(
                        fc_tmp[:, nc*n_ions:(nc + 1)*n_ions], axes=[1, 0, 2, 3])
                cell_origins = np.reshape(
                    read_entry(file_obj, int_type), (n_cells_in_sc, 3))
                fc_row = read_entry(file_obj, int_type)

        # Get ion_r in correct form
        # CASTEP stores ion positions as 3D array (3,
        # max_ions_in_species, n_species) so need to slice data to get
        # correct information
        ion_begin = np.insert(np.cumsum(n_ions_in_species[:-1]), 0, 0)
        ion_end = np.cumsum(n_ions_in_species)
        ion_r = np.zeros((n_ions, 3))
        for i in range(n_species):
                ion_r[ion_begin[i]:ion_end[i], :] = ion_r_tmp[
                    i,:n_ions_in_species[i], :]
        # Get ion_type in correct form
        ion_type = np.array([])
        ion_mass = np.array([])
        for ion in range(n_species):
            ion_type = np.append(ion_type, [ion_type_tmp[ion] for i in
                range(n_ions_in_species[ion])])
            ion_mass = np.append(ion_mass, [ion_mass_tmp[ion] for i in
                range(n_ions_in_species[ion])])

        cell_vec = cell_vec*ureg.bohr
        cell_vec.ito('angstrom')
        ion_mass = ion_mass*ureg.e_mass
        ion_mass.ito('amu')
        force_constants = force_constants*ureg.hartree/(ureg.bohr**2)

        self.n_ions = n_ions
        self.n_branches = 3*n_ions
        self.cell_vec = cell_vec
        self.ion_r = ion_r
        self.ion_type = ion_type
        self.ion_mass = ion_mass
        self.sc_matrix = sc_matrix
        self.n_cells_in_sc = n_cells_in_sc
        self.force_constants = force_constants
        self.cell_origins = cell_origins


    def calculate_fine_phonons(self, qpts, asr=True):
        """
        Calculate phonon frequencies and eigenvectors at specified q-points
        from a supercell force constant matrix via interpolation, and set
        InterpolationData freqs and eigenvecs attributes. For more information
        on the method see section 2.5:
        http://www.tcm.phy.cam.ac.uk/castep/Phonons_Guide/Castep_Phonons.html

        Parameters
        ----------
        qpts : ndarray
            The q-points to interpolate onto
            dtype = 'float'
            shape = (n_qpts, 3)
        asr : boolean, optional, default True
            Whether to apply an acoustic sum rule correction to the force
            constant matrix

        Returns
        -------
        freqs : ndarray
            The phonon frequencies (same as set to InterpolationData.freqs)
            dtype = 'float'
            shape = (n_qpts, 3*n_ions)
        eigenvecs : ndarray
            The phonon eigenvectors (same as set to
            InterpolationData.eigenvecs)
            dtype = 'complex'
            shape = (n_qpts, 3*n_ions, n_ions, 3)
        """

        if asr:
            force_constants = self._enforce_acoustic_sum_rule().magnitude
        else:
            force_constants = self.force_constants.magnitude
        ion_mass = self.ion_mass.to('e_mass').magnitude
        sc_matrix = self.sc_matrix
        cell_origins = self.cell_origins
        n_cells_in_sc = self.n_cells_in_sc
        n_ions = self.n_ions
        n_branches = self.n_branches
        n_qpts = qpts.shape[0]
        freqs = np.zeros((n_qpts, n_branches))
        eigenvecs = np.zeros((n_qpts, n_branches, n_ions, 3),
                             dtype=np.complex128)

        # Build list of all possible supercell image coordinates
        lim = 2 # Supercell image limit
        sc_image_r = self._calculate_supercell_image_r(lim)
        # Calculate origin coordinates of each cell in each supercell image
        sc_offsets = np.einsum('ji,kj->ki', sc_matrix, sc_image_r)
        cell_r = np.tile(sc_offsets, (n_cells_in_sc, 1)) + np.repeat(
            cell_origins, len(sc_image_r), axis=0)
        # Construct list of supercell ion images
        if not hasattr(self, 'sc_image_i'):
            self._calculate_supercell_images(lim)
        n_sc_images = self.n_sc_images
        max_sc_images = np.max(self.n_sc_images)
        sc_image_i = self.sc_image_i

        # Precompute fc matrix weighted by number of supercell ion images
        # (for cumulant method)
        fc_img_weighted = np.divide(
            force_constants,
            n_sc_images[:, :, np.newaxis, np.newaxis],
            where=n_sc_images[:, :, np.newaxis, np.newaxis] != 0)

        # Precompute dynamical matrix mass weighting
        masses = np.tile(np.repeat(ion_mass, 3), (3*n_ions, 1))
        dyn_mat_weighting = 1/np.sqrt(masses*np.transpose(masses))

        for q in range(n_qpts):
            qpt = qpts[q, :]
            dyn_mat = np.zeros((n_ions*3, n_ions*3), dtype=np.complex128)

            # Cumulant method: For each cell in the supercell, sum phases for
            # all supercell images and multiply by image weighted fc matrix
            # for each 3 x 3 ij displacement 
            for nc in range(n_cells_in_sc):
                phases = self._calculate_phases(
                    qpt, cell_r[nc*len(sc_image_r):(nc+1)*len(sc_image_r)])

                phase_sum = np.sum(phases[sc_image_i[:,
                    nc*n_ions:(nc+1)*n_ions, 0:max_sc_images]], axis=2)
                terms = np.einsum('ij,ijkl->ijkl',
                                  phase_sum, 
                                  fc_img_weighted[:, nc*n_ions:(nc+1)*n_ions])
                dyn_mat += np.reshape(np.transpose(
                    terms, axes=[0, 2, 1, 3]), (3*n_ions, 3*n_ions))

            # Mass weight dynamical matrix
            dyn_mat *= dyn_mat_weighting

            try:
                evals, evecs = np.linalg.eigh(dyn_mat)
            # Fall back to zheev if eigh fails (eigh calls zheevd)
            except np.linalg.LinAlgError:
                evals, evecs, info = zheev(dyn_mat)

            eigenvecs[q, :] = np.reshape(np.transpose(evecs), (n_branches, n_ions, 3))
            freqs[q, :] = np.sqrt(np.abs(evals))

            # Set imaginary frequencies to negative
            imag_freqs = np.where(evals < 0)
            freqs[q, imag_freqs] *= -1

        self.n_qpts = n_qpts
        self.qpts = qpts
        self.freqs = freqs*self.freqs.units
        self.eigenvecs = eigenvecs

        return self.freqs, self.eigenvecs


    def _enforce_acoustic_sum_rule(self):
        """
        Apply a transformation to the force constants matrix so that it
        satisfies the acousic sum rule. For more information on the method
        see section 2.3.4:
        http://www.tcm.phy.cam.ac.uk/castep/Phonons_Guide/Castep_Phonons.html

        Returns
        -------
        force_constants : ndarray
            The corrected force constants matrix
            dtype = 'float'
            shape = (n_ions, n_ions*n_cells_in_sc, 3, 3)
        """
        cell_vec = self.cell_vec
        cell_origins = self.cell_origins
        sc_matrix = self.sc_matrix
        n_cells_in_sc = self.n_cells_in_sc
        n_ions = self.n_ions
        n_branches = self.n_branches

        # Compute square matrix giving relative index of cells in sc
        # Get all possible cell-cell vector combinations
        inter_cell_vectors = (np.tile(cell_origins, (n_cells_in_sc, 1)) -
                             np.repeat(cell_origins, n_cells_in_sc, axis=0))
        # Compare cell-cell vectors with origin-cell vectors and determine
        # which are equivalent
        offset = (inter_cell_vectors[:, np.newaxis, :] -
                 cell_origins[np.newaxis, :, :])
        inv_sc_matrix = np.linalg.inv(np.transpose(sc_matrix))
        dist = np.einsum('ijk,lk->ijl', offset, inv_sc_matrix)
        dist_frac, _ = np.modf(dist)
        dist_frac_sum = np.sum(np.abs(dist_frac), axis=2)
        sc_relative_index = np.argmin(dist_frac_sum, axis=1)
        if (np.any(dist_frac_sum[range(n_cells_in_sc**2), sc_relative_index])
            > 16*sys.float_info.epsilon):
            print('Error correcting FC matrix for acoustic sum rule, ' +
                  'supercell relative index couldn\'t be found. Returning ' +
                  'uncorrected FC matrix')
            return self.force_constants

        # Construct square FC matrix
        n_ions_in_sc = n_ions*n_cells_in_sc
        sq_fc = np.zeros((3*n_ions_in_sc, 3*n_ions_in_sc))
        for nc in range(n_cells_in_sc):
            cell_indices = np.repeat(sc_relative_index[
                nc*n_cells_in_sc:(nc+1)*n_cells_in_sc], n_ions)*n_ions
            ion_indices = np.tile(range(n_ions), n_cells_in_sc)
            fc_indices = cell_indices + ion_indices
            sq_fc[3*nc*n_ions:3*(nc+1)*n_ions, :] = np.reshape(np.transpose(
                self.force_constants[:, fc_indices],
                axes=[0, 2, 1, 3]), (3*n_ions, 3*n_ions*n_cells_in_sc))

        # Find acoustic modes, they should have the sum of c of m amplitude
        # squared = mass (note: have not actually included mass weighting
        # here so assume mass = 1.0)
        evals, evecs = np.linalg.eigh(sq_fc)
        n_sc_branches = n_ions_in_sc*3
        evec_reshape = np.reshape(
            np.transpose(evecs), (n_sc_branches, n_ions_in_sc, 3))
        # Sum displacements for all ions in each branch
        c_of_m_disp = np.sum(evec_reshape, axis=1)
        c_of_m_disp_sq = np.sum(np.abs(c_of_m_disp)**2, axis=1)
        sensitivity = 0.5
        sc_mass = 1.0*n_cells_in_sc
        # Check number of acoustic modes
        if np.sum(c_of_m_disp_sq > sensitivity*sc_mass) < 3:
            print('Error correcting FC matrix for acoustic sum rule, could' +
                  'not find 3 acoustic modes. Returning uncorrected FC matrix')
            return self.force_constants
        # Find indices of acoustic modes (3 largest c of m displacements)
        ac_i = np.argsort(c_of_m_disp_sq)[-3:]
        # Correct force constant matrix - set acoustic modes to almost zero
        fc_tol = 1e-8*np.min(np.abs(evals))
        for ac in ac_i:
            sq_fc -= (fc_tol + evals[ac])*np.einsum('i,j->ij', evecs[:, ac], evecs[:, ac])

        force_constants = np.zeros((n_ions, n_ions_in_sc, 3, 3))
        for i in range(n_ions):
            for j in range(n_ions_in_sc):
                force_constants[i, j] = sq_fc[3*i:(3*i + 3), 3*j:3*j + 3]
        force_constants = force_constants*self.force_constants.units
        return force_constants


    def _calculate_supercell_image_r(self, lim):
        """
        Calculate a list of all the possible supercell image coordinates up to
        a certain limit

        Parameters
        ----------
        lim : int
            The supercell image limit

        Returns
        -------
        sc_image_r : ndarray
            A list of the possible supercell image coordinates
            e.g. if lim = 2: [[-2, -2, -2], [-2, -2, -1] ... [2, 2, 2]]
            dtype = 'int'
            shape = ((2*lim + 1)**3, 3)
        """
        irange = range(-lim, lim + 1)
        inum = 2*lim + 1
        scx = np.repeat(irange, inum**2)
        scy = np.tile(np.repeat(irange, inum), inum)
        scz = np.tile(irange, inum**2)

        return np.column_stack((scx, scy, scz))


    def _calculate_phases(self, qpt, cell_r):
        """
        Calculate the dynamical matrix phase factor

        Parameters
        ----------
        qpt : ndarray
            The q-point to calculate the phase for
            dtype = 'float'
            shape = (3,)
        cell_r : ndarray
            The cell coordinates to calculate the phase for. These are the
            origin coordinates of a unit cell in each supercell image in
            unit cell vector units
            dtype = 'int'
            shape = ((2*lim + 1)**3, 3)

        Returns
        -------
        phases : ndarray
            Phase factors exp(iq.r) for each cell coordinate in cell_r. The
            last entry is a phase of 0 so that when summing supercell image
            phases for the cumulant method, if there isn't an image for that
            ion a phase of zero can be used
            dtype = 'float'
            shape = (cell_r + 1,)
        """
        phases = np.zeros(len(cell_r) + 1, dtype=np.complex128)
        phases[:-1] = np.exp(2j*math.pi*np.einsum('ij,j->i', cell_r, qpt))

        return phases


    def _calculate_supercell_images(self, lim):
        """
        For each displacement of ion i in the unit cell and ion j in the
        supercell, calculate the number of supercell periodic images there are
        and which supercells they reside in, and sets the sc_image_i and
        n_sc_images InterpolationData attributes

        Parameters
        ----------
        lim : int
            The supercell image limit
        """

        n_ions = self.n_ions
        cell_vec = self.cell_vec.to(ureg.bohr).magnitude
        ion_r = self.ion_r - np.floor(self.ion_r)
        cell_origins = self.cell_origins
        n_cells_in_sc = self.n_cells_in_sc
        sc_matrix = self.sc_matrix

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

        # Get Cartesian coords of supercell images and ions in supercell
        sc_image_r = self._calculate_supercell_image_r(lim)
        sc_image_cart = np.einsum('ij,jk->ik', sc_image_r, sc_vecs)
        sc_ion_r = np.dot(np.tile(ion_r, (n_cells_in_sc, 1)) + np.repeat(
            cell_origins, n_ions, axis=0), np.linalg.inv(np.transpose(sc_matrix)))
        sc_ion_cart = np.einsum('ij,jk->ik', sc_ion_r, sc_vecs)

        sc_image_i = np.full((self.n_ions, 
                             self.n_ions*self.n_cells_in_sc,
                             (2*lim + 1)**3), -1, dtype=np.int8)
        n_sc_images = np.zeros((self.n_ions, self.n_ions*self.n_cells_in_sc),
                               dtype=np.int8)
        for i in range(n_ions):
            for j in range(n_ions*n_cells_in_sc):
                # Get distance between ions in every supercell
                dist = sc_ion_cart[i] - sc_ion_cart[j] - sc_image_cart
                for k in range(len(sc_image_r)):
                    dist_ws_point = np.absolute(
                        np.dot(ws_list[1:], dist[k, :])*inv_ws_sq)
                    if np.max(dist_ws_point) <= (0.5*cutoff_scale + 0.001):
                        sc_image_i[i, j, n_sc_images[i, j]] = k
                        n_sc_images[i, j] += 1

        self.sc_image_i = sc_image_i
        self.n_sc_images = n_sc_images


    def convert_e_units(self, units):
        """
        Convert energy units of relevant attributes in place e.g. freqs,
        dos_bins

        Parameters
        ----------
        units : str
            The units to convert to e.g. '1/cm', 'hartree', 'eV'
        """
        super(InterpolationData, self).convert_e_units(units)

        if hasattr(self, 'freqs'):
            self.freqs.ito(units, 'spectroscopy')
