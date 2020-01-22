import re
import os
import struct
import numpy as np
from euphonic import ureg
from euphonic.util import reciprocal_lattice, is_gamma


def _read_phonon_data(seedname, path):
    """
    Reads data from a .phonon file and returns it in a dictionary

    Parameters
    ----------
    seedname : str
        Seedname of file(s) to read
    path : str
        Path to dir containing the file(s), if in another directory

    Returns
    -------
    data_dict : dict
        A dict with the following keys: 'n_ions', 'n_branches', 'n_qpts'
        'cell_vec', 'recip_vec', 'ion_r', 'ion_type', 'ion_mass', 'qpts',
        'weights', 'freqs', 'eigenvecs', 'split_i', 'split_freqs',
        'split_eigenvecs'
        Meta information: 'seedname', 'path' and 'model'.
    """
    file = os.path.join(path, seedname + '.phonon')
    with open(file, 'r') as f:

        (n_ions, n_branches, n_qpts, cell_vec, ion_r,
         ion_type, ion_mass) = _read_phonon_header(f)

        qpts = np.zeros((n_qpts, 3))
        weights = np.zeros(n_qpts)
        freqs = np.zeros((n_qpts, n_branches))
        ir = np.array([])
        raman = np.array([])
        eigenvecs = np.zeros((n_qpts, n_branches, n_ions, 3),
                             dtype='complex128')
        split_i = np.array([], dtype=np.int32)
        split_freqs = np.empty((0, n_branches))
        split_eigenvecs = np.empty((0, n_branches, n_ions, 3))

        # Need to loop through file using while rather than number of q-points
        # as sometimes points are duplicated
        first_qpt = True
        qpt_line = f.readline()
        prev_qpt_num = -1
        qpt_num_patt = re.compile('q-pt=\s*(\d+)')
        float_patt = re.compile('-?\d+\.\d+')
        while qpt_line:
            qpt_num = int(re.search(qpt_num_patt, qpt_line).group(1)) - 1
            floats = re.findall(float_patt, qpt_line)
            qpts[qpt_num] = [float(x) for x in floats[:3]]
            weights[qpt_num] = float(floats[3])

            freq_lines = [f.readline().split() for i in range(n_branches)]
            tmp = np.array([float(line[1]) for line in freq_lines])
            if qpt_num != prev_qpt_num:
                freqs[qpt_num, :] = tmp
            elif is_gamma(qpts[qpt_num]):
                split_i = np.concatenate((split_i, [qpt_num]))
                split_freqs = np.concatenate((split_freqs, tmp[np.newaxis]))
            ir_index = 2
            raman_index = 3
            if is_gamma(qpts[qpt_num]):
                ir_index += 1
                raman_index += 1
            if len(freq_lines[0]) > ir_index:
                if first_qpt:
                    ir = np.zeros((n_qpts, n_branches))
                ir[qpt_num, :] = [float(
                    line[ir_index]) for line in freq_lines]
            if len(freq_lines[0]) > raman_index:
                if first_qpt:
                    raman = np.zeros((n_qpts, n_branches))
                raman[qpt_num, :] = [float(
                    line[raman_index]) for line in freq_lines]

            [f.readline() for x in range(2)]  # Skip 2 label lines
            lines = np.array([f.readline().split()[2:]
                              for x in range(n_ions*n_branches)],
                             dtype=np.float64)
            lines_i = np.column_stack(([lines[:, 0] + lines[:, 1]*1j,
                                        lines[:, 2] + lines[:, 3]*1j,
                                        lines[:, 4] + lines[:, 5]*1j]))
            tmp = np.zeros((n_branches, n_ions, 3), dtype=np.complex128)
            for i in range(n_branches):
                    tmp[i, :, :] = lines_i[i*n_ions:(i+1)*n_ions, :]
            if qpt_num != prev_qpt_num:
                eigenvecs[qpt_num] = tmp
            elif is_gamma(qpts[qpt_num]):
                split_eigenvecs = np.concatenate(
                    (split_eigenvecs, tmp[np.newaxis]))
            first_qpt = False
            qpt_line = f.readline()
            prev_qpt_num = qpt_num

    data_dict = {}
    data_dict['n_ions'] = n_ions
    data_dict['n_branches'] = n_branches
    data_dict['n_qpts'] = n_qpts
    data_dict['cell_vec'] = (cell_vec*ureg('angstrom').to('bohr')).magnitude
    data_dict['recip_vec'] = ((reciprocal_lattice(cell_vec)/ureg.angstrom)
                              .to('1/bohr')).magnitude
    data_dict['ion_r'] = ion_r
    data_dict['ion_type'] = ion_type
    data_dict['ion_mass'] = ion_mass*(ureg('amu')).to('e_mass').magnitude
    data_dict['qpts'] = qpts
    data_dict['weights'] = weights
    data_dict['freqs'] = ((freqs*(1/ureg.cm)).
                          to('E_h', 'spectroscopy')).magnitude
    data_dict['eigenvecs'] = eigenvecs
    data_dict['split_i'] = split_i
    data_dict['split_freqs'] = ((split_freqs*(1/ureg.cm))
                                .to('E_h', 'spectroscopy')).magnitude
    data_dict['split_eigenvecs'] = split_eigenvecs

    # Meta information
    data_dict['model'] = 'CASTEP'
    data_dict['seedname'] = seedname
    data_dict['path'] = path

    return data_dict


def _read_phonon_header(f):
    """
    Reads the header of a .phonon file

    Parameters
    ----------
    f : file object
        File object in read mode for the .phonon file containing the data

    Returns
    -------
    n_ions : integer
        The number of ions per unit cell
    n_branches : integer
        The number of phonon branches (3*n_ions)
    n_qpts : integer
        The number of q-points in the .phonon file
    cell_vec : (3, 3) float ndarray
        The unit cell vectors in Angstroms.
    ion_r : (n_ions, 3) float ndarray
        The fractional position of each ion within the unit cell
    ion_type : (n_ions,) string ndarray
        The chemical symbols of each ion in the unit cell. Ions are in the
        same order as in ion_r
    ion_mass : (n_ions,) float ndarray
        The mass of each ion in the unit cell in atomic units
    """
    f.readline()  # Skip BEGIN header
    n_ions = int(f.readline().split()[3])
    n_branches = int(f.readline().split()[3])
    n_qpts = int(f.readline().split()[3])
    [f.readline() for x in range(4)]  # Skip units and label lines
    cell_vec = np.array([[float(x) for x in f.readline().split()[0:3]]
                         for i in range(3)])
    f.readline()  # Skip fractional co-ordinates label
    ion_info = np.array([f.readline().split() for i in range(n_ions)])
    ion_r = np.array([[float(x) for x in y[1:4]] for y in ion_info])
    ion_type = np.array([x[4] for x in ion_info])
    ion_mass = np.array([float(x[5]) for x in ion_info])
    f.readline()  # Skip END header line

    return n_ions, n_branches, n_qpts, cell_vec, ion_r, ion_type, ion_mass


def _read_interpolation_data(seedname, path):
    """
    Reads data from a .castep_bin or .check file and returns it in a dictionary

    Parameters
    ----------
    seedname : str
        Seedname of file(s) to read
    path : str
        Path to dir containing the file(s), if in another directory

    Returns
    -------
    data_dict : dict
        A dict with the following keys: 'n_ions', 'n_branches', 'cell_vec',
        'recip_vec', 'ion_r', 'ion_type', 'ion_mass', 'force_constants',
        'sc_matrix', 'n_cells_in_sc' and 'cell_origins'. Also contains 'born'
        and 'dielectric' if they are present in the .castep_bin or .check file.
        Meta information: 'seedname', 'path' and 'model'.
    """
    file = os.path.join(path, seedname + '.castep_bin')
    if not os.path.isfile(file):
        print(
            '{:s}.castep_bin file not found, trying to read {:s}.check'
            .format(seedname, seedname))
        file = os.path.join(path, seedname + '.check')

    with open(file, 'rb') as f:
        int_type = '>i4'
        float_type = '>f8'
        header = ''
        first_cell_read = True
        while header.strip() != b'END':
            header = _read_entry(f)
            if header.strip() == b'BEGIN_UNIT_CELL':
                # CASTEP writes the cell twice: the first is the geometry
                # optimised cell, the second is the original cell. We only
                # want the geometry optimised cell.
                if first_cell_read:
                    n_ions, cell_vec, ion_r, ion_mass, ion_type = _read_cell(
                        f, int_type, float_type)
                    first_cell_read = False
            elif header.strip() == b'FORCE_CON':
                sc_matrix = np.transpose(np.reshape(
                    _read_entry(f, int_type), (3, 3)))
                n_cells_in_sc = int(np.rint(np.absolute(
                    np.linalg.det(sc_matrix))))
                # Transpose and reshape fc so it is indexed [nc, i, j]
                force_constants = np.ascontiguousarray(np.transpose(
                    np.reshape(_read_entry(f, float_type),
                               (n_cells_in_sc, 3*n_ions, 3*n_ions)),
                    axes=[0, 2, 1]))
                cell_origins = np.reshape(
                    _read_entry(f, int_type), (n_cells_in_sc, 3))
                fc_row = _read_entry(f, int_type)
            elif header.strip() == b'BORN_CHGS':
                born = np.reshape(
                    _read_entry(f, float_type), (n_ions, 3, 3))
            elif header.strip() == b'DIELECTRIC':
                dielectric = np.transpose(np.reshape(
                    _read_entry(f, float_type), (3, 3)))

    data_dict = {}
    data_dict['n_ions'] = n_ions
    data_dict['n_branches'] = 3*n_ions
    data_dict['cell_vec'] = cell_vec
    data_dict['recip_vec'] = reciprocal_lattice(cell_vec)
    data_dict['ion_r'] = ion_r - np.floor(ion_r)  # Normalise ion coordinates
    data_dict['ion_type'] = ion_type
    data_dict['ion_mass'] = ion_mass

    # Set entries relating to 'FORCE_CON' block
    try:
        data_dict['force_constants'] = force_constants
        data_dict['sc_matrix'] = sc_matrix
        data_dict['n_cells_in_sc'] = n_cells_in_sc
        data_dict['cell_origins'] = cell_origins
    except NameError:
        raise Exception((
            'Force constants matrix could not be found in {:s}.\n Ensure '
            'PHONON_WRITE_FORCE_CONSTANTS: true has been set when running '
            'CASTEP').format(file))

    # Set entries relating to dipoles
    try:
        data_dict['born'] = born
        data_dict['dielectric'] = dielectric
    except UnboundLocalError:
        pass

    # Meta information
    data_dict['model'] = 'CASTEP'
    data_dict['seedname'] = seedname
    data_dict['path'] = path

    return data_dict


def _read_cell(file_obj, int_type, float_type):
    """
    Read cell data from a .castep_bin or .check file

    Parameters
    ----------
    f : file object
        File object in read mode for the .castep_bin or .check file
    int_type : str
        Python struct format string describing the size and endian-ness of
        ints in the file
    float_type : str
        Python struct format string describing the size and endian-ness of
        floats in the file

        Returns
    -------
    n_ions : int
        Number of ions in the unit cell
    cell_vec : (3, 3) float ndarray
        The unit cell vectors in bohr
    ion_r : (n_ions, 3) float ndarray
        The fractional position of each ion within the unit cell
    ion_mass : (n_ions,) float ndarray
        The mass of each ion in the unit cell in units of electron mass
    ion_type : (n_ions,) string ndarray
        The chemical symbols of each ion in the unit cell. Ions are in the
        same order as in ion_r
    """
    header = ''
    while header.strip() != b'END_UNIT_CELL':
        header = _read_entry(file_obj)
        if header.strip() == b'CELL%NUM_IONS':
            n_ions = _read_entry(file_obj, int_type)
        elif header.strip() == b'CELL%REAL_LATTICE':
            cell_vec = np.transpose(np.reshape(
                _read_entry(file_obj, float_type), (3, 3)))
        elif header.strip() == b'CELL%NUM_SPECIES':
            n_species = _read_entry(file_obj, int_type)
        elif header.strip() == b'CELL%NUM_IONS_IN_SPECIES':
            n_ions_in_species = _read_entry(file_obj, int_type)
            if n_species == 1:
                n_ions_in_species = np.array([n_ions_in_species])
        elif header.strip() == b'CELL%IONIC_POSITIONS':
            max_ions_in_species = max(n_ions_in_species)
            ion_r_tmp = np.reshape(_read_entry(file_obj, float_type),
                                   (n_species, max_ions_in_species, 3))
        elif header.strip() == b'CELL%SPECIES_MASS':
            ion_mass_tmp = _read_entry(file_obj, float_type)
            if n_species == 1:
                ion_mass_tmp = np.array([ion_mass_tmp])
        elif header.strip() == b'CELL%SPECIES_SYMBOL':
            # Need to decode binary string for Python 3 compatibility
            if n_species == 1:
                ion_type_tmp = [_read_entry(file_obj, 'S8')
                                .strip().decode('utf-8')]
            else:
                ion_type_tmp = [x.strip().decode('utf-8')
                                for x in _read_entry(file_obj, 'S8')]
    # Get ion_r in correct form
    # CASTEP stores ion positions as 3D array (3,
    # max_ions_in_species, n_species) so need to slice data to get
    # correct information
    ion_begin = np.insert(np.cumsum(n_ions_in_species[:-1]), 0, 0)
    ion_end = np.cumsum(n_ions_in_species)
    ion_r = np.zeros((n_ions, 3))
    for i in range(n_species):
        ion_r[ion_begin[i]:ion_end[i], :] = ion_r_tmp[
            i, :n_ions_in_species[i], :]
    # Get ion_type in correct form
    ion_type = np.array([])
    ion_mass = np.array([])
    for ion in range(n_species):
        ion_type = np.append(
            ion_type,
            [ion_type_tmp[ion] for i in range(n_ions_in_species[ion])])
        ion_mass = np.append(
            ion_mass,
            [ion_mass_tmp[ion] for i in range(n_ions_in_species[ion])])

    return n_ions, cell_vec, ion_r, ion_mass, ion_type


def _read_entry(file_obj, dtype=''):
    """
    Read a record from a Fortran binary file, including the beginning
    and end record markers and return the data inbetween

    Parameters
    ----------
    f : file object
        File object in read mode for the Fortran binary file
    dtype : str, optional, default ''
        String determining what order and type to unpack the bytes as. See
        'Format Strings' in Python struct documentation

    Returns
    -------
    data : str, int, float or ndarray
        Data type returned depends on dtype specified. If dtype is not
        specified, return type is a string. If there is more than one
        element in the record, it is returned as an ndarray of floats or
        integers

    """
    def record_mark_read(file_obj):
        # Read 4 byte Fortran record marker
        rawdata = file_obj.read(4)
        if rawdata == b'':
            raise EOFError(
                'Problem reading binary file: unexpected EOF reached')
        return struct.unpack('>i', rawdata)[0]

    begin = record_mark_read(file_obj)
    if dtype:
        n_bytes = int(dtype[-1])
        n_elems = int(begin/n_bytes)
        if n_elems > 1:
            data = np.fromfile(file_obj, dtype=dtype, count=n_elems)
            if 'i' in dtype:
                data = data.astype(np.int32)
            elif 'f' in dtype:
                data = data.astype(np.float64)
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
        raise IOError("""Problem reading binary file: beginning and end
                         record markers do not match""")

    return data


def _read_bands_data(seedname, path):
    """
    Reads data from a .bands file (and a .castep file if available) and
    returns it in a dictionary

    Parameters
    ----------
    seedname : str
        Seedname of file(s) to read
    path : str
        Path to dir containing the file(s), if in another directory

    Returns
    -------
    data_dict : dict
        A dict with the following keys: 'n_qpts', 'n_spins', 'n_branches',
        'fermi', 'cell_vec', 'recip_vec', 'qpts', 'weights', 'freqs',
        'freq_down'. If a .castep file is available to read, the keys 'n_ions',
        'ion_r' and 'ion_type' are also present.
        Meta information: 'seedname', 'path' and 'model'.
    """

    file = os.path.join(path, seedname + '.bands')
    with open(file, 'r') as f:
        n_qpts = int(f.readline().split()[3])
        n_spins = int(f.readline().split()[4])
        f.readline()  # Skip number of electrons line
        n_branches = int(f.readline().split()[3])
        fermi = np.array([float(x) for x in f.readline().split()[5:]])
        f.readline()  # Skip unit cell vectors line
        cell_vec = [[float(x) for x in f.readline().split()[0:3]]
                    for i in range(3)]

        qpts = np.zeros((n_qpts, 3))
        weights = np.zeros(n_qpts)
        freqs_qpt = np.zeros(n_branches)
        freqs = np.zeros((n_qpts, n_branches))
        if n_spins == 2:
            freq_down = np.zeros((n_qpts, n_branches))
        else:
            freq_down = np.array([])

        # Need to loop through file using while rather than number of k-points
        # as sometimes points are duplicated
        line = f.readline().split()
        while line:
            qpt_num = int(line[1]) - 1
            qpts[qpt_num, :] = [float(x) for x in line[2:5]]
            weights[qpt_num] = float(line[5])

            for j in range(n_spins):
                spin = int(f.readline().split()[2])

                # Read frequencies
                for k in range(n_branches):
                    freqs_qpt[k] = float(f.readline())

                if spin == 1:
                    freqs[qpt_num, :] = freqs_qpt
                elif spin == 2:
                    freq_down[qpt_num, :] = freqs_qpt

            line = f.readline().split()

    data_dict = {}
    data_dict['n_qpts'] = n_qpts
    data_dict['n_spins'] = n_spins
    data_dict['n_branches'] = n_branches
    data_dict['fermi'] = fermi
    data_dict['cell_vec'] = cell_vec
    data_dict['recip_vec'] = reciprocal_lattice(cell_vec)
    data_dict['qpts'] = qpts
    data_dict['weights'] = weights
    data_dict['freqs'] = freqs
    data_dict['freq_down'] = freq_down

    # Try to get extra data (ionic species, coords) from .castep file
    try:
        n_ions, ion_r, ion_type = _read_castep_data(seedname, path)
        data_dict['n_ions'] = n_ions
        data_dict['ion_r'] = ion_r
        data_dict['ion_type'] = ion_type
    except IOError:
        pass


    # Meta information
    data_dict['model'] = 'CASTEP'
    data_dict['seedname'] = seedname
    data_dict['path'] = path

    return data_dict


def _read_castep_data(seedname, path):
    """
    Reads extra data from .castep file (ionic species, coords) and returns them

    Parameters
    ----------
    seedname : str
        Seedname of file(s) to read
    path : str
        Path to dir containing the file(s), if in another directory

    Returns
    -------
    n_ions : int
        Number of ions in the unit cell
    ion_r : (n_ions, 3) float ndarray
        The fractional position of each ion within the unit cell
    ion_type : (n_ions,) string ndarray
        The chemical symbols of each ion in the unit cell. Ions are in the
        same order as in ion_r
    """
    n_ions_read = False
    ion_info_read = False

    castep_file = os.path.join(path, seedname + '.castep')
    with open(castep_file, 'r') as f:
        line = f.readline()
        while line:
            if all([n_ions_read, ion_info_read]):
                break
            if 'Total number of ions in cell' in line:
                n_ions = int(line.split()[-1])
                n_ions_read = True
            if 'Fractional coordinates of atoms' in line:
                f.readline()  # Skip uvw line
                f.readline()  # Skip --- line
                ion_info = [f.readline().split() for i in range(n_ions)]
                ion_r = np.array([[float(x) for x in line[-4:-1]]
                                  for line in ion_info])
                ion_type = np.array([x[1] for x in ion_info])
                ion_info_read = True
            line = f.readline()

    return n_ions, ion_r, ion_type
