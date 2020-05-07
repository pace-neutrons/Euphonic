import re
import os
import struct
import numpy as np
from euphonic import ureg
from euphonic.util import is_gamma


def _read_phonon_data(filename, cell_vectors_unit='angstrom',
                      atom_mass_unit='amu', frequencies_unit='meV'):
    """
    Reads data from a .phonon file and returns it in a dictionary

    Parameters
    ----------
    filename : str
        The path and name of the .phonon file to read

    Returns
    -------
    data_dict : dict
        A dict with the following keys: 'n_atoms', 'cell_vectors',
        'cell_vectors_unit', 'atom_r', 'atom_type', 'atom_mass',
        'atom_mass_unit', 'qpts', 'weights', 'frequencies',
        'frequencies_unit', 'eigenvectors'
    """
    with open(filename, 'r') as f:

        (n_atoms, n_branches, n_qpts, cell_vectors, atom_r,
         atom_type, atom_mass) = _read_phonon_header(f)

        qpts = np.zeros((n_qpts, 3))
        weights = np.zeros(n_qpts)
        freqs = np.zeros((n_qpts, n_branches))
        eigenvecs = np.zeros((n_qpts, n_branches, n_atoms, 3),
                             dtype='complex128')

        # Need to loop through file using while rather than number of
        # q-points as sometimes points are duplicated
        qpt_line = f.readline()
        float_patt = re.compile('-?\d+\.\d+')
        idx = 0
        while qpt_line:
            floats = re.findall(float_patt, qpt_line)
            qpt = [float(x) for x in floats[:3]]
            qweight = float(floats[3])

            freq_lines = [f.readline().split()
                          for i in range(n_branches)]
            qfreq = np.array([float(line[1]) for line in freq_lines])
            ir_index = 2
            raman_index = 3
            if is_gamma(qpt):
                ir_index += 1
                raman_index += 1
            if len(freq_lines[0]) > ir_index:
                qir = [float(line[ir_index]) for line in freq_lines]
            if len(freq_lines[0]) > raman_index:
                qraman = [float(line[raman_index])
                          for line in freq_lines]

            [f.readline() for x in range(2)]  # Skip 2 label lines
            lines = np.array([f.readline().split()[2:]
                              for x in range(n_atoms*n_branches)],
                             dtype=np.float64)
            lines_i = np.column_stack(([lines[:, 0] + lines[:, 1]*1j,
                                        lines[:, 2] + lines[:, 3]*1j,
                                        lines[:, 4] + lines[:, 5]*1j]))
            qeigenvec = np.zeros((n_branches, n_atoms, 3),
                                 dtype=np.complex128)
            for i in range(n_branches):
                    qeigenvec[i, :, :] = lines_i[i*n_atoms:(i+1)*n_atoms, :]
            qpt_line = f.readline()
            # Sometimes there are more than n_qpts q-points in the file
            # due to LO-TO splitting
            if idx < len(qpts):
                qpts[idx] = qpt
                freqs[idx] = qfreq
                weights[idx] = qweight
                eigenvecs[idx] = qeigenvec
            else:
                qpts = np.concatenate((qpts, [qpt]))
                freqs = np.concatenate((freqs, [qfreq]))
                weights = np.concatenate((weights, [qweight]))
                eigenvecs = np.concatenate((eigenvecs, [qeigenvec]))
            idx += 1

    data_dict = {}
    data_dict['crystal'] = {}
    cry_dict = data_dict['crystal']
    cry_dict['n_atoms'] = n_atoms
    cry_dict['cell_vectors'] = (cell_vectors*ureg('angstrom').to(
        cell_vectors_unit)).magnitude
    cry_dict['cell_vectors_unit'] = cell_vectors_unit
    cry_dict['atom_r'] = atom_r
    cry_dict['atom_type'] = atom_type
    cry_dict['atom_mass'] = atom_mass*(ureg('amu')).to(
        atom_mass_unit).magnitude
    cry_dict['atom_mass_unit'] = atom_mass_unit
    data_dict['qpts'] = qpts
    data_dict['weights'] = weights
    data_dict['frequencies'] = ((freqs*(1/ureg.cm)).to(
        frequencies_unit)).magnitude
    data_dict['frequencies_unit'] = frequencies_unit
    data_dict['eigenvectors'] = eigenvecs

    return data_dict


def _read_phonon_header(f):
    """
    Reads the header of a .phonon file

    Parameters
    ----------
    f : file object
        File object in read mode for the .phonon file containing the
        data

    Returns
    -------
    n_atoms : integer
        The number of atoms per unit cell
    n_branches : integer
        The number of phonon branches (3*n_atoms)
    n_qpts : integer
        The number of q-points in the .phonon file
    cell_vectors : (3, 3) float ndarray
        The unit cell vectors in Angstroms.
    atom_r : (n_atoms, 3) float ndarray
        The fractional position of each atom within the unit cell
    atom_type : (n_atoms,) string ndarray
        The chemical symbols of each atom in the unit cell. Atoms are in
        the same order as in atom_r
    atom_mass : (n_atoms,) float ndarray
        The mass of each atom in the unit cell in atomic mass units
    """
    f.readline()  # Skip BEGIN header
    n_atoms = int(f.readline().split()[3])
    n_branches = int(f.readline().split()[3])
    n_qpts = int(f.readline().split()[3])
    while not 'Unit cell vectors' in f.readline():
        pass
    cell_vectors = np.array([[float(x) for x in f.readline().split()[0:3]]
                         for i in range(3)])
    f.readline()  # Skip fractional co-ordinates label
    atom_info = np.array([f.readline().split() for i in range(n_atoms)])
    atom_r = np.array([[float(x) for x in y[1:4]] for y in atom_info])
    atom_type = np.array([x[4] for x in atom_info])
    atom_mass = np.array([float(x[5]) for x in atom_info])
    f.readline()  # Skip END header line

    return (n_atoms, n_branches, n_qpts, cell_vectors, atom_r, atom_type,
            atom_mass)


def _read_interpolation_data(filename, cell_vectors_unit='angstrom',
                             atom_mass_unit='amu',
                             force_constants_unit='hartree/bohr**2',
                             born_unit='e',
                             dielectric_unit='(e**2)/(bohr*hartree)'):
    """
    Reads data from a .castep_bin or .check file and returns it in a
    dictionary

    Parameters
    ----------
    filename : str
        The path and name of the file to read

    Returns
    -------
    data_dict : dict
        A dict with the following keys: 'n_atoms', 'cell_vectors',
        'cell_vectors_unit', 'atom_r', 'atom_type', 'atom_mass',
        'atom_mass_unit', 'force_constants', 'force_constants_unit',
        'sc_matrix' and 'cell_origins'. Also contains 'born',
        'born_unit', 'dielectric' and 'dielectric_unit' if they are
        present in the .castep_bin or .check file.
    """
    with open(filename, 'rb') as f:
        int_type = '>i4'
        float_type = '>f8'
        header = ''
        first_cell_read = True
        while header.strip() != b'END':
            header = _read_entry(f)
            if header.strip() == b'BEGIN_UNIT_CELL':
                # CASTEP writes the cell twice: the first is the
                # geometry optimised cell, the second is the original
                # cell. We only want the geometry optimised cell.
                if first_cell_read:
                    (n_atoms, cell_vectors, atom_r, atom_mass,
                    atom_type) = _read_cell(f, int_type, float_type)
                    first_cell_read = False
            elif header.strip() == b'FORCE_CON':
                sc_matrix = np.transpose(np.reshape(
                    _read_entry(f, int_type), (3, 3)))
                n_cells_in_sc = int(np.rint(np.absolute(
                    np.linalg.det(sc_matrix))))
                # Transpose and reshape fc so it is indexed [nc, i, j]
                force_constants = np.ascontiguousarray(np.transpose(
                    np.reshape(_read_entry(f, float_type),
                               (n_cells_in_sc, 3*n_atoms, 3*n_atoms)),
                    axes=[0, 2, 1]))
                cell_origins = np.reshape(
                    _read_entry(f, int_type), (n_cells_in_sc, 3))
                fc_row = _read_entry(f, int_type)
            elif header.strip() == b'BORN_CHGS':
                born = np.reshape(
                    _read_entry(f, float_type), (n_atoms, 3, 3))
            elif header.strip() == b'DIELECTRIC':
                dielectric = np.transpose(np.reshape(
                    _read_entry(f, float_type), (3, 3)))

    data_dict = {}
    data_dict['crystal'] = {}
    cry_dict = data_dict['crystal']
    cry_dict['n_atoms'] = n_atoms
    cry_dict['cell_vectors'] = cell_vectors*ureg(
        'bohr').to(cell_vectors_unit).magnitude
    cry_dict['cell_vectors_unit'] = cell_vectors_unit
    # Normalise atom coordinates
    cry_dict['atom_r'] = atom_r - np.floor(atom_r)
    cry_dict['atom_type'] = atom_type
    cry_dict['atom_mass'] = atom_mass*ureg(
        'electron_mass').to(atom_mass_unit).magnitude
    cry_dict['atom_mass_unit'] = atom_mass_unit

    # Set entries relating to 'FORCE_CON' block
    try:
        data_dict['force_constants'] = force_constants*ureg(
            'hartree/bohr**2').to(force_constants_unit).magnitude
        data_dict['force_constants_unit'] = force_constants_unit
        data_dict['sc_matrix'] = sc_matrix
        data_dict['cell_origins'] = cell_origins
    except NameError:
        raise Exception((
            f'Force constants matrix could not be found in {filename}. '
            f'\nEnsure PHONON_WRITE_FORCE_CONSTANTS: true and a '
            f'PHONON_FINE_METHOD has been chosen when running CASTEP'))

    # Set entries relating to dipoles
    try:
        data_dict['born'] = born*ureg('e').to(born_unit).magnitude
        data_dict['born_unit'] = born_unit
        data_dict['dielectric'] = dielectric*ureg(
            'e**2/(hartree*bohr)').to(dielectric_unit).magnitude
        data_dict['dielectric_unit'] = dielectric_unit
    except UnboundLocalError:
        pass

    return data_dict


def _read_cell(file_obj, int_type, float_type):
    """
    Read cell data from a .castep_bin or .check file

    Parameters
    ----------
    f : file object
        File object in read mode for the .castep_bin or .check file
    int_type : str
        Python struct format string describing the size and endian-ness
        of ints in the file
    float_type : str
        Python struct format string describing the size and endian-ness
        of floats in the file

        Returns
    -------
    n_atoms : int
        Number of atoms in the unit cell
    cell_vectors : (3, 3) float ndarray
        The unit cell vectors in bohr
    atom_r : (n_atoms, 3) float ndarray
        The fractional position of each atom within the unit cell
    atom_mass : (n_atoms,) float ndarray
        The mass of each atom in the unit cell in units of electron mass
    atom_type : (n_atoms,) string ndarray
        The chemical symbols of each atom in the unit cell. Atoms are in
        the same order as in atom_r
    """
    header = ''
    while header.strip() != b'END_UNIT_CELL':
        header = _read_entry(file_obj)
        if header.strip() == b'CELL%NUM_IONS':
            n_atoms = _read_entry(file_obj, int_type)
        elif header.strip() == b'CELL%REAL_LATTICE':
            cell_vectors = np.transpose(np.reshape(
                _read_entry(file_obj, float_type), (3, 3)))
        elif header.strip() == b'CELL%NUM_SPECIES':
            n_species = _read_entry(file_obj, int_type)
        elif header.strip() == b'CELL%NUM_IONS_IN_SPECIES':
            n_atoms_in_species = _read_entry(file_obj, int_type)
            if n_species == 1:
                n_atoms_in_species = np.array([n_atoms_in_species])
        elif header.strip() == b'CELL%IONIC_POSITIONS':
            max_atoms_in_species = max(n_atoms_in_species)
            atom_r_tmp = np.reshape(_read_entry(file_obj, float_type),
                                   (n_species, max_atoms_in_species, 3))
        elif header.strip() == b'CELL%SPECIES_MASS':
            atom_mass_tmp = _read_entry(file_obj, float_type)
            if n_species == 1:
                atom_mass_tmp = np.array([atom_mass_tmp])
        elif header.strip() == b'CELL%SPECIES_SYMBOL':
            # Need to decode binary string for Python 3 compatibility
            if n_species == 1:
                atom_type_tmp = [_read_entry(file_obj, 'S8')
                                .strip().decode('utf-8')]
            else:
                atom_type_tmp = [x.strip().decode('utf-8')
                                for x in _read_entry(file_obj, 'S8')]
    # Get atom_r in correct form
    # CASTEP stores atom positions as 3D array (3,
    # max_atoms_in_species, n_species) so need to slice data to get
    # correct information
    atom_begin = np.insert(np.cumsum(n_atoms_in_species[:-1]), 0, 0)
    atom_end = np.cumsum(n_atoms_in_species)
    atom_r = np.zeros((n_atoms, 3))
    for i in range(n_species):
        atom_r[atom_begin[i]:atom_end[i], :] = atom_r_tmp[
            i, :n_atoms_in_species[i], :]
    # Get atom_type in correct form
    atom_type = np.array([])
    atom_mass = np.array([])
    for at in range(n_species):
        atom_type = np.append(
            atom_type,
            [atom_type_tmp[at] for i in range(n_atoms_in_species[at])])
        atom_mass = np.append(
            atom_mass,
            [atom_mass_tmp[at] for i in range(n_atoms_in_species[at])])

    return n_atoms, cell_vectors, atom_r, atom_mass, atom_type


def _read_entry(file_obj, dtype=''):
    """
    Read a record from a Fortran binary file, including the beginning
    and end record markers and return the data inbetween

    Parameters
    ----------
    f : file object
        File object in read mode for the Fortran binary file
    dtype : str, optional, default ''
        String determining what order and type to unpack the bytes as.
        See 'Format Strings' in Python struct documentation

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
