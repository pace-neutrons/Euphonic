import re
import struct
from typing import Dict, Any, TextIO, BinaryIO, Tuple, Optional, List, Union

import numpy as np

from euphonic import ureg
from euphonic.util import is_gamma


def read_phonon_dos_data(
        filename: str,
        cell_vectors_unit: str = 'angstrom',
        atom_mass_unit: str = 'amu',
        frequencies_unit: str = 'meV',
        mode_gradients_unit: str = 'meV*angstrom') -> Dict[str, Any]:
    """
    Reads data from a .phonon_dos file and returns it in a dictionary

    Parameters
    ----------
    filename
        The path and name of the .phonon file to read
    cell_vectors_unit
        The unit to return the cell vectors in
    atom_mass_unit
        The unit to return the atom masses in
    frequencies_unit
        The unit to return the frequencies in

    Returns
    -------
    data_dict : dict
        A dict with the following keys: 'n_atoms', 'cell_vectors',
        'cell_vectors_unit', 'atom_r', 'atom_type', 'atom_mass',
        'atom_mass_unit', 'qpts', 'weights', 'frequencies',
        'frequencies_unit', 'mode_gradients', 'mode_gradients_unit',
        'dos_bins', 'dos_bins_unit', 'dos', 'pdos'
    """
    with open(filename, 'r') as f:

        f.readline()  # Skip BEGIN header
        n_atoms = int(f.readline().split()[-1])
        n_species = int(f.readline().split()[-1])
        n_branches = int(f.readline().split()[-1])
        n_bins = int(f.readline().split()[-1])
        (cell_vectors, atom_r, atom_type, atom_mass) = _read_crystal_info(
            f, n_atoms)
        f.readline()  # Skip BEGIN GRADIENTS

        qpts = np.empty((0, 3))
        weights = np.empty((0,))
        freqs = np.empty((0, n_branches))
        mode_grads = np.empty((0, n_branches))
        while True:
            try:
                qpt, qweight, qfreq, qmode_grad = _read_frequency_block(
                    f, n_branches, extra_columns=[0])
            except IndexError:
                # This should indicate we've reached 'END GRADIENTS' line
                break
            qpts = np.concatenate((qpts, [qpt]))
            weights = np.concatenate((weights, [qweight]))
            freqs = np.concatenate((freqs, [qfreq]))
            mode_grads = np.concatenate((mode_grads, [qmode_grad.squeeze()]))
        line = f.readline()
        if not 'BEGIN DOS' in line:
            raise RuntimeError(
                f'Expected "BEGIN DOS" in {filename}, got {line}')
        # max_rows arg not available until Numpy 1.16.0
        try:
            dos_data = np.loadtxt(f, max_rows=n_bins)
        except TypeError:
            data = f.readlines()
            dos_data = np.array([[float(elem) for elem in line.split()]
                                      for line in data[:n_bins]])

    data_dict: Dict[str, Any] = {}
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
    # Pint conversion 'angstrom*(1/cm)' to 'angstrom*(meV)' doesn't
    # work so convert separately
    mode_grads = mode_grads*ureg('angstrom').to(cell_vectors_unit).magnitude
    data_dict['mode_gradients'] = mode_grads*ureg('1/cm').to(
            frequencies_unit).magnitude
    data_dict['mode_gradients_unit'] = (frequencies_unit + '*'
                                        + cell_vectors_unit)
    data_dict['dos_bins'] = dos_data[:, 0]*ureg('1/cm').to(
            frequencies_unit).magnitude
    data_dict['dos_bins_unit'] = frequencies_unit

    data_dict['dos'] = {}
    data_dict['dos_unit'] = frequencies_unit
    # Avoid issues in converting DOS, Pint allows
    # cm^-1 -> meV but not cm -> 1/meV
    dos_conv = (1*ureg('1/cm').to(frequencies_unit)).magnitude
    dos_dict = data_dict['dos']
    dos_dict['Total'] = dos_data[:, 1]/dos_conv
    _, idx = np.unique(atom_type, return_index=True)
    unique_types = atom_type[np.sort(idx)]
    for i, species in enumerate(unique_types):
        dos_dict[species] = dos_data[:, i + 2]/dos_conv

    return data_dict


def read_phonon_data(
        filename: str,
        cell_vectors_unit: str = 'angstrom',
        atom_mass_unit: str = 'amu',
        frequencies_unit: str = 'meV',
        read_eigenvectors: bool = True) -> Dict[str, Any]:
    """
    Reads data from a .phonon file and returns it in a dictionary

    Parameters
    ----------
    filename
        The path and name of the .phonon file to read
    cell_vectors_unit
        The unit to return the cell vectors in
    atom_mass_unit
        The unit to return the atom masses in
    frequencies_unit
        The unit to return the frequencies in
    read_eigenvectors
        Whether to read the eigenvectors and return them in the
        dictionary

    Returns
    -------
    data_dict : dict
        A dict with the following keys: 'n_atoms', 'cell_vectors',
        'cell_vectors_unit', 'atom_r', 'atom_type', 'atom_mass',
        'atom_mass_unit', 'qpts', 'weights', 'frequencies',
        'frequencies_unit'

        If read_eigenvectors is True, there will also be an
        'eigenvectors' key
    """
    with open(filename, 'r') as f:

        f.readline()  # Skip BEGIN header
        n_atoms = int(f.readline().split()[-1])
        n_branches = int(f.readline().split()[-1])
        n_qpts = int(f.readline().split()[-1])
        (cell_vectors, atom_r, atom_type, atom_mass) = _read_crystal_info(
            f, n_atoms)

        qpts = np.zeros((n_qpts, 3))
        weights = np.zeros(n_qpts)
        freqs = np.zeros((n_qpts, n_branches))
        if read_eigenvectors:
            eigenvecs = np.zeros((n_qpts, n_branches, n_atoms, 3),
                                 dtype='complex128')

        # Need to loop through file using while rather than number of
        # q-points as points are duplicated on LO-TO splitting
        idx = 0
        while True:
            try:
                qpt, qweight, qfreq, _ = _read_frequency_block(f, n_branches)
            except EOFError:
                break
            [f.readline() for x in range(2)]  # Skip 2 label lines
            evec_lines = np.array(
                [f.readline() for x in range(n_atoms*n_branches)])
            if read_eigenvectors:
                evec_lines = np.array([x.split()[2:] for x in evec_lines],
                                      dtype=np.float64)
                lines_i = np.column_stack((
                    [evec_lines[:, 0] + evec_lines[:, 1]*1j,
                     evec_lines[:, 2] + evec_lines[:, 3]*1j,
                     evec_lines[:, 4] + evec_lines[:, 5]*1j]))
                qeigenvec = np.zeros((n_branches, n_atoms, 3),
                                     dtype=np.complex128)
                for i in range(n_branches):
                    qeigenvec[i, :, :] = lines_i[i*n_atoms:(i+1)*n_atoms, :]
            # Sometimes there are more than n_qpts q-points in the file
            # due to LO-TO splitting
            if idx < len(qpts):
                qpts[idx] = qpt
                freqs[idx] = qfreq
                weights[idx] = qweight
                if read_eigenvectors:
                    eigenvecs[idx] = qeigenvec
            else:
                qpts = np.concatenate((qpts, [qpt]))
                freqs = np.concatenate((freqs, [qfreq]))
                weights = np.concatenate((weights, [qweight]))
                if read_eigenvectors:
                    eigenvecs = np.concatenate((eigenvecs, [qeigenvec]))
            idx += 1

    data_dict: Dict[str, Any] = {}
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
    if read_eigenvectors:
        data_dict['eigenvectors'] = eigenvecs

    return data_dict


def _read_crystal_info(f: TextIO, n_atoms: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads the header crystal information from a CASTEP text file, from
    'Unit cell vectors' to 'END header'

    Parameters
    ----------
    f
        File object in read mode for the .phonon file containing the
        data
    n_atoms
        Number of atoms in the cell

    Returns
    -------
    cell_vectors
        Shape (3, 3) float ndarray. The unit cell vectors in Angstroms
    atom_r
        Shape (n_atoms, 3) float ndarray. The fractional position of
        each atom within the unit cell
    atom_type
        Shape (n_atoms,) string ndarray. The chemical symbols of each
        atom in the unit cell
    atom_mass
        Shape (n_atoms,) float ndarray. The mass of each atom in the
        unit cell in atomic mass units
    """
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

    return cell_vectors, atom_r, atom_type, atom_mass


def _read_frequency_block(
            f: TextIO, n_branches: int, extra_columns: Optional[List] = None
    ) -> Tuple[np.ndarray, float, np.ndarray, Optional[np.ndarray]]:
    """
    For a single q-point reads the q-point, weight, frequencies
    and optionally any extra columns

    Parameters
    ----------
    f
        File object in read mode for the file containing the data
    extra_columns
        The index(es) of extra columns to read after the frequencies
        column. e.g. to read the first column after frequencies
        extra_columns = [0]. To read the first and second columns after
        frequencies extra_columns = [0, 1]. For reference, in .phonon
        IR intensities = 0, Raman intensities = 1, and in .phonon_dos
        mode gradients = 0

    Returns
    -------
    qpt
        Shape (3,) float ndarray. The q-point in reciprocal fractional
        coordinates
    weight
        The weight of this q-point
    frequencies
        Shape (n_modes,) float ndarray. The phonon frequencies for this
        q-point in 1/cm
    extra
        None if extra_columns is None, otherwise a shape
        (len(extra_columns), n_modes) float ndarray
    """
    qpt_line = f.readline()
    if qpt_line == '':
        raise EOFError
    float_patt = re.compile('-?\d+\.\d+')
    floats = re.findall(float_patt, qpt_line)
    qpt = np.array([float(x) for x in floats[:3]])
    qweight = float(floats[3])
    freq_lines = [f.readline().split()
                  for i in range(n_branches)]
    freq_col = 1
    qfreq = np.array([float(line[freq_col]) for line in freq_lines])
    extra: Optional[np.ndarray] = None
    if extra_columns is not None:
        if len(qpt_line.split()) > 6:
            # Indicates a split gamma point in .phonon, so there is an
            # additional column
            extra_columns = [x + 1 for x in extra_columns]
        extra = np.zeros((len(extra_columns), n_branches))
        for i, col in enumerate(extra_columns):
            extra[i] = np.array(
                [float(line[freq_col + col + 1]) for line in freq_lines])
    return qpt, qweight, qfreq, extra


def read_interpolation_data(
        filename: str,
        cell_vectors_unit: str = 'angstrom',
        atom_mass_unit: str = 'amu',
        force_constants_unit: str = 'hartree/bohr**2',
        born_unit: str = 'e',
        dielectric_unit: str = '(e**2)/(bohr*hartree)') -> Dict[str, Any]:
    """
    Reads data from a .castep_bin or .check file and returns it in a
    dictionary

    Parameters
    ----------
    filename
        The path and name of the file to read
    cell_vectors_unit
        The unit to return the cell vectors in
    atom_mass_unit
        The unit to return the atom masses in
    force_constants_unit
        The unit to return the force constants in
    born_unit
        The unit to return the Born charges in
    dielectric_unit
        The unit to return the dielectric permittivity tensor in

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

    data_dict: Dict[str, Any] = {}
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
        raise RuntimeError((
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


def _read_cell(file_obj: BinaryIO, int_type: str, float_type: str
               ) -> Tuple[int, np.ndarray, np.ndarray,
                          np.ndarray, np.ndarray]:
    """
    Read cell data from a .castep_bin or .check file

    Parameters
    ----------
    file_obj
        File object in read mode for the .castep_bin or .check file
    int_type
        Python struct format string describing the size and endian-ness
        of ints in the file
    float_type
        Python struct format string describing the size and endian-ness
        of floats in the file

    Returns
    -------
    n_atoms
        Number of atoms in the unit cell
    cell_vectors
        Shape (3, 3) float ndarray. The unit cell vectors in bohr
    atom_r
        Shape (n_atoms, 3) float ndarray. The fractional position of
        each atom within the unit cell
    atom_mass
        Shape (n_atoms,) float ndarray. The mass of each atom in the
        unit cell in units of electron mass
    atom_type
        Shape (n_atoms,) string ndarray. The chemical symbols of each
        atom in the unit cell
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


def _read_entry(file_obj: BinaryIO, dtype: str = ''
                ) -> Union[str, int, float, np.ndarray]:
    """
    Read a record from a Fortran binary file, including the beginning
    and end record markers and return the data inbetween

    Parameters
    ----------
    f
        File object in read mode for the Fortran binary file
    dtype
        String determining what order and type to unpack the bytes as.
        See 'Format Strings' in Python struct documentation

    Returns
    -------
    data
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
