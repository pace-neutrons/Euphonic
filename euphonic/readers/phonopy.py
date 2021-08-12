import os
import warnings
from typing import Optional, Dict, Any, Union, Tuple, TextIO, Sequence, List

import numpy as np

from euphonic import ureg
from euphonic.util import get_all_origins, _get_supercell_relative_idx


# h5py can't be called from Matlab, so import as late as possible to
# minimise impact. Do the same with yaml for consistency
class ImportPhonopyReaderError(ModuleNotFoundError):

    def __init__(self):
        self.message = (
            '\n\nCannot import yaml, h5py to read Phonopy files, maybe '
            'they are not installed. To install the optional '
            'dependencies for Euphonic\'s Phonopy reader, try:\n\n'
            'pip install euphonic[phonopy_reader]\n')

    def __str__(self):
        return self.message


def _convert_weights(weights: np.ndarray) -> np.ndarray:
    """
    Convert q-point weights to normalised convention

    Parameters
    ----------
    weights
        Shape (n_qpts,) float ndarray. Weights in Phonopy convention

    Returns
    ----------
    norm_weights
        Shape (n_qpts,) float ndarray. Normalised weights
    """
    total_weight = weights.sum()
    return weights/total_weight


def _extract_phonon_data_yaml(filename: str,
                              read_eigenvectors: bool = True
                              ) -> Dict[str, np.ndarray]:
    """
    From a mesh/band/qpoint.yaml file, extract the relevant information
    as a dict of Numpy arrays. No unit conversion is done at this point,
    so they are in the same units as in the .yaml file

    Parameters
    ----------
    filename
        Path and name of the mesh/band/qpoint.yaml file
    read_eigenvectors
        Whether to try and read the eigenvectors and return them in the
        dictionary

    Returns
    -------
    data_dict
        A dict with the following keys:
            'qpts', 'frequencies'
        It may also have the following keys if they are present in the
        .yaml file:
            'eigenvectors', 'weights', 'cell_vectors', 'atom_r',
            'atom_mass', 'atom_type'
    """
    try:
        import yaml
        try:
            from yaml import CSafeLoader as SafeLoader
        except ImportError:
            from yaml import SafeLoader
    except ModuleNotFoundError as e:
        raise ImportPhonopyReaderError from e

    with open(filename, 'r') as yaml_file:
        phonon_data = yaml.load(yaml_file, Loader=SafeLoader)

    data_dict = {}
    phonons = [phon for phon in phonon_data['phonon']]
    bands_data_each_qpt = [bands_data['band'] for bands_data in phonons]

    data_dict['qpts'] = np.array([phon['q-position'] for phon in phonons])
    data_dict['frequencies'] = np.array(
        [[band_data['frequency'] for band_data in bands_data]
         for bands_data in bands_data_each_qpt])
    if read_eigenvectors:
        # Eigenvectors may not be present if users haven't set
        # --eigvecs when running Phonopy - deal with this later
        try:
            data_dict['eigenvectors'] = np.squeeze(np.array(
                [[band_data['eigenvector'] for band_data in bands_data]
                 for bands_data in bands_data_each_qpt]).view(np.complex128))
        except KeyError:
            pass

    # Weights only present in mesh
    try:
        data_dict['weights'] = np.array([phon['weight'] for phon in phonons])
    except KeyError:
        pass

    # Crystal information only appears to be present in mesh/band
    try:
        data_dict['cell_vectors'] = np.array(phonon_data['lattice'])
        data_dict['atom_r'] = np.array(
            [atom['coordinates'] for atom in phonon_data['points']])
        data_dict['atom_mass'] = np.array(
            [atom['mass'] for atom in phonon_data['points']])
        data_dict['atom_type'] = np.array(
            [atom['symbol'] for atom in phonon_data['points']])
    except KeyError:
        pass
    return data_dict


def _extract_phonon_data_hdf5(filename: str,
                              read_eigenvectors: bool = True
                              ) -> Dict[str, np.ndarray]:
    """
    From a mesh/band/qpoint.hdf5 file, extract the relevant information
    as a dict of Numpy arrays. No unit conversion is done at this point,
    so they are in the same units as in the .hdf5 file

    Parameters
    ----------
    filename
        Path and name of the mesh/band/qpoint.hdf5 file
    read_eigenvectors
        Whether to try and read the eigenvectors and return them in the
        dictionary

    Returns
    -------
    data_dict
        A dict with the following keys:
            'qpts', 'frequencies'
        It may also have the following keys if they are present in the
        .hdf5 file:
            'eigenvectors', 'weights'
    """
    try:
        import h5py
    except ModuleNotFoundError as e:
        raise ImportPhonopyReaderError from e

    with h5py.File(filename, 'r') as hdf5_file:
        data_dict = {}
        if 'qpoint' in hdf5_file.keys():
            data_dict['qpts'] = hdf5_file['qpoint'][()]
            data_dict['frequencies'] = hdf5_file['frequency'][()]
            if read_eigenvectors:
                # Eigenvectors may not be present if users haven't set
                # --eigvecs when running Phonopy - deal with this later
                try:
                    data_dict['eigenvectors'] = hdf5_file['eigenvector'][()]
                except KeyError:
                    pass
            # Only mesh.hdf5 has weights
            try:
                data_dict['weights'] = hdf5_file['weight'][()]
            except KeyError:
                pass
        # Is a band.hdf5 file - q-points are stored in 'path' and need
        # special treatment
        else:
            data_dict['qpts'] = hdf5_file['path'][()].reshape(
                -1, hdf5_file['path'][()].shape[-1])
            data_dict['frequencies'] = hdf5_file['frequency'][()].reshape(
                -1, hdf5_file['frequency'][()].shape[-1])
            try:
                # The last 2 dimensions of eigenvectors in bands.hdf5 are for
                # some reason transposed compared to mesh/qpoints.hdf5, so also
                # transpose to handle this
                data_dict['eigenvectors'] = hdf5_file[
                    'eigenvector'][()].reshape(
                        -1, *hdf5_file['eigenvector'][()].shape[-2:]
                        ).transpose([0,2,1])
            except KeyError:
                pass

    return data_dict

def read_phonon_data(
        path: str = '.',
        phonon_name: str = 'band.yaml',
        phonon_format: Optional[str] = None,
        summary_name: str = 'phonopy.yaml',
        cell_vectors_unit: str = 'angstrom',
        atom_mass_unit: str = 'amu',
        frequencies_unit: str = 'meV',
        read_eigenvectors: bool = True
        ) -> Dict[str, Union[int, str, np.ndarray]]:
    """
    Reads precalculated phonon mode data from a Phonopy
    mesh/band/qpoints.yaml/hdf5 file and returns it in a dictionary. May
    also read from phonopy.yaml for structure information.

    Parameters
    ----------
    path
        Path to directory containing the file(s)
    phonon_name
        Name of Phonopy file including the frequencies and eigenvectors
    phonon_format
        Format of the phonon_name file if it isn't obvious from the
        phonon_name extension. One of {'yaml', 'hdf5'}
    summary_name
        Name of Phonopy summary file to read the crystal information
        from. Crystal information in the phonon_name file takes
        priority, but if it isn't present, crystal information is read
        from summary_name instead
    cell_vectors_unit
        The unit to return the cell vectors in
    atom_mass_unit
        The unit to return the atom masses in
    frequencies_unit
        The unit to return the frequencies in
    read_eigenvectors
        Whether to read the eigenvectors and return them in the
        dictionary. In this case, if eigenvectors = False it is assumed
        a QpointFrequencies object is being created so detailed structure
        information such as atom locations, masses, species is not
        explicitly required, although the cell vectors are required.

    Returns
    -------
    data_dict
        A dict with the following keys: 'n_atoms', 'cell_vectors',
        'cell_vectors_unit', 'atom_r', 'atom_type', 'atom_mass',
        'atom_mass_unit', 'qpts', 'weights', 'frequencies',
        'frequencies_unit'.

        If read_eigenvectors is True, there will also be an
        'eigenvectors' key
    """
    phonon_pathname = os.path.join(path, phonon_name)
    summary_pathname = os.path.join(path, summary_name)

    hdf5_exts = ['hdf5', 'hd5', 'h5']
    yaml_exts = ['yaml', 'yml', 'yl']
    if phonon_format is None:
        phonon_format = os.path.splitext(phonon_name)[1].strip('.')

    if phonon_format in hdf5_exts:
        phonon_dict = _extract_phonon_data_hdf5(
            phonon_pathname, read_eigenvectors=read_eigenvectors)
    elif phonon_format in yaml_exts:
        phonon_dict = _extract_phonon_data_yaml(
            phonon_pathname, read_eigenvectors=read_eigenvectors)
    else:
        raise ValueError((f'File format {phonon_format} of {phonon_name}'
                          f' is not recognised'))

    if read_eigenvectors and not 'eigenvectors' in phonon_dict.keys():
        raise RuntimeError((f'Eigenvectors couldn\'t be found in '
                            f'{phonon_pathname}, ensure --eigvecs was '
                            f'set when running Phonopy'))

    # Since units are not explicitly defined in
    # mesh/band/qpoints.yaml/hdf5 assume:
    ulength = 'angstrom'
    umass = 'amu'
    ufreq = 'THz'

    crystal_keys = ['cell_vectors', 'atom_r', 'atom_mass', 'atom_type']
    # Check if crystal structure has been read from phonon_file, if not
    # get structure from summary_file
    if len(crystal_keys & phonon_dict.keys()) != len(crystal_keys):
        summary_dict = _extract_summary(summary_pathname)
        phonon_dict['cell_vectors'] = summary_dict['cell_vectors']
        phonon_dict['atom_r'] = summary_dict['atom_r']
        phonon_dict['atom_mass'] = summary_dict['atom_mass']
        phonon_dict['atom_type'] = summary_dict['atom_type']
        # Overwrite assumed units if they are found in summary file
        ulength = summary_dict['ulength']
        umass = summary_dict['umass']
        # Check phonon_file and summary_file are commensurate
        if 3*len(phonon_dict['atom_r']) != len(phonon_dict['frequencies'][0]):
            raise ValueError((
                f'Phonon file {phonon_pathname} not commensurate '
                f'with summary file {summary_pathname}. Please '
                'check contents'))

    data_dict: Dict[str, Any] = {}
    data_dict['crystal'] = {}
    cry_dict = data_dict['crystal']
    cry_dict['n_atoms'] = len(phonon_dict['atom_r'])
    cry_dict['cell_vectors'] = phonon_dict['cell_vectors']*ureg(
        ulength).to(cell_vectors_unit).magnitude
    cry_dict['cell_vectors_unit'] = cell_vectors_unit
    cry_dict['atom_r'] = phonon_dict['atom_r']
    cry_dict['atom_type'] = phonon_dict['atom_type']
    cry_dict['atom_mass'] = phonon_dict['atom_mass']*ureg(
        umass).to(atom_mass_unit).magnitude
    cry_dict['atom_mass_unit'] = atom_mass_unit
    n_qpts = len(phonon_dict['qpts'])
    data_dict['n_qpts'] = n_qpts
    data_dict['qpts'] = phonon_dict['qpts']
    data_dict['frequencies'] = phonon_dict['frequencies']*ureg(
        ufreq).to(frequencies_unit).magnitude
    data_dict['frequencies_unit'] = frequencies_unit
    if read_eigenvectors:
        # Convert Phonopy conventions to Euphonic conventions
        data_dict['eigenvectors'] = convert_eigenvector_phases(phonon_dict)
    if 'weights' in phonon_dict.keys():
        data_dict['weights'] = _convert_weights(phonon_dict['weights'])
    return data_dict


def convert_eigenvector_phases(phonon_dict: Dict[str, np.ndarray]
                               ) -> np.ndarray:
    """
    When interpolating the force constants matrix, Euphonic uses a phase
    convention of e^iq.r_a, where r_a is the coordinate of each CELL in
    the supercell, whereas Phonopy uses e^iq.r_k, where r_k is the
    coordinate of each ATOM in the supercell. This must be accounted for
    when reading Phonopy eigenvectors by applying a phase of
    e^-iq(r_k - r_k')
    """
    atom_r = phonon_dict['atom_r']
    n_atoms = len(atom_r)
    qpts = phonon_dict['qpts']
    n_qpts = len(qpts)

    eigvecs = np.reshape(phonon_dict['eigenvectors'],
                         (n_qpts, n_atoms, 3, n_atoms, 3))
    na = np.newaxis
    rk_diff = atom_r[:, na, :] - atom_r[na, :, :]
    conversion = np.exp(-2j*np.pi*np.einsum('il,jkl->ijk', qpts, rk_diff))
    eigvecs = np.einsum('ijklm,ijl->ijklm', eigvecs, conversion)
    return np.reshape(eigvecs, (n_qpts, 3*n_atoms, n_atoms, 3))


def _extract_force_constants(fc_pathname: str, n_atoms: int, n_cells: int,
                             summary_name: str, cell_origins_map: np.ndarray,
                             sc_relative_idx: np.ndarray, p2s_map: np.ndarray,
                             ) -> np.ndarray:
    """
    Reads force constants from a Phonopy FORCE_CONSTANTS file

    Parameters
    ----------
    fc_pathname
        The FORCE_CONSTANTS file to read from
    n_atoms
        Number of atoms in the unit cell
    n_cells
        Number of unit cells in the supercell
    summary_name
        Name of input phonopy.yaml file
    cell_origins_map
        Shape (n_atoms*n_cells, 2) int ndarray. In the case of
        non-diagonal supercell_matrices, the cell origins are not the
        same for each atom. This is a map of the equivalent cell origins
        for each atom, which is required to reorder the force constants
        matrix so that all atoms in the unit cell share equivalent
        origins
    sc_relative_idx
        Shape (n_cells, n_cells) int ndarray. The index n of the
        equivalent vector in cell_origins for each
        cell_origins[i] -> cell_origins[j] vector in the supercell. See
        _get_supercell_relative_idx
    p2s_map
        Shape (n_atoms,) int ndarray. The index of the primitive atoms
        in the atoms of the supercell. Used if fc_format = 'full'

    Returns
    -------
    fc
        Shape (n_cells, 3*n_atoms, 3*n_atoms) float ndarray. The force
        constants, in Euphonic convention
    """
    with open(fc_pathname, 'r') as f:
        fc_dims =  [int(dim) for dim in f.readline().split()]
    # single shape specifier implies full format
    if len(fc_dims) == 1:
        fc_dims.append(fc_dims[0])
    _check_fc_shape(fc_dims, n_atoms, n_cells, fc_pathname, summary_name)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = np.genfromtxt(fc_pathname, skip_header=1,
                           max_rows=4*(n_atoms*n_cells)**2, usecols=(0,1,2),
                           invalid_raise=False)
    if fc_dims[0] == fc_dims[1]:  # full fc
        fc = fc.reshape(n_atoms*n_cells, n_atoms, n_cells, 3, 3)
        fc = fc[p2s_map]

    fc = fc.reshape((n_atoms, n_cells*n_atoms, 3, 3))

    return _reshape_fc(fc, n_atoms, n_cells, cell_origins_map, sc_relative_idx)


def _extract_force_constants_hdf5(
        filename: str, n_atoms: int, n_cells: int, summary_name: str,
        cell_origins_map: np.ndarray, sc_relative_idx: np.ndarray
        ) -> np.ndarray:
    try:
        import h5py
    except ModuleNotFoundError as e:
        raise ImportPhonopyReaderError from e

    with h5py.File(filename, 'r') as fc_file:
        fc = fc_file['force_constants'][:]
        _check_fc_shape(fc.shape, n_atoms, n_cells, fc_file.filename,
                        summary_name)
        p2s_map = list(fc_file['p2s_map'])
    if fc.shape[0] == fc.shape[1]: # FULL FC, convert down to COMPACT
        fc = fc[p2s_map, :, :, :]
    fc_unfolded = fc.reshape(n_atoms, n_atoms*n_cells, 3, 3)
    return _reshape_fc(fc_unfolded, n_atoms, n_cells, cell_origins_map,
                       sc_relative_idx)


def _extract_force_constants_summary(
        summary_object: Dict[str, Any],
        cell_origins_map: np.ndarray,
        sc_relative_idx: np.ndarray,
        p2s_map: np.ndarray) -> np.ndarray:
    """
    Get force constants from phonopy yaml summary file.

    Parameters
    ----------
    summary_object
        Dict containing contents of phonopy.yaml
    cell_origins_map
        Shape (n_atoms*n_cells, 2) int ndarray. In the case of of
        non-diagonal supercell_matrices, the cell origins are not the
        same for each atom. This is a map of the equivalent cell
        origins for each atom, which is required to reorder the force
        constants matrix so that all atoms in the unit cell share
        equivalent origins
    sc_relative_idx
        Shape (n_cells, n_cells) int ndarray. The index n of the
        equivalent vector in cell_origins for each
        cell_origins[i] -> cell_origins[j] vector in the supercell.
        See _get_supercell_relative_idx
    p2s_map
        Shape (n_atoms,) int ndarray. The index of the primitive atoms
        in the atoms of the supercell. Used if fc_format = 'full'

    Returns
    -------
    fc
        Shape (n_cells, 3*n_atoms, 3*n_atoms) float ndarray. Force
        constants in Euphonic format
    """
    fc_entry = summary_object['force_constants']
    fc_format = fc_entry['format']

    n_cells = sc_relative_idx.shape[0]
    n_atoms = int(cell_origins_map.shape[0]/n_cells)
    if fc_format == 'compact':
        fc = np.array(fc_entry['elements'])
    else:  # convert to compact
        fc = np.array(fc_entry['elements']).reshape(
            [n_atoms*n_cells, n_atoms, n_cells, 3, 3])[p2s_map, :, :, :, :]

    fc = _reshape_fc(fc, n_atoms, n_cells, cell_origins_map, sc_relative_idx)
    return fc

def _reshape_fc(fc: np.ndarray, n_atoms: int, n_cells: int,
                cell_origins_map: np.ndarray, sc_relative_idx: np.ndarray
                ) -> np.ndarray:
    """
    Reshape force constants from Phonopy convention to Euphonic
    convention

    Parameters
    ----------
    fc
        Shape (n_atoms, n_atoms*n_cells, 3, 3) float ndarray. Force
        constants matrix in Phonopy shape
    n_atoms
        Number of atoms in the unit cell
    n_cells
        Number of cells in the supercell
    cell_origins_map
        Shape (n_atoms*n_cells, 2) int ndarray. This describes how the
        supercell atoms in Phonopy map onto the cells and atoms in
        Euphonic. The first column is the index of the equivalent atom
        in the unit cell, and the second column is the index of the
        cell in the supercell
    sc_relative_idx
        Shape (n_cells, n_cells) int ndarray. The index n of the
        equivalent vector in cell_origins for each
        cell_origins[i] -> cell_origins[j] vector in the supercell.
        See _get_supercell_relative_idx

    Returns
    -------
    fc
        Shape (n_cells, 3*n_atoms, 3*n_atoms) float ndarray. Force
        constants matrix in Euphonic shape
    """
    fc_phonopy = np.reshape(fc, (n_atoms, n_atoms*n_cells, 3, 3))
    fc_euphonic = np.full((n_atoms, n_cells, n_atoms, 3, 3), -1.0)

    for i in range(n_atoms):
        fc_tmp = np.zeros((n_cells, n_atoms, 3, 3))
        # Put fc's in correct atom and cell in supercell
        fc_tmp[cell_origins_map[:, 1],
               cell_origins_map[:, 0]] = fc_phonopy[i]
        # For Phonopy force constants, the n_atoms in the (n_atoms, ...)
        # shaped array may not be in the same cell within the supercell,
        # but Euphonic's interpolation requires this, so use equivalent
        # cell vector indices to arrange the force constants correctly
        atom_idx = np.where(cell_origins_map[:, 0] == i)[0][0]
        cell_idx = cell_origins_map[atom_idx, 1]
        fc_euphonic[i, sc_relative_idx[cell_idx]] = fc_tmp

    return np.reshape(np.transpose(
        fc_euphonic,
        axes=[1, 0, 3, 2, 4]), (n_cells, 3*n_atoms, 3*n_atoms))


def _check_fc_shape(fc_shape: Tuple[int, int], n_atoms: int,
                    n_cells: int, fc_filename: str,
                    summary_filename: str) -> None:
    """
    Check if force constants has the correct shape
    """
    if (not ((fc_shape[0] == n_atoms or fc_shape[0] == n_cells*n_atoms) and
             fc_shape[1] == n_cells*n_atoms)):
        raise ValueError((
            f'Force constants matrix with shape {fc_shape} read from '
            f'{fc_filename} is not compatible with crystal read from '
            f'{summary_filename} which has {n_atoms} atoms in the cell,'
            f' and {n_cells} cells in the supercell'))


def _extract_born(born_file_obj: TextIO) -> Dict[str, Union[float, np.ndarray]]:
    """
    Parse and convert dielectric tensor and born effective
    charge from BORN file

    Parameters
    ----------
    born_file_obj
        Pointer to BORN file

    Returns
    ----------
    born_dict
        Dict containing dielectric tensor and born effective charge
    """
    born_lines_str = born_file_obj.readlines()
    born_lines: List[List[float]] = []
    for line in born_lines_str:
        # Ignore comments
        if line.startswith('#'):
            pass
        else:
            born_lines.append([float(x) for x in line.split()])

    born_dict: Dict[str, Union[float, np.ndarray]] = {}

    idx0 = 0
    if len(born_lines[0]) == 1:
        # Then this is the NAC conversion factor
        born_dict['factor'] = born_lines[0][0]
        idx0 = 1

    # dielectric first line after factor
    # xx, xy, xz, yx, yy, yz, zx, zy, zz.
    born_dict['dielectric'] = np.array(born_lines[idx0]).reshape([3,3])

    # born charges after dielectric
    # xx, xy, xz, yx, yy, yz, zx, zy, zz.
    born_dict['born'] = np.array(
        [np.array(bl).reshape([3,3]) for bl in born_lines[idx0+1:]])

    return born_dict


def _extract_summary(filename: str, fc_extract: bool = False
                     ) -> Dict[str, Union[str, int, np.ndarray]]:
    """
    Read phonopy.yaml for summary data produced during the Phonopy
    post-process

    Parameters
    ----------
    filename
        Path and name of the Phonopy summary file (usually phonopy.yaml)
    fc_extract
        Whether to attempt to read force constants and related
        information from summary_object

    Returns
    ----------
    summary_dict
        A dict with the following keys: n_atoms, cell_vectors, atom_r,
        atom_type, atom_mass, ulength, umass. Also optionally has the
        following keys: sc_matrix, n_cells_in_sc, cell_origins,
        cell_origins_map, force_constants, ufc, born, dielectric
    """
    try:
        import yaml
        try:
            from yaml import CSafeLoader as SafeLoader
        except ImportError:
            from yaml import SafeLoader
    except ModuleNotFoundError as e:
        raise ImportPhonopyReaderError from e

    with open(filename, 'r') as summary_file:
        summary_object = yaml.load(summary_file, Loader=SafeLoader)
    (cell_vectors, n_atoms, atom_r, atom_mass,
     atom_type, _) = _extract_crystal_data(summary_object['primitive_cell'])

    summary_dict = {}
    try:
        pu = summary_object['physical_unit']
    except KeyError:
        default_units = {'atomic_mass': 'AMU',
                         'length': 'Angstrom',
                         'force_constants': 'eV/Angstrom^2'}
        print(f'physical_unit key not found in {filename}, assuming '
              f'the following units: {default_units}')
        pu = default_units
    summary_dict['ulength'] = pu['length'].lower()
    summary_dict['umass'] = pu['atomic_mass'].lower()

    summary_dict['n_atoms'] = n_atoms
    summary_dict['cell_vectors'] = cell_vectors
    summary_dict['atom_r'] = atom_r
    summary_dict['atom_type'] = atom_type
    summary_dict['atom_mass'] = atom_mass

    if fc_extract:
        u_to_sc_matrix = np.array(summary_object['supercell_matrix'])
        if 'primitive_matrix' in summary_object.keys():
            u_to_p_matrix = np.array(summary_object['primitive_matrix'])
            # Matrix to convert from primitive to supercell
            p_to_u_matrix = np.linalg.inv(u_to_p_matrix).transpose()
            p_to_sc_matrix = np.rint(
                np.matmul(u_to_sc_matrix, p_to_u_matrix)).astype(np.int32)
        else:
            u_to_p_matrix = np.identity(3, dtype=np.int32)
            p_to_sc_matrix = u_to_sc_matrix

        _, _, satom_r, _, _, sc_idx_in_pc = _extract_crystal_data(
            summary_object['supercell'])
        n_pcells_in_sc = int(np.rint(np.absolute(
            np.linalg.det(p_to_sc_matrix))))
        # Coords of supercell atoms in frac coords of the prim cell
        satom_r_pcell = np.einsum('ij,jk->ik', satom_r, p_to_sc_matrix)
        # Determine mapping from atoms in the supercell to the prim cell
        _, p2s_map_idx, sc_to_pc_atom_idx = np.unique(
            sc_idx_in_pc, return_index=True, return_inverse=True)
        # Get cell origins for all atoms
        cell_origins_per_atom = np.rint((
            satom_r_pcell - atom_r[sc_to_pc_atom_idx])).astype(np.int32)
        # Recenter cell origins onto atom 0
        atom0_idx = np.where(sc_to_pc_atom_idx == 0)[0]
        cell_origins_per_atom -= cell_origins_per_atom[[atom0_idx[0]]]
        # Build unique cell origins by getting all cell origins
        # associated with primitive atom 0
        cell_origins = cell_origins_per_atom[atom0_idx]
        # For some supercells, cell origins aren't always the
        # same for each atom in a supercell, and the cell origins are
        # sometimes outside the supercell. Create a mapping of cell
        # origins for atoms 1..n onto the equivalent cell origins for
        # atom 0, so the same cell origins can be used for all atoms
        cell_origins_map = np.zeros((n_atoms*n_pcells_in_sc, 2),
                                    dtype=np.int32)
        # Get origins of adjacent supercells in prim cell frac coords
        sc_origins =  get_all_origins((2,2,2), min_xyz=(-1,-1,-1))
        sc_origins_pcell = np.einsum('ij,jk->ik', sc_origins, p_to_sc_matrix)
        for i in range(n_pcells_in_sc*n_atoms):
            co_idx = np.where(
                (cell_origins_per_atom[i] == cell_origins).all(axis=1))[0]
            if len(co_idx) != 1:
                # Get equivalent cell origin in surrounding supercells
                origin_in_scs = cell_origins_per_atom[i] - sc_origins_pcell
                co_idx = -1
                # Find which of the 'unique' cell origins is equivalent
                for j, cell_origin in enumerate(cell_origins):
                    if np.any((origin_in_scs == cell_origin).all(axis=1)):
                        co_idx = j
                        break
                if co_idx == -1:
                    raise Exception((
                        'Couldn\'t determine cell origins for '
                        'force constants matrix'))
            cell_origins_map[i, 0] = sc_to_pc_atom_idx[i]
            cell_origins_map[i, 1] = co_idx

        summary_dict['sc_matrix'] = p_to_sc_matrix
        summary_dict['n_cells_in_sc'] = n_pcells_in_sc
        summary_dict['cell_origins'] = cell_origins[:n_pcells_in_sc]
        summary_dict['cell_origins'] = cell_origins[:n_pcells_in_sc]
        summary_dict['cell_origins_map'] = cell_origins_map
        p2s_map = sc_idx_in_pc[np.sort(p2s_map_idx)] - 1
        summary_dict['p2s_map'] = p2s_map
        sc_relative_idx = _get_supercell_relative_idx(cell_origins,
                                                      p_to_sc_matrix)
        summary_dict['sc_relative_idx'] = sc_relative_idx

        summary_dict['ufc'] = pu['force_constants'].replace(
            'Angstrom', 'angstrom')
        try:
            summary_dict['force_constants'] = _extract_force_constants_summary(
                summary_object, cell_origins_map, sc_relative_idx, p2s_map)
        except KeyError:
            pass

        try:
            summary_dict['born'] = np.array(
                summary_object['born_effective_charge'])
            summary_dict['dielectric'] = np.array(
                summary_object['dielectric_constant'])
        except KeyError:
            pass

    return summary_dict


def _extract_crystal_data(crystal: Dict[str, Any]
                          ) -> Tuple[np.ndarray, int, np.ndarray,
                                     np.ndarray, np.ndarray, np.ndarray]:
    """
    Gets relevant data from a section of phonopy.yaml

    Parameters
    ----------
    crystal
        Part of the dict obtained from reading a phonopy.yaml file. e.g.
        summary_dict['unit_cell']

    Returns
    -------
    cell_vectors
        Shape (3,3) float ndarray. Cell vectors, in same units as in
        the phonopy.yaml file
    n_atoms
        Number of atoms
    atom_r
        Shape (n_atoms, 3) float ndarray. Fractional position of each
        atom
    atom_mass
        Shape (n_atoms,) float ndarray. Mass of each atom, in same
        units as in the phonopy.yaml file
    atom_type
        Shape (n_atoms,) str ndarray. String specifying the species
        of each atom
    idx_in_pcell
        Shape (n_atoms,) int ndarray. Maps the atom index on the
        unit/supercell to the primitive cell
    """
    n_atoms = len(crystal['points'])
    cell_vectors = np.array(crystal['lattice'])
    atom_r = np.zeros((n_atoms, 3))
    atom_mass = np.zeros(n_atoms)
    atom_type = np.array([])
    for i in range(n_atoms):
        atom_mass[i] = crystal['points'][i]['mass']
        atom_r[i] = crystal['points'][i]['coordinates']
        atom_type = np.append(atom_type, crystal['points'][i]['symbol'])

    if 'reduced_to' in crystal['points'][0]:
        idx_in_pcell = np.zeros(n_atoms, dtype=np.int32)
        for i in range(n_atoms):
            idx_in_pcell[i] = crystal['points'][i]['reduced_to']
    else:
        # If reduced_to isn't present it is already the primitive cell
        idx_in_pcell = np.arange(n_atoms, dtype=np.int32)

    return cell_vectors, n_atoms, atom_r, atom_mass, atom_type, idx_in_pcell


def read_interpolation_data(
        path: str = '.',
        summary_name: str = 'phonopy.yaml',
        born_name: Optional[str] = None,
        fc_name: str = 'FORCE_CONSTANTS',
        fc_format: Optional[str] = None,
        cell_vectors_unit: str = 'angstrom',
        atom_mass_unit: str = 'amu',
        force_constants_unit: str = 'hartree/bohr**2',
        born_unit: str = 'e',
        dielectric_unit: str = '(e**2)/(bohr*hartree)') -> Dict[str, Any]:
    """
    Reads data from the phonopy summary file (default phonopy.yaml) and
    optionally born and force constants files. Only attempts to read
    from born or force constants files if these can't be found in the
    summary file.

    Parameters
    ----------
    path
        Path to directory containing the file(s)
    summary_name
        Filename of phonopy summary file, default phonopy.yaml. By
        default any information (e.g. force constants) read from this
        file takes priority
    born_name
        Name of the Phonopy file containing born charges and dielectric
        tensor, (by convention in Phonopy this would be called BORN). Is
        only read if Born charges can't be found in the summary_name
        file
    fc_name
        Name of file containing force constants. Is only read if force
        constants can't be found in summary_name
    fc_format
        One of {'phonopy', 'hdf5'}. Format of file containing force
        constants data. FORCE_CONSTANTS is type 'phonopy'
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
    data_dict
        A dict with the following keys: 'n_atoms', 'cell_vectors',
        'cell_vectors_unit', 'atom_r', 'atom_type', 'atom_mass',
        'atom_mass_unit', 'force_constants', 'force_constants_unit',
        'sc_matrix' and 'cell_origins'. Also contains 'born',
        'born_unit', 'dielectric' and 'dielectric_unit' if they are
        present in phonopy.yaml
    """
    summary_pathname = os.path.join(path, summary_name)
    summary_dict = _extract_summary(summary_pathname, fc_extract=True)

    # Only read force constants if it's not in summary file
    if not 'force_constants' in summary_dict:
        hdf5_exts = ['hdf5', 'hd5', 'h5']
        if fc_format is None:
            fc_format = os.path.splitext(fc_name)[1].strip('.')
            if fc_format not in hdf5_exts:
                fc_format = 'phonopy'
        fc_pathname = os.path.join(path, fc_name)
        print((f'Force constants not found in {summary_pathname}, '
               f'attempting to read from {fc_pathname}'))
        n_atoms = summary_dict['n_atoms']
        n_cells = summary_dict['n_cells_in_sc']
        if fc_format == 'phonopy':
            with open(fc_pathname, 'r') as fc_file:
                summary_dict['force_constants'] = _extract_force_constants(
                    fc_pathname, n_atoms, n_cells, summary_pathname,
                    summary_dict['cell_origins_map'],
                    summary_dict['sc_relative_idx'],
                    summary_dict['p2s_map'])
        elif fc_format in 'hdf5':
            summary_dict['force_constants'] =  _extract_force_constants_hdf5(
                fc_pathname, n_atoms, n_cells, summary_pathname,
                summary_dict['cell_origins_map'],
                summary_dict['sc_relative_idx'])
        else:
            raise ValueError((
                f'Force constants file format "{fc_format}" of '
                f'"{fc_name}" is not recognised'))

    # Only read born/dielectric if they're not in summary file and the
    # user has specified a Born file
    dipole_keys = ['born', 'dielectric']
    if (born_name is not None and
            len(dipole_keys & summary_dict.keys()) != len(dipole_keys)):
        born_pathname = os.path.join(path, born_name)
        print((f'Born, dielectric not found in {summary_pathname}, '
               f'attempting to read from {born_pathname}'))
        with open(born_pathname, 'r') as born_file:
            born_dict = _extract_born(born_file)
        summary_dict['born'] = born_dict['born']
        summary_dict['dielectric'] = born_dict['dielectric']

    # Units from summary_file
    ulength = summary_dict['ulength']
    umass = summary_dict['umass']
    ufc = summary_dict['ufc']

    data_dict: Dict[str, Any] = {}
    data_dict['crystal'] = {}
    cry_dict = data_dict['crystal']
    cry_dict['cell_vectors'] = summary_dict['cell_vectors']*ureg(
        ulength).to(cell_vectors_unit).magnitude
    cry_dict['cell_vectors_unit'] = cell_vectors_unit
    # Normalise atom coordinates
    cry_dict['atom_r'] = (summary_dict['atom_r']
                          - np.floor(summary_dict['atom_r']))
    cry_dict['atom_type'] = summary_dict['atom_type']
    cry_dict['atom_mass'] = summary_dict['atom_mass']*ureg(
        umass).to(atom_mass_unit).magnitude
    cry_dict['atom_mass_unit'] = atom_mass_unit

    data_dict['force_constants'] = summary_dict['force_constants']*ureg(
        ufc).to(force_constants_unit).magnitude
    data_dict['force_constants_unit'] = force_constants_unit
    data_dict['sc_matrix'] = summary_dict['sc_matrix']
    data_dict['cell_origins'] = summary_dict['cell_origins']

    try:
        data_dict['born'] = summary_dict['born']*ureg(
            'e').to(born_unit).magnitude
        data_dict['born_unit'] = born_unit
        data_dict['dielectric'] = summary_dict['dielectric']*ureg(
            'e**2/(hartree*bohr)').to(dielectric_unit).magnitude
        data_dict['dielectric_unit'] = dielectric_unit
    except KeyError:
        pass

    return data_dict
