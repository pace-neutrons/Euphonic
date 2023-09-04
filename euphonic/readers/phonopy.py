import os
import re
import warnings
from typing import Optional, Dict, Any, Union, Tuple, TextIO, Sequence, List

import numpy as np

from euphonic import ureg
from euphonic.util import convert_fc_phases


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
        if 'qpoint' in hdf5_file:
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
    mesh/band/qpoints.yaml/hdf5 file and returns it in a dictionary.
    May also read from phonopy.yaml for structure information. The
    output eigenvectors will have their phase transformed from
    per atom to per cell phases.

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

    if read_eigenvectors and not 'eigenvectors' in phonon_dict:
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
    if 'weights' in phonon_dict:
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

    rk_diff = atom_r[:, None, :] - atom_r[None, :, :]
    conversion = np.exp(-2j*np.pi*np.einsum('il,jkl->ijk', qpts, rk_diff))
    eigvecs = np.einsum('ijklm,ijl->ijklm', eigvecs, conversion)
    return np.reshape(eigvecs, (n_qpts, 3*n_atoms, n_atoms, 3))


def _extract_force_constants(fc_pathname: str, n_atoms: int, n_cells: int,
                             summary_name: str) -> np.ndarray:
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

    Returns
    -------
    fc
        Shape (n_atoms, n_atoms*n_cells, 3, 3) or
        (n_atoms*n_cells, n_atoms*n_cells, 3, 3) float ndarray. The
        force constants in Phonopy convention
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
    return fc.reshape(fc_dims + [3, 3])


def _extract_force_constants_hdf5(
        filename: str, n_atoms: int, n_cells: int, summary_name: str,
        ) -> np.ndarray:
    try:
        import h5py
    except ModuleNotFoundError as e:
        raise ImportPhonopyReaderError from e

    with h5py.File(filename, 'r') as fc_file:
        fc = fc_file['force_constants'][:]
        _check_fc_shape(fc.shape, n_atoms, n_cells, fc_file.filename,
                        summary_name)
    return fc


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
        born_dict['nac_factor'] = born_lines[0][0]
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
    if 'physical_unit' in summary_object:
        pu = summary_object['physical_unit']
    else:
        default_units = {'atomic_mass': 'AMU',
                         'length': 'Angstrom',
                         'force_constants': 'eV/Angstrom^2'}
        print(f'physical_unit key not found in {filename}, assuming '
              f'the following units: {default_units}')
        pu = default_units
    # Format units so they can be read by Pint
    pu['atomic_mass'] = pu['atomic_mass'].replace('AMU', 'amu')
    # Ensure denominator is surrounded by brackets to handle
    # cases like 'eV/Angstrom.au'
    divs = re.findall(r'/([^/]+)', pu['force_constants'])
    pu['force_constants'] = (pu['force_constants'].split('/')[0]
                             + '/(' + '/('.join([d+')' for d in divs]))
    for key, value in pu.items():
        pu[key] = value.replace('au', 'bohr').replace('Angstrom', 'angstrom')

    summary_dict['ulength'] = pu['length']
    summary_dict['umass'] = pu['atomic_mass']

    summary_dict['n_atoms'] = n_atoms
    summary_dict['cell_vectors'] = cell_vectors
    summary_dict['atom_r'] = atom_r
    summary_dict['atom_type'] = atom_type
    summary_dict['atom_mass'] = atom_mass

    if fc_extract:
        # Get matrix to convert from primitive cell to unit cell
        u_to_sc_matrix = np.array(summary_object['supercell_matrix'])
        if 'primitive_matrix' in summary_object:
            u_to_p_matrix = np.array(summary_object['primitive_matrix'])
            # Matrix to convert from primitive to supercell
            p_to_u_matrix = np.linalg.inv(u_to_p_matrix).transpose()
            p_to_sc_matrix = np.rint(
                np.matmul(u_to_sc_matrix, p_to_u_matrix)).astype(np.int32)
        else:
            u_to_p_matrix = np.identity(3, dtype=np.int32)
            p_to_sc_matrix = u_to_sc_matrix

        # Get supercell info
        _, _, satom_r, _, _, sc_idx_in_pc = _extract_crystal_data(
            summary_object['supercell'])
        # Coords of supercell atoms in frac coords of the prim cell
        satom_r_pcell = np.einsum('ij,jk->ik', satom_r, p_to_sc_matrix)
        # Determine mapping from atoms in the supercell to the prim cell
        # Note: -1 to get 0-indexed instead of 1-indexed values
        pc_to_sc_atom_idx, sc_to_pc_atom_idx = np.unique(
            sc_idx_in_pc - 1, return_inverse=True)

        summary_dict['sc_matrix'] = p_to_sc_matrix
        summary_dict['sc_atom_r'] = satom_r_pcell
        summary_dict['pc_to_sc_atom_idx'] = pc_to_sc_atom_idx
        summary_dict['sc_to_pc_atom_idx'] = sc_to_pc_atom_idx
        summary_dict['ufc'] = pu['force_constants']

        if 'force_constants' in summary_object:
            fc = summary_object['force_constants']
            summary_dict['force_constants'] = np.array(
                fc['elements']).reshape(fc['shape'] + [3,3])

        # NAC factor may be present even if born is not in phonopy.yaml
        # (born may be in separate BORN file), and the BORN file may
        # not necessarily contain NAC factor, so just try reading it
        # from phonopy.yaml anyway

        # New format
        if 'nac' in summary_object:
            nac = summary_object['nac']
            summary_dict['nac_factor'] = nac.get('unit_conversion_factor')
            summary_dict['born'] = nac.get('born_effective_charge')
            summary_dict['dielectric'] = nac.get('dielectric_constant')

        # Old format
        else:
            summary_dict['nac_factor'] = summary_object[
                'phonopy'].get('nac_unit_conversion_factor')
            summary_dict['born'] = summary_object.get('born_effective_charge')
            summary_dict['dielectric'] = summary_object.get(
                'dielectric_constant')

        # Cast arrays and drop keys if data not found
        for key in ('nac_factor', 'born', 'dielectric'):
            if summary_dict[key] is None:
                del summary_dict[key]

        for key in ('born', 'dielectric'):
            if key in summary_dict:
                summary_dict[key] = np.array(summary_dict[key])

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
    summary file. Note that the output force constants will be transformed
    to a Euphonic-like shape, but if it is a polar material it will still
    be the long-ranged force constants matrix (and not the short-ranged one
    that Euphonic requires in the case of polar materials).

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
        n_cells = int(len(summary_dict['sc_atom_r'])/n_atoms)
        if fc_format == 'phonopy':
            with open(fc_pathname, 'r') as fc_file:
                summary_dict['force_constants'] = _extract_force_constants(
                    fc_pathname, n_atoms, n_cells, summary_pathname)
        elif fc_format in 'hdf5':
            summary_dict['force_constants'] =  _extract_force_constants_hdf5(
                fc_pathname, n_atoms, n_cells, summary_pathname)
        else:
            raise ValueError((
                f'Force constants file format "{fc_format}" of '
                f'"{fc_name}" is not recognised'))

    # Only read born/dielectric if they're not in summary file and the
    # user has specified a Born file
    dipole_keys = ['born', 'dielectric', 'nac_factor']
    if (born_name is not None and
            len(dipole_keys & summary_dict.keys()) != len(dipole_keys)):
        born_pathname = os.path.join(path, born_name)
        print((f'Born, dielectric not found in {summary_pathname}, '
               f'attempting to read from {born_pathname}'))
        with open(born_pathname, 'r') as born_file:
            born_dict = _extract_born(born_file)
        # Let BORN file take priority, but merge because the 'nac_factor'
        # key may not always be present in BORN
        summary_dict.update(born_dict)
    # Check if born key is present, then factor is also present. It
    # may not always be e.g. you can run Phonopy so that 'born',
    # 'dielectric' are written to phonopy.yaml, but if NAC = .FALSE.
    # 'nac_factor' will not be written. In this case raise error.
    if ('born' in summary_dict
            and 'nac_factor' not in summary_dict):
        raise KeyError(f'nac_unit_conversion_factor could not be found in '
                       f'{summary_pathname} or the BORN file (if given), so '
                       f'units of the dielectric tensor cannot be determined.')

    # Units from summary_file
    ulength = summary_dict['ulength']
    umass = summary_dict['umass']
    ufc = summary_dict['ufc']

    data_dict: Dict[str, Any] = {}
    cry_dict = data_dict['crystal'] = {}
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

    fc, cell_origins = convert_fc_phases(
         summary_dict['force_constants'], summary_dict['atom_r'],
         summary_dict['sc_atom_r'], summary_dict['pc_to_sc_atom_idx'],
         summary_dict['sc_to_pc_atom_idx'], summary_dict['sc_matrix'])
    data_dict['force_constants'] = fc*ureg(
        ufc).to(force_constants_unit).magnitude
    data_dict['force_constants_unit'] = force_constants_unit
    data_dict['sc_matrix'] = summary_dict['sc_matrix']
    data_dict['cell_origins'] = cell_origins

    try:
        data_dict['born'] = summary_dict['born']*ureg(
            'e').to(born_unit).magnitude
        data_dict['born_unit'] = born_unit
        # Phonopy NAC conversion factor converts the following NAC
        # term (eq. 75 in Gonze & Lee 1997, ignoring dimensionless
        # units)
        # born**2/(volume x dielectric)
        #     = 1/(ulength**3 x di_energy x di_length)
        # to the same units as the force constants (given by ufc)
        # i.e. factor*nac_units = force constants units
        # So allow pint to do this unit conversion automatically
        phonopy_dielectric_unit = ureg('e**2')/(ureg(ulength)**3*ureg(ufc))
        data_dict['dielectric'] = (
            summary_dict['dielectric']/summary_dict['nac_factor']
            *phonopy_dielectric_unit.to(dielectric_unit).magnitude)
        data_dict['dielectric_unit'] = dielectric_unit
    except KeyError:
        pass

    return data_dict
