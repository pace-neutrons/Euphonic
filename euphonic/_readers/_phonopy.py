import os
import warnings
import numpy as np
from euphonic import ureg
from euphonic.util import reciprocal_lattice, is_gamma
try:
    import yaml
    import h5py
except ImportError as e:
    raise ImportError(('Cannot import yaml, h5py to read Phonopy files, maybe '
                       'they are not installed. To install the optional '
                       'dependencies for Euphonic\'s Phonopy reader, '
                       'try:\n\npip install euphonic[phonopy_reader]\n')) from e


def _convert_weights(weights):
    """
    Convert q-point weights to normalised convention

    Parameters
    ----------
    weights : (n_qpts,) float ndarray
        Weights in Phonopy convention

    Returns
    ----------
    norm_weights : (n_qpts,) float ndarray
        Normalised weights
    """
    total_weight = weights.sum()
    return weights/total_weight


def _extract_phonon_data_yaml(phonon_data):
    """
    From a dictionary obtained from reading a mesh/band/qpoint.yaml file,
    extract the relevant information as a dict of Numpy arrays. No unit
    conversion is done at this point, so they are in the same units as in the
    .yaml file

    Parameters
    ----------
    phonon_data : dict
        Obtained from reading a Phonopy mesh/band/qpoint.yaml file

    Returns
    -------
    data_dict : dict
        A dict with the following keys:
            'qpts', 'freqs'
        It may also have the following keys if they are present in the .yaml
        file:
            'eigenvecs', 'weights', 'cell_vec', 'ion_r', 'ion_mass', 'ion_type'
    """

    data_dict = {}
    phonons = [phon for phon in phonon_data['phonon']]
    bands_data_each_qpt = [bands_data['band'] for bands_data in phonons]

    data_dict['qpts'] = np.array([phon['q-position'] for phon in phonons])
    data_dict['freqs'] = np.array(
        [[band_data['frequency'] for band_data in bands_data]
            for bands_data in bands_data_each_qpt])
    try:
        data_dict['eigenvecs'] = np.squeeze(np.array(
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
        data_dict['cell_vec'] = np.array(phonon_data['lattice'])
        data_dict['ion_r'] = np.array(
            [ion['coordinates'] for ion in phonon_data['points']])
        data_dict['ion_mass'] = np.array(
            [ion['mass'] for ion in phonon_data['points']])
        data_dict['ion_type'] = np.array(
            [ion['symbol'] for ion in phonon_data['points']])
    except KeyError:
        pass

    return data_dict

def _extract_phonon_data_hdf5(hdf5_file):
    """
    From a h5py.File obtained from reading a mesh/band/qpoint.hdf5 file,
    extract the relevant information as a dict of Numpy arrays. No unit
    conversion is done at this point, so they are in the same units as in the
    .hdf5 file

    Parameters
    ----------
    hdf5_file : h5py.File
        Obtained from reading a Phonopy mesh/band/qpoint.hdf5 file

    Returns
    -------
    data_dict : dict
        A dict with the following keys:
            'qpts', 'freqs'
        It may also have the following keys if they are present in the .hdf5
        file:
            'eigenvecs', 'weights'
    """
    if 'qpoint' in hdf5_file.keys():
        data_dict = {}
        data_dict['qpts'] = hdf5_file['qpoint'][()]
        data_dict['freqs'] = hdf5_file['frequency'][()]
        # Eigenvectors may not be present if users haven't set --eigvecs when
        # running Phonopy
        try:
            data_dict['eigenvecs'] = hdf5_file['eigenvector'][()]
        except KeyError:
            pass
        # Only mesh.hdf5 has weights
        try:
            data_dict['weights'] = hdf5_file['weight'][()]
        except KeyError:
            pass
    # Is a band.hdf5 file - q-points are stored in 'path' and need special
    # treatment
    else:
        data_dict = _extract_band_data_hdf5(hdf5_file)

    return data_dict

def _extract_band_data_hdf5(hdf5_file):
    """
    Read data from a Phonopy band.hdf5 file. All information is stored in the
    shape (n_paths, n_qpoints_in_path, x) rather than (n_qpts, x) so needs
    special treatment
    """
    data_dict = {}
    data_dict['qpts'] = hdf5_file['path'][()].reshape(
        -1, hdf5_file['path'][()].shape[-1])
    data_dict['freqs'] = hdf5_file['frequency'][()].reshape(
        -1, hdf5_file['frequency'][()].shape[-1])
    try:
        # The last 2 dimensions of eigenvectors in bands.hdf5 are for some
        # reason transposed compared to mesh/qpoints.hdf5, so also transpose to
        # handle this
        data_dict['eigenvecs'] = hdf5_file['eigenvector'][()].reshape(
            -1, *hdf5_file['eigenvector'][()].shape[-2:]).transpose([0,2,1])
    except KeyError:
        pass
    return data_dict


def _read_phonon_data(path='.', phonon_name='band.yaml', phonon_format=None,
                      summary_name='phonopy.yaml'):
    """
    Reads precalculated phonon mode data from a Phonopy
    mesh/band/qpoints.yaml/hdf5 file and returns it in a dictionary. May also
    read from phonopy.yaml for structure information.

    Parameters
    ----------
    path : str, optional, default '.'
        Path to directory containing the file(s)
    phonon_name : str, optional, default 'band.yaml'
        Name of Phonopy file including the frequencies and eigenvectors
    phonon_format : {'yaml', 'hdf5'} str, optional, default None
        Format of the phonon_name file if it isn't obvious from the phonon_name
        extension
    summary_name : str, optional, default 'phonopy.yaml'
        Name of Phonopy summary file to read the crystal information from.
        Crystal information in the phonon_name file takes priority, but if it
        isn't present, crystal information is read from summary_name instead

    Returns
    -------
    data_dict : dict
        A dict with the following keys: 'n_ions', 'n_branches', 'n_qpts'
        'cell_vec', 'recip_vec', 'ion_r', 'ion_type', 'ion_mass', 'qpts',
        'weights', 'freqs', 'eigenvecs', 'split_i', 'split_freqs',
        'split_eigenvecs'. The returned data is in Hartree atomic units
    """
    phonon_pathname = os.path.join(path, phonon_name)
    summary_pathname = os.path.join(path, summary_name)

    hdf5_exts = ['hdf5', 'hd5', 'h5']
    yaml_exts = ['yaml', 'yml', 'yl']
    if phonon_format is None:
        phonon_format = os.path.splitext(phonon_name)[1].strip('.')
        if phonon_format == '':
            raise Exception(f'Format of {phonon_name} couldn\'t be determined')

    if phonon_format in hdf5_exts:
        with h5py.File(phonon_pathname, 'r') as hdf5_file:
            phonon_dict = _extract_phonon_data_hdf5(hdf5_file)
    elif phonon_format in yaml_exts:
        with open(phonon_pathname, 'r') as yaml_file:
            phonon_data = yaml.safe_load(yaml_file)
        phonon_dict = _extract_phonon_data_yaml(phonon_data)
    else:
        raise Exception((f'File format {phonon_format} of {phonon_name} is not '
                          'recognised'))

    if not 'eigenvecs' in phonon_dict.keys():
        raise Exception((f'Eigenvectors couldn\'t be foud in {phonon_pathname},'
                          ' ensure --eigvecs was set when running Phonopy'))

    # Since units are not explicitly defined in mesh/band/qpoints.yaml/hdf5
    # assume:
    ulength = 'angstrom'
    umass = 'amu'
    ufreq = 'THz'

    crystal_keys = ['cell_vec', 'ion_r', 'ion_mass', 'ion_type']
    # Check if crystal structure has been read from phonon_file, if not get
    # structure from summary_file
    if len(crystal_keys & phonon_dict.keys()) != len(crystal_keys):
        with open(summary_pathname, 'r') as summary_file:
            summary_data = yaml.safe_load(summary_file)
            summary_dict = _extract_summary(summary_data)
        phonon_dict['cell_vec'] = summary_dict['cell_vec']
        phonon_dict['ion_r'] = summary_dict['ion_r']
        phonon_dict['ion_mass'] = summary_dict['ion_mass']
        phonon_dict['ion_type'] = summary_dict['ion_type']
        # Overwrite assumed units if they are found in summary file
        ulength = summary_dict['ulength']
        umass = summary_dict['umass']
        # Check phonon_file and summary_file are commensurate
        if 3*len(phonon_dict['ion_r']) != len(phonon_dict['freqs'][0]):
            raise Exception((f'Phonon file {phonon_pathname} not commensurate '
                             f'with summary file {summary_pathname}. Please '
                              'check contents'))

    # Add extra derived keys
    n_ions = len(phonon_dict['ion_r'])
    phonon_dict['n_ions'] = n_ions
    phonon_dict['n_branches'] = 3*n_ions
    n_qpts = len(phonon_dict['qpts'])
    phonon_dict['n_qpts'] = n_qpts
    # Convert Phonopy conventions to Euphonic conventions
    phonon_dict['eigenvecs'] = convert_eigenvector_phases(phonon_dict)
    # If weights not specified, assume equal weights
    if not 'weights' in phonon_dict.keys():
        phonon_dict['weights'] = np.full(n_qpts, 1)
    phonon_dict['weights'] = _convert_weights(phonon_dict['weights'])
    # Convert units to atomic
    phonon_dict['cell_vec'] = phonon_dict['cell_vec']*ureg(
        ulength).to('bohr').magnitude
    phonon_dict['recip_vec'] = reciprocal_lattice(phonon_dict['cell_vec'])
    phonon_dict['ion_mass'] = phonon_dict['ion_mass']*ureg(
        umass).to('e_mass').magnitude
    phonon_dict['freqs'] = phonon_dict['freqs']*ureg(
        ufreq).to('hartree', 'spectroscopy').magnitude
    # Reading LO-TO split frequencies from Phonopy files is currently not
    # supported, return empty arrays
    phonon_dict['split_i'] = np.empty((0,), dtype=np.int32)
    phonon_dict['split_freqs'] = np.empty((0, 3*n_ions))
    phonon_dict['split_eigenvecs'] = np.empty((0, 3*n_ions, n_ions, 3),
                                            dtype=np.complex128)

    metadata = type('', (), {})()
    metadata.model = 'phonopy'
    metadata.phonon_name = phonon_name
    metadata.phonon_format = phonon_format
    metadata.summary_name = summary_name

    phonon_dict['metadata'] = metadata
    return phonon_dict


def convert_eigenvector_phases(phonon_dict):
    """
    When interpolating the force constants matrix, Euphonic uses a phase
    convention of e^iq.r_a, where r_a is the coordinate of each CELL in the
    supercell, whereas Phonopy uses e^iq.r_k, where r_k is the coordinate of
    each ATOM in the supercell. This must be accounted for when reading Phonopy
    eigenvectors by applying a phase of e^-iq(r_k - r_k')
    """
    ion_r = phonon_dict['ion_r']
    n_ions = len(ion_r)
    qpts = phonon_dict['qpts']
    n_qpts = len(qpts)

    eigvecs = np.reshape(phonon_dict["eigenvecs"],
                         (n_qpts, n_ions, 3, n_ions, 3))
    na = np.newaxis
    rk_diff = ion_r[:, na, :] - ion_r[na, :, :]
    conversion = np.exp(-2j*np.pi*np.einsum('il,jkl->ijk', qpts, rk_diff))
    eigvecs = np.einsum('ijklm,ijl->ijklm', eigvecs, conversion)
    return np.reshape(eigvecs, (n_qpts, 3*n_ions, n_ions, 3))


def _check_fc_file(filename):
    """
    Check for basic formatting of force constants file generated by
    Phonopy during pre-process, then make a quick validity check.
    --write-fc or equivalent must be set at least in Phonopy for the
    file to be made available.

    Parameters
    ----------
    filename : str
        Pathname of target force constants file.

    Returns
    ----------
    success : bool
        True/false if the file meets/fails basic specification.
    """
    #TODO deep check
    with open(filename, 'r') as fco:
        fco.readline() # first line can vary
        line = fco.readline() # always two numbers
        n = np.array(line.strip('\n').split(' ')).astype(np.int)

    if n.size != 2:
        return False

    return True

def _check_born_file(filename):
    """
    Check for formatting the born file created by the user during
    the pre-process when using non-analytical term correction
    activated using the --nac option.

    Parameters
    ----------
    filename : str
        Pathname of target born file.

    Returns
    ----------
    success : bool
        Returns true if born file data is valid
    """

    return True #TODO BORN formatting

    with open(filename, 'r') as bfo:
        #first is single number
        try:
            line = float(bfo.readline())
        except:
            return False

        #lines 2+ are always 9 numbers
        line = bfo.readline()
        while line:
            try:
                tensor = line.split(' ')
                tensor = [float(t) for t in tensor]
            except:
                return False

            if len(tensor) != 9:
                return False

            line = bfo.readline()

    return True

def _check_born_summary(filename):
    """ DOC
    Check for presence of the dielectric tensor and born effective
    charge in the phonopy summary file.

    Parameters
    ----------
    filename : str
        Pathname of target summary file.

    Returns
    ----------
    success : bool
        Returns true if born data is present.
    """
    with open(filename, 'r') as bfo:
        bec_present = False
        dt_present = False

        line = bfo.readline()
        while line in bfo:
            if 'born_effective_charge' in line:
                bec_present = True

            if 'dielectric_constant' in line:
                dt_present = True

            if bec_present and dt_present:
                return True

            line = bfo.readline()

    return False


def _extract_force_constants(fc_object, n_ions, n_cells):
    warnings.filterwarnings('ignore')
    sc_block = n_ions*n_cells

    data = np.array([])
    fc = np.array([])

    fc_dims =  [dim for dim in fc_object.readline().split()]
    if len(fc_dims) == 1: # single shape specifier implies full format
        is_full_fc = True
        fc_dims = [fc_dims[0], fc_dims[0]]
    elif fc_dims[0] == fc_dims[1]:
        full_fc = True
    else:
        full_fc = False

    first = True
    skip_blocks = 0
    max_rows = 3*n_ions*n_cells
    for n_ion in range(n_ions):
        skip_header = 4*skip_blocks*sc_block

        # Prints line fmt errors to stderr with invalid_raise=False
        # stopped with warnings
        data = np.genfromtxt(fc_object,
                         skip_header=skip_header,
                         max_rows=max_rows,
                         usecols=(0,1,2),
                         invalid_raise=False)

        if full_fc:
            skip_blocks = n_cells - 1

        if first:
            fc = data
            first = False
        else:
            fc = np.concatenate([fc, data])

    return _reshape_fc(fc, n_ions, n_cells)

def _extract_force_constants_hdf5(fc_object, n_ions, n_cells):
    fc = fc_object['force_constants'][:]
    p2s_map = list(fc_object['p2s_map']) # 'primitive' to supercell indexing
    physical_units = list(fc_object['physical_unit'])[0].decode('utf-8')

    if fc.shape[0] == fc.shape[1]: # FULL FC, convert down to COMPACT
        fc = fc[p2s_map, :, :, :]

    fc_unfolded = fc.reshape(n_ions*n_ions*n_cells, 3, 3)
    return _reshape_fc(fc_unfolded, n_ions, n_cells)

def _extract_force_constants_summary(summary_object):
    """
    Get force constants from phonopy yaml summary file.

    Parameters
    ----------
    summary_object : dict
        Dict containing contents of phonopy.yaml

    Returns
    ----------
    units : dict
        Dict containing force constants in Euphonic format.
    """
    fc_entry = summary_object['force_constants']

    fc_dims = fc_entry['shape']
    fc_format = fc_entry['format']

    sc_matrix = summary_object['supercell_matrix']
    n_cells = int(np.rint(np.absolute(np.linalg.det(sc_matrix))))
    n_ions = len(summary_object['unit_cell']['points'])

    if fc_format == 'compact':
        fc = np.array(fc_entry['elements'])

    elif fc_format == 'full': # FULL FC, convert down to COMPACT
        p2s_map = [pi for pi in range(0, n_ions*n_cells, n_ions)]
        fc = np.array(fc_entry['elements']).reshape(
                [n_ions*n_cells, n_ions, n_cells, 3, 3])[p2s_map, :, :, :, :]

    return _reshape_fc(fc, n_ions, n_cells)

def _reshape_fc(fc, n_ions, n_cells):
    return np.reshape( np.transpose(
        np.reshape(fc, (n_ions, n_ions, n_cells, 3, 3)),
        axes=[2,0,3,1,4]), (n_cells, 3*n_ions, 3*n_ions))

def _extract_born(born_object):
    """
    Parse and convert dielectric tensor and born effective
    charge from BORN file

    Parameters
    ----------
    born_object : file-like object
        Object representing contents of BORN

    Returns
    ----------
    born_dict : float ndarray
        Dict containing dielectric tensor and born effective charge
    """

    born_lines_str = born_object.readlines()
    born_lines = [narr.split() for narr in born_lines_str]

    for l_i, line in enumerate(born_lines):
        try:
            born_lines[l_i] = [np.float(i) for i in line]
            continue
        except:
            pass

    n_lines = len(born_lines)
    n_entries = (n_lines - 1) // 4

    # factor first line
    factor = born_lines[0][0]

    # dielectric second line
    # xx, xy, xz, yx, yy, yz, zx, zy, zz.
    dielectric = np.array(born_lines[1]).reshape([3,3])

    # born ec third line onwards
    # xx, xy, xz, yx, yy, yz, zx, zy, zz.
    born_lines = born_lines[2:]
    born = np.array([np.array(bl).reshape([3,3]) for bl in born_lines])

    born_dict = {}
    born_dict['factor'] = factor
    born_dict['dielectric'] = dielectric
    born_dict['born'] = born

    return born_dict


def _extract_summary(summary_object, fc_extract=False):
    """
    Read phonopy.yaml for summary data produced during the Phonopy
    post-process.

    Parameters
    ----------
    summary_object : dict-like object
        The Phonopy data object which contains phonon data.

    Returns
    ----------
    summary_dict : dict
        A dict containing: sc_matrix, n_cells_in_sc, n_ions, cell_vec,
        ion_r, ion_mass, ion_type, n_ions, cell_origins.
    """
    n_ions = len(summary_object['unit_cell']['points'])
    cell_vec = np.array(summary_object['unit_cell']['lattice'])
    ion_r = np.zeros((n_ions, 3))
    ion_mass = np.zeros(n_ions)
    ion_type = np.array([])
    for i in range(n_ions):
        ion_mass[i] = summary_object['unit_cell']['points'][i]['mass']
        ion_r[i] = summary_object['unit_cell']['points'][i]['coordinates']
        ion_type = np.append(
            ion_type, summary_object['unit_cell']['points'][i]['symbol'])

    sc_matrix = np.array(summary_object['supercell_matrix'])
    n_cells_in_sc = int(np.rint(np.absolute(np.linalg.det(sc_matrix))))
    sc_n_ions = len(summary_object['supercell']['points'])
    sc_ion_r = np.zeros((sc_n_ions, 3))
    for i in range(sc_n_ions):
        sc_ion_r[i] = summary_object['supercell']['points'][i]['coordinates']

    # Coordinates of supercell ions in fractional coords of the unit cell
    sc_ion_r_ucell = np.einsum('ij,jk->ij', sc_ion_r, sc_matrix)
    cell_origins = np.rint(
        sc_ion_r_ucell[:n_cells_in_sc] - ion_r[0]).astype(np.int32)

    summary_dict = {}
    pu = summary_object['physical_unit']
    summary_dict['ulength'] = pu['length'].lower()
    summary_dict['umass'] = pu['atomic_mass'].lower()
    summary_dict['ufc'] = pu['force_constants'].replace('Angstrom', 'angstrom')

    summary_dict['n_ions'] = n_ions
    summary_dict['cell_vec'] = cell_vec
    summary_dict['ion_r'] = ion_r
    summary_dict['ion_type'] = ion_type
    summary_dict['ion_mass'] = ion_mass

    summary_dict['cell_origins'] = cell_origins
    summary_dict['sc_matrix'] = sc_matrix
    summary_dict['n_cells_in_sc'] = n_cells_in_sc

    try:
        summary_dict['force_constants'] = _extract_force_constants_summary(
            summary_object)
    except KeyError:
        pass

    try:
        summary_dict['born'] = np.array(summary_object['born_effective_charge'])
        summary_dict['dielectric'] = np.array(
            summary_object['dielectric_constant'])
    except KeyError:
        pass

    return summary_dict

def _check_fc_summary(filename):
    """
    Checks summary file for force constants.

    Parameters
    ----------
    filename : str
        Pathname of target summary file containing force constants.

    Returns
    -------
    success : bool
        Returns true/false If the file contains/lacks a reference to
        force constants
    """
    with open(filename, 'r') as fco:
        line = fco.readline()
        while line:
            if 'force_constants' in line:
                return True
            line = fco.readline()
    return False


def _read_interpolation_data(path='.', summary_name='phonopy.yaml',
                             born_name=None, fc_name='FORCE_CONSTANTS',
                             fc_format=None):
    """
    Reads data from the phonopy summary file (default phonopy.yaml) and
    optionally born (default BORN) and force constants (default FORCE_CONSTANTS)
    files. Only attempts to read from born or force constants files if these
    can't be found in the summary file.

    Parameters
    ----------
    path : str, optional, default '.'
        Path to directory containing the file(s)
    summary_name : str, optional, default 'phonpy.yaml'
        Filename of phonopy summary file, default phonopy.yaml. By default any
        information (e.g. force constants) read from this file takes priority
    born_name : str, optional, default None
        Name of the Phonopy file containing born charges and dielectric tensor,
        (by convention in Phonopy this would be called BORN). Is only read if
        Born charges can't be found in the summary_name file
    fc_name : str, optional, default 'FORCE_CONSTANTS'
        Name of file containing force constants. Is only read if force constants
        can't be found in summary_name
    fc_format : {'phonopy', 'hdf5'} str, optional, default None
        Format of file containing force constants data. FORCE_CONSTANTS is type
        'phonopy'

    Returns
    -------
    data_dict : dict
        A dict with the following keys: 'n_ions', 'n_branches', 'cell_vec',
        'recip_vec', 'ion_r', 'ion_type', 'ion_mass', 'force_constants',
        'sc_matrix', 'n_cells_in_sc' and 'cell_origins'. Also contains 'born'
        and 'dielectric' if they are present in the .castep_bin or .check file
    """
    summary_pathname = os.path.join(path, summary_name)

    with open(summary_pathname, 'r') as summary_file:
        summary_data = yaml.safe_load(summary_file)
        summary_dict = _extract_summary(summary_data, fc_extract=True)

    # Only read force constants if it's not in summary file
    if not 'force_constants' in summary_dict:
        hdf5_exts = ['hdf5', 'hd5', 'h5']
        if fc_format is None:
            fc_format = os.path.splitext(fc_name)[1].strip('.')
            if fc_format not in hdf5_exts:
                fc_format = 'phonopy'
        fc_pathname = os.path.join(path, fc_name)
        if fc_format == 'phonopy':
            if not _check_fc_file(fc_pathname):
                raise Exception(f'Incorrect formatting for {fc_pathname}.')
            with open(fc_pathname, 'r') as fc_file:
                summary_dict['force_constants'] = _extract_force_constants(
                    fc_file, summary_dict['n_ions'],
                    summary_dict['n_cells_in_sc'])
        elif fc_format in 'hdf5':
            with h5py.File(fc_pathname, 'r') as fc_file:
                summary_dict[
                    'force_constants'] =  _extract_force_constants_hdf5(
                        fc_file, summary_dict['n_ions'],
                        summary_dict['n_cells_in_sc'])
        else:
            raise Exception((f'Force constants file format {fc_format} of '
                             f'{fc_name} is not recognised'))

    # Only read born/dielectric if they're not in summary file and the user has
    # specified a Born file
    dipole_keys = ['born', 'dielectric']
    if (born_name is not None and
            len(dipole_keys & summary_dict.keys()) != len(dipole_keys)):
        born_pathname = os.path.join(path, born_name)
        if not _check_born_file(born_pathname):
            raise Exception(f'Incorrect born formatting for {born_pathname}.')
        with open(born_pathname, 'r') as born_file:
            born_dict = _extract_born(born_file)
        summary_dict['born'] = born_dict['born']
        summary_dict['dielectric'] = born_dict['dielectric']

    # Units from summary_file
    ulength = summary_dict['ulength']
    umass = summary_dict['umass']
    ufc = summary_dict['ufc']

    data_dict = {}
    data_dict['n_ions'] = summary_dict['n_ions']
    data_dict['n_branches'] = 3*summary_dict['n_ions']
    data_dict['cell_vec'] = summary_dict['cell_vec']*ureg(
        ulength).to('bohr').magnitude
    data_dict['recip_vec'] = reciprocal_lattice(data_dict['cell_vec'])
    data_dict['ion_r'] = summary_dict['ion_r']
    data_dict['ion_type'] = summary_dict['ion_type']
    data_dict['ion_mass'] = summary_dict['ion_mass']*ureg(
        umass).to('e_mass').magnitude
    data_dict['force_constants'] = summary_dict['force_constants']*ureg(
        ufc).to('hartree/bohr**2').magnitude
    data_dict['sc_matrix'] = summary_dict['sc_matrix']
    data_dict['n_cells_in_sc'] = summary_dict['n_cells_in_sc']
    data_dict['cell_origins'] = summary_dict['cell_origins']

    try:
        data_dict['born'] = summary_dict['born']
        data_dict['dielectric'] = summary_dict['dielectric']
    except KeyError:
        pass

    metadata = type('', (), {})()
    metadata.model = 'phonopy'
    metadata.path = path
    metadata.fc_name = fc_name
    metadata.fc_format = fc_format
    metadata.born_name = born_name
    metadata.summary_name = summary_name

    data_dict['metadata'] = metadata

    return data_dict