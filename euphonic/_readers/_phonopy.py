import re
import os
import struct
import yaml
import h5py
import glob

import numpy as np
from euphonic import ureg
from euphonic.util import reciprocal_lattice, is_gamma



def _convert_weights(weights):
    """ DOC
    Convert atom weights to CASTEP normalised convention.

    Parameters
    ----------
    weights : list
        List of weights

    Returns
    ----------
    norm_weights : list
        List of normalised weights
    """
    weights = np.array(weights)
    total_weight = weights.sum()
    norm_weights = list(weights / total_weight)
    return norm_weights

def _extract_phonon_data(phonon_data):
    """ DOC
    Search a given data object (dict or hdf5) for phonon data.

    Parameters
    ----------
    phonon_data : dict
        The Phonopy data object which contains phonon data.

    Returns
    -------
    data_dict : dict
        A dict with the following keys: 'n_ions', 'n_branches', 'n_qpts'
        'cell_vec', 'recip_vec', 'ion_r', 'ion_type', 'ion_mass', 'qpts',
        'weights', 'freqs', 'eigenvecs'. The return data has original
        Phonopy units.
    """

    #n_qpts = phonon_data['nqpoint']
    #n_ions = phonon_data['natom']
    #cell_vec = np.array(phonon_data['lattice'])
    #recip_vec = np.array(phonon_data['reciprocal_lattice'])
    qpts = np.array([phon['q-position'] for phon in phonon_data['phonon']])

    #TODO not found in bands. cant find, give equal sum = 1
    weights = _convert_weights([phon['weight'] for phon in phonon_data['phonon']])

    phonons = [phon for phon in phonon_data['phonon']]
    bands_data_each_qpt = [bands_data['band']
                            for bands_data in phonons]

    # The frequency for each band at each q-point
    freqs = np.array([[band_data['frequency']
                            for band_data in bands_data]
                              for bands_data in bands_data_each_qpt])

    # The eigenvector for each atom for each band at each q-point
    #TODO one extra axis size 1, remove np squeeze
    eigenvecs = np.array([ [band_data['eigenvector']
                            for band_data in bands_data]
                              for bands_data in bands_data_each_qpt]).view(np.complex128)

    #n_branches = freqs.shape[1]

    #ion_type = [ion['symbol'] for ion in phonon_data['points']]
    #ion_r = np.array([ion['coordinates'] for ion in phonon_data['points']])
    #ion_mass = np.array([ion['mass'] for ion in phonon_data['points']])

    data_dict = {}
    #data_dict['n_qpts'] = n_qpts
    #data_dict['n_branches'] = n_branches
    #data_dict['cell_vec'] = cell_vec
    #data_dict['recip_vec'] = recip_vec
    #data_dict['n_ions'] = n_ions
    #data_dict['ion_r'] = ion_r
    #data_dict['ion_type'] = ion_type
    #data_dict['ion_mass'] = ion_mass

    data_dict['qpts'] = qpts
    data_dict['weights'] = weights
    data_dict['freqs'] = freqs
    data_dict['eigenvecs'] = eigenvecs

    #TODO implement empty split arrays: split_freqs = np.empty((0, n_branches))

    return data_dict

def _read_phonon_data(path='.', phonon_name='', phonon_format=None,
                        summary_name='phonopy.yaml'):
    """ DOC
    Reads data from a mesh.yaml/hdf5 file and returns it in a dictionary

    Parameters
    ----------
    path : str
        Path to dir containing the file(s), default : ./
    phonon_name : str
        Seedname of phonopy mesh file to read, default : mesh.yaml/mesh.hdf5
    summary_name : str
        Seedname of phonopy summary file to read, default : phonopy.yaml

    Returns
    -------
    data_dict : dict
        A dict with the following keys: 'n_ions', 'n_branches', 'n_qpts'
        'cell_vec', 'recip_vec', 'ion_r', 'ion_type', 'ion_mass', 'qpts',
        'weights', 'freqs', 'eigenvecs', 'split_i', 'split_freqs',
        'split_eigenvecs'. The return data has Euphonic default units.
    """

    phonon_pathname = os.path.join(path, phonon_name)
    summary_pathname = os.path.join(path, summary_name)

    phonon_ext = os.path.splitext(phonon_name)[1].strip('.')

    hdf5_exts = ['hdf5', 'hd5', 'h5']
    yaml_exts = ['yaml', 'yml', 'yl']
    if (phonon_format is None and phonon_ext in hdf5_exts) or\
            (phonon_format in hdf5_exts):
        phonon_format = 'hdf5'
    elif (phonon_format is None and phonon_ext in yaml_exts) or\
            (phonon_format in yaml_exts):
        phonon_format = 'yaml'
    elif phonon_ext is None:
        if phonon_format is None:
            raise Exception((
                'Phonon file format is not set. '
                f'Please specify file format from {yaml_exts} or {hdf5_exts}'))
        else:
            raise Exception((
                'Phonon file format not allowed. Please specify \n'
                f'file format from {yaml_exts} or {hdf5_exts}'))
    else:
        raise Exception('Failure to establish phonon file format.')



    if phonon_format == 'hdf5':
        try:
            with h5py.File(phonon_pathname, 'r') as hdf5_file:
                phonon_dict = _extract_phonon_data_hdf5(hdf5_file)
        except OSError as ose:
            if 'file or directory' in ose.__str__():
                raise Exception((
                        f'Could not find phonon file {phonon_pathname}'))
            elif 'to open file' in ose.__str__():
                raise Exception((
                        f'Could not open phonon file {phonon_pathname}'))
            else:
                raise ose
        except KeyError:
            raise Exception((
                f'Phonon file {phonon_pathname} missing data or not in correct format.'))

    elif phonon_format == 'yaml':
        try:
            with open(phonon_pathname, 'r') as phonon_file:
                phonon_data = yaml.safe_load(phonon_file)
                phonon_dict = _extract_phonon_data(phonon_data)
        except FileNotFoundError:
            raise Exception(f'Phonon file {phonon_pathname} not found.')
        except UnicodeDecodeError:
            raise Exception((
                f'Phonon file {phonon_pathname} could not be loaded.'
                'Data type may be hdf5.'))
        except KeyError:
            raise Exception((
                f'Phonon file {phonon_pathname} missing data or not in correct format.'))
    else:
        raise Exception(f'Phonon file format not currently supported: {phonon_format}.')


    try: # load summary file
        with open(summary_pathname, 'r') as summary_file:
            summary_data = yaml.safe_load(summary_file)
            if type(summary_data) is not dict:
                raise TypeError
            summary_dict = _extract_summary(summary_data)

    except FileNotFoundError:
        raise Exception(f'Summary file {summary_pathname} not found.')

    except TypeError:
        raise Exception(('Summary file data in incorrect format; type must be dict'
            'and not {type(summary_data)}.'))
    except:
        raise

    # Handle units
    try:
        units = summary_dict['physical_unit']
        ulength = ureg(units['length'].lower()).to('bohr').magnitude
        umass = ureg(units['atomic_mass'].lower()).to('e_mass').magnitude
        ufreq = units['frequency_unit_conversion_factor']*ureg('THz')\
                                .to('E_h', 'spectroscopy').magnitude #TODO check unit conversion 1*
    except KeyError as ke:
        missing_unit = 'UNKNOWN UNIT'

        if 'physical_unit' in ke:
            missing_unit = 'physical units'
        if 'length' in ke:
            missing_unit = 'length'
        if 'atomic_mass' in ke:
            missing_unit = 'atomic mass'
        if 'frequency_unit_conversion_factor' in ke:
            missing_unit = 'frequency factor'

        raise Exception((
            f'{missing_unit} missing from summary file {summary_pathname}.\n'
            'Check output files and run configs for errors or missing data,\n'
            'then rerun Phonopy with correct configs'))

    # Output object
    data_dict = {}
    data_dict['n_ions'] = summary_dict['n_ions']
    data_dict['n_branches'] = summary_dict['n_ions'] * 3 #TODO
    data_dict['n_qpts'] = phonon_dict['qpts'].shape[0]
    data_dict['cell_vec'] = summary_dict['cell_vec']*ulength
    data_dict['recip_vec'] = reciprocal_lattice(summary_dict['cell_vec'])/ulength
    data_dict['ion_r'] = summary_dict['ion_r']
    data_dict['ion_type'] = summary_dict['ion_type']
    data_dict['ion_mass'] = summary_dict['ion_mass']*umass
    data_dict['qpts'] = phonon_dict['qpts']
    data_dict['weights'] = phonon_dict['weights']
    data_dict['freqs'] = phonon_dict['freqs']*ufreq
    data_dict['eigenvecs'] = phonon_dict['eigenvecs']

    # Metadata
    data_dict['model'] = 'phonopy'
    data_dict['phonon_name'] = phonon_name
    data_dict['phonon_format'] = phonon_format
    data_dict['summary_name'] = summary_name

    return data_dict


def _check_fc_file_format(filename):
    """ DOC
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
    """ DOC
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

            if 'dielectric_tensor' in line:
                dt_present = True

            if bec_present and dt_present:
                return True

            line = bfo.readline()

    return False



def _extract_force_constants(fc_object):
    """ DOC
    Parse, reshape, and convert FC from FORCE_CONSTANTS file.

    Parameters
    ----------
    fc_object : dict-like object
        Object representing contents of FORCE_CONSTANTS.

    Returns
    ----------
    fc_resh : float ndarray
        Force constants matrix reshaped into Euphonic sc-first format.
    """
    fc_lines_all = [narr.split() for narr in
                [line for line in fc_object.read().split('\n') if line]]

    # convert string numerals as float/int
    for l_i, line in enumerate(fc_lines_all):
        try:
            fc_lines_all[l_i] = [np.int(i) for i in line]
            continue
        except:
            pass

        try:
            fc_lines_all[l_i] = [np.float(i) for i in line]
            continue
        except:
            pass

    fc_dims = fc_lines_all[0]
    fc_lines = fc_lines_all[1:]

    n_lines = len(fc_lines)
    n_entries = n_lines / 4

    if n_entries != int(n_entries):
        raise Exception('Incorrect file format.')
    else:
        n_entries = int(n_entries)

    if len(fc_dims) == 1: # single shape specifier implies full format
        is_full_fc = True
        fc_dims = [fc_dims[0], fc_dims[0]]
    elif fc_dims[0] == fc_dims[1]:
        is_full_fc = True
    else:
        is_full_fc = False

    n_ions = 8 # fc_dims[0]
    sc_n_ions = fc_dims[1]
    n_cells = sc_n_ions // n_ions # not correct for 64 64

    if is_full_fc:
        p2s_map = [pi for pi in range(0, n_ions*n_cells, n_ions)]

    inds = []
    fc_vecs = np.zeros([n_entries, 3, 3], dtype=np.float)

    for sc_ion in range(n_entries):
        entry_ind = 4*sc_ion

        # line 1
        #    (i_entry, i_patom, i_satom)
        inds.append([sc_ion] + fc_lines[entry_ind])

        # lines 2-4
        for j in range(1,4):
            row_in_entry = fc_lines[entry_ind + j]
            fc_vecs[sc_ion, j-1, :] = row_in_entry

    fc = np.array(fc_vecs)

    if is_full_fc: # FULL FC, convert down to COMPACT
        #TODO truncate full fc on read

        #TODO investigate how reshape is working
        fc_phonopy = fc.reshape([64,64,3,3])
        fc_reduced = fc_phonopy[p2s_map, :, :, :]
        fc_unfolded = fc_reduced.reshape(n_ions*n_ions*n_cells, 3, 3)

    return _reshape_fc(fc, n_ions, n_cells)

def _extract_force_constants_hdf5(fc_object):
    fc = fc_object['force_constants'][:]
    p2s_map = list(fc_object['p2s_map']) # 'primitive' to supercell indexing
    physical_units = list(fc_object['physical_unit'])[0].decode('utf-8')

    if fc.shape[0] == fc.shape[1]: # FULL FC, convert down to COMPACT
        fc = fc[p2s_map, :, :, :]

    n_ions = len(p2s_map)
    n_cells = p2s_map[1]

    fc_unfolded = fc.reshape(n_ions*n_ions*n_cells, 3, 3)
    return _reshape_fc(fc_unfolded, n_ions, n_cells), physical_units

def _extract_force_constants_summary(summary_object):
    """ DOC
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

    n_ions = fc_dims[0]
    n_cells = fc_dims[1] // n_ions # TODO if full, this is incorrect

    if fc_format == 'compact':
        fc = np.array(fc_entry['elements'])

    elif fc_format == 'full': # FULL FC, convert down to COMPACT
        p2s_map = [pi for pi in range(0, n_ions*n_cells, n_ions)]
        fc = np.array(fc_entry['elements'])[p2s_map, :]

    return _reshape_fc(fc, n_ions, n_cells)

def _reshape_fc(fc, n_ions, n_cells):
    return np.reshape( np.transpose( np.reshape(fc, (n_ions, n_ions, n_cells, 3, 3)), axes=[2,0,3,1,4]), (n_cells, 3*n_ions, 3*n_ions))

def _old_reshape_fc(fc, n_ions, n_cells):
    """ DOC
    Reshape FORCE_CONSTANTS to conform to Euphonic format.
    Into [N_sc, 3*N_pc, 3*N_pc]

    Parameters
    ----------
    fc : float ndarray
        Force constants matrix with Phonopy indexing
    n_ions : int
        Number of ions in unit cell
    n_cells : int
        Number of cells in supercell

    Returns
    -------
    fc_euph : float ndarray
    """

    sc_n_ions = n_cells * n_ions

    # Construct convenient array of indices
    K = [k for k in range(1, 1 + sc_n_ions)]
    J = [j for j in range(1, 1 + n_ions)]

    inds = np.zeros([n_ions*sc_n_ions, 3]).astype(np.int)
    for j in J:
        for k in K:
            i = (j-1)*sc_n_ions + (k-1)
            inds[i, :] = [i, j, k]

    # Indices: sc index, ion index in sc, Fi, Fj
    fc_resh = np.zeros([n_cells, n_ions, n_ions, 3, 3])

    # Reshape to arrange FC sub matrices by cell index,
    # atom_i index, atom_j index

    for i, j, k in inds:
        # ion index within primitive cell
        pci_ind = (j-1) // n_cells

        # target ion index within primitive cell
        pcj_ind = (k-1) // n_cells

        # target cell index within sc
        scj_ind = (k-1) % n_cells

        fc_resh[scj_ind, pci_ind, pcj_ind, :, :] = fc[i, :, :]

    # tile nxnx3x3 matrices into 3nx3n 
    fc_euph = fc_resh.transpose([0,1,3,2,4]).reshape([n_cells, 3*n_ions, 3*n_ions])

    #TODO replace above
    #fc_euph = np.reshape(np.transpose(np.reshape(fc, (n_cells, n_ions, n_ions, 3, 3)),
                    #axes=[1,0,3,2,4]), (n_cells, 3*n_ions, 3*n_ions))

    return fc_euph

def _extract_born(born_object):
    """ DOC
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
    born_dict['dielectric_constant'] = dielectric
    born_dict['born_effective_charge'] = born

    return born_dict

def _extract_physical_units(summary_object):
    """ DOC
    Retrieve physical units from phonopy summary file object.

    Parameters
    ----------
    summary_object : dict-like object
        Object representing contents of phonopy.yaml

    Returns
    ----------
    units : dict
        Dict containing physical unit, frequency factor, and
        non-analytical term correction factor.
    """

    if 'physical_unit' in summary_object.keys():
        units = summary_object['physical_unit']
    else:
        raise Exception('Physical units not present.')

    if 'phonopy' in summary_object.keys():
        if 'frequency_unit_conversion_factor' in summary_object['phonopy']:
            units['frequency_unit_conversion_factor'] =\
                summary_object['phonopy']['frequency_unit_conversion_factor']

        if 'nac_unit_conversion_factor' in summary_object['phonopy']:
            units['nac_unit_conversion_factor'] =\
                summary_object['phonopy']['nac_unit_conversion_factor']

    return units

def _extract_summary(summary_object):
    """ DOC
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
    ion_type = []
    for i in range(n_ions):
        ion_mass[i] = summary_object['unit_cell']['points'][i]['mass']
        ion_r[i] = summary_object['unit_cell']['points'][i]['coordinates']
        ion_type.append(summary_object['unit_cell']['points'][i]['symbol'])

    sc_matrix = np.array(summary_object['supercell_matrix'])
    n_cells_in_sc = int(np.rint(np.absolute(np.linalg.det(sc_matrix))))
    sc_n_ions = len(summary_object['supercell']['points'])
    sc_ion_r = np.zeros((sc_n_ions, 3))
    for i in range(sc_n_ions):
        sc_ion_r[i] = summary_object['supercell']['points'][i]['coordinates']

    # Coordinates of supercell ions in fractional coords of the unit cell
    sc_ion_r_ucell = np.einsum('ij,jk->ij', sc_ion_r, sc_matrix)
    cell_origins = sc_ion_r_ucell[:n_ions] - ion_r[0] #TODO numpy astype integer array

    summary_dict = {}
    summary_dict['n_ions'] = n_ions
    summary_dict['cell_vec'] = cell_vec

    summary_dict['ion_r'] = ion_r
    summary_dict['ion_type'] = ion_type
    summary_dict['ion_mass'] = ion_mass

    summary_dict['cell_origins'] = cell_origins
    summary_dict['sc_matrix'] = sc_matrix
    summary_dict['n_cells_in_sc'] = n_cells_in_sc

    if 'force_constants' in summary_object.keys():
        summary_dict['force_constants'] = _extract_force_constants_summary(summary_object)

    if 'born_effective_charge' in summary_object.keys():
        summary_dict['born_effective_charge'] = np.array(summary_object['born_effective_charge'])

    if 'dielectric_constant' in summary_object.keys():
        summary_dict['dielectric_constant'] = np.array(summary_object['dielectric_constant'])

    if 'physical_unit' in summary_object.keys():
        summary_dict['physical_unit'] = _extract_physical_units(summary_object)

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
            line.readline()
    return False


def _read_interpolation_data(path='.', summary_name='phonopy.yaml',
                                born_name='BORN', born_format=None,
                                    fc_name='FORCE_CONSTANTS', fc_format=None):
    """
    Reads data from Phonopy output files.

    Parameters
    ----------
    path : str
        Path to dir containing the file(s), if in another directory
    summary_name : str
        Filename of phonopy user script summary file, default phonopy.yaml
    born_name : str
        Filename of born file, default BORN
    born_format : {str, None}
        Format of file containing born data [None|'phonopy'|'yaml'], default None (BORN)
    fc_name : str
        Seedname of force constants file, default FORCE_CONSTANTS/force_constants.hdf5
    fc_format : {str, None}
        Format of file containing force constants data. FORCE_CONSTANTS is type 'phonopy'.
        [None|'phonopy'|'yaml'|'hdf5'], default : None

    Returns
    -------
    data_dict : dict
        A dict with the following keys: 'n_ions', 'n_branches', 'cell_vec',
        'recip_vec', 'ion_r', 'ion_type', 'ion_mass', 'force_constants',
        'sc_matrix', 'n_cells_in_sc' and 'cell_origins'. Also contains 'born'
        and 'dielectric' if they are present in the .castep_bin or .check file
    """

    fc_pathname = os.path.join(path, fc_name)
    born_pathname = os.path.join(path, born_name)
    summary_pathname = os.path.join(path, summary_name)

    # Check FORCE CONSTANTS (phonopy, hdf5, yaml summary)
    fc_ext = os.path.splitext(fc_name)[1].strip('.')
    born_ext = os.path.splitext(born_name)[1].strip('.')

    hdf5_exts = ['hdf5', 'hd5', 'h5']
    yaml_exts = ['yaml', 'yml', 'yl']

    # hdf5
    if (fc_format is None and fc_ext in hdf5_exts)\
             or (fc_format in hdf5_exts):
        fc_format = 'hdf5'

    # phonopy FORCE_CONSTANTS
    elif (fc_format == 'phonopy' or fc_format is None and not fc_ext)\
            or (fc_pathname == summary_pathname):
        fc_format = 'phonopy'

    # summary file
    elif (fc_format is None and fc_ext in yaml_exts)\
            or (fc_format in yaml_exts):
        fc_format = 'yaml'

    # otherwise
    else:
        raise Exception('Failure to establish force constants file format.')


    # text file formatting
    if fc_format == 'phonopy':
        if not _check_fc_file_format(fc_pathname):
            raise Exception(f'Incorrect formatting for {fc_pathname}.')

    elif fc_format == 'yaml':
        if not _check_fc_summary(summary_pathname):
            raise Exception((
                f'Force constants not found in summary file {summary_pathname}'))


    # Check BORN (phonopy, or yaml summary)
    # summary file
    if (born_name == summary_name) or (born_format in yaml_exts):
        born_format = 'yaml'

    # phonopy BORN file
    elif not born_ext and (born_format in [None,'','phonopy']):
        born_format = 'phonopy'

    # otherwise
    else:
        raise Exception('Failure to establish born file format.')


    if born_format == 'phonopy':
        if not _check_born_file(born_pathname):
            raise Exception(f'Incorrect born formatting for {born_pathname}.')
    elif born_format == 'yaml':
        if not _check_born_summary(summary_pathname):
            raise Exception(f'Born data not present in {summary_pathname}.')


    ## Load files:
    # SUMMARY
    try: # yaml
        with open(summary_pathname, 'r') as summary_file:
            summary_data = yaml.safe_load(summary_file)
            summary_dict = _extract_summary(summary_data)
    except Exception as e:
        #TODO
        raise Exception(f'Failed to load summary file {summary_name}.')


    # BORN
    if born_format == 'phonopy':
        try: # phonopy
            with open(born_pathname, 'r') as born_file:
                born_dict = _extract_born(born_file)
            dielectric_constant = born_dict['dielectric_constant']
            born_effective_charge = born_dict['born_effective_charge']
        except:
            raise Exception((
                f'Could not extract born data from {born_pathname}.'))

    elif born_format == 'yaml':
        try:
            dielectric_constant = summary_dict['dielectric_constant']
            born_effective_charge = summary_dict['born_effective_charge']
        except:
            raise Exception((
                f'Could not extract born data from {born_pathname}.'))

    # FORCE CONSTANTS
    if fc_format == 'hdf5':
        try:
            with h5py.File(fc_pathname, 'r') as fc_file:
                force_constants =  _extract_force_constants_hdf5(fc_file)
        except:
            raise Exception((
                f'Could not extract force constants from {fc_pathname}.'))

    elif fc_format == 'yaml':
        try:
            force_constants = summary_dict['force_constants']
        except:
            raise Exception((
                f'Force constants not present in loaded summary data.'))

    elif fc_format == 'phonopy':
        try: # compact or full, returns compact
            with open(fc_pathname, 'r') as fc_file:
                force_constants = _extract_force_constants(fc_file)
        except:
            raise Exception((
                f'Could not extract force constants from {fc_pathname}.'))


    # Construct unit conversion factors
    try:
        units = summary_dict['physical_unit']
        ulength = ureg(units['length'].lower()).to('bohr').magnitude
        umass = ureg(units['atomic_mass'].lower()).to('e_mass').magnitude
        unac = units['nac_unit_conversion_factor']

        # Check force constants string so that it matches format required for ureg
        ufc_str = units['force_constants'].replace('Angstrom', 'angstrom')
        ufc = ureg(ufc_str).to('hartree/bohr^2').magnitude
    except:
        missing_unit = 'UNKNOWN'

        if 'physical_unit' in ke:
            missing_unit = 'physical units'
        if 'length' in ke:
            missing_unit = 'length'
        if 'atomic_mass' in ke:
            missing_unit = 'atomic mass'
        if 'nac_unit_conversion_factor' in ke:
            missing_unit = 'non-analytic term factor'
        if 'force_consants' in ke:
            missing_unit = 'force constants factor'

        if missing_unit == 'UNKOWN':
            raise Exception((
                f'Unkown error while setting unit conversion factors.\n'
                'Check output files and run configs for errors or missing data,\n'))
        else:
            raise Exception((
                f'{missing_unit} missing from summary file {summary_pathname}.\n'
                'Check output files and run configs for errors or missing data'))

    data_dict = {}
    data_dict['n_ions'] = summary_dict['n_ions']
    data_dict['n_branches'] = 3*summary_dict['n_ions'] #TODO
    data_dict['cell_vec'] = summary_dict['cell_vec']*ulength
    data_dict['recip_vec'] = reciprocal_lattice(summary_dict['cell_vec']*ulength)

    data_dict['ion_r'] = summary_dict['ion_r']
    data_dict['ion_type'] = summary_dict['ion_type']
    data_dict['ion_mass'] = summary_dict['ion_mass']*umass

    try: # FORCE CONSTANTS
        data_dict['force_constants'] = force_constants*ufc
        data_dict['sc_matrix'] = summary_dict['sc_matrix']
        data_dict['n_cells_in_sc'] = np.int(np.round(np.linalg.det(np.array(summary_dict['sc_matrix']))))
        data_dict['cell_origins'] = summary_dict['cell_origins']
    except NameError:
        raise Exception((
            'Force constants matrix could not be found in {:s}. '
            'Ensure WRITE_FC = .TRUE. or --write-fc has been set. '
            ).format(fc_pathname))

    try: # NAC
        data_dict['born'] = born_effective_charge # *unac
    except UnboundLocalError as eb:
        print('NOTE: Born effective charges not present.')

    try:
        data_dict['dielectric'] = dielectric_constant # *unac
    except UnboundLocalError as eb:
        print('NOTE: Dielectric tensor not present.')

    return data_dict

