import re
import os
import struct
import yaml
import h5py
import glob

import numpy as np
from euphonic import ureg
from euphonic.util import reciprocal_lattice, is_gamma


def _match_phonopy_file(path='.', name='*'):
    """ DOC
    Search target path for name with standard yaml and hdf5 extensions.

    Parameters
    ----------
    path : str
        Path to directory containing target hdf5/yaml file.
    name : str
        Filename of target hdf5/yaml file.

    Returns
    ----------
    file : str
        Filename of first matching file. Returns None if no file is
        found.
    """

    HDF5_EXT = '.hdf5' # ['.hdf5', '.he5', '.h5'] phonopy currently defaults to hdf5
    YAML_EXT = '.yaml' # ['.yaml', '.yml'] phonopy currently defaults to yaml

    file_glob = os.path.join(path, name + '.*')
    file_globs = glob.glob(file_glob)

    h5_matches = [name for name in file_globs if HDF5_EXT in name]
    yml_matches = [name for name in file_globs if YAML_EXT in name]

    if h5_matches: # take the first hdf5 match
        file = h5_matches[0]
    elif yml_matches: # take the first yaml match
        file = yml_matches[0]
    elif name and name != '*': # try just name given by the user
        file = os.path.join(path, name)
    else:
        return None

    if os.path.exists(file):
        return file
    else:
        return None

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

def _extract_phonon_data(data_object):
    """ DOC
    Search a given data object (dict or hdf5) for phonon data.

    Parameters
    ----------
    data_object : dict-like
        The Phonopy data object which contains phonon data.

    Returns
    -------
    data_dict : dict
        A dict with the following keys: 'n_ions', 'n_branches', 'n_qpts'
        'cell_vec', 'recip_vec', 'ion_r', 'ion_type', 'ion_mass', 'qpts',
        'weights', 'freqs', 'eigenvecs'. The return data has original
        Phonopy units.
    """

    n_qpts = data_object['nqpoint']
    n_ions = data_object['natom']
    cell_vec = data_object['lattice']
    recip_vec = data_object['reciprocal_lattice']
    qpts = [phon['q-position'] for phon in data_object['phonon']]

    weights = _convert_weights([phon['weight'] for phon in data_object['phonon']])

    phonon_data = [phon for phon in data_object['phonon']]
    bands_data_each_qpt = [bands_data['band']
                            for bands_data in phonon_data]

    # The frequency for each band at each q-point
    freqs = np.array([ [band_data['frequency']
                            for band_data in bands_data]
                              for bands_data in bands_data_each_qpt])

    # The eigenvector for each atom for each band at each q-point
    eigenvecs = np.array([ [band_data['eigenvector']
                            for band_data in bands_data]
                              for bands_data in bands_data_each_qpt]).view(np.complex128)

    n_branches = freqs.shape[1]

    ion_type = [ion['symbol'] for ion in data_object['points']]
    ion_r = [ion['coordinates'] for ion in data_object['points']]
    ion_mass = [ion['mass'] for ion in data_object['points']]

    data_dict = {}
    data_dict['n_qpts'] = n_qpts
    data_dict['n_spins'] = NotImplemented #n_spins, electronic
    data_dict['n_branches'] = n_branches
    data_dict['fermi'] = NotImplemented #(fermi*ureg.hartree).to('eV'), electronic
    data_dict['cell_vec'] = cell_vec #(cell_vec*ureg.bohr).to('angstrom')
    data_dict['recip_vec'] = recip_vec #((reciprocal_lattice(cell_vec)/ureg.bohr).to('1/angstrom'))
    data_dict['qpts'] = qpts
    data_dict['weights'] = weights #weights
    data_dict['freqs'] = freqs #(freqs*ureg.hartree).to('eV')
    data_dict['freq_down'] = NotImplemented #(freq_down*ureg.hartree).to('eV'), electronic
    data_dict['eigenvecs'] = eigenvecs
    data_dict['n_ions'] = n_ions
    data_dict['ion_r'] = ion_r
    data_dict['ion_type'] = ion_type
    data_dict['ion_mass'] = ion_mass

    return data_dict

def _read_phonon_data(path='.', phonon_name='mesh', summary_name='phonopy.yaml'):
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

    phonon_name = _match_phonopy_file(path=path, name=phonon_name)
    summary_name = _match_summary_file(path=path, name=summary_name)

    try:
        with h5py.File(phonon_name, 'r') as hdf5o:
            phonon_data = _extract_phonon_data(hdf5o)
    except:
        pass

    try:
        with open(phonon_name, 'r') as ymlo:
            yaml_data = yaml.safe_load(ymlo)
            phonon_data = _extract_phonon_data(yaml_data)
    except:
        pass

    # Extract units
    try:
        with open(summary_name, 'r') as ppo:
            yaml_data = yaml.safe_load(ppo)
            units = _extract_physical_units(yaml_data)
    except:
        pass

    # Handle units
    ulength = ureg(units['length'].lower()).to('bohr')
    #urlength = 1/ureg(units['length'].lower()).to('1/bohr')
    umass = ureg(units['atomic_mass'].lower()).to('e_mass')
    ufreq = units['frequency_unit_conversion_factor']*ureg('THz')\
                                .to('E_h', 'spectroscopy')

    data_dict = {}
    data_dict['n_ions'] = phonon_data['n_ions']
    data_dict['n_branches'] = phonon_data['n_branches']
    data_dict['n_qpts'] = phonon_data['n_qpts']
    data_dict['cell_vec'] = (phonon_data['cell_vec']*ulength).magnitude
    data_dict['recip_vec'] = (phonon_data['recip_vec']/ulength).magnitude
    data_dict['ion_r'] = phonon_data['ion_r']
    data_dict['ion_type'] = phonon_data['ion_type']
    data_dict['ion_mass'] = (phonon_data['ion_mass']*umass).magnitude
    data_dict['qpts'] = phonon_data['qpts']
    data_dict['weights'] = phonon_data['weights']
    data_dict['freqs'] = (phonon_data['freqs']*ufreq).magnitude
    data_dict['eigenvecs'] = phonon_data['eigenvecs']
    data_dict['split_i'] = NotImplemented
    data_dict['split_freqs'] = NotImplemented
    data_dict['split_eigenvecs'] = NotImplemented

    return data_dict


def _match_summary_file(path='.', name='phonopy.yaml'):
    """ DOC
    Search for the output summary file generated by Phonopy at the
    end of the workflow when using the cli.

    Parameters
    ----------
    path : str
        Path to directory containing target summary yaml file.
    name : str
        Filename of target summary file, default : phonopy.yaml

    Returns
    ----------
    file : str
        Filename of first matching summary file. Returns None if no file
        is found.
    """

    file = os.path.join(path, name)

    try:
        with open(file, 'r') as summ:
            summary_contents = yaml.safe_load(summ)
        atom_names = summary_contents['phonopy']['configuration']['atom_name']
    except KeyError:
        return None

    if os.path.exists(file):
        return file
    else:
        return None

def _match_disp_file(path='.', name='phonopy_disp.yaml'):
    """ DOC
    Search for the atom displacements summary generated by Phonopy at
    the start of the workflow when using the cli.

    Parameters
    ----------
    path : str
        Path to directory containing target displacements file.
    name : str
        Filename of target displacements file, default : phonopy_disp.yaml.

    Returns
    ----------
    file : str
        Filename of firstfile matching specification of displacements
        file. Returns None if no file is found.
    """

    file = os.path.join(path, name)

    try:
        with open(file, 'r') as dspo:
            disp_contents = yaml.safe_load(dspo)
        create_disp = disp_contents['phonopy']['configuration']['create_displacements']

        if create_disp != '.true.':
            return None
    except KeyError:
        return None

    if os.path.exists(file):
        return file
    else:
        return None

def _match_force_constants_file(path='.', name='FORCE_CONSTANTS'):
    """ DOC
    Search for the force constants file generated by Phonopy during
    the pre-process, then make a quick validity check. The --write-fc
    flag or equivalent must be set in Phonopy for the file to be made
    available.

    Parameters
    ----------
    path : str
        Path to directory containing target force constants file.
    name : str
        Filename of target force constants file, default : FORCE_CONSTANTS.

    Returns
    ----------
    file : str
        Filename of first file matching specification of force constants
        file. Returns None if no file is found.
    """

    fc_name = os.path.join(path, name)
    fc5_name = os.path.join(path, (name + '.hdf5').lower())

    if os.path.exists(fc_name):
        try:
            with open(fc_name, 'r') as fco:
                fco.readline() # first line can vary
                line = fco.readline() # always two numbers
                n = np.array(line.strip('\n').split(' ')).astype(np.int)
        except:
            print(f'Incorrect formatting for force constants file {fc_name}.')
            return None

        if n.size == 2:
            return fc_name
        else:
            return None
    elif os.path.exists(fc5_name):
        try:
            with h5py.File(fc5_name, 'r') as fc5o:
                if 'force_constants' in list(fc5o):
                    return fc5_name
                else:
                    return None
        except:
            pass

        return None

def _match_born_file(path='.', name='BORN'):
    """ DOC
    Search for the born file created by the user during the
    pre-process when using non-analytical term correction (nac).

    Parameters
    ----------
    path : str
        Path to directory containing target born file.
    name : str
        File basename of target born file, default : BORN.

    Returns
    ----------
    file : str
        Filename of first file matching specification of born
        file. Returns None if no file is found.
    """
    born_name = os.path.join(path, name)

    if os.path.exists(born_name):
        return born_name
    else:
        return None

    return born_name


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
    print(n_entries)

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
        # TODO read full fc directly truncated rather 
        # than full read then truncate

        #TODO investigate how reshape is working
        print(fc.shape)
        fc_phonopy = fc.reshape([64,64,3,3])
        print(fc_phonopy.shape)
        fc_reduced = fc_phonopy[p2s_map, :, :, :]
        print(fc_reduced.shape)
        fc_unfolded = fc_reduced.reshape(n_ions*n_ions*n_cells, 3, 3)
        print(fc.shape)

    return _reshape_fc(fc, n_ions, n_cells)

def _extract_force_constants_hdf5(fc_object):
    fc = fc_object['force_constants'][:]
    p2s_map = list(fc_object['p2s_map']) # 'primitive' to supercell indexing
    physical_units = list(fc_object['physical_unit'])[0].decode('utf-8')

    print(fc.shape)
    if fc.shape[0] == fc.shape[1]: # FULL FC, convert down to COMPACT
        fc = fc[p2s_map, :, :, :]
    print(fc.shape)

    n_ions = len(p2s_map)
    n_cells = p2s_map[1]

    fc_unfolded = fc.reshape(n_ions*n_ions*n_cells, 3, 3)
    return _reshape_fc(fc_unfolded, n_ions, n_cells), physical_units

def _extract_force_constants_summary(summary):
    """ DOC
    Get force constants from phonopy yaml summary file.

    Parameters
    ----------
    summary : dict
        Dict containing contents of phonopy.yaml

    Returns
    ----------
    units : dict
        Dict containing force constants in Euphonic format.
    """
    fc_entry = summary['force_constants']

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

    #TODO incorporate p2s_map

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

    # ignore blank lines 
    born_lines = [narr.split() for narr in
                [line for line in born_object.read().split(' ') if line]]

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
    born_dict['born_effective_charge'] = born_effective_charge

    return born_dict

def _extract_physical_units(summary):
    """ DOC
    Retrieve physical units from phonopy summary file object.

    Parameters
    ----------
    summary : file-like object
        Object representing contents of phonopy.yaml

    Returns
    ----------
    units : dict
        Dict containing physical unit, frequency factor, and
        non-analytical term correction factor.
    """

    if 'physical_unit' in summary.keys():
        units = summary['physical_unit']

    if 'phonopy' in summary.keys():
        if 'frequency_unit_conversion_factor' in summary['phonopy']:
            units['frequency_unit_conversion_factor'] =\
                summary['phonopy']['frequency_unit_conversion_factor']

        if 'nac_unit_conversion_factor' in summary['phonopy']:
            units['nac_unit_conversion_factor'] =\
                summary['phonopy']['nac_unit_conversion_factor']

    return units

def _extract_summary(summary):
    """ DOC
    Read phonopy.yaml for summary data produced during the Phonopy
    post-process.

    Parameters
    ----------
    summary : dict, hdf5
        The Phonopy data object which contains phonon data.

    Returns
    ----------
    summary_dict : dict
        A dict containing: sc_matrix, n_cells_in_sc, n_ions, cell_vec,
        ion_r, ion_mass, ion_type, n_ions, cell_origins.
    """
    n = lambda: None
    n.sc_matrix = np.array(summary['supercell_matrix'])
    #n.p_matrix = np.array(summary['primitive_matrix']) # Not found
    n.n_cells_in_sc = int(np.rint(np.absolute(np.linalg.det(n.sc_matrix))))

    n.p_n_ions = len(summary['primitive_cell']['points'])
    n.p_cell_vec = np.array(summary['primitive_cell']['lattice'])
    n.p_recip_vec = np.array(summary['primitive_cell']['reciprocal_lattice'])
    n.p_ion_r = np.zeros((n.p_n_ions, 3))
    n.p_ion_mass = np.zeros(n.p_n_ions)
    n.p_ion_type = []
    for i in range(n.p_n_ions):
        n.p_ion_mass[i] = summary['primitive_cell']['points'][i]['mass']
        n.p_ion_r[i] = summary['primitive_cell']['points'][i]['coordinates']
        n.p_ion_type.append(summary['primitive_cell']['points'][i]['symbol'])

    n.n_ions = len(summary['unit_cell']['points'])
    n.cell_vec = np.array(summary['unit_cell']['lattice'])
    n.ion_r = np.zeros((n.n_ions, 3))
    n.ion_mass = np.zeros(n.n_ions)
    n.ion_type = []
    for i in range(n.n_ions):
        n.ion_mass[i] = summary['unit_cell']['points'][i]['mass']
        n.ion_r[i] = summary['unit_cell']['points'][i]['coordinates']
        n.ion_type.append(summary['unit_cell']['points'][i]['symbol'])

    n.sc_n_ions = len(summary['supercell']['points'])
    n.sc_cell_vec = np.array(summary['supercell']['lattice'])
    n.sc_ion_r = np.zeros((n.sc_n_ions, 3))
    n.sc_ion_mass = np.zeros(n.sc_n_ions)
    n.sc_ion_type = []
    for i in range(n.sc_n_ions):
        n.sc_ion_mass[i] = summary['supercell']['points'][i]['mass']
        n.sc_ion_r[i] = summary['supercell']['points'][i]['coordinates']
        n.sc_ion_type.append(summary['supercell']['points'][i]['symbol'])

    # Coordinates of supercell ions in fractional coords of the unit cell
    sc_ion_r_ucell = np.einsum('ij,jk->ij', n.sc_ion_r, n.sc_matrix)
    cell_origins = sc_ion_r_ucell[:n.n_ions] - n.ion_r[0]

    n.cell_origins = cell_origins
    n.sc_ion_r_ucell = sc_ion_r_ucell

    n.units = summary['physical_unit']

    summary_dict = {}
    summary_dict['n_ions'] = n.n_ions
    summary_dict['cell_vec'] = n.cell_vec

    summary_dict['ion_r'] = n.ion_r
    summary_dict['ion_type'] = n.ion_type
    summary_dict['ion_mass'] = n.ion_mass

    summary_dict['cell_origins'] = n.cell_origins
    summary_dict['sc_matrix'] = n.sc_matrix
    summary_dict['n_cells_in_sc'] = n.n_cells_in_sc

    summary_dict['units'] = n.units

    if 'force_constants' in summary.keys():
        summary_dict['force_constants'] = _extract_force_constants_summary(summary)

    if 'born_effective_charge' in summary.keys():
        summary_dict['born_effective_charge'] = np.array(summary['born_effective_charge'])

    if 'dielectric_constant' in summary.keys():
        summary_dict['dielectric_constant'] = np.array(summary['dielectric_constant'])

    return summary_dict


def _extract_qpts_data(qpts_object):
    """ DOC
    Retrieve n_qpts, n_ions, recip_vec, qpts, etc from qpoint.yaml or
    qpoint.hdf5 which is created during a Phonopy qpoint session.

    Parameters
    ----------
    qpts_object : dict-like object
        The Phonopy data object which contains phonon data.

    Returns
    ----------
    data_dict : dict
        A dict containing: sc_matrix, n_cells_in_sc, n_ions, cell_vec,
        ion_r, ion_mass, ion_type, n_ions, cell_origins.
    """

    n_qpts = qpts_object['nqpoint']
    n_ions = qpts_object['natom']
    #cell_vec = qpts_object['lattice']
    recip_vec = np.array(qpts_object['reciprocal_lattice'])
    qpts = np.array([phon['q-position'] for phon in qpts_object['phonon']])

    #weights = [phon['weight'] for phon in qpts_object['phonon']]

    phonon_data = [phon for phon in qpts_object['phonon']]
    bands_data_each_qpt = [bands_data['band']
                            for bands_data in phonon_data]

    dyn_mat_data = [phon['dynamical_matrix'] for phon in phonon_data]

    # The frequency for each band at each q-point
    freqs = np.array([ [band_data['frequency']
                            for band_data in bands_data]
                              for bands_data in bands_data_each_qpt])

    n_branches = freqs.shape[1]

    data_dict = {}
    data_dict['n_ions'] = n_ions
    data_dict['n_qpts'] = n_qpts
    data_dict['n_branches'] = n_branches
    data_dict['recip_vec'] = recip_vec #((reciprocal_lattice(cell_vec)/ureg.bohr).to('1/angstrom'))
    data_dict['qpts'] = qpts
    data_dict['freqs'] = freqs #(freqs*ureg.hartree).to('eV')

    return data_dict



def _read_interpolation_data(path='.', qpts_name='qpoints',
                            disp_name='phonopy_disp.yaml', summary_name='phonopy.yaml',
                                born_name='BORN', fc_name='FORCE_CONSTANTS'):
    """
    Reads data from phonopy.yaml and qpoints files.

    Parameters
    ----------
    path : str
        Path to dir containing the file(s), if in another directory
    qpts_name : str
        Seedname of phonopy qpoints file
    disp_name : str
        Seedname of phonopy displacements file
    summary_name : str
        Seedname of phonopy user script summary file
    born_name : str
        Seedname of BORN file
    fc_name : str
        Seedname of FORCE_CONSTANTS file

    Returns
    -------
    data_dict : dict
        A dict with the following keys: 'n_ions', 'n_branches', 'cell_vec',
        'recip_vec', 'ion_r', 'ion_type', 'ion_mass', 'force_constants',
        'sc_matrix', 'n_cells_in_sc' and 'cell_origins'. Also contains 'born'
        and 'dielectric' if they are present in the .castep_bin or .check file
    """

    # If name not found, returns None
    qpts_name = _match_phonopy_file(path=path, name=qpts_name)
    disp_name = _match_disp_file(path=path, name=disp_name)
    summary_name = _match_summary(path=path, name=summary_name)

    fc_name = _match_force_constants_file(path=path, name=fc_name)
    born_name = _match_born_file(path=path, name=born_name)

    # Load data from qpoint mode output file
    try:
        with h5py.File(qpts_name, 'r') as hdf5o:
            qpoint_dict = _extract_qpts_data(hdf5o)
    except Exception as e:
        pass

    try:
        with open(qpts_name) as ymlo:
            yaml_data = yaml.safe_load(ymlo)
            qpoint_dict =  _extract_qpts_data(yaml_data)
    except Exception as e:
        pass

    try:
        with open(summary_name, 'r') as ppo:
            summary_data = yaml.safe_load(ppo)
            summary_dict = _extract_summary(summary_data)
    except Exception as e: # differentiate "find" from "load"
        print(f"Failed to load summary file {summary_name}.")
        print(e)

    try:
        with open(disp_name, 'r') as dspo:
            disp_dict = yaml.safe_load(dspo)
    except Exception as e:
        print(f"Failed to load displacements summary file {disp_name}.")
        print(e)

    # check summary_name for force_constants, dielectric_constant, 
    # born_effective_charge
    if 'force_constants' in summary_dict.keys():
        force_constants = summary_dict['force_constants']
    else:
        try: # csv compact or full, returns compact
            with open(fc_name, 'r') as fco:
                force_constants = _extract_force_constants(fco)
        except:
            pass

        try:
            with h5py.File(fc_name, 'r') as fco:
                force_constants =  _extract_force_constants_hdf5(fco)
        except:
            pass


    try:
        with open(born_name, 'r') as born:
            born_dict = _extract_born(born)
    except:
         pass

    if 'dielectric_constant' in summary_dict.keys():
        dielectric_constant = summary_dict['dielectric_constant']
    else:
        try:
            dielectric_constant = born_dict['dielectric_constant']
        except:
            pass

    if 'born_effective_charge' in summary_dict.keys():
        born_effective_charge = summary_dict['born_effective_charge']
    else:
        try:
            born_effective_charge = born_dict['born_effective_charge']
        except:
            pass

    # Extract units
    try:
        with open(summary_name, 'r') as ppo:
            yaml_data = yaml.safe_load(ppo)
            units = _extract_physical_units(yaml_data)
    except:
        pass

    if 'frequency_unit_conversion_factor' not in units:
        #TODO decide what to set for default value vs warn
        # Nowhere else to check for fucf
        units['frequency_unit_conversion_factor'] = None

    if 'nac_unit_conversion_factor' not in units:
        try: # Check BORN
            with open(born_name, 'r') as bbo:
                units['nac_unit_conversion_factor'] = int(bbo.readline())
        except:
            units['nac_unit_conversion_factor'] = None

    ulength = ureg(units['length'].lower()).to('bohr').magnitude
    umass = ureg(units['atomic_mass'].lower()).to('e_mass').magnitude
    ufreq = (ureg('THz')*units['frequency_unit_conversion_factor'])\
                .to('E_h', 'spectroscopy').magnitude
    unac = units['nac_unit_conversion_factor']

    # Check force constants string so that it matches format required for ureg
    ufc_str = units['force_constants'].replace('Angstrom', 'angstrom')
    ufc = ureg(ufc_str).to('hartree/bohr^2').magnitude

    #TODO tidy comments
    data_dict = {}
    data_dict['n_ions'] = summary_dict['n_ions'] #n_ions
    data_dict['n_branches'] = 3*summary_dict['n_ions'] #3*n_ions
    data_dict['cell_vec'] = summary_dict['cell_vec']*ulength #(cell_vec*ureg.bohr).to('angstrom')
    data_dict['recip_vec'] = reciprocal_lattice(summary_dict['cell_vec']*ulength) # qpoint_dict['recip_vec']/ulength #((reciprocal_lattice(cell_vec)/ureg.bohr).to('1/angstrom'))

    data_dict['ion_r'] = summary_dict['ion_r']*ulength #ion_r - np.floor(ion_r)  # Normalise ion coordinates
    data_dict['ion_type'] = summary_dict['ion_type'] #ion_type
    data_dict['ion_mass'] = summary_dict['ion_mass']*umass #(ion_mass*ureg.e_mass).to('amu')

    try: # Set entries relating to 'FORCE_CONSTANTS' block
        data_dict['force_constants'] = force_constants*ufc #(force_constants*ureg.hartree/(ureg.bohr**2))
        data_dict['sc_matrix'] = summary_dict['sc_matrix'] #sc_matrix
        data_dict['n_cells_in_sc'] = np.int(np.round(np.linalg.det(np.array(summary_dict['sc_matrix']))))
        data_dict['cell_origins'] = summary_dict['cell_origins']*ulength #cell_origins
    except NameError:
        raise Exception((
            'Force constants matrix could not be found in {:s} or {:s}. '
            'Ensure WRITE_FC: true or --write-fc has been set when running '
            'Phonopy').format(summary_name, fc_name))

    try: # Set entries relating to dipoles
        data_dict['born'] = born_effective_charge*unac #born*ureg.e
        data_dict['dielectric'] = dielectric_constant*unac #dielectric
    except UnboundLocalError:
        print('No bec or dielectric') #TODO Warn if either are not found.
        pass

    return data_dict

