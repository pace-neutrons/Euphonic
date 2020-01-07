import re
import os
import struct
import yaml
import h5py
import glob

import numpy as np
from euphonic import ureg
from euphonic.util import reciprocal_lattice, is_gamma

def _convert_units(data_dict):
    #TODO
    """ DOC

    """

    pass

def _match_file(path='.', name='*'):
    """DOC
    Search target path for seed with standard yaml and hdf5 extensions.
    """

    #TODO remove filechecking redundancy

    HDF5_EXT = '.hdf5' # ['.hdf5', '.he5', '.h5'] phonopy currently defaults to hdf5
    YAML_EXT = '.yaml' # ['.yaml', '.yml'] phonopy currently defaults to yaml

    file_glob = os.path.join(path, seed + '.*')
    file_globs = glob.glob(file_glob)

    h5_matches = [name for name in file_globs if HDF5_EXT in name]
    yml_matches = [name for name in file_globs if YAML_EXT in name]

    if h5_matches: # take the first hdf5 match
        file = h5_matches[0]
    elif yml_matches: # take the first yaml match
        file = yml_matches[0]
    elif seed and seed != '*': # try just seed given by the user
        file = os.path.join(path, seed)
    else:
        return None

    if os.path.exists(file):
        return file
    else:
        return None

def _convert_weights(weights):

    weights = np.array(weights)
    total_weight = weights.sum()

    norm_weights = weights / total_weight

    return list(norm_weights)

def _extract_phonon_data(data_object):
    """ DOC
    Search a given data object (dict or hdf5) for phonon data.

    Parameters
    ----------
    data_object : dict, hdf5 (, other key:value addressable object)
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

def _read_phonon_data(path='.', phonon_file='mesh', summary_file='phonopy.yaml'):
    """
    Reads data from a mesh.yaml/hdf5 file and returns it in a dictionary

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
        'split_eigenvecs'. The return data has Euphonic default units.
    """

    phonon_file = _match_file(path=path, name=phonon_file)
    summary_file = _match_summary(path=path, name=summary_file)

    try:
        with h5py.File(phonon_file, 'r') as hdf5o:
            phonon_data = _extract_phonon_data(hdf5o)
    except:
        pass

    try:
        with open(phonon_file, 'r') as ymlo:
            yaml_data = yaml.safe_load(ymlo)
            phonon_data = _extract_phonon_data(yaml_data)
    except:
        pass

    # Extract units
    try:
        with open(summary_file, 'r') as ppo:
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
    data_dict['eigenvecs'] = phonon_data['eigenvecs'] #TODO Scaling
    data_dict['split_i'] = NotImplemented
    data_dict['split_freqs'] = NotImplemented
    data_dict['split_eigenvecs'] = NotImplemented

    return data_dict



def _match_summary(path='.', name='phonopy.yaml'):
    """ DOC
    Search for the output summary file generated by Phonopy at the
    end of the workflow when using the cli.
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

def _match_disp(path='.', name='phonopy_disp.yaml'):
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
        Filename of existing file matching specification of displacements
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

def _match_fc(path='.', name='FORCE_CONSTANTS'):
    """ DOC
    Search for the force constants file generated by Phonopy during
    the pre-process. The --write-fc flag or equivalent must be set
    in Phonopy for the file to be made available.

    Parameters
    ----------
    path : str
        Path to directory containing target force constants file.
    name : str
        Filename of target force constants file,
        default : FORCE_CONSTANTS.

    Returns
    ----------
    file : str
        Filename of existing file matching specification of force
        constants file. Returns None if no file is found.
    """

    #TODO match glob for FORCE_CONSTANTS*
    fc_file = os.path.join(path, name)
    fc5_file = os.path.join(path, (name + '.hdf5').lower())

    if os.path.exists(fc_file):
        try:
            with open(fc_file, 'r') as fco:
                fco.readline() # first line can vary
                line = fco.readline() # always two numbers
                n = np.array(line.strip('\n').split(' ')).astype(np.int)
        except:
            print(f'Incorrect formatting for force constants file {fc_file}.')
            return None

        if n.size == 2:
            return fc_file
        else:
            return None
    elif os.path.exists(fc5_file):
        try:
            with h5py.File(fc5_file, 'r') as fc5o:
                if 'force_constants' in list(fc5o):
                    return fc5_file
                else:
                    return None
        except:
            pass

        return None

def _match_born(path='.', name='BORN'):
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
        Filename of existing file matching specification of born
        file. Returns None if no file is found.
    """

    born_file = os.path.join(path, name)
    #TODO match glob for BORN*

    if os.path.exists(born_file):
        return born_file
    else:
        return None

    return born_file


def _extract_force_constants(fc_object):
    # TODO
    """ DOC
    Parse, reshape, and convert FC from FORCE_CONSTANTS
    """

    #TODO implement numpy focused index assignment, no .append

    k_lines = [narr.split() for narr in
                [line for line in fc_object.read().split('\n') if line]]

    # convert string numerals as float/int
    for l_i, line in enumerate(k_lines):
        #TODO better way to read values with correct type:
        # switching between coordinates and indices every 4 lines.
        try:
            k_lines[l_i] = [np.int(i) for i in line]
            continue
        except:
            pass

        try:
            k_lines[l_i] = [np.float(i) for i in line]
            continue
        except:
            pass

    n_lines = len(k_lines)
    n_entries = (n_lines - 1)/4

    if n_entries != int(n_entries):
        print('File is in incorrect format.')
    else:
        n_entries = int(n_entries)

    dims = k_lines[0]
    if len(dims) == 1: # one number implies both the same
        dims = 2*dims

    entry_lines = k_lines[1:]

    inds = []
    fc_vecs = np.zeros([n_entries, 3, 3], dtype=np.float)
    for atom_i in np.arange(0, n_entries):
        # line 1
        #    (i_entry, i_patom, i_satom)
        inds.append([atom_i] + entry_lines[atom_i*4])

        # lines 2-4
        for j in range(1,4):
            row_in_entry = entry_lines[atom_i*4 + j]
            fc_vecs[atom_i, j-1, :] = row_in_entry

    fc = np.array(fc_vecs)
    inds = np.array(inds)
    dims = np.array(dims)

    fc_resh = _reshape_fc(fc, inds, dims)

    return fc_resh

def _extract_born(born_object):
    # TODO
    """ DOC
    Parse and convert dielectric tensor and born effective
    charge from BORN file
    """

    # ignore blank lines 
    k_lines = [narr.split() for narr in
                [line for line in born_object.read().split(' ') if line]]

    for l_i, line in enumerate(k_lines):
        try:
            k_lines[l_i] = [np.float(i) for i in line]
            continue
        except:
            pass

    n_lines = len(k_lines)
    n_entries = (n_lines - 1)/4

    # factor first line
    factor = k_lines[0][0]

    # dielectric second line
    # xx, xy, xz, yx, yy, yz, zx, zy, zz.
    dielectric = np.array(k_lines[1]).reshape([3,3])

    # born ec third line onwards
    # xx, xy, xz, yx, yy, yz, zx, zy, zz.
    born_lines = k_lines[2:]
    born = np.array([np.array(bl).reshape([3,3]) for bl in born_lines])

    born_dict = {}
    born_dict['factor'] = factor
    born_dict['dielectric_constant'] = dielectric
    born_dict['born_effective_charge'] = born_effective_charge

    return born_dict

def _get_fc_summary(summary):
        fc_entry = summary['force_constants']

        fc_dims = fc_entry['shape']
        fc_format = fc_entry['format']

        K = [k for k in range(1,1+fc_dims[1])]
        J = [j for j in range(1,1+fc_dims[0])]

        inds = np.zeros([fc_dims[0]*fc_dims[1], 3]).astype(np.int)
        for j in J:
            for k in K:
                i = (j-1)*fc_dims[1] + (k-1)
                inds[i, :] = [i, j, k]

        if fc_format == 'compact':
            fc = np.array(fc_entry['elements'])
        elif fc_format == 'full': # Truncate to compact size.
            fc = np.array(fc_entry['elements'])[:, 0:fc_dims[0]]
        return _reshape_fc(fc, inds, fc_dims)

def _extract_physical_units(summary):
    #TODO
    """ DOC
    Retrieve physical units from phonopy summary file object.
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
    summary_dict['recip_vec'] = NotImplemented

    summary_dict['ion_r'] = n.ion_r
    summary_dict['ion_type'] = n.ion_type
    summary_dict['ion_mass'] = n.ion_mass

    summary_dict['cell_origins'] = n.cell_origins
    summary_dict['sc_matrix'] = n.sc_matrix
    summary_dict['n_cells_in_sc'] = n.n_cells_in_sc

    summary_dict['units'] = n.units

    if 'force_constants' in summary.keys():
        summary_dict['force_constants'] = _get_fc_summary(summary)

    return summary_dict

def _extract_qpts_data(data_object):
    # TODO
    """ DOC
    Retrieve n_qpts, n_ions, recip_vec, qpts, etc from qpoint.yaml or
    qpoint.hdf5 which is created during a Phonopy qpoint session.
    """

    n_qpts = data_object['nqpoint']
    n_ions = data_object['natom']
    #cell_vec = data_object['lattice']
    recip_vec = data_object['reciprocal_lattice']
    qpts = [phon['q-position'] for phon in data_object['phonon']]

    #weights = [phon['weight'] for phon in data_object['phonon']]

    phonon_data = [phon for phon in data_object['phonon']]
    bands_data_each_qpt = [bands_data['band']
                            for bands_data in phonon_data]

    dyn_mat_data = [phon['dynamical_matrix'] for phon in phonon_data]

    # The frequency for each band at each q-point
    freqs = np.array([ [band_data['frequency']
                            for band_data in bands_data]
                              for bands_data in bands_data_each_qpt])

    n_branches = freqs.shape[1]

    data_dict = {}
    #data_dict['dynamical_matrix'] = dyn_mat_data 
    data_dict['n_ions'] = n_ions
    data_dict['n_qpts'] = n_qpts
    data_dict['n_branches'] = n_branches
    data_dict['recip_vec'] = recip_vec #((reciprocal_lattice(cell_vec)/ureg.bohr).to('1/angstrom'))
    data_dict['qpts'] = qpts
    data_dict['freqs'] = freqs #(freqs*ureg.hartree).to('eV')

    return data_dict

def _reshape_fc(fc, inds, dims):
    # TODO
    """ DOC
    Reshape FORCE_CONSTANTS to conform to Euphonic format.
    Into [N_sc, 3*N_pc, 3*N_pc]
    """

    n_ions = dims[0]
    sc_n_ions = dims[1]
    n_cells = sc_n_ions // n_ions #NOTE is there a convenient safer way?

    # supercell index, atom index sc, Fi, Fj
    fc_resh = np.zeros([n_cells, n_ions, n_ions, 3, 3])

    # Reshape to arrange FC sub matrices by cell index,
    # atom_i index, atom_j index

    for i, j, k in inds: 
        # index within primitive cell
        pci_ind = (j-1) // n_cells

        # target index within primitive cell
        pcj_ind = (k-1) // n_cells

        # target cell index within sc
        scj_ind = (k-1)  % n_cells

        fc_resh[scj_ind, pci_ind, pcj_ind, :, :] = fc[i, :, :]

    # tile nxnx3x3 matrices into 3nx3n 
    fc_euph = fc_resh.transpose([0,1,3,2,4]).reshape([n_cells, 3*n_ions, 3*n_ions])

    return fc_euph

def _read_interpolation_data(path='.', qpts_file='qpoints',
                            disp_file='phonopy_disp.yaml', summary_file='phonopy.yaml',
                                born_file='BORN', fc_file='FORCE_CONSTANTS'):
    """
    Reads data from phonopy.yaml and qpoints files.

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
        and 'dielectric' if they are present in the .castep_bin or .check file
    """

    qpts_file = _match_file(path=path, name=qpts_file)
    disp_file = _match_disp(path=path, name=disp_file)
    summary_file = _match_summary(path=path, name=summary_file)

    fc_file = _match_fc(path=path, name=fc_file)
    born_file = _match_born(path=path, name=born_file)

    # load data from qpoint mode output file
    try:
        print(qpts_file)
        with h5py.File(qpts_file, 'r') as hdf5o:
            qpoint_dict = _extract_qpts_data(hdf5o)
    except Exception as e:
        pass

    try:
        with open(qpts_file) as ymlo:
            yaml_data = yaml.safe_load(ymlo)
            qpoint_dict =  _extract_qpts_data(yaml_data)
    except Exception as e:
        pass

    try:
        print(summary_file)
        with open(summary_file, 'r') as ppo:
            summary_data = yaml.safe_load(ppo)
            summary_dict = _extract_summary(summary_data)
    except Exception as e: # differentiate "find" from "load"
        print(f"Failed to load summary file {summary_file}.")
        print(e)

    try:
        print(disp_file)
        with open(disp_file, 'r') as dspo:
            disp_dict = yaml.safe_load(dspo)
    except Exception as e:
        print(f"Failed to load displacements summary file {disp_file}.")
        print(e)

    # check summary_file for force_constants, dielectric_constant, 
    # born_effective_charge
    if 'force_constants' in summary_dict.keys():
        #TODO reshape without indices
        force_constants = summary_dict['force_constants']
    else:
        try:
            with open(fc_file, 'r') as fco:
                force_constants = _extract_force_constants(fco)
        except:
            pass

    try:
        with open(born_file, 'r') as born:
            born_dict = _extract_born(born)
    except:
         pass

    if 'dielectric_constant' in summary_dict.keys():
        dielectric_constant = summary_dict['dielectric_constant']
    else:
        try:
            dielectric = born_dict['dielectric_constant']
        except:
            pass

    if 'born_effective_charge' in summary_dict.keys():
        born_effective_charge = summary_dict['born_effective_charge']
    else:
        try:
            born = born_dict['born_effective_charge']
        except:
            pass


    # Extract units
    try:
        with open(summary_file, 'r') as ppo:
            yaml_data = yaml.safe_load(ppo)
            units = _extract_physical_units(yaml_data)
    except:
        pass

    if 'frequency_unit_conversion_factor' not in units:
        # nowhere to check, warn 
        pass

    if 'nac_unit_conversion_factor' not in units:
        # check BORN
        pass

    # TODO Unit conversion factors
    ulength = ureg(units['length'].lower()).to('bohr').magnitude
    umass = ureg(units['atomic_mass'].lower()).to('e_mass').magnitude
    ufreq = (ureg('THz')*units['frequency_unit_conversion_factor'])\
                .to('E_h', 'spectroscopy').magnitude
    unac = ureg[units['nac_unit_conversion_factor']].magnitude

    # Check force constants string so that it matches format required for ureg
    ufc_str = units['force_constants'].replace('Angstrom', 'angstrom').magnitude
    ufc = ureg(ufc_str).to('hartree/bohr^2').magnitude

    data_dict = {}
    data_dict['n_ions'] = qpoint_dict['n_ions'] #n_ions
    data_dict['n_branches'] = qpoint_dict['n_branches'] #3*n_ions 
    data_dict['cell_vec'] = summary_dict['cell_vec']*ulength #(cell_vec*ureg.bohr).to('angstrom')
    data_dict['recip_vec'] = qpoint_dict['recip_vec']/ulength #((reciprocal_lattice(cell_vec)/ureg.bohr).to('1/angstrom'))

    data_dict['ion_r'] = summary_dict['ion_r'].ulength #ion_r - np.floor(ion_r)  # Normalise ion coordinates
    data_dict['ion_type'] = summary_dict['ion_type'] #ion_type
    data_dict['ion_mass'] = summary_dict['ion_mass']*umass #(ion_mass*ureg.e_mass).to('amu')

    # Set entries relating to 'FORCE_CONSTANTS' block
    try:
        data_dict['force_constants'] = force_constants*ufc #(force_constants*ureg.hartree/(ureg.bohr**2))
        data_dict['sc_matrix'] = summary_dict['sc_matrix'] #sc_matrix
        data_dict['n_cells_in_sc'] = np.int(np.round(np.linalg.det(np.array(summary_dict['sc_matrix']))))
        data_dict['cell_origins'] = summary_dict['cell_origins']*ulength #cell_origins
    except NameError:
        raise Exception((
            'Force constants matrix could not be found in {:s} or {:s}. '
            'Ensure WRITE_FC: true or --write-fc has been set when running '
            'Phonopy').format(summary_file, fc_file))

    # Set entries relating to dipoles
    try:
        data_dict['born'] = born*unac #born*ureg.e
        data_dict['dielectric'] = dielectric*unac #dielectric
    except UnboundLocalError:
        print('No bec or dielectric') #TODO remove when fixed soft fail error 
        pass

    data_dict['dynamical_matrix'] = qpoint_dict['dynamical_matrix']

    return data_dict

