import re
import os
import struct
import yaml
import h5py
import glob

import numpy as np
from euphonic import ureg
from euphonic.util import reciprocal_lattice, is_gamma

#TODO documentation / docstrings
#TODO test _extract_*_data
#TODO test _read_*_data
#TODO phonopy unit conversion, search phonopy.yaml for 'physical_unit'
#TODO separate necessary, conditional, and optional variable getting processes
#TODO appropriate error responses
#TODO parse hdf5 file with p2s_map
#TODO find the units for a given file set

#TODO
def _convert_units(data_dict):
    pass

def _match_seed(path='.', seed='*'):
    # TODO
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

    #TODO open file to check formatting, fallback if fail

    # check that final name exists
    if os.path.exists(file):
        return file
    else:
        return None

def _extract_phonon_data(data_object):
    #TODO
    """ DOC
    """

    n_qpts = data_objec['nqpoint']
    n_ions = data_objec['natom']
    cell_vec = data_objec['lattice']
    recip_vec = data_objec['reciprocal_lattice']
    qpts = [phon['q-position'] for phon in data_object['phonon']]

    weights = [phon['weight'] for phon in data_object['phonon']]

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

    #TODO supply units
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

    return data_dict

def _read_phonon_data(path='.', seedname='mesh'):
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
        'split_eigenvecs'
    """

    """
    data_dict = {}
    data_dict['n_ions'] = None #n_ions
    data_dict['n_branches'] = None #n_branches
    data_dict['n_qpts'] = None #n_qpts
    data_dict['cell_vec'] = None #cell_vec*ureg.angstrom
    data_dict['recip_vec'] = None #reciprocal_lattice(cell_vec)/ureg.angstrom
    data_dict['ion_r'] = None #ion_r
    data_dict['ion_type'] = None #ion_type
    data_dict['ion_mass'] = None #ion_mass*ureg.amu
    data_dict['qpts'] = None #qpts
    data_dict['weights'] = None #weights
    data_dict['freqs'] = None #(freqs*(1/ureg.cm)).to('meV', 'spectroscopy')
    data_dict['eigenvecs'] = None #eigenvecs
    data_dict['split_i'] = None #split_i
    data_dict['split_freqs'] = None #(split_freqs*(1/ureg.cm)).to('meV', 'spectroscopy')
    data_dict['split_eigenvecs'] = None #split_eigenvecs
    """

    file = _match_seed(path='.', seed=seed)

    try:
        with h5py.File(file, 'r') as hdf5o:
            return _extract_phonon_data(hdf5o)
    except:
        pass

    try:
        with open(file, 'r') as ymlo:
            yaml_data = yaml.safe_load(ymlo)
            return _extract_phonon_data(yaml_data)
    except:
        pass

    #TODO unpack data for unit conversion and easy reading 

    return None



def _match_ppyaml(path='.', seed='phonopy.yaml'):
    # TODO
    """ DOC
    Search for the output summary generated by phonopy at the end of the
    workflow when using the cli.
    """

    file = os.path.join(path, seed)

    try:
        with open(file, 'r') as summ:
            ppyaml_contents = yaml.safe_load(summ)
        atom_names = ppyaml_contents['phonopy']['configuration']['atom_name']
    except KeyError:
        return None

    if os.path.exists(file):
        return file
    else:
        return None

def _match_disp(path='.', seed='phonopy_disp.yaml'):
    # TODO
    """ DOC
    Search for the atom displacements summary generated by phonopy at
    the start of the workflow when using the cli.
    """

    file = os.path.join(path, seed)

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

def _match_fc(path='.', seed='FORCE_CONSTANTS'):
    # TODO
    """ DOC
    Name 'FORCE_CONSTANTS' is default in Phonopy, but in some cases
    renamed to e.g. FORCE_CONSTANTS_NaCl to distinguish between them.
    """

    #TODO match glob for FORCE_CONSTANTS*
    fc_file = os.path.join(path, seed)
    fc5_file = os.path.join(path, (seed + '.hdf5').lower())

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

def _match_born(path='.', seed='BORN'):
    # TODO
    """ DOC
    Name BORN is default in Phonopy, but in some cases they're
    renamed to e.g. BORN_NaCl to distinguish between them.
    """

    born_file = os.path.join(path, seed)
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

    # ignore blank lines #TODO split('') or split(' ')
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

def _get_fc_ppyaml(ppyaml):
        fc_entry = ppyaml['force_constants']

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
            #TODO check that 'full' is the label given
            fc = np.array(fc_entry['elements'])[:, 0:fc_dims[0]]
        return _reshape_fc(fc, inds, fc_dims)

def _extract_ppyaml(ppyaml):
    # TODO
    """DOC
    Read phonopy.yaml.
    """
    n = lambda: None
    n.sc_matrix = np.array(ppyaml['supercell_matrix'])
    #n.p_matrix = np.array(ppyaml['primitive_matrix']) # Not found
    n.n_cells_in_sc = int(np.rint(np.absolute(np.linalg.det(n.sc_matrix))))

    n.p_n_ions = len(ppyaml['primitive_cell']['points'])
    n.p_cell_vec = np.array(ppyaml['primitive_cell']['lattice'])
    n.p_ion_r = np.zeros((n.p_n_ions, 3))
    n.p_ion_mass = np.zeros(n.p_n_ions)
    n.p_ion_type = []
    for i in range(n.p_n_ions):
        n.p_ion_mass[i] = ppyaml['primitive_cell']['points'][i]['mass']
        n.p_ion_r[i] = ppyaml['primitive_cell']['points'][i]['coordinates']
        n.p_ion_type.append(ppyaml['primitive_cell']['points'][i]['symbol'])

    n.n_ions = len(ppyaml['unit_cell']['points'])
    n.cell_vec = np.array(ppyaml['unit_cell']['lattice'])
    n.ion_r = np.zeros((n.n_ions, 3))
    n.ion_mass = np.zeros(n.n_ions)
    n.ion_type = []
    for i in range(n.n_ions):
        n.ion_mass[i] = ppyaml['unit_cell']['points'][i]['mass']
        n.ion_r[i] = ppyaml['unit_cell']['points'][i]['coordinates']
        n.ion_type.append(ppyaml['unit_cell']['points'][i]['symbol'])

    n.sc_n_ions = len(ppyaml['supercell']['points'])
    n.sc_cell_vec = np.array(ppyaml['supercell']['lattice'])
    n.sc_ion_r = np.zeros((n.sc_n_ions, 3))
    n.sc_ion_mass = np.zeros(n.sc_n_ions)
    n.sc_ion_type = []
    for i in range(n.sc_n_ions):
        n.sc_ion_mass[i] = ppyaml['supercell']['points'][i]['mass']
        n.sc_ion_r[i] = ppyaml['supercell']['points'][i]['coordinates']
        n.sc_ion_type.append(ppyaml['supercell']['points'][i]['symbol'])

    # Coordinates of supercell ions in fractional coords of the unit cell
    sc_ion_r_ucell = np.einsum('ij,jk->ij', n.sc_ion_r, n.sc_matrix)
    cell_origins = sc_ion_r_ucell[:n.n_ions] - n.ion_r[0]

    n.cell_origins = cell_origins
    n.sc_ion_r_ucell = sc_ion_r_ucell

    ppyaml_dict = {}
    ppyaml_dict['n_ions'] = n.n_ions
    ppyaml_dict['cell_vec'] = n.cell_vec
    ppyaml_dict['recip_vec'] = NotImplemented

    ppyaml_dict['ion_r'] = n.ion_r
    ppyaml_dict['ion_type'] = n.ion_type
    ppyaml_dict['ion_mass'] = n.ion_mass

    ppyaml_dict['cell_origins'] = n.cell_origins
    ppyaml_dict['sc_matrix'] = n.sc_matrix
    ppyaml_dict['n_cells_in_sc'] = n.n_cells_in_sc

    if 'force_constants' in ppyaml.keys():
        ppyaml_dict['force_constants'] = _get_fc_ppyaml(ppyaml)

    return ppyaml_dict


def _extract_qpoints_data(data_object):
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
    data_dict['dynamical_matrix'] = dyn_mat_data #TODO remove dynamical matrix if not necessary
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


def _read_interpolation_data(path='.', qpointsseed='qpoints',
                            dispseed='phonopy_disp.yaml', ppyamlseed='phonopy.yaml',
                                bornseed='BORN', fcseed='FORCE_CONSTANTS'):
    """
    Reads data from a qpoints.yaml file and returns it in a dictionary

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

    qpoints_file = _match_seed(path=path, seed=qpointsseed)
    disp_file = _match_disp(path=path, seed=dispseed)
    ppyaml_file = _match_ppyaml(path=path, seed=ppyamlseed)

    fc_file = _match_fc(path=path, seed=fcseed)
    born_file = _match_born(path=path, seed=bornseed)

    # load data from qpoint mode output file
    try:
        print(qpoints_file)
        with h5py.File(qpoints_file, 'r') as hdf5o:
            qpoint_dict = _extract_qpoints_data(hdf5o)
    except Exception as e:
        pass

    try:
        with open(qpoints_file) as ymlo:
            yaml_data = yaml.safe_load(ymlo)
            qpoint_dict =  _extract_qpoints_data(yaml_data)
    except Exception as e:
        pass

    try: #NOTE This might be done more concisely from just ppyaml_dict
        print(ppyaml_file)
        with open(ppyaml_file, 'r') as ppo:
            ppyaml_data = yaml.safe_load(ppo)
            ppyaml_dict = _extract_ppyaml(ppyaml_data)
    except Exception as e: # differentiate "find" from "load"
        print(f"Failed to load summary file {ppyaml_file}.")
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
    if 'force_constants' in ppyaml_dict.keys():
        #TODO reshape without indices
        force_constants = ppyaml_dict['force_constants']
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

    if 'dielectric_constant' in ppyaml_dict.keys():
        dielectric_constant = ppyaml_dict['dielectric_constant']
    else:
        try:
            dielectric = born_dict['dielectric_constant']
        except:
            pass

    if 'born_effective_charge' in ppyaml_dict.keys():
        born_effective_charge = ppyaml_dict['born_effective_charge']
    else:
        try:
            born = born_dict['born_effective_charge']
        except:
            pass

    # Unit conversion
    #TODO see how Phonopy handles units: same as calculator, or standard internal unit?
    data_dict = {}
    data_dict['n_ions'] = qpoint_dict['n_ions'] #n_ions
    data_dict['n_branches'] = qpoint_dict['n_branches'] #3*n_ions #TODO
    data_dict['cell_vec'] = ppyaml_dict['cell_vec'] #(cell_vec*ureg.bohr).to('angstrom')
    data_dict['recip_vec'] = qpoint_dict['recip_vec'] #((reciprocal_lattice(cell_vec)/ureg.bohr).to('1/angstrom'))

    data_dict['ion_r'] = ppyaml_dict['ion_r'] #ion_r - np.floor(ion_r)  # Normalise ion coordinates
    data_dict['ion_type'] = ppyaml_dict['ion_type'] #ion_type
    data_dict['ion_mass'] = ppyaml_dict['ion_mass'] #(ion_mass*ureg.e_mass).to('amu')

    # Set entries relating to 'FORCE_CONSTANTS' block
    try:
        data_dict['force_constants'] = force_constants #(force_constants*ureg.hartree/(ureg.bohr**2))
        data_dict['sc_matrix'] = ppyaml_dict['sc_matrix'] #sc_matrix
        data_dict['n_cells_in_sc'] = np.int(np.round(np.linalg.det(np.array(ppyaml_dict['sc_matrix'])))) 
        data_dict['cell_origins'] = ppyaml_dict['cell_origins'] #cell_origins
    except NameError:
        raise Exception((
            'Force constants matrix could not be found in {:s} or {:s}. '
            'Ensure WRITE_FC: true or --write-fc has been set when running '
            'Phonopy').format(ppyamlseed, fcseed))

    # Set entries relating to dipoles
    try:
        data_dict['born'] = born #born*ureg.e
        data_dict['dielectric'] = dielectric #dielectric
    except UnboundLocalError:
        print('No bec or dielectric') #TODO remove when fixed soft fail error 
        pass

    data_dict['dynamical_matrix'] = qpoint_dict['dynamical_matrix']

    return data_dict


#TODO remove bands readers if not needed
def _extract_bands_data(data_object):
    n_qpts = data_object['nqpoint']
    n_ions = data_object['natom']
    cell_vec = data_object['lattice']
    recip_vec = data_object['reciprocal_lattice']
    qpts = [phon['q-position'] for phon in data_object['phonon']]

    weights = [phon['weight'] for phon in data_object['phonon']]

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

    ion_type = [ion['symbol'] for ion in data_object ['points']]
    ion_r = [ion['coordinates'] for ion in data_object ['points']]
    ion_mass = [ion['mass'] for ion in data_object ['points']]

    #TODO supply units
    data_dict = {}
    data_dict['n_qpts'] = n_qpts
    data_dict['n_spins'] = NotImplemented #n_spins, electronic
    data_dict['n_branches'] = n_branches
    data_dict['fermi'] = NotImplemented #(fermi*ureg.hartree).to('eV')
    data_dict['cell_vec'] = cell_vec #(cell_vec*ureg.bohr).to('angstrom')
    data_dict['recip_vec'] = recip_vec #((reciprocal_lattice(cell_vec)/ureg.bohr).to('1/angstrom'))
    data_dict['qpts'] = qpts
    data_dict['weights'] = weights #weights
    data_dict['freqs'] = freqs #(freqs*ureg.hartree).to('eV')
    data_dict['freq_down'] = NotImplemented #(freq_down*ureg.hartree).to('eV'), electronic
    #data_dict['eigenvecs'] = eigenvecs
    data_dict['n_ions'] = n_ions
    data_dict['ion_r'] = ion_r
    data_dict['ion_type'] = ion_type

    return data_dict

def _read_bands_data(path='.', seedname='mesh'):
    """
    Reads data from a band.yaml file and
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
    """

    """ define unassigned defaults
    data_dict = {}
    data_dict['n_ions'] = None #n_ions
    data_dict['n_branches'] = None #n_branches
    data_dict['n_qpts'] = None #n_qpts
    data_dict['cell_vec'] = None #cell_vec*ureg.angstrom
    data_dict['recip_vec'] = None #reciprocal_lattice(cell_vec)/ureg.angstrom
    data_dict['ion_r'] = None #ion_r
    data_dict['ion_type'] = None #ion_type
    data_dict['ion_mass'] = None #ion_mass*ureg.amu
    data_dict['qpts'] = None #qpts
    data_dict['weights'] = None #weights
    data_dict['freqs'] = None #(freqs*(1/ureg.cm)).to('meV', 'spectroscopy')
    data_dict['eigenvecs'] = None #eigenvecs
    data_dict['split_i'] = None #split_i
    data_dict['split_freqs'] = None #(split_freqs*(1/ureg.cm)).to('meV', 'spectroscopy')
    data_dict['split_eigenvecs'] = None #split_eigenvecs
    """

    file = _match_seed(path='.', seed=seed)

    try:
        with h5py.File(file, 'r') as hdf5o:
            return _extract_bands_data(hdf5o)
    except:
        pass

    try:
        with open(file, 'r') as ymlo:
            yaml_data = yaml.safe_load(ymlo)
            return _extract_bands_data(yaml_data)
    except:
        pass

    return None

