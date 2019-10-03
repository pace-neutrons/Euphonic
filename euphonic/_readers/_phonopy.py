import re
import os
import struct
import yaml
import h5py

import numpy as np
from euphonic import ureg
from euphonic.util import reciprocal_lattice, is_gamma


def _read_bands_data(seedname='mesh', path='.'):
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

    #seedname = "band" # does not contain all required information

    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(f'Phonopy seed path {path} does not exist.')
        phonopy_files = os.listdir(path)
    else:
        phonopy_files = os.listdir('.')
        raise NotImplementedError('handling empty path')

    if seedname != 'mesh':
        raise NotImplementedError('seedname format inconsistent with castep style. Extension is determined by phonopy output type hdf5/yaml.')

    seed_files = [m for m in phonopy_files if seedname in m]

    #TODO make this more clear, enforce yaml, hdf5 only?
    # search for mesh files with any of these extensions
    yaml_exts = ['.yaml', '.yml']
    hdf5_exts = ['.hdf5', '.he5', '.h5']

    present_hdf5 = [[ext for ext in hdf5_exts if ext in name]
                    for name in seed_files][0]
    present_yaml = [[ext for ext in yaml_exts if ext in name]
                    for name in seed_files][0]

    use_hdf5 = False
    use_yaml = False
    if present_hdf5: # default to hdf5
        seedname_ext = present_hdf5[0]
        use_hdf5 = True
    elif present_yaml:
        seedname_ext = present_yaml[0]
        use_yaml = True
    else:
        raise NotImplementedError('handling no yaml or hdf5 found')

    bands_name = os.path.join(path, seedname + seedname_ext)


    def extract(bands_obj):
        n_qpts = bands_obj['nqpoint']
        n_ions = bands_obj['natom']
        cell_vec = bands_obj['lattice']
        recip_vec = bands_obj['reciprocal_lattice']
        qpts = [phon['q-position'] for phon in bands_obj['phonon']]

        weights = [phon['weight'] for phon in bands_obj['phonon']]

        phonon_data = [phon for phon in bands_obj['phonon']]
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

        ion_type = [ion['symbol'] for ion in bands_obj['points']]
        ion_r = [ion['coordinates'] for ion in bands_obj['points']]
        ion_mass = [ion['mass'] for ion in bands_obj['points']]

        data_dict = {}
        data_dict['n_qpts'] = n_qpts
        data_dict['n_spins'] = NotImplemented #n_spins
        data_dict['n_branches'] = n_branches
        data_dict['fermi'] = NotImplemented #(fermi*ureg.hartree).to('eV')
        data_dict['cell_vec'] = cell_vec #(cell_vec*ureg.bohr).to('angstrom')
        data_dict['recip_vec'] = recip_vec #((reciprocal_lattice(cell_vec)/ureg.bohr).to('1/angstrom'))
        data_dict['qpts'] = qpts
        data_dict['weights'] = weights #weights
        data_dict['freqs'] = freqs #(freqs*ureg.hartree).to('eV')
        data_dict['freq_down'] = NotImplemented #(freq_down*ureg.hartree).to('eV')
        #data_dict['eigenvecs'] = eigenvecs
        data_dict['n_ions'] = n_ions
        data_dict['ion_r'] = ion_r
        data_dict['ion_type'] = ion_type

        return data_dict


    if use_hdf5: #use hdf5
        with h5py.File(bands_name) as hdf5_data:
            return extract(hdf5_data)
    elif use_yaml: #use yaml
        with open(bands_name) as yaml_obj:
            yaml_data = yaml.safe_load(yaml_obj)
            return extract(yaml_data)
    else:
        raise NotImplementedError('handling no use_hdf5 / use_yaml')
        return NotImplemented



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

    n_ions = None
    n_branches = None
    n_qpts = None
    cell_vec = None
    ion_r = None
    ion_type = None
    ion_mass = None

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
        and 'dielectric' if they are present in the .castep_bin or .check file
    """

    data_dict = {}
    data_dict['n_ions'] = None #n_ions
    data_dict['n_branches'] = None #3*n_ions
    data_dict['cell_vec'] = None #(cell_vec*ureg.bohr).to('angstrom')
    data_dict['recip_vec'] = None #((reciprocal_lattice(cell_vec)/ureg.bohr).to('1/angstrom'))
    data_dict['ion_r'] = None #ion_r - np.floor(ion_r)  # Normalise ion coordinates
    data_dict['ion_type'] = None #ion_type
    data_dict['ion_mass'] = None #(ion_mass*ureg.e_mass).to('amu')

    # Set entries relating to 'FORCE_CON' block
    try:
        data_dict['force_constants'] = None #(force_constants*ureg.hartree/(ureg.bohr**2))
        data_dict['sc_matrix'] = None #sc_matrix
        data_dict['n_cells_in_sc'] = None #n_cells_in_sc
        data_dict['cell_origins'] = None #cell_origins
    except NameError:
        raise Exception((
            'Force constants matrix could not be found in {:s}.\n Ensure '
            'PHONON_WRITE_FORCE_CONSTANTS: true has been set when running '
            'CASTEP').format(file))

    # Set entries relating to dipoles
    try:
        data_dict['born'] = None #born*ureg.e
        data_dict['dielectric'] = None #dielectric
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
        The mass of each ion in the unit cell in amu
    ion_type : (n_ions,) string ndarray
        The chemical symbols of each ion in the unit cell. Ions are in the
        same order as in ion_r
    """

    n_ions = None
    cell_vec = None
    ion_r = None
    ion_mass = None
    ion_type = None

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

    data = None

    return data


