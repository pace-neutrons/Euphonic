import re
import numpy as np
from simphony import ureg
from simphony.util import is_gamma


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
    cell_vec : ndarray
        The unit cell vectors in Angstroms.
        dtype = 'float'
        shape = (3, 3)
    ion_r : ndarray
        The fractional position of each ion within the unit cell
        dtype = 'float'
        shape = (n_ions, 3)
    ion_type : ndarray
        The chemical symbols of each ion in the unit cell. Ions are in the
        same order as in ion_r
        dtype = 'string'
        shape = (n_ions,)
    ion_mass : ndarray
        The mass of each ion in the unit cell in atomic units
        dtype = 'float'
        shape = (n_ions,)
    """
    f.readline()  # Skip BEGIN header
    n_ions = int(f.readline().split()[3])
    n_branches = int(f.readline().split()[3])
    n_qpts = int(f.readline().split()[3])
    [f.readline() for x in range(4)]  # Skip units and label lines
    cell_vec = np.array([[float(x) for x in f.readline().split()[0:3]]
                         for i in range(3)])
    f.readline()  # Skip fractional co-ordinates label
    ion_info = np.array([f.readline().split() for i in range(n_ions)])
    ion_r = np.array([[float(x) for x in y[1:4]] for y in ion_info])
    ion_type = np.array([x[4] for x in ion_info])
    ion_mass = np.array([float(x[5]) for x in ion_info])
    f.readline()  # Skip END header line

    return n_ions, n_branches, n_qpts, cell_vec, ion_r, ion_type, ion_mass


def _read_phonon_data(data, f):
    """
    Reads data from a .phonon file and sets attributes of the input
    PhononData object

    Parameters
    ----------
    data : PhononData object
        Empty PhononData object that will have its attributes set
    f : file object
        File object in read mode for the .phonon file containing the data
    """
    (n_ions, n_branches, n_qpts, cell_vec, ion_r,
     ion_type, ion_mass) = _read_phonon_header(f)

    qpts = np.zeros((n_qpts, 3))
    weights = np.zeros(n_qpts)
    freqs = np.zeros((n_qpts, n_branches))
    ir = np.array([])
    raman = np.array([])
    eigenvecs = np.zeros((n_qpts, n_branches, n_ions, 3),
                         dtype='complex128')
    split_i = np.array([], dtype=np.int32)
    split_freqs = np.empty((0, n_branches))
    split_eigenvecs = np.empty((0, n_branches, n_ions, 3))

    # Need to loop through file using while rather than number of q-points
    # as sometimes points are duplicated
    first_qpt = True
    qpt_line = f.readline()
    prev_qpt_num = -1
    qpt_num_patt = re.compile('q-pt=\s*(\d+)')
    float_patt = re.compile('-?\d+\.\d+')
    while qpt_line:
        qpt_num = int(re.search(qpt_num_patt, qpt_line).group(1)) - 1
        floats = re.findall(float_patt, qpt_line)
        qpts[qpt_num] = [float(x) for x in floats[:3]]
        weights[qpt_num] = float(floats[3])

        freq_lines = [f.readline().split() for i in range(n_branches)]
        tmp = np.array([float(line[1]) for line in freq_lines])
        if qpt_num != prev_qpt_num:
            freqs[qpt_num, :] = tmp
        elif is_gamma(qpts[qpt_num]):
            split_i = np.concatenate((split_i, [qpt_num]))
            split_freqs = np.concatenate((split_freqs, tmp[np.newaxis]))
        ir_index = 2
        raman_index = 3
        if is_gamma(qpts[qpt_num]):
            ir_index += 1
            raman_index += 1
        if len(freq_lines[0]) > ir_index:
            if first_qpt:
                ir = np.zeros((n_qpts, n_branches))
            ir[qpt_num, :] = [float(
                line[ir_index]) for line in freq_lines]
        if len(freq_lines[0]) > raman_index:
            if first_qpt:
                raman = np.zeros((n_qpts, n_branches))
            raman[qpt_num, :] = [float(
                line[raman_index]) for line in freq_lines]

        [f.readline() for x in range(2)]  # Skip 2 label lines
        lines = np.array([f.readline().split()[2:]
                          for x in range(n_ions*n_branches)],
                         dtype=np.float64)
        lines_i = np.column_stack(([lines[:, 0] + lines[:, 1]*1j,
                                    lines[:, 2] + lines[:, 3]*1j,
                                    lines[:, 4] + lines[:, 5]*1j]))
        tmp = np.zeros((n_branches, n_ions, 3), dtype=np.complex128)
        for i in range(n_branches):
                tmp[i, :, :] = lines_i[i*n_ions:(i+1)*n_ions, :]
        if qpt_num != prev_qpt_num:
            eigenvecs[qpt_num] = tmp
        elif is_gamma(qpts[qpt_num]):
            split_eigenvecs = np.concatenate(
                (split_eigenvecs, tmp[np.newaxis]))
        first_qpt = False
        qpt_line = f.readline()
        prev_qpt_num = qpt_num

    cell_vec = cell_vec*ureg.angstrom
    ion_mass = ion_mass*ureg.amu
    freqs = freqs*(1/ureg.cm)
    freqs.ito('meV', 'spectroscopy')
    split_freqs = split_freqs*(1/ureg.cm)
    split_freqs.ito('meV', 'spectroscopy')

    data.n_ions = n_ions
    data.n_branches = n_branches
    data.n_qpts = n_qpts
    data.cell_vec = cell_vec
    data.ion_r = ion_r
    data.ion_type = ion_type
    data.ion_mass = ion_mass
    data.qpts = qpts
    data.weights = weights
    data.freqs = freqs
    data.ir = ir
    data.raman = raman
    data.eigenvecs = eigenvecs

    data.split_i = split_i
    data.split_freqs = split_freqs
    data.split_eigenvecs = split_eigenvecs
