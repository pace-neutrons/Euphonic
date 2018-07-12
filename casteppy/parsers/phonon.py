import numpy as np

def read_dot_phonon(f, ureg, units='1/cm', ir=False, raman=False,
                    read_eigenvecs=False):
    """
    Reads band structure from a *.phonon file

    Parameters
    ----------
    f : file object
        File object in read mode for the .phonon file containing the data
    ureg : UnitRegistry
        Unit registry (from Pint module) for converting frequencies to units
        specified by units argument
    units : string
        String specifying the units of the output frequencies. For valid
        values see the Pint docs: http://pint.readthedocs.io/en/0.8.1/.
        Default = '1/cm'
    read_eigenvecs : boolean
        Whether to read and store the eigenvectors. Default: False
    ir : boolean
        Whether to read and store IR intensities. Default: False
    raman : boolean
        Whether to read and store Raman intensities. Default: False

    Returns
    -------
    cell_vec : list of floats
        3 x 3 list of the unit cell vectors
    ion_pos : list of floats
        n_ions x 3 list of the fractional position of each ion within the
        unit cell
    ion_type : list of strings
        n_ions length list of the chemical symbols of each ion in the unit
        cell. Ions are in the same order as in ion_pos
    qpts : list of floats
        M x 3 list of q-point coordinates, where M = number of q-points
    weights : list of floats
        List of length M containing the weight for each q-point, where
        M = number of q-points
    freqs: list of floats
        M x N list of phonon band frequencies in units specified by the
        'units parameter, where M = number of q-points and N = number of
        bands, ordered according to increasing q-point number
    i_intens: list of floats
        M x N list of IR intensities, where M = number of q-points and
        N = number of bands. Empty if no IR intensities in .phonon file
    r_intens: list of floats
        M x N list of Raman intensities, where M = number of q-points and
        N = number of bands. Empty if no Raman intensities in .phonon file
    eigenvecs: list of complex floats
        M x L x 3 list of the atomic displacements (dynamical matrix
        eigenvectors), where M = number of q-points and
        L = number of ions*number of bands. Empty if read_eigenvecs is False
    """
    (n_ions, n_branches, n_qpts, cell_vec, ion_pos,
        ion_type) = read_dot_phonon_header(f)

    qpts = np.zeros((n_qpts, 3))
    weights = np.zeros(n_qpts)
    freqs = np.zeros((n_qpts, n_branches))
    if ir:
        i_intens = np.zeros((n_qpts, n_branches))
    else:
        i_intens = np.array([])
    if raman:
        r_intens = np.zeros((n_qpts, n_branches))
    else:
        r_intens = np.array([])
    if read_eigenvecs:
        eigenvecs = np.zeros((n_qpts, n_branches*n_ions, 3),
                             dtype='complex128')
    else:
        eigenvecs = np.array([])

    # Need to loop through file using while rather than number of q-points
    # as sometimes points are duplicated
    while True:
        line = f.readline().split()
        if not line:
            break
        qpt_num = int(line[1]) - 1
        qpts[qpt_num,:] = [float(x) for x in line[2:5]]
        weights[qpt_num] = float(line[5])

        freq_lines = [f.readline().split() for i in range(n_branches)]
        freqs[qpt_num, :] = [float(line[1]) for line in freq_lines]
        if ir:
            i_intens[qpt_num, :] = [float(line[2]) for line in freq_lines]
        if raman:
            r_intens[qpt_num, :] = [float(line[3]) for line in freq_lines]

        if read_eigenvecs:
            [f.readline() for x in range(2)] # Skip 2 label lines
            lines = [f.readline().split()[2:]
                for x in range(n_ions*n_branches)]
            eigenvecs[qpt_num, :, :] = [[complex(float(x[0]), float(x[1])),
                                         complex(float(x[2]), float(x[3])),
                                         complex(float(x[4]), float(x[5]))]
                                             for x in lines]
        else:
            # Don't bother reading eigenvectors
            # Skip eigenvectors and 2 label lines
            [f.readline() for x in range(n_ions*n_branches + 2)]

    freqs = freqs*(1/ureg.cm)
    freqs.ito(units, 'spectroscopy')

    return (cell_vec, ion_pos, ion_type, qpts, weights, freqs, i_intens,
            r_intens, eigenvecs)


def read_dot_phonon_header(f):
    """
    Reads the header of a *.phonon file

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
    cell_vec : list of floats
        3 x 3 list of the unit cell vectors
    ion_pos : list of floats
        n_ions x 3 list of the fractional position of each ion within the
        unit cell
    ion_type : list of strings
        n_ions length list of the chemical symbols of each ion in the unit
        cell. Ions are in the same order as in ion_pos
    """
    f.readline() # Skip BEGIN header
    n_ions = int(f.readline().split()[3])
    n_branches = int(f.readline().split()[3])
    n_qpts = int(f.readline().split()[3])
    [f.readline() for x in range(4)] # Skip units and label lines
    cell_vec = [[float(x) for x in f.readline().split()[0:3]]
        for i in range(3)]
    f.readline() # Skip fractional co-ordinates label
    ion_info = [f.readline().split() for i in range(n_ions)]
    ion_pos = [[float(x) for x in y[1:4]] for y in ion_info]
    ion_type = [x[4] for x in ion_info]
    f.readline() # Skip END header line

    return n_ions, n_branches, n_qpts, cell_vec, ion_pos, ion_type
