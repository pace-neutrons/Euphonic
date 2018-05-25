"""
Parse a *.castep, *.phonon or *.band output file from new CASTEP for
vibrational frequency data and output a matplotlib plot of the electronic
or vibrational band structure or dispersion.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import math
import argparse
import seekpath
from pint import UnitRegistry


def main():
    args = parse_arguments()
    ureg = set_up_unit_registry()

    # Read data
    with open(args.filename, 'r') as f:
        read_eigenvecs = args.reorder
        (cell_vec, ion_pos, ion_type, qpts, weights, freqs, freq_down,
            i_intens, r_intens, eigenvecs, fermi) = read_input_file(
                f, ureg, args.units, args.up, args.down, args.ir, args.raman,
                read_eigenvecs)

    # Calculate and plot DOS
    if args.dos:
        # Set default DOS bin and broadening width based on whether it's
        # electronic or vibrational
        if args.b == None:
            if f.name.endswith('.bands'):
                bwidth = 0.05*ureg.eV
            else:
                bwidth = 1.0*(1/ureg.cm)
            bwidth.ito(args.units, 'spectroscopy')
        else:
            bwidth = float(args.b)*ureg[args.units]
        if args.w == None:
            if f.name.endswith('.bands'):
                gwidth = 0.1*ureg.eV
            else:
                gwidth = 10.0*(1/ureg.cm)
            gwidth.ito(args.units, 'spectroscopy')
        else:
            gwidth = float(args.w)*ureg[args.units]

        if args.ir:
            dos, dos_down, bins = calculate_dos(
                freqs, freq_down, weights, bwidth.magnitude, gwidth.magnitude,
                args.lorentz, intensities=i_intens)
        else:
            dos, dos_down, bins = calculate_dos(
                freqs, freq_down, weights, bwidth.magnitude, gwidth.magnitude,
                args.lorentz)

        fig = plot_dos(dos, dos_down, bins, args.units, args.filename,
                       fermi=[f.magnitude for f in fermi], mirror=args.mirror)

    # Calculate and plot dispersion
    else:
        # Reorder frequencies if eigenvectors have been read and the flag
        # has been set
        if eigenvecs.size > 0 and args.reorder:
            if freqs.size > 0:
                freqs = reorder_freqs(freqs, qpts, eigenvecs)
            if freq_down.size > 0:
                freq_down = reorder_freqs(freq_down, qpts, eigenvecs)

        # Get positions of q-points along x-axis
        recip_latt = reciprocal_lattice(cell_vec)
        abscissa = calc_abscissa(qpts, recip_latt)

        # Get labels for high symmetry / fractional q-point coordinates
        labels, qpts_with_labels = recip_space_labels(
            qpts, cell_vec, ion_pos, ion_type)

        fig = plot_dispersion(abscissa, freqs, freq_down, args.units,
                              args.filename, xticks=abscissa[qpts_with_labels],
                              xlabels=labels,
                              fermi=[f.magnitude for f in fermi])

    plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="""Extract phonon or bandstructure data from .castep,
                       .phonon or .bands files and plot the band structure
                       (default) or density of states with matplotlib""")
    parser.add_argument(
        'filename',
        help="""The .castep, .phonon or .bands file to extract the
                bandstructure data from""")
    parser.add_argument(
        '-v',
        action='store_true',
        help='Be verbose about progress')
    parser.add_argument(
        '-units',
        default='eV',
        help="""Convert frequencies to specified units for plotting (e.g
                1/cm, Ry). Default: eV""")

    spin_group = parser.add_mutually_exclusive_group()
    spin_group.add_argument(
        '-up',
        action='store_true',
        help='Extract and plot only spin up from *.castep or *.bands')
    spin_group.add_argument(
        '-down',
        action='store_true',
        help='Extract and plot only spin down from *.castep or *.bands')

    disp_group = parser.add_argument_group(
        'Dispersion arguments',
        'Arguments specific to plotting the band structure')
    disp_group.add_argument(
        '-reorder',
        action='store_true',
        help="""Try to determine branch crossings from eigenvectors and
                rearrange frequencies accordingly (only applicable if
                using a .phonon file)""")

    dos_group = parser.add_argument_group(
        'DOS arguments',
        'Arguments specific to plotting the density of states')
    dos_group.add_argument(
        '-dos',
        action='store_true',
        help='Plot density of states instead of a dispersion plot')
    dos_group.add_argument(
        '-ir',
        action='store_true',
        help='Extract IR intensities from .phonon and use to weight DOS')
    dos_group.add_argument(
        '-raman',
        action='store_true',
        help="""Extract Raman intensities from .phonon and calculate a Raman
                spectrum""")
    dos_group.add_argument(
        '-w',
        default=None,
        help="""Set Gaussian/Lorentzian FWHM for broadening (in units specified
                by -units argument or default eV). Default: 0.1 eV for
                electronic DOS, 10.0/cm for vibrational DOS""")
    dos_group.add_argument(
        '-b',
        default=None,
        help="""Set histogram resolution for binning (in units specified by
                -units argument or default eV). Default: 0.05 eV for electronic
                DOS, 1.0/cm for vibrational DOS""")
    dos_group.add_argument(
        '-lorentz',
        action='store_true',
        help='Use Lorentzian broadening instead of Gaussian')
    dos_group.add_argument(
        '-mirror',
        action='store_true',
        help='Plot spin down electronic DOS mirrored in the x axis')


    args = parser.parse_args()
    return args


def set_up_unit_registry():
    ureg = UnitRegistry()
    ureg.define('rydberg = 13.605693009*eV = Ry') # CODATA 2014
    return ureg


def read_input_file(f, ureg, units, up=False, down=False, ir=False,
                    raman=False, read_eigenvecs=False):
    """
    Reads data from a .bands, .phonon, or .castep_bin file. If the specified
    file doesn't contain all the required data (e.g. ionic positions) looks in
    other .bands, .phonon or .castep_bin files of the same structure name

    Parameters
    ----------
    f : file object
        File object in read mode for the file containing the data
    ureg : UnitRegistry
        Unit registry (from Pint module) for converting frequencies and Fermi
        energies to units specified by units argument
    units : string
        String specifying the units of the output frequencies. For valid
        values see the Pint docs: http://pint.readthedocs.io/en/0.8.1/
    up : boolean
        Read only spin up frequencies from the .bands file. Default: False
    down : boolean
        Read only spin down frequencies from the .bands file. Default: False
    ir : boolean
        Whether to read and store IR intensities (only applicable if
        reading a .phonon file). Default: False
    raman : boolean
        Whether to read and store Raman intensities (only applicable if
        reading a .phonon file). Default: False
    read_eigenvecs : boolean
        Whether to read and store the eigenvectors (only applicable if
        reading a .phonon file). Default: False

    Returns
    -------
    cell_vec : list of floats
        3 x 3 list of the unit cell vectors
    ion_pos : list of floats
        n_ions x 3 list of the fractional position of each ion within the
        unit cell. If the ion positions can't be found, ion_pos is empty
    ion_type : list of strings
        n_ions length list of the chemical symbols of each ion in the unit
        cell. Ions are in the same order as in ion_pos. If the ion types can't
        be found, ion_type is empty
    qpts : list of floats
        M x 3 list of q-point coordinates, where M = number of q-points
    weights : list of floats
        List of length M containing the weight for each q-point, where
        M = number of q-points
    freqs : list of floats
        M x N list of band frequencies in units specified by the
        'units' parameter, where M = number of q-points and N = number of
        bands. If there are 2 spin components, these are the spin up
        frequencies. If -down is specified, freqs is empty
    freq_down : list of floats
        M x N list of spin down band frequencies in units specified by the
        'units' parameter, where M = number of q-points and N = number of
        bands. If there are no spin down frequencies, or if -up is specified
        freq_down is empty
    i_intens: list of floats
        M x N list of IR intensities, where M = number of q-points and
        N = number of bands. Empty if no IR intensities in .phonon file
    r_intens: list of floats
        M x N list of Raman intensities, where M = number of q-points and
        N = number of bands. Empty if no Raman intensities in .phonon file
    eigenvecs: list of complex floats
        M x L x 3 list of the atomic displacements (dynamical matrix
        eigenvectors), where M = number of q-points and
        L = number of ions*number of bands. If read_eigenvecs is False,
        eigenvecs is empty
    fermi : list of floats
        List of length 1 or 2 containing the Fermi energy/energies in atomic
        units. If the file doesn't contain the fermi energy, fermi is empty
    """
    ion_pos = []
    ion_type = []
    freqs = np.array([])
    freq_down = np.array([])
    i_intens = np.array([])
    r_intens = np.array([])
    eigenvecs = np.array([])
    fermi = np.array([])

    # If file is in this directory
    if f.name.rfind('/') == -1:
        dir_name = '.'
        structure_name = f.name[:f.name.rfind('.')]
    # If file is in another directory
    else:
        dir_name = f.name[:f.name.rfind('/')]
        structure_name = f.name[f.name.rfind('/') + 1:f.name.rfind('.')]

    if f.name.endswith('.bands'):
        (fermi, cell_vec, qpts, weights, freqs,
            freq_down) = read_dot_bands(f, ureg, up, down, units)
        # Try and get point group info by reading ionic positions from .phonon
        phonon_file = dir_name + '/' + structure_name + '.phonon'
        try:
            with open(phonon_file, 'r') as pf:
                ion_pos, ion_type = read_dot_phonon_header(pf)[4:6]
        except IOError:
            pass

    elif f.name.endswith('.phonon'):
        (cell_vec, ion_pos, ion_type, qpts, weights, freqs, i_intens, r_intens,
            eigenvecs) = read_dot_phonon(f, ureg, units, ir, raman,
                                         read_eigenvecs)

    else:
        sys.exit('Error: Please supply a .bands or .phonon file')

    return (cell_vec, ion_pos, ion_type, qpts, weights, freqs, freq_down,
            i_intens, r_intens, eigenvecs, fermi)


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

    ureg = set_up_unit_registry()
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


def read_dot_bands(f, ureg, up=False, down=False, units='eV'):
    """
    Reads band structure from a *.bands file

    Parameters
    ----------
    f : file object
        File object in read mode for the .bands file containing the data
    ureg : UnitRegistry
        Unit registry (from Pint module) for converting frequencies and Fermi
        energies to units specified by units argument
    up : boolean
        Read only spin up frequencies from the .bands file. Default: False
    down : boolean
        Read only spin down frequencies from the .bands file. Default: False
    units : string
        String specifying the units of the output frequencies. For valid
        values see the Pint docs: http://pint.readthedocs.io/en/0.8.1/.
        Default: 'eV'

    Returns
    -------
    fermi : list of floats
        List of length 1 or 2 containing the Fermi energy/energies in atomic
        units
    cell_vec : list of floats
        3 x 3 list of the unit cell vectors
    kpts : list of floats
        M x 3 list of k-point coordinates, where M = number of k-points
    weights : list of floats
        List of length M containing the weight for each k-point, where
        M = number of k-points
    freq_up : list of floats
        M x N list of spin up band frequencies in units specified by the
        'units' parameter, where M = number of k-points and N = number of
        bands, ordered according to increasing k-point number
    freq_down : list of floats
        M x N list of spin down band frequencies in units specified by the
        'units' parameter, where M = number of k-points and N = number of
        bands, ordered according to increasing k-point number
    """
    n_kpts = int(f.readline().split()[3])
    n_spins = int(f.readline().split()[4])
    f.readline() # Skip number of electrons line
    n_freqs = int(f.readline().split()[3])
    fermi = [float(x) for x in f.readline().split()[5:]]
    f.readline() # Skip unit cell vectors line
    cell_vec = [[float(x) for x in f.readline().split()[0:3]]
        for i in range(3)]

    freq_up = np.array([])
    freq_down = np.array([])
    freqs = np.zeros(n_freqs)
    kpts = np.zeros((n_kpts, 3))
    weights = np.zeros(n_kpts)

    # Need to loop through file using while rather than number of k-points
    # as sometimes points are duplicated
    first_kpt = True
    while True:
        line = f.readline().split()
        if not line:
            break
        kpt_num = int(line[1]) - 1
        kpts[kpt_num,:] = [float(x) for x in line[2:5]]
        weights[kpt_num] = float(line[5])

        for j in range(n_spins):
            spin = int(f.readline().split()[2])

            # Read frequencies
            for k in range(n_freqs):
                freqs[k] = float(f.readline())

            # Allocate spin up freqs as long as -down hasn't been specified
            if spin == 1 and not down:
                if first_kpt:
                    freq_up = np.zeros((n_kpts, n_freqs))
                freq_up[kpt_num, :] = freqs
            # Allocate spin down freqs as long as -up hasn't been specified
            elif spin == 2 and not up:
                if first_kpt:
                    freq_down = np.zeros((n_kpts, n_freqs))
                freq_down[kpt_num, :] = freqs

        if first_kpt:
            if freq_up.size == 0 and freq_down.size == 0:
                sys.exit('Error: requested spin not found in .bands file')
        first_kpt = False

    freq_up = freq_up*ureg.hartree
    freq_up.ito(units, 'spectroscopy')
    freq_down = freq_down*ureg.hartree
    freq_down.ito(units, 'spectroscopy')
    fermi = fermi*ureg.hartree
    fermi.ito(units, 'spectroscopy')

    return fermi, cell_vec, kpts, weights, freq_up, freq_down


def reorder_freqs(freqs, qpts, eigenvecs):
    """
    Reorders frequencies across q-points in order to join branches

    Parameters
    ----------
    freqs: list of floats
        M x N list of phonon band frequencies, where M = number of q-points
        and N = number of bands, ordered according to increasing q-point number
    qpts : list of floats
        M x 3 list of q-point coordinates, where M = number of q-points
    eigenvecs: list of complex floats
        M x L x 3 list of the atomic displacements (dynamical matrix
        eigenvectors), where M = number of q-points and
        L = number of ions*number of bands

    Returns
    -------
    ordered_freqs: list of floats
        Ordered M x N list of phonon band frequencies, where M = number of
        q-points and N = number of bands
    """

    n_qpts = qpts.shape[0]
    n_branches = freqs.shape[1]
    n_ions = int(eigenvecs.shape[1]/n_branches)
    ordered_freqs = np.zeros((n_qpts,n_branches))
    qmap = np.arange(n_branches)

    # Only calculate qmap and reorder freqs if the direction hasn't changed
    calculate_qmap = np.concatenate(([True], np.logical_not(
        direction_changed(qpts))))
    # Don't reorder first q-point
    ordered_freqs[0,:] = freqs[0,:]
    for i in range(1,n_qpts):
        # Initialise q-point mapping for this q-point
        qmap_tmp = np.arange(n_branches)
        if calculate_qmap[i-1]:
            # Compare eigenvectors for each mode for this q-point with every
            # mode for the previous q-point
            # Reshape eigenvector arrays for this and previous q-point so that
            # each mode is a row and each ion is a column, for efficient
            # summing over modes later. Then explicitly broadcast arrays with
            # repeat and tile to ensure correct multiplication of modes
            current_eigenvecs = np.reshape(eigenvecs[i, :, :],
                                           (n_branches, n_ions, 3))
            current_eigenvecs = np.repeat(current_eigenvecs,
                                          n_branches, axis=0)
            prev_eigenvecs = np.reshape(eigenvecs[i-1, :, :],
                                        (n_branches, n_ions, 3))
            prev_eigenvecs = np.tile(prev_eigenvecs,
                                     (n_branches, 1, 1))
            # Compute complex conjugated dot product of every mode of this
            # q-point with every mode of previous q-point, and sum the dot
            # products over ions (i.e. multiply eigenvectors elementwise, then
            # sum over the last 2 dimensions)
            dots = np.absolute(np.einsum('ijk,ijk->i',
                                         np.conj(prev_eigenvecs),
                                         current_eigenvecs))

            # Create matrix of dot products for each mode of this q-point with
            # each mode of the previous q-point
            dot_mat = np.reshape(dots, (n_branches, n_branches))

            # Find greates exp(-iqr)-weighted dot product
            for j in range(n_branches):
                max_i = (np.argmax(dot_mat))
                mode = int(max_i/n_branches) # Modes are dot_mat rows
                prev_mode = max_i%n_branches # Prev q-pt modes are columns
                # Ensure modes aren't mapped more than once
                dot_mat[mode, :] = 0
                dot_mat[:, prev_mode] = 0
                qmap_tmp[mode] = prev_mode
        # Map q-points according to previous q-point mapping
        qmap = qmap[qmap_tmp]

        # Reorder frequencies
        ordered_freqs[i,qmap] = freqs[i,:]

    return ordered_freqs


def calc_abscissa(qpts, recip_latt):
    """
    Calculates the distance between q-points (for the plot x-coordinate)
    """

    # Get distance between q-points in each dimension
    # Note: length is nqpts - 1
    delta = np.diff(qpts, axis=0)

    # Determine how close delta is to being an integer. As q = q + G where G
    # is a reciprocal lattice vector based on {1,0,0},{0,1,0},{0,0,1}, this
    # is used to determine whether 2 q-points are equivalent ie. they differ
    # by a multiple of the reciprocal lattice vector
    delta_rem = np.sum(np.abs(delta - np.rint(delta)), axis=1)

    # Create a boolean array that determines whether to calculate the distance
    # between q-points,taking into account q-point equivalence. If delta is
    # more than the tolerance, but delta_rem is less than the tolerance, the
    # q-points differ by G so are equivalent and the distance shouldn't be
    # calculated
    TOL = 0.001
    calc_modq = np.logical_not(np.logical_and(
        np.sum(np.abs(delta), axis=1) > TOL,
        delta_rem < TOL))

    # Multiply each delta by the reciprocal lattice to get delta in Cartesian
    deltaq = np.einsum('ji,kj->ki', recip_latt, delta)

    # Get distance between q-points for all valid pairs of q-points
    modq = np.zeros(np.size(delta, axis=0))
    modq[calc_modq] = np.sqrt(np.sum(np.square(deltaq), axis=1))

    # Prepend initial x axis value of 0
    abscissa = np.insert(modq, 0, 0.)

    # Do cumulative some to get position along x axis
    abscissa = np.cumsum(abscissa)

    return abscissa


def calculate_dos(freqs, freq_down, weights, bwidth, gwidth, lorentz=False, intensities=[]):
    """
    Calculates a density of states with Gaussian/Lorentzian broadening from a
    list of frequencies

    Parameters
    ----------
    freqs : list of floats
        M x N list of spin up band frequencies, where M = number of q-points
        and N = number of bands, can be empty if only spin down frequencies are
        present
    freq_down : list of floats
        M x N list of spin down band frequencies, where M = number of q-points
        and N = number of bands, can be empty if only spin up frequencies are
        present
    weights : list of floats
        List of length M containing the weight for each q-point, where
        M = number of q-points
    bwidth : float
        Width of each bin for the DOS, in the same units as freqs/freq_down
    gwidth : float
        FWHM of Gaussian/Lorentzian for broadening the DOS bins. Set to 0 if
        no broadening is desired
    lorentz : boolean
        Whether to use a Lorentzian or Gaussian broadening function.
        Default: False
    intensities : list of floats
        M x N list of IR intensities for each frequency, is used to weight the
        DOS binning in addition to the weights parameter, where M = number of
        q-points and N = number of bands. Default: []

    Returns
    -------
    dos : list of floats
        L - 1 length list of the spin up density of states for each bin, where
        L is the lengh of the bins return value. Can be empty if only spin down
        frequencies are present
    dos_down : list of floats
        L - 1 length list of the spin down density of states for each bin,
        where L is the lengh of the bins return value. Can be empty if only
        spin up frequencies are present
    bins : list of floats
        One dimensional list of the energy bin edges, in the same units as
        freqs/freq_down and determined by the max/min values of freqs/freq_down
    """

    n_branches = len(freqs[0]) if len(freqs) > 0 else len(freq_down[0])
    hist = np.array([])
    hist_down = np.array([])
    dos = np.array([])
    dos_down = np.array([])

    # Calculate bin edges
    all_freqs = np.append(freqs, freq_down)
    freq_max = np.amax(all_freqs)
    freq_min = np.amin(all_freqs)
    bins = np.arange(freq_min, freq_max + bwidth, bwidth)

    # Calculate weight for each q-point and branch
    freq_weights = np.repeat(np.array(weights)[:,np.newaxis],
                             n_branches, axis=1)
    if len(intensities) > 0:
        freq_weights *= intensities

    # Bin frequencies
    if len(freqs) > 0:
        hist, bin_edges = np.histogram(freqs, bins,
                                       weights=freq_weights)
    if len(freq_down) > 0:
        hist_down, bin_edges = np.histogram(freq_down, bins,
                                            weights=freq_weights)

    # Only broaden if broadening is more than bin width
    if gwidth > bwidth:
        # Calculate broadening for adjacent nbin_broaden bins
        if lorentz:
            # 25 * Lorentzian FWHM
            nbin_broaden = int(math.floor(25.0*gwidth/bwidth))
            broadening = lorentzian(
                np.arange(-nbin_broaden, nbin_broaden)*bwidth, gwidth)
        else:
            # 3 * Gaussian FWHM
            nbin_broaden = int(math.floor(3.0*gwidth/bwidth))
            sigma = gwidth/(2*math.sqrt(2*math.log(2)))
            broadening = gaussian(
                np.arange(-nbin_broaden, nbin_broaden)*bwidth, sigma)

        if hist.size > 0:
            # Allow broadening beyond edge of bins
            dos = np.zeros(len(hist) + 2*nbin_broaden)
            for i, h in enumerate(hist):
                # Broaden each hist bin value to adjacent bins
                bhist = h*broadening
                dos[i:i+2*nbin_broaden] += bhist
            # Slice dos array to same size as bins
            dos = dos[nbin_broaden:-nbin_broaden]
        if hist_down.size > 0:
            dos_down = np.zeros(len(hist_down) + 2*nbin_broaden)
            for i, h in enumerate(hist_down):
                bhist = h*broadening
                dos_down[i:i+2*nbin_broaden] += bhist
            dos_down = dos_down[nbin_broaden:-nbin_broaden]

    else:
        dos = hist
        dos_down = hist_down

    return dos, dos_down, bins


def gaussian(x, sigma):
    return np.exp(-np.square(x)/(2*sigma**2))/(math.sqrt(2*math.pi)*sigma)


def lorentzian(x, gamma):
    return gamma/(2*math.pi*(np.square(x) + (gamma/2)**2))


def recip_space_labels(qpts, cell_vec, ion_pos, ion_type):
    """
    Gets high symmetry point labels (e.g. GAMMA, X, L) for the q-points at
    which the path through reciprocal space changes direction

    Parameters
    ----------
    qpts : list of floats
        N x 3 list of the q-point coordinates, where N = number of q-points
    cell_vec : list of floats
        3 x 3 list of the unit cell vectors
    ion_pos : list of floats
        n_ions x 3 list of the fractional position of each ion within the
        unit cell
    ion_type : list of strings
        n_ions length list of the chemical symbols of each ion in the unit
        cell. Ions are in the same order as in ion_pos

    Returns
    -------
    labels : list of strings
        List of the labels for each q-point at which the path through
        reciprocal space changes direction
    qpts_with_labels : list of integers
        List of the indices of the q-points at which the path through
        reciprocal space changes direction
    """

    # First and last q-points should always be labelled
    qpt_has_label = np.concatenate(([True], direction_changed(qpts), [True]))
    qpts_with_labels = np.where(qpt_has_label)[0]

    # Get dict of high symmetry point labels to their coordinates for this
    # space group. If space group can't be determined use a generic dictionary
    # of fractional points
    sym_label_to_coords = {}
    if len(ion_pos) > 0:
        _, ion_num = np.unique(ion_type, return_inverse=True)
        cell = (cell_vec, ion_pos, ion_num)
        sym_label_to_coords = seekpath.get_path(cell)["point_coords"]
    else:
        sym_label_to_coords = generic_qpt_labels()

    # Get labels for each q-point
    labels = np.array([])

    for qpt in qpts[qpts_with_labels]:
        labels = np.append(labels, get_qpt_label(qpt, sym_label_to_coords))

    return labels, qpts_with_labels


def generic_qpt_labels():
    """
    Returns a dictionary relating fractional q-point label strings to their
    coordinates e.g. '1/4 1/2 1/4' = [0.25, 0.5, 0.25]. Used for labelling
    q-points when the space group can't be calculated
    """
    label_strings = ['0', '1/4', '3/4', '1/2', '1/3', '2/3', '3/8', '5/8']
    label_coords = [0., 0.25, 0.75, 0.5, 1./3., 2./3., 0.375, 0.625]

    generic_labels = {}
    for i, s1 in enumerate(label_strings):
        for j, s2 in enumerate(label_strings):
            for k, s3 in enumerate(label_strings):
                key = s1 + ' ' + s2 + ' ' + s3
                value = [label_coords[i], label_coords[j], label_coords[k]]
                generic_labels[key] = value
    return generic_labels


def get_qpt_label(qpt, point_labels):
    """
    Gets a label for a particular q-point, based on the high symmetry points
    of a particular space group. Used for labelling the dispersion plot x-axis

    Parameters
    ----------
    qpt : list of floats
        3 dimensional coordinates of a q-point
    point_labels : dictionary
        A dictionary with N entries, relating high symmetry point lables (e.g.
        'GAMMA', 'X'), to their 3-dimensional coordinates (e.g. [0.0, 0.0,
        0.0]) where N = number of high symmetry points for a particular space
        group

    Returns
    -------
    label : string
        The label for this q-point. If the q-point isn't a high symmetry point
        label is just an empty string
    """

    # Normalise qpt to [0,1]
    qpt_norm = [x - math.floor(x) for x in qpt]

    # Split dict into keys and values so labels can be looked up by comparing
    # q-point coordinates with the dict values
    labels = list(point_labels)
    # Ensure symmetry points in label_keys and label_values are in the same
    # order (not guaranteed if using .values() function)
    label_coords = [point_labels[x] for x in labels]

    # Check for matching symmetry point coordinates (roll q-point coordinates
    # if no match is found)
    TOL = 1e-6
    matching_label_index = np.where((np.isclose(
        label_coords, qpt_norm, atol=TOL)).all(axis=1))[0]
    if matching_label_index.size == 0:
        matching_label_index = np.where((np.isclose(
            label_coords, np.roll(qpt_norm, 1), atol=TOL)).all(axis=1))[0]
    if matching_label_index.size == 0:
        matching_label_index = np.where((np.isclose(
            label_coords, np.roll(qpt_norm, 2), atol=TOL)).all(axis=1))[0]

    label = '';
    if matching_label_index.size > 0:
        label = labels[matching_label_index[0]]

    return label


def direction_changed(qpts, tolerance=5e-6):
    """
    Takes a N length list of q-points and returns an N - 2 length list of
    booleans indicating whether the direction has changed between each pair
    of q-points
    """

    # Get vectors between q-points
    delta = np.diff(qpts, axis=0)

    # Dot each vector with the next to determine the relative direction
    dot = np.einsum('ij,ij->i', delta[1:,:], delta[:-1,:])
    # Get magnitude of each vector
    modq = np.linalg.norm(delta, axis=1)
    # Determine how much the direction has changed (dot) relative to the
    # vector sizes (modq)
    direction_changed = (np.abs(np.abs(dot) - modq[1:]*modq[:-1]) > tolerance)

    return direction_changed


def reciprocal_lattice(unit_cell):
    """
    Calculates the reciprocal lattice from a unit cell
    """

    a = np.array(unit_cell[0])
    b = np.array(unit_cell[1])
    c = np.array(unit_cell[2])

    bxc = np.cross(b, c)
    cxa = np.cross(c, a)
    axb = np.cross(a, b)

    adotbxc = np.vdot(a, bxc) # Unit cell volume
    norm = 2*math.pi/adotbxc # Normalisation factor

    astar = norm*bxc
    bstar = norm*cxa
    cstar = norm*axb

    return np.array([astar, bstar, cstar])


def plot_dispersion(abscissa, freq_up, freq_down, units, title='', xticks=None,
                    xlabels=None, fermi=[]):
    """
    Creates a Matplotlib figure of the band structure

    Parameters
    ----------
    abscissa: list of floats
        M length list of the position of each q-point along the x-axis based
        on the distance between each q-point (can be calculated by the
        calc_abscissa function), where M = number of q-points
    freq_up : list of floats
        M x N list of spin up band frequencies, where M = number of q-points
        and N = number of bands, can be empty if only spin down frequencies are
        present
    freq_down : list of floats
        M x N list of spin down band frequencies, where M = number of q-points
        and N = number of bands, can be empty if only spin up frequencies are
        present
    units : string
        String specifying the frequency units. Used for axis labels
    title : string
        The figure title. Default: ''
    xticks : list of floats
        List of floats specifying the x-axis tick label locations. Usually
        they are located where the q-point direction changes, this can be
        calculated using abscissa[qpts_with_labels], where abscissa has been
        calculated from the calc_abscissa function, and qpts_with_labels from
        the recip_space_labels function. Default: None
    xlabels : list of strings
        List of strings specifying the x-axis tick labels. Should be the same
        length as xlabels, and can be calculated using the recip_space_labels
        function. Default: None
    fermi : list of floats
        1 or 2 length list specifying the fermi energy/energies. Default: []

    Returns
    -------
    fig : Matplotlib Figure
        Figure containing subplot(s) for the plotted band structure. If there
        is a large gap between some q-points there will be multiple subplots
    """

    # Create figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)

    # Y-axis formatting
    # Replace 1/cm with cm^-1
    inverse_unit_index = units.find('/')
    if inverse_unit_index > -1:
        units = units[inverse_unit_index+1:]
        ax.set_ylabel('Energy (' + units + r'$^{-1}$)')
    else:
        ax.set_ylabel('Energy (' + units + ')')
    ax.ticklabel_format(style='sci', scilimits=(-2, 2), axis='y')

    # X-axis formatting
    # Set high symmetry point x-axis ticks/labels
    if xticks is not None:
        ax.set_xticks(xticks)
        ax.xaxis.grid(True, which='major')
        # Set labels (rotate long tick labels)
        if len(max(xlabels, key=len)) >= 11:
            ax.set_xticklabels(xlabels, rotation=90)
        else:
            ax.set_xticklabels(xlabels)
    ax.set_xlim(left=0)

    # Plot frequencies and Fermi energy
    if len(freq_up) > 0:
        ax.plot(abscissa, freq_up, lw=1.0)
    if len(freq_down) > 0:
        ax.plot(abscissa, freq_down, lw=1.0)
    if len(fermi) > 0:
        for i, ef in enumerate(fermi):
            if i == 0:
                ax.axhline(y=ef, ls='dashed', c='k', label=r'$\epsilon_F$')
            else:
                ax.axhline(y=ef, ls='dashed', c='k')
        ax.legend()
    plt.tight_layout()

    return fig


def plot_dos(dos, dos_down, bins, units, title='', fermi=[], mirror=False):
    """
    Creates a Matplotlib figure of the density of states

    Parameters
    ----------
    dos : list of floats
        L - 1 length list of the spin up density of states for each bin, where
        L is the lengh of the bins parameter. Can be empty if only spin down
        frequencies are present
    dos_down : list of floats
        L - 1 length list of the spin down density of states for each bin,
        where L is the lengh of the bins parameter. Can be empty if only spin
        up frequencies are present
    bins : list of floats
        One dimensional list of the energy bin edges
    units : string
        String specifying the energy bin units. Used for axis labels
    title : string
        The figure title. Default: ''
    fermi : list of floats
        1 or 2 length list specifying the fermi energy/energies. Default: []
    mirror : boolean
        Whether to reflect the dos_down frequencies in the x-axis.
        Default: False

    Returns
    -------
    fig : Matplotlib Figure
        Figure containing the subplot containing the plotted density of states
    """

    # Create figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)

    # X-axis label formatting
    # Replace 1/cm with cm^-1
    inverse_unit_index = units.find('/')
    if inverse_unit_index > -1:
        units = units[inverse_unit_index+1:]
        ax.set_xlabel('Energy (' + units + r'$^{-1}$)')
    else:
        ax.set_xlabel('Energy (' + units + ')')
    ax.set_ylabel('g(E)')
    ax.minorticks_on()

    # Calculate bin centres
    bwidth = bins[1] - bins[0]
    bin_centres = bins[:-1] + bwidth/2

    # Plot dos and Fermi energy
    if len(dos) > 0:
        ax.plot(bin_centres, dos, label='alpha', lw=1.0)
    if len(dos_down) > 0:
        if mirror:
            ax.plot(bin_centres, -dos_down, label='beta', lw=1.0)
            ax.axhline(y=0, c='k', lw=1.0)
        else:
            ax.plot(bin_centres, dos_down, label='beta', lw=1.0)
    if len(fermi) > 0:
        for i, ef in enumerate(fermi):
            if i == 0:
                ax.axvline(x=ef, ls='dashed', c='k', label=r'$\epsilon_F$')
            else:
                ax.axvline(x=ef, ls='dashed', c='k')
    ax.legend()
    if not mirror:
        ax.set_ylim(bottom=0) # Need to set limits after plotting the data
    plt.tight_layout()

    return fig


if __name__ == '__main__':
    main()
