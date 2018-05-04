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
from pint import UnitRegistry


def main():
    args = parse_arguments()

    with open(args.filename, 'r') as f:
        if args.filename.endswith('.bands'):
            freq_up, freq_down, kpts, fermi, weights, cell = read_dot_bands(
                f, args.up, args.down, args.units)
        elif args.filename.endswith('.phonon'):
            freq_up, kpts, cell, cell_pos = read_dot_phonon(f, args.units)
            freq_down = np.array([])
        else:
            sys.exit('Error: Please supply a .bands or .phonon file')

    recip_latt = reciprocal_lattice(cell)
    abscissa = calc_abscissa(kpts, recip_latt)

    plot_dispersion(abscissa, freq_up, freq_down, args.filename, args.units)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="""Extract phonon or bandstructure data from .castep,
                       .phonon or .bands files and prepare a band structure
                       plot with matplotlib""")
    parser.add_argument(
        'filename',
        help="""The .castep, .phonon or .bands file to extract the
                bandstructure data from""")
    parser.add_argument(
        '-v',
        action='store_true',
        help='Be verbose about progress')
    parser.add_argument(
        '-bs',
        action='store_true',
        help='Read band-structure from *.castep or *.bands')
    parser.add_argument(
        '-units',
        default='eV',
        help='Convert frequencies to specified units for plotting')

    spin_group = parser.add_mutually_exclusive_group()
    spin_group.add_argument(
        '-up',
        action='store_true',
        help='Extract and plot only spin up from *.castep or *.bands')
    spin_group.add_argument(
        '-down',
        action='store_true',
        help='Extract and plot only spin down from *.castep or *.bands')

    args = parser.parse_args()
    return args


def set_up_unit_registry():
    ureg = UnitRegistry()
    ureg.define('rydberg = 13.605693009*eV = Ry') # CODATA 2014
    return ureg


def read_dot_phonon(f, units='1/cm'):
    """
    Reads band structure from a *.phonon file

    Parameters
    ----------
    f : file object
        File object in read mode for the .phonon file containing the data
    up : boolean
        Read only spin up frequencies from the .phonon file
    down : boolean
        Read only spin down frequencies from the .phonon file
    units : string
        String specifying the units of the output frequencies. For valid
        values see the Pint docs: http://pint.readthedocs.io/en/0.8.1/

    Returns
    -------
    freqs: list of floats
        M x N list of phonon band frequencies in units specified by the
        'units parameter, where M = number of q-points and N = number of
        bands, ordered according to increasing q-point number
    qpts : list of floats
        M x 3 list of q-point coordinates, where M = number of q-points
    cell : list of floats
        3 x 3 list of the unit cell vectors
    """
    n_ions, n_branches, n_qpts, cell, cell_pos = read_dot_phonon_header(f)
    qpts = np.zeros((n_qpts, 3))
    freqs = np.zeros((n_qpts, n_branches))

    # Need to loop through file using while rather than number of k-points
    # as sometimes points are duplicated
    while True:
        line = f.readline().split()
        if not line:
            break
        qpt_num = int(line[1]) - 1
        qpts[qpt_num,:] = [float(x) for x in line[2:5]]
        freqs_tmp = [float(f.readline().split()[1]) for i in range(n_branches)]
        freqs[qpt_num,:] = freqs_tmp
        # Skip eigenvectors and 2 label lines
        [f.readline() for x in range(n_ions*n_branches + 2)]

    ureg = set_up_unit_registry()
    freqs = freqs*(1/ureg.cm)
    freqs.ito(units, 'spectroscopy')

    return freqs, qpts, cell, cell_pos


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
    cell : list of floats
        3 x 3 list of the unit cell vectors
    cell_pos : list of floats
        n_ions x 3 list of the fractional position of each atom within the
        unit cell
    """
    f.readline() # Skip BEGIN header
    n_ions = int(f.readline().split()[3])
    n_branches = int(f.readline().split()[3])
    n_qpts = int(f.readline().split()[3])
    [f.readline() for x in range(4)] # Skip units and label lines
    cell = [[float(x) for x in f.readline().split()[0:3]] for i in range(3)]
    f.readline() # Skip fractional co-ordinates label
    cell_pos = [[float(x) for x in f.readline().split()[1:4]]
                 for i in range(n_ions)]
    f.readline() # Skip END header line

    return n_ions, n_branches, n_qpts, cell, cell_pos

def read_dot_bands(f, up=False, down=False, units='eV'):
    """
    Reads band structure from a *.bands file

    Parameters
    ----------
    f : file object
        File object in read mode for the .bands file containing the data
    up : boolean
        Read only spin up frequencies from the .bands file
    down : boolean
        Read only spin down frequencies from the .bands file
    units : string
        String specifying the units of the output frequencies. For valid
        values see the Pint docs: http://pint.readthedocs.io/en/0.8.1/

    Returns
    -------
    freq_up : list of floats
        M x N list of spin up band frequencies in units specified by the
        'units' parameter, where M = number of k-points and N = number of
        bands, ordered according to increasing k-point number
    freq_down : list of floats
        M x N list of spin down band frequencies in units specified by the
        'units' argument, where M = number of k-points and N = number of
        bands, ordered according to increasing k-point number
    kpts : list of floats
        M x 3 list of k-point coordinates, where M = number of k-points
    fermi : list of floats
        List of length 1 or 2 containing the Fermi energy/energies in atomic
        units
    weights : list of floats
        List of length M containing the weight for each k-point, where
        M = number of k-points
    cell : list of floats
        3 x 3 list of the unit cell vectors 
    """
    n_kpts = int(f.readline().split()[3])
    n_spins = int(f.readline().split()[4])
    f.readline() # Skip number of electrons line
    n_freqs = int(f.readline().split()[3])
    fermi = [float(x) for x in f.readline().split()[5:]]
    f.readline() # Skip unit cell vectors line
    cell = [[float(x) for x in f.readline().split()[0:3]] for i in range(3)]

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

    ureg = set_up_unit_registry()
    freq_up = freq_up*ureg.hartree
    freq_up.ito(units, 'spectroscopy')
    freq_down = freq_down*ureg.hartree
    freq_down.ito(units, 'spectroscopy')

    return freq_up, freq_down, kpts, fermi, weights, cell


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
    KVEC_TOL = 0.001
    calc_modq = np.logical_not(np.logical_and(
        np.sum(np.abs(delta), axis=1) > KVEC_TOL,
        delta_rem < KVEC_TOL))

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


def plot_dispersion(abscissa, freq_up, freq_down, filename, units):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    inverse_unit_index = units.find('/')
    if inverse_unit_index > -1:
        units = units[inverse_unit_index+1:]
        ax.set_ylabel('Energy (' + units + r'$^{-1}$)')
    else:
        ax.set_ylabel('Energy (' + units + ')')
    ax.ticklabel_format(style='sci', scilimits=(-2, 2), axis='y')
    ax.set_title(filename)

    if freq_up.size != 0:
        ax.plot(abscissa, freq_up)
    if freq_down.size != 0:
        ax.plot(abscissa, freq_down)
    plt.show()


if __name__ == '__main__':
    main()
