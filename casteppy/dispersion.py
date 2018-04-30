"""
Parse a *.castep, *.phonon or *.band output file from new CASTEP for vibrational frequency 
data and output a matplotlib plot of the electronic or vibrational band structure or dispersion.
"""

import numpy as np
import math
import argparse

def main():
    args = parse_arguments()
    freq_up, freq_down, kpts, fermi, weights, cell = read_dot_bands(args.filename)

    recip_latt = reciprocal_lattice(cell)
    abscissa = calc_abscissa(kpts, recip_latt)

def parse_arguments():
    parser = argparse.ArgumentParser(description="""Extract phonon or bandstructure data
                                                    from .castep, .phonon or .bands files
                                                    and prepare a bandstructure/dispersion
                                                    plot with matplotlib""")
    parser.add_argument("filename",
                        help="""The .castep, .phonon or .bands file to
                                extract the bandstructure data from""")
    parser.add_argument("-v",
                        help="Be verbose about progress",
                        action="store_true")
    parser.add_argument("-bs",
                        help="Read band-structure from *.castep or *.bands",
                        action="store_true")
    parser.add_argument("-up",
                        help="Extract and plot only spin up from *.castep or *.bands",
                        action="store_true")
    parser.add_argument("-down",
                        help="Extract and plot only spin down from *.castep or *.bands",
                        action="store_true")
    return parser.parse_args()

def read_dot_bands(bands_file):
    """
    Reads band structure from a *.bands file

    Parameters
    ----------
    bands_file : str
        The *.bands file containing the data

    Returns
    -------
    freq_up : list of floats
        M x N list of spin up band frequencies in eV, where M = number of k-points and
        N = number of bands, ordered according to increasing k-point number
    freq_down : list of floats
        M x N list of spin down band frequencies in eV, where M = number of k-points and
        N = number of bands, ordered according to increasing k-point number
    kpts: list of floats
        M x 3 list of k-point coordinates, where M = number of k-points
    fermi: list of floats
        List of length 1 or 2 containing the Fermi energy/energies in atomic units
    weights: list of floats
        List of length M containing the weight for each k-point, where M = number of k-points
    cell: list of floats
        3 x 3 list of the unit cell vectors 
    """
    f = open(bands_file, 'r')
    n_kpts = int(f.readline().split()[3]) # Read number of k-points
    n_spins = int(f.readline().split()[4]) # Read number of spin components
    f.readline() # Skip number of electrons line
    n_freqs = int(f.readline().split()[3]) # Read number of eigenvalues
    fermi = [float(x) for x in f.readline().split()[5:]] # Read number of eigenvalues
    f.readline() # Skip unit cell vectors line
    cell = [[float(x) for x in f.readline().split()[0:3]] for i in range(3)] # Read cell vectors

    freq_up = np.zeros((n_kpts, n_freqs))
    freq_down = np.zeros((n_kpts, n_freqs))
    kpts = np.zeros((n_kpts, 3))
    weights = np.zeros(n_kpts)

    for i in range(n_kpts):
        line = f.readline().split()
        kpt = int(line[1]) - 1
        kpts[kpt,:] = [float(x) for x in line[2:5]]
        weights[kpt] = float(line[5])

        for j in range(n_spins):
            spin = int(f.readline().split()[2])

            for k in range(n_freqs):
                freq = float(f.readline())
                if spin == 1:
                    freq_up[kpt, k] = freq
                elif spin == 2:
                    freq_down[kpt, k] = freq

    f.close()
    return freq_up, freq_down, kpts, fermi, weights, cell

def calc_abscissa(qpts, recip_latt):
    """
    Calculates the distance between q-points (for the plot x-coordinate)
    """

    # Get distance between q-points in each dimension. Note: length is nqpts - 1
    delta = np.diff(qpts, axis=0)

    # Determine how close delta is to being an integer. As q = q + G where G is a reciprocal
    # lattice vector based on {1,0,0},{0,1,0},{0,0,1}, this is used to determine whether 2
    # q-points are equivalent ie. they differ by a multiple of the reciprocal lattice vector
    delta_rem = np.sum(np.abs(delta - np.rint(delta)), axis=1)

    # Create a boolean array that determines whether to calculate the distance between q-points,
    # taking into account q-point equivalence. If delta is more than the tolerance, but delta_rem
    # is less than the tolerance, the q-points differ by G so are equivalent and the distance
    # shouldn't be calculated
    KVEC_TOL = 0.001
    calc_modq = np.logical_not(np.logical_and(np.sum(np.abs(delta), axis=1) > KVEC_TOL, delta_rem < KVEC_TOL))

    # Multiply each delta by the reciprocal lattice to get delta in cartesian coords
    deltaq = np.einsum('ji,kj->ki', recip_latt, delta)

    # Get distance between qpoints for all valid pairs of qpoints
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

if __name__ == "__main__":
    main()
