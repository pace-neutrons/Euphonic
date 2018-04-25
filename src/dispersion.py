"""
Parse a *.castep, *.phonon or *.band output file from new CASTEP for vibrational frequency 
data and output a matplotlib plot of the electronic or vibrational band structure or dispersion.
"""
import numpy as np
import argparse

def main():
    args = parse_arguments()
    read_dot_bands(args.filename)

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
    qpts: list of floats
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
    qpts = np.zeros((n_kpts, 3))
    weights = np.zeros(n_kpts)

    for i in range(n_kpts):
        line = f.readline().split()
        kpt = int(line[1]) - 1
        qpts[kpt,:] = [float(x) for x in line[2:5]]
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
    return freq_up, freq_down, qpts, fermi, weights, cell

main()
