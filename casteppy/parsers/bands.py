import numpy as np

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
