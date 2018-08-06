import os
import numpy as np
from casteppy import ureg
from casteppy.data.data import Data

class BandsData(Data):
    """
    A class to read and store data from a .bands file

    Attributes
    ----------
    n_qpts : int
        Number of k-points in the .bands file
    n_spins : int
        Number of spin components
    n_branches : int
        Number of electronic dispersion branches
    fermi : list of floats
        List of length 1 or 2 containing the Fermi energy/energies. Default
        units eV
    cell_vec : list of floats
        3 x 3 list of the unit cell vectors. Default units Angstroms
    qpts : list of floats
        M x 3 list of k-point coordinates, where M = number of k-points
    weights : list of floats
        List of length M containing the weight for each k-point, where
        M = number of k-points
    freqs : list of floats
        M x N list of spin up band frequencies, where
        M = number of k-points and N = number of bands, ordered according to
        increasing k-point number. Default units eV.
    freq_down : list of floats
        M x N list of spin down band frequencies, where
        M = number of k-points and N = number of bands, ordered according to
        increasing k-point number. Can be empty if there are no spin down
        frequencies present in .bands file. Default units eV.
    """


    def __init__(self, seedname, path=''):
        """"
        Reads .bands file and sets attributes

        Parameters
        ----------
        seedname : str
            Name of .bands file to read
        path : str, optional
            Path to dir containing the .bands file, if it is in another 
            directory
        """
        self._get_data(seedname, path)


    def _get_data(self, seedname, path=''):
        """"
        Open.bands file for reading

        Parameters
        ----------
        seedname : str
            Name of .bands file to read
        path : str, optional
            Path to dir containing the .bands file, if it is in another 
            directory
        """
        file = os.path.join(path, seedname + '.bands')
        with open(file, 'r') as f:
            self._read_bands_data(f)

        # Try to get extra data (ionic species, coords) from .castep file
        castep_file = os.path.join(path, seedname + '.castep')
        try:
            with open(castep_file, 'r') as f:
                self._read_castep_data(f)
        except IOError:
            pass


    def _read_bands_data(self, f):
        """
        Reads data from .bands file and sets attributes

        Parameters
        ----------
        f : file object
            File object in read mode for the .bands file containing the data
        """
        n_qpts = int(f.readline().split()[3])
        n_spins = int(f.readline().split()[4])
        f.readline() # Skip number of electrons line
        n_branches = int(f.readline().split()[3])
        fermi = [float(x) for x in f.readline().split()[5:]]
        f.readline() # Skip unit cell vectors line
        cell_vec = [[float(x) for x in f.readline().split()[0:3]]
            for i in range(3)]

        freqs = np.array([])
        freq_down = np.array([])
        freqs_qpt = np.zeros(n_branches)
        qpts = np.zeros((n_qpts, 3))
        weights = np.zeros(n_qpts)

        # Need to loop through file using while rather than number of k-points
        # as sometimes points are duplicated
        first_qpt = True
        line = f.readline().split()
        while line:
            qpt_num = int(line[1]) - 1
            qpts[qpt_num,:] = [float(x) for x in line[2:5]]
            weights[qpt_num] = float(line[5])

            for j in range(n_spins):
                spin = int(f.readline().split()[2])
    
                # Read frequencies
                for k in range(n_branches):
                    freqs_qpt[k] = float(f.readline())

                # Allocate spin up freqs as long as -down hasn't been specified
                if spin == 1:
                    if first_qpt:
                        freqs = np.zeros((n_qpts, n_branches))
                    freqs[qpt_num, :] = freqs_qpt
                # Allocate spin down freqs as long as -up hasn't been specified
                elif spin == 2:
                    if first_qpt:
                        freq_down = np.zeros((n_qpts, n_branches))
                    freq_down[qpt_num, :] = freqs_qpt

            first_qpt = False
            line = f.readline().split()

        freqs = freqs*ureg.hartree
        freqs.ito('eV')
        freq_down = freq_down*ureg.hartree
        freq_down.ito('eV')
        fermi = fermi*ureg.hartree
        fermi.ito('eV')
        cell_vec = cell_vec*ureg.bohr
        cell_vec.ito('angstrom')

        self.n_qpts = n_qpts
        self.n_spins = n_spins
        self.n_branches = n_branches
        self.fermi = fermi
        self.cell_vec = cell_vec
        self.qpts = qpts
        self.weights = weights
        self.freqs = freqs
        self.freq_down = freq_down


    def _read_castep_data(self, f):
        """
        Reads extra data from .castep file (ionic species, coords) and sets
        attributes

        Parameters
        ----------
        f : file object
            File object in read mode for the .castep file containing the data
        """
        n_ions_read = False
        ion_info_read = False
        line = f.readline()
        while line:
            if all([n_ions_read, ion_info_read]):
                break
            if 'Total number of ions in cell' in line:
                 n_ions = int(line.split()[-1])
                 n_ions_read = True
            if 'Fractional coordinates of atoms' in line:
                 f.readline() # Skip uvw line
                 f.readline() # Skip --- line
                 ion_info = [f.readline().split() for i in range(n_ions)]
                 ion_r = np.array([[float(x) for x in line[-4:-1]] 
                                    for line in ion_info])
                 ion_type = np.array([x[1] for x in ion_info])
            line = f.readline()

        self.n_ions = n_ions
        self.ion_r = ion_r
        self.ion_type = ion_type

    def convert_e_units(self, units):
        super(BandsData, self).convert_e_units(units)
        self.freq_down.ito(units, 'spectroscopy')
        self.fermi.ito(units, 'spectroscopy')