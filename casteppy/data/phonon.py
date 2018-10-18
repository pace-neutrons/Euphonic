import os
import sys
import numpy as np
from casteppy import ureg
from casteppy.util import direction_changed
from casteppy.data.data import Data


class PhononData(Data):
    """
    A class to read and store data from a .phonon file

    Attributes
    ----------
    seedname : str
        Seedname specifying .phonon file to read from
    n_ions : int
        Number of ions in the unit cell
    n_branches : int
        Number of phonon dispersion branches
    n_qpts : int
        Number of q-points in the .phonon file
    cell_vec : ndarray
        The unit cell vectors. Default units Angstroms.
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
    qpts : ndarray
        Q-point coordinates
        dtype = 'float'
        shape = (n_qpts, 3)
    weights : ndarray
        The weight for each q-point
        dtype = 'float'
        shape = (n_qpts,)
    freqs: ndarray
        Phonon frequencies, ordered according to increasing q-point
        number. Default units eV
        dtype = 'float'
        shape = (n_qpts, 3*n_ions)
    ir: ndarray
        IR intensities, empty if no IR intensities in .phonon file
        dtype = 'float'
        shape = (n_qpts, 3*n_ions)
    raman: ndarray
        Raman intensities, empty if no Raman intensities in .phonon file
        dtype = 'float'
        shape = (n_qpts, 3*n_ions)
    eigenvecs: ndarray
        M x N x L x 3 list of the atomic displacements (dynamical matrix
        eigenvectors), where M = number of q-points, N = number of branches
        and L = number of ions. Empty if read_eigenvecs is False
        dtype = 'complex'
        shape = (n_qpts, 3*n_ions, n_ions, 3)
    """


    def __init__(self, seedname, path='', read_eigenvecs=True, read_ir=True,
                 read_raman=True):
        """"
        Reads .phonon file and sets attributes

        Parameters
        ----------
        seedname : str
            Name of .phonon file to read
        path : str, optional
            Path to dir containing the .phonon file, if it is in another 
            directory
        read_eigenvecs : boolean, optional
            Whether to read and store the eigenvectors from the .phonon file.
            Default: True
        read_ir : boolean, optional
            Whether to read and store IR intensities from the .phonon file.
            Default: True
        read_raman : boolean, optional
            Whether to read and store Raman intensities from the .phonon file.
            Default: True
        """
        self._get_data(seedname, path, read_eigenvecs, read_ir, read_raman)
        self.seedname = seedname


    def _get_data(self, seedname, path, read_eigenvecs, read_ir, read_raman):
        """"
        Opens .phonon file for reading

        Parameters
        ----------
        seedname : str
            Name of .phonon file to read
        path : str
            Path to dir containing the .phonon file, if it is in another 
            directory
        read_eigenvecs : boolean
            Whether to read and store the eigenvectors from the .phonon file
        read_ir : boolean
            Whether to read and store IR intensities from the .phonon file
        read_raman : boolean
            Whether to read and store Raman intensities from the .phonon file
        """
        file = os.path.join(path, seedname + '.phonon')
        with open(file, 'r') as f:
            self._read_phonon_data(f, read_eigenvecs, read_ir, read_raman)


    def _read_phonon_data(self, f, read_eigenvecs, read_ir, read_raman):
        """
        Reads data from .phonon file and sets attributes

        Parameters
        ----------
        f : file object
            File object in read mode for the .phonon file containing the data
        read_eigenvecs : boolean
            Whether to read and store the eigenvectors from the .phonon file
        read_ir : boolean
            Whether to read and store IR intensities from the .phonon file
        read_raman : boolean
            Whether to read and store Raman intensities from the .phonon file
        """
        (n_ions, n_branches, n_qpts, cell_vec, ion_r,
        ion_type, ion_mass) = self._read_phonon_header(f)

        qpts = np.zeros((n_qpts, 3))
        weights = np.zeros(n_qpts)
        freqs = np.zeros((n_qpts, n_branches))
        ir = np.array([])
        raman = np.array([])
        if read_eigenvecs:
            eigenvecs = np.zeros((n_qpts, n_branches, n_ions, 3),
                                 dtype='complex128')
        else:
            eigenvecs = np.array([])

        # Need to loop through file using while rather than number of q-points
        # as sometimes points are duplicated
        first_qpt = True
        line = f.readline().split()
        while line:
            qpt_num = int(line[1]) - 1
            qpts[qpt_num,:] = [float(x) for x in line[2:5]]
            weights[qpt_num] = float(line[5])

            freq_lines = [f.readline().split() for i in range(n_branches)]
            freqs[qpt_num, :] = [float(line[1]) for line in freq_lines]
            ir_index = 2
            raman_index = 3
            if np.all(qpts[qpt_num] == 0.):
                ir_index += 1
                raman_index += 1
            if read_ir and len(freq_lines[0]) > ir_index:
                if first_qpt:
                    ir = np.zeros((n_qpts, n_branches))
                ir[qpt_num, :] = [float(
                    line[ir_index]) for line in freq_lines]
            if read_raman and len(freq_lines[0]) > raman_index:
                if first_qpt:
                     raman = np.zeros((n_qpts, n_branches))
                raman[qpt_num, :] = [float(
                    line[raman_index]) for line in freq_lines]

            if read_eigenvecs:
                [f.readline() for x in range(2)]  # Skip 2 label lines
                lines = np.array([f.readline().split()[2:]
                    for x in range(n_ions*n_branches)]).astype(float)
                lines_i = np.column_stack(([lines[:, 0] + lines[:, 1]*1j,
                                            lines[:, 2] + lines[:, 3]*1j,
                                            lines[:, 4] + lines[:, 5]*1j]))
                for i in range(n_branches):
                    eigenvecs[qpt_num, i, :, :] = lines_i[
                        i*n_ions:(i+1)*n_ions, :]
            else:
                # Don't bother reading eigenvectors
                # Skip eigenvectors and 2 label lines
                [f.readline() for x in range(n_ions*n_branches + 2)]
            first_qpt = False
            line = f.readline().split()

        cell_vec = cell_vec*ureg.angstrom
        ion_mass = ion_mass*ureg.amu
        freqs = freqs*(1/ureg.cm)
        freqs.ito('eV', 'spectroscopy')

        self.n_ions = n_ions
        self.n_branches = n_branches
        self.n_qpts = n_qpts
        self.cell_vec = cell_vec
        self.ion_r = ion_r
        self.ion_type = ion_type
        self.ion_mass = ion_mass
        self.qpts = qpts
        self.weights = weights
        self.freqs = freqs
        self.ir = ir
        self.raman = raman
        self.eigenvecs = eigenvecs


    def _read_phonon_header(self, f):
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
        cell_vec : ndarray
            The unit cell vectors. Default units Angstroms.
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


    def convert_e_units(self, units):
        """
        Convert energy units of relevant attributes in place e.g. freqs,
        dos_bins

        Parameters
        ----------
        units : str
            The units to convert to e.g. '1/cm', 'hartree', 'eV'
        """
        super(PhononData, self).convert_e_units(units)
        self.freqs.ito(units, 'spectroscopy')
