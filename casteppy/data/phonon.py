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
    n_ions : int
        Number of ions in the unit cell
    n_branches : int
        Number of phonon dispersion branches
    n_qpts : int
        Number of q-points in the .phonon file
    cell_vec : list of floats
        3 x 3 list of the unit cell vectors. Default units Angstroms.
    ion_r : list of floats
        n_ions x 3 list of the fractional position of each ion within the
        unit cell
    ion_type : list of strings
        n_ions length list of the chemical symbols of each ion in the unit
        cell. Ions are in the same order as in ion_r
    ion_mass : list of floats
        n_ions length list of the mass of each ion in the unit cell in atomic
        units. Default units amu.
    qpts : list of floats
        M x 3 list of q-point coordinates, where M = number of q-points
    weights : list of floats
        List of length M containing the weight for each q-point, where
        M = number of q-points
    freqs: list of floats
        M x N list of phonon frequencies, where M = number of q-points
        and N = number of branches, ordered according to increasing q-point
        number. Default units eV.
    ir: list of floats
        M x N list of IR intensities, where M = number of q-points and
        N = number of branches. Empty if no IR intensities in .phonon file
    raman: list of floats
        M x N list of Raman intensities, where M = number of q-points and
        N = number of branches. Empty if no Raman intensities in .phonon file
    eigenvecs: list of complex floats
        M x N x L x 3 list of the atomic displacements (dynamical matrix
        eigenvectors), where M = number of q-points, N = number of branches
        and L = number of ions. Empty if read_eigenvecs is False
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
        cell_vec : list of floats
            3 x 3 list of the unit cell vectors
        ion_r : list of floats
            n_ions x 3 list of the fractional position of each ion within the
            unit cell
        ion_type : list of strings
            n_ions length list of the chemical symbols of each ion in the unit
            cell. Ions are in the same order as in ion_r
        """
        f.readline()  # Skip BEGIN header
        n_ions = int(f.readline().split()[3])
        n_branches = int(f.readline().split()[3])
        n_qpts = int(f.readline().split()[3])
        [f.readline() for x in range(4)]  # Skip units and label lines
        cell_vec = [[float(x) for x in f.readline().split()[0:3]]
            for i in range(3)]
        f.readline()  # Skip fractional co-ordinates label
        ion_info = [f.readline().split() for i in range(n_ions)]
        ion_r = [[float(x) for x in y[1:4]] for y in ion_info]
        ion_type = [x[4] for x in ion_info]
        ion_mass = [float(x[5]) for x in ion_info]
        f.readline()  # Skip END header line

        return n_ions, n_branches, n_qpts, cell_vec, ion_r, ion_type, ion_mass


    def reorder_freqs(self):
        """
        Reorders frequencies across q-points in order to join branches, and sets
        the freqs attribute to the newly ordered frequencies
        """
        n_qpts = self.n_qpts
        n_branches = self.n_branches
        n_ions = self.n_ions
        qpts = self.qpts
        freqs = self.freqs
        eigenvecs = self.eigenvecs

        if eigenvecs.size == 0:
            print("""No eigenvectors in PhononData object, cannot reorder
	             frequencies""")
            return

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
                # Explicitly broadcast arrays with repeat and tile to ensure
                # correct multiplication of modes
                current_eigenvecs = np.repeat(eigenvecs[i, :, :, :],
                                              n_branches, axis=0)
                prev_eigenvecs = np.tile(eigenvecs[i-1, :, :, :],
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

        ordered_freqs = ordered_freqs*freqs.units
        self.freqs = ordered_freqs

    def convert_e_units(self, units):
        super(PhononData, self).convert_e_units(units)
        self.freqs.ito(units, 'spectroscopy')
