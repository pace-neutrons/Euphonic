import os
import numpy as np
from simphony.util import direction_changed
from simphony.data.data import Data
from simphony._readers import _castep


class PhononData(Data):
    """
    A class to read and store vibrational data from model (e.g. CASTEP) output
    files

    Attributes
    ----------
    seedname : str
        Seedname specifying file(s) to read from
    model : str
        Records what model the data came from
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
        number. Default units meV
        dtype = 'float'
        shape = (n_qpts, 3*n_ions)
    eigenvecs: ndarray
        Dynamical matrix eigenvectors. Empty if read_eigenvecs is False
        dtype = 'complex'
        shape = (n_qpts, 3*n_ions, n_ions, 3)
    split_i : ndarray
        The q-point indices where there is LO-TO splitting, if applicable.
        Otherwise empty.
        dtype = 'int'
        shape = (n_splits,)
    split_freqs : ndarray
        Holds the additional LO-TO split phonon frequencies for the q-points
        specified in split_i. Empty if no LO-TO splitting. Default units meV
        dtype = 'float'
        shape = (n_splits, 3*n_ions)
    split_eigenvecs : ndarray
        Holds the additional LO-TO split dynamical matrix eigenvectors for the
        q-points specified in split_i. Empty if no LO-TO splitting
        dtype = 'complex'
        shape = (n_splits, 3*n_ions, n_ions, 3)
    """

    def __init__(self, seedname, model='CASTEP', path=''):
        """"
        Calls functions to read the correct file(s) and sets PhononData
        attributes

        Parameters
        ----------
        seedname : str
            Seedname of file(s) to read
        model : {'CASTEP'}, optional, default 'CASTEP'
            Which model has been used. e.g. if seedname = 'quartz' and
            model='CASTEP', the 'quartz.phonon' file will be read
        path : str, optional
            Path to dir containing the file(s), if in another directory
        """
        self._get_data(seedname, model, path)
        self.seedname = seedname
        self.model = model

    def _get_data(self, seedname, model, path):
        """"
        Opens the correct file(s) for reading

        Parameters
        ----------
        seedname : str
            Seedname of file(s) to read
        model : {'CASTEP'}, optional, default 'CASTEP'
            Which model has been used. e.g. if seedname = 'quartz' and
            model='CASTEP', the 'quartz.phonon' file will be read
        path : str
            Path to dir containing the file(s), if in another directory
        """
        if model.lower() == 'castep':
            file = os.path.join(path, seedname + '.phonon')
            with open(file, 'r') as f:
                _castep._read_phonon_data(self, f)
        else:
            raise ValueError(
                "{:s} is not a valid model, please use one of {{'CASTEP'}}"
                .format(model))

    def reorder_freqs(self):
        """
        Reorders frequencies across q-points in order to join branches, and
        sets the freqs and eigenvecs attributes to the newly ordered
        frequencies
        """
        n_qpts = self.n_qpts
        n_branches = self.n_branches
        qpts = self.qpts
        freqs = self.freqs.magnitude
        eigenvecs = self.eigenvecs

        ordered_freqs = np.zeros(freqs.shape)
        ordered_eigenvecs = np.zeros(eigenvecs.shape, dtype=np.complex128)
        qmap = np.arange(n_branches)

        # Only calculate qmap and reorder freqs if the direction hasn't changed
        # and there is no LO-TO splitting
        calculate_qmap = np.concatenate(([True], np.logical_not(
            direction_changed(qpts))))
        if hasattr(self, 'split_i'):
            split_freqs = self.split_freqs.magnitude
            split_eigenvecs = self.split_eigenvecs
            ordered_split_freqs = np.zeros(split_freqs.shape)
            ordered_split_eigenvecs = np.zeros(
                split_eigenvecs.shape, dtype=np.complex128)
            calculate_qmap[self.split_i + 1] = False

        # Don't reorder first q-point
        ordered_freqs[0, :] = freqs[0, :]
        ordered_eigenvecs[0, :] = eigenvecs[0, :]
        prev_evecs = eigenvecs[0, :, :, :]
        for i in range(1, n_qpts):
            # Initialise q-point mapping for this q-point
            qmap_tmp = np.arange(n_branches)
            # Compare eigenvectors for each mode for this q-point with every
            # mode for the previous q-point
            # Explicitly broadcast arrays with repeat and tile to ensure
            # correct multiplication of modes
            curr_evecs = eigenvecs[i, :, :, :]
            current_eigenvecs = np.repeat(curr_evecs, n_branches, axis=0)
            prev_eigenvecs = np.tile(prev_evecs, (n_branches, 1, 1))

            if calculate_qmap[i-1]:
                # Compute complex conjugated dot product of every mode of this
                # q-point with every mode of previous q-point, and sum the dot
                # products over ions (i.e. multiply eigenvectors elementwise,
                # then sum over the last 2 dimensions)
                dots = np.absolute(np.einsum('ijk,ijk->i',
                                             np.conj(prev_eigenvecs),
                                             current_eigenvecs))

                # Create matrix of dot products for each mode of this q-point
                # with each mode of the previous q-point
                dot_mat = np.reshape(dots, (n_branches, n_branches))

                # Find greates exp(-iqr)-weighted dot product
                for j in range(n_branches):
                    max_i = (np.argmax(dot_mat))
                    mode = int(max_i/n_branches)  # Modes are dot_mat rows
                    prev_mode = max_i % n_branches  # Prev q-pt modes are cols
                    # Ensure modes aren't mapped more than once
                    dot_mat[mode, :] = 0
                    dot_mat[:, prev_mode] = 0
                    qmap_tmp[mode] = prev_mode
            # Map q-points according to previous q-point mapping
            qmap = qmap[qmap_tmp]

            prev_evecs = curr_evecs

            # Reorder frequencies and eigenvectors
            ordered_eigenvecs[i, qmap] = eigenvecs[i, :]
            ordered_freqs[i, qmap] = freqs[i, :]

            if hasattr(self, 'split_i') and i in self.split_i:
                idx = np.where(i == self.split_i)
                ordered_split_eigenvecs[idx, qmap] = split_eigenvecs[idx]
                ordered_split_freqs[idx, qmap] = split_freqs[idx]

        ordered_freqs = ordered_freqs*self.freqs.units
        self.eigenvecs = ordered_eigenvecs
        self.freqs = ordered_freqs
        if hasattr(self, 'split_i'):
            self.split_freqs = ordered_split_freqs*self.split_freqs.units
            self.split_eigenvecs = ordered_split_eigenvecs

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
        self.split_freqs.ito(units, 'spectroscopy')
        if hasattr(self, 'sqw_ebins'):
            self.sqw_ebins.ito(units, 'spectroscopy')
