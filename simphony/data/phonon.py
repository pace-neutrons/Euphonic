import os
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
