import numpy as np
from euphonic.data.data import Data
from euphonic._readers import _castep


class BandsData(Data):
    """
    A class to read and store electronic data from model (e.g. CASTEP) output
    files

    Attributes
    ----------
    seedname : str
        Seedname specifying file(s) to read from
    model : str
        Records what model the data came from
    n_qpts : int
        Number of k-points
    n_spins : int
        Number of spin components
    n_branches : int
        Number of electronic dispersion branches
    fermi : (n_spins,) float ndarray
        The Fermi energy/energies. Default units eV
    cell_vec : (3, 3) float ndarray
        The unit cell vectors. Default units Angstroms
    recip_vec : (3, 3) float ndarray
        The reciprocal lattice vectors. Default units inverse Angstroms
    qpts : (n_qpts, 3) float ndarray
        K-point coordinates
    weights : (n_qpts,) float ndarray
        The weight for each k-point
    freqs: (n_qpts, 3*n_ions) float ndarray
        Band frequencies, ordered according to increasing k-point
        number. Default units eV
    freq_down: (n_qpts, 3*n_ions) float ndarray
        Spin down band frequencies, ordered according to increasing k-point
        number. Can be empty if there are no spin down frequencies present.
        Default units eV
    """

    def __init__(self, seedname, model='CASTEP', path=''):
        """"
        Calls functions to read the correct file(s) and sets BandsData
        attributes

        Parameters
        ----------
        seedname : str
            Seedname of file(s) to read
        model : {'CASTEP'}, optional, default 'CASTEP'
            Which model has been used. e.g. if seedname = 'Fe' and
            model='CASTEP', the 'Fe.bands' file will be read
        path : str, optional
            Path to dir containing the file(s), if in another directory
        """
        self._get_data(seedname, model, path)
        self.seedname = seedname
        self.model = model

    def _get_data(self, seedname, model, path):
        """"
        Calls the correct reader to get the required data, and sets the
        BandsData attributes

        Parameters
        ----------
        seedname : str
            Seedname of file(s) to read
        model : {'CASTEP'}, optional, default 'CASTEP'
            Which model has been used. e.g. if seedname = 'Fe' and
            model='CASTEP', the 'Fe.bands' file will be read
        path : str
            Path to dir containing the file(s), if in another directory
        """
        if model.lower() == 'castep':
            data = _castep._read_bands_data(seedname, path)
        else:
            raise ValueError(
                "{:s} is not a valid model, please use one of {{'CASTEP'}}"
                .format(model))

        self.n_qpts = data['n_qpts']
        self.n_spins = data['n_spins']
        self.n_branches = data['n_branches']
        self.fermi = data['fermi']
        self.cell_vec = data['cell_vec']
        self.qpts = data['qpts']
        self.weights = data['weights']
        self.freqs = data['freqs']
        self.freq_down = data['freq_down']

        try:
            self.n_ions = data['n_ions']
            self.ion_r = data['ion_r']
            self.ion_type = data['ion_type']
        except KeyError:
            pass

    def calculate_dos(self, dos_bins, gwidth, lorentz=False,
                      weights=None, set_attrs=True):
        """
        Calculates a density of states with fixed width Gaussian/Lorentzian
        broadening. Extends the Data.calculate_dos function and has the same
        input parameters, but additionally calculates DOS for spin down
        frequencies

        Parameters
        ----------
        dos_bins : (n_ebins + 1,) float ndarray
            The energy bin edges to use for calculating the DOS, in the same
            units as freqs
        gwidth : float
            FWHM of Gaussian/Lorentzian for broadening the DOS bins, in the
            same units as freqs. Set to 0 if
            no broadening is desired
        lorentz : boolean, optional
            Whether to use a Lorentzian or Gaussian broadening function.
            Default: False
        weights : (n_qpts, n_branches) float ndarray, optional
            The weights to use for each q-points and branch. If unspecified,
            uses the q-point weights stored in the Data object
        set_attrs : boolean, optional, default True
            Whether to set the dos, dos_down and dos_bins attributes to the
            newly calculated values

        Returns
        -------
        dos : (n_ebins,) float ndarray
            The spin up density of states for each bin
        dos_down : (n_ebins,) float ndarray
            The spin down density of states for each bin
        """
        dos = super(BandsData, self).calculate_dos(
            dos_bins, gwidth, lorentz=lorentz, weights=weights,
            set_attrs=set_attrs)

        if self.freq_down.size > 0:
            dos_down = super(BandsData, self).calculate_dos(
                dos_bins, gwidth, lorentz=lorentz, weights=weights,
                set_attrs=False, _freqs=self.freq_down)
        else:
            dos_down = np.array([])
        if set_attrs:
            self.dos_down = dos_down

        return dos, dos_down

    def convert_e_units(self, units):
        """
        Convert energy units of relevant attributes in place e.g. freqs,
        dos_bins

        Parameters
        ----------
        units : str
            The units to convert to e.g. 'hartree', 'eV'
        """
        super(BandsData, self).convert_e_units(units)
        self.freq_down.ito(units, 'spectroscopy')
        self.fermi.ito(units, 'spectroscopy')
