import numpy as np
from euphonic import ureg
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

    def __init__(self, data):
        """
        Calls functions to read the correct file(s) and sets BandsData
        attributes

        Parameters
        ----------
        data : dict
            A dict containing the following keys: n_qpts, n_spins,
            n_branches, fermi, cell_vec, recip_vec, qpts, weights,
            freqs, freq_down, and optional: n_ions, ion_r, ion_type.
            meta :
                model:{'CASTEP'}
                    Which model has been used
                path : str, default ''
                    Location of seed files on filesystem
            meta (CASTEP) :
                seedname : str
                    Seedname of file that is read
        """
        if type(data) is str:
            raise Exception('The old interface is now replaced by',
                            'BandsData.from_castep(seedname, path="<path>").',
                            '(Please see documentation for more information.)')

        self._set_data(data)

        self._l_units = 'angstrom'
        self._e_units = 'eV'

    @property
    def cell_vec(self):
        return self._cell_vec*ureg('bohr').to(self._l_units)

    @property
    def recip_vec(self):
        return self._recip_vec*ureg('1/bohr').to('1/' + self._l_units)

    @property
    def freqs(self):
        return self._freqs*ureg('hartree').to(self._e_units, 'spectroscopy')

    @property
    def freq_down(self):
        return self._freq_down*ureg('hartree').to(self._e_units, 'spectroscopy')

    @property
    def fermi(self):
        return self._fermi*ureg('hartree').to(self._e_units, 'spectroscopy')

    @classmethod
    def from_castep(self, seedname, path=''):
        """
        Calls the CASTEP bands data reader and sets the BandsData attributes

        Parameters
        ----------
        seedname : str
            Seedname of file(s) to read e.g. if seedname = 'Fe' then
                the 'Fe.bands' file will be read
        path : str
            Path to dir containing the file(s), if in another directory
        """
        data = _castep._read_bands_data(seedname, path)
        return self(data)

    def _set_data(self, data):
        self.n_qpts = data['n_qpts']
        self.n_spins = data['n_spins']
        self.n_branches = data['n_branches']
        self._fermi = data['fermi']
        self._cell_vec = data['cell_vec']
        self._recip_vec = data['recip_vec']
        self.qpts = data['qpts']
        self.weights = data['weights']
        self._freqs = data['freqs']
        self._freq_down = data['freq_down']

        try:
            self.n_ions = data['n_ions']
            self.ion_r = data['ion_r']
            self.ion_type = data['ion_type']
        except KeyError:
            pass

        try:
            self.model = data['model']
            if data['model'].lower() == 'castep':
                self.seedname = data['seedname']
                self.model = data['model']
                self.path = data['path']
        except KeyError:
            pass


    def calculate_dos(self, dos_bins, gwidth, lorentz=False,
                      weights=None):
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

        Returns
        -------
        dos : (n_ebins,) float ndarray
            The spin up density of states for each bin
        dos_down : (n_ebins,) float ndarray
            The spin down density of states for each bin
        """
        dos = super(BandsData, self).calculate_dos(
            dos_bins, gwidth, lorentz=lorentz, weights=weights)

        if self._freq_down.size > 0:
            dos_down = super(BandsData, self).calculate_dos(
                dos_bins, gwidth, lorentz=lorentz, weights=weights,
                _freqs=self._freq_down)
        else:
            dos_down = np.array([])

        self.dos_down = dos_down

        return dos, dos_down
