import math
import numpy as np
from euphonic import ureg
from euphonic.util import gaussian, lorentzian


class Data(object):
    """
    A general superclass for both vibrational and electronic data. Has
    functions that can apply to both vibrational and electronic data (e.g.
    CASTEP .bands and .phonon file data)
    """

    @property
    def dos_bins(self):
        return self._dos_bins*ureg('E_h').to(self._e_units, 'spectroscopy')

    def calculate_dos(self, dos_bins, gwidth=0, lorentz=False, weights=None,
                      _freqs=None):
        """
        Calculates a density of states with fixed width Gaussian/Lorentzian
        broadening

        Parameters
        ----------
        dos_bins : (n_ebins + 1,) float ndarray
            The energy bin edges to use for calculating the DOS, in the same
            units as freqs
        gwidth : float, optional, default 0
            FWHM of Gaussian/Lorentzian for broadening the DOS bins, in the
            same units as freqs
        lorentz : boolean, optional
            Whether to use a Lorentzian or Gaussian broadening function.
            Default: False
        weights : (n_qpts, n_branches) float ndarray, optional
            The weights to use for each q-points and branch. If unspecified,
            uses the q-point weights stored in the Data object

        Returns
        -------
        dos : (n_ebins,) float ndarray
            The density of states for each bin
        """

        # Use freqs in data if _freqs aren't specified. This allows BandsData
        # to calculate DOS for freqs_down too
        if _freqs is not None:
            freqs = _freqs
        else:
            freqs = self._freqs

        # Convert dos_bins to Hartree. If no units are specified, assume
        # dos_bins is in same units as freqs
        try:
            dos_bins = dos_bins.to('E_h', 'spectroscopy').magnitude
        except AttributeError:
            dos_bins = (dos_bins*ureg(self._e_units).to('E_h', 'spectroscopy')).magnitude
        try:
            gwidth = gwidth.to('E_h', 'spectroscopy').magnitude
        except AttributeError:
            gwidth = (gwidth*ureg(self._e_units).to('E_h', 'spectroscopy')).magnitude

        # Bin frequencies
        if weights is None:
            weights = np.repeat(self.weights[:, np.newaxis], self.n_branches,
                                axis=1)
        hist, bin_edges = np.histogram(freqs, dos_bins, weights=weights)

        bwidth = np.mean(np.diff(dos_bins))
        # Only broaden if broadening is more than bin width
        if gwidth > bwidth:
            # Calculate broadening for adjacent nbin_broaden bins
            if lorentz:
                # 25 * Lorentzian FWHM
                nbin_broaden = int(math.floor(25.0*gwidth/bwidth))
                broadening = lorentzian(
                    np.arange(-nbin_broaden, nbin_broaden)*bwidth, gwidth)
            else:
                # 3 * Gaussian FWHM
                nbin_broaden = int(math.floor(3.0*gwidth/bwidth))
                sigma = gwidth/(2*math.sqrt(2*math.log(2)))
                broadening = gaussian(
                    np.arange(-nbin_broaden, nbin_broaden)*bwidth, sigma)

            if hist.size > 0:
                # Allow broadening beyond edge of bins
                dos = np.zeros(len(hist) + 2*nbin_broaden)
                for i, h in enumerate(hist):
                    # Broaden each hist bin value to adjacent bins
                    bhist = h*broadening
                    dos[i:i+2*nbin_broaden] += bhist
                # Slice dos array to same size as bins
                dos = dos[nbin_broaden:-nbin_broaden]
        else:
            dos = hist

        # Don't set self.dos if this is for freqs_down (i.e. if _freqs has
        # been specified)
        if _freqs is None:
            self.dos = dos
            self._dos_bins = dos_bins

        return dos

    def convert_e_units(self, units):
        """
        Redefine the units to be used when displaying values with energy units
        e.g. freqs

        Parameters
        ----------
        units : str
            The units to use e.g. '1/cm', 'hartree', 'eV'
        """
        self._e_units = units

    def convert_l_units(self, units):
        """
        Redefine the units to be used when displaying value with length units
        e.g. cell_vec

        Parameters
        ----------
        units : str
            The units to use e.g. 'angstrom', 'bohr'
        """
        self._l_units = units
