import math
import numpy as np
from simphony.util import gaussian, lorentzian


class Data(object):
    """
    A general superclass for both vibrational and electronic data. Has
    functions that can apply to both vibrational and electronic data (e.g.
    CASTEP .bands and .phonon file data)
    """

    def calculate_dos(self, dos_bins, gwidth, lorentz=False, weights=None,
                      set_attrs=True, _freqs=None):
        """
        Calculates a density of states with fixed width Gaussian/Lorentzian
        broadening

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
            Whether to set the dos and dos_bins attributes to the newly
            calculated values

        Returns
        -------
        dos : (n_ebins,) float ndarray
            The density of states for each bin
        """

        # Convert quantities to magnitudes
        # Use freqs in data if _freqs aren't specified
        try:
            freqs = _freqs.magnitude
        except AttributeError:
            freqs = self.freqs.magnitude
        # Convert dos_bins units to same units as freqs. If no units are
        # specified, assume dos_bins is in same units as freqs
        try:
            dos_bins = dos_bins.to(self.freqs.units).magnitude
        except AttributeError:
            pass
        try:
            gwidth = gwidth.to(self.freqs.units).magnitude
        except AttributeError:
            pass

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

        if set_attrs:
            self.dos = dos
            self.dos_bins = dos_bins*self.freqs.units

        return dos

    def convert_e_units(self, units):
        """
        Convert energy units of relevant attributes in place e.g. dos_bins

        Parameters
        ----------
        units : str
            The units to convert to e.g. '1/cm', 'hartree', 'eV'
        """
        self.freqs.ito(units, 'spectroscopy')
        if hasattr(self, 'dos_bins'):
            self.dos_bins.ito(units, 'spectroscopy')

    def convert_l_units(self, units):
        """
        Convert length units of relevant attributes in place e.g. cell_vec

        Parameters
        ----------
        units : str
            The units to convert to e.g. 'angstrom', 'bohr'
        """
        self.cell_vec.ito(units)
