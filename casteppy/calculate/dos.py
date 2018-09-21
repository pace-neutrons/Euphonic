import math
import numpy as np


def calculate_dos(data, bwidth, gwidth, lorentz=False):
    """
    Calculates a density of states with fixed width Gaussian/Lorentzian 
    broadening from a PhononData or BandsData object and sets the bins and
    dos/dos_down attributes

    Parameters
    ----------
    data: PhononData or BandsData object
        Data object containing the frequencies, weights, and optionally IR
        intensities for bin weighting
    bwidth : float
        Width of each bin for the DOS, in the same units as freqs/freq_down
    gwidth : float
        FWHM of Gaussian/Lorentzian for broadening the DOS bins, in the same
        units as freqs/freq_down. Set to 0 if
        no broadening is desired
    lorentz : boolean, optional
        Whether to use a Lorentzian or Gaussian broadening function.
        Default: False

    Returns
    -------
    bins : ndarray
        One dimensional list of the energy bin edges, in the same units as
        freqs/freq_down and determined by the max/min values of freqs/freq_down
        dtype = 'float'
        shape = (n_bins + 1,)
    dos : ndarray
        The spin up density of states for each bin
        dtype = 'float'
        shape = (n_bins,)
    dos_down : ndarray
        The spin down density of states for each bin. Can be empty if only
        spin up frequencies are present
        dtype = 'float'
        shape = (n_bins,)

    """
    hist = np.array([])
    hist_down = np.array([])
    dos = np.array([])
    dos_down = np.array([])

    # Calculate bin edges
    if hasattr(data, 'freq_down') and len(data.freq_down) > 0:
        all_freqs = np.append(data.freqs, data.freq_down)
    else:
        all_freqs = data.freqs.magnitude
    freq_max = np.amax(all_freqs)
    freq_min = np.amin(all_freqs)
    bins = np.arange(freq_min, freq_max + bwidth, bwidth)

    # Calculate weight for each q-point and branch
    freq_weights = np.repeat(np.array(data.weights)[:,np.newaxis],
                             data.n_branches, axis=1)
    if hasattr(data, 'ir') and len(data.ir) > 0:
        freq_weights *= data.ir

    # Bin frequencies
    hist, bin_edges = np.histogram(data.freqs, bins,
                                   weights=freq_weights)
    if hasattr(data, 'freq_down') and len(data.freq_down) > 0:
        hist_down, bin_edges = np.histogram(data.freq_down, bins,
                                            weights=freq_weights)

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
        if hist_down.size > 0:
            dos_down = np.zeros(len(hist_down) + 2*nbin_broaden)
            for i, h in enumerate(hist_down):
                bhist = h*broadening
                dos_down[i:i+2*nbin_broaden] += bhist
            dos_down = dos_down[nbin_broaden:-nbin_broaden]

    else:
        dos = hist
        dos_down = hist_down

    bins = bins*data.freqs.units

    data.dos = dos
    data.dos_bins = bins
    if hasattr(data, 'freq_down'):
        data.dos_down = dos_down

    return dos, dos_down, bins


def gaussian(x, sigma):
    return np.exp(-np.square(x)/(2*sigma**2))/(math.sqrt(2*math.pi)*sigma)


def lorentzian(x, gamma):
    return gamma/(2*math.pi*(np.square(x) + (gamma/2)**2))
