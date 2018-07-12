import math
import numpy as np


def calculate_dos(freqs, freq_down, weights, bwidth, gwidth, lorentz=False, intensities=[]):
    """
    Calculates a density of states with Gaussian/Lorentzian broadening from a
    list of frequencies

    Parameters
    ----------
    freqs : list of floats
        M x N list of spin up band frequencies, where M = number of q-points
        and N = number of bands, can be empty if only spin down frequencies are
        present
    freq_down : list of floats
        M x N list of spin down band frequencies, where M = number of q-points
        and N = number of bands, can be empty if only spin up frequencies are
        present
    weights : list of floats
        List of length M containing the weight for each q-point, where
        M = number of q-points
    bwidth : float
        Width of each bin for the DOS, in the same units as freqs/freq_down
    gwidth : float
        FWHM of Gaussian/Lorentzian for broadening the DOS bins. Set to 0 if
        no broadening is desired
    lorentz : boolean
        Whether to use a Lorentzian or Gaussian broadening function.
        Default: False
    intensities : list of floats
        M x N list of IR intensities for each frequency, is used to weight the
        DOS binning in addition to the weights parameter, where M = number of
        q-points and N = number of bands. Default: []

    Returns
    -------
    dos : list of floats
        L - 1 length list of the spin up density of states for each bin, where
        L is the lengh of the bins return value. Can be empty if only spin down
        frequencies are present
    dos_down : list of floats
        L - 1 length list of the spin down density of states for each bin,
        where L is the lengh of the bins return value. Can be empty if only
        spin up frequencies are present
    bins : list of floats
        One dimensional list of the energy bin edges, in the same units as
        freqs/freq_down and determined by the max/min values of freqs/freq_down
    """

    n_branches = len(freqs[0]) if len(freqs) > 0 else len(freq_down[0])
    hist = np.array([])
    hist_down = np.array([])
    dos = np.array([])
    dos_down = np.array([])

    # Calculate bin edges
    all_freqs = np.append(freqs, freq_down)
    freq_max = np.amax(all_freqs)
    freq_min = np.amin(all_freqs)
    bins = np.arange(freq_min, freq_max + bwidth, bwidth)

    # Calculate weight for each q-point and branch
    freq_weights = np.repeat(np.array(weights)[:,np.newaxis],
                             n_branches, axis=1)
    if len(intensities) > 0:
        freq_weights *= intensities

    # Bin frequencies
    if len(freqs) > 0:
        hist, bin_edges = np.histogram(freqs, bins,
                                       weights=freq_weights)
    if len(freq_down) > 0:
        hist_down, bin_edges = np.histogram(freq_down, bins,
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

    return dos, dos_down, bins


def gaussian(x, sigma):
    return np.exp(-np.square(x)/(2*sigma**2))/(math.sqrt(2*math.pi)*sigma)


def lorentzian(x, gamma):
    return gamma/(2*math.pi*(np.square(x) + (gamma/2)**2))
