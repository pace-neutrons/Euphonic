"""
Functions for fast adaptive broadening of density of states spectra
"""
from scipy.optimize import nnls
from scipy.signal import convolve
import numpy as np

def fast_broaden(dos_bins_hartree: np.ndarray,
                 freqs: np.ndarray,
                 mode_widths_hartree: np.ndarray,
                 weights: np.ndarray,
                 mode_weights: np.ndarray,
                 adaptive_error: float) -> np.ndarray:
    """
    Uses a fast, approximate method to adaptively broaden a density
    of states spectrum

    Parameters
    ----------
    dos_bins_hartree
        Shape (n_e_bins + 1,) float ndarray in hartree units. The energy bin
        edges to use for calculating the DOS
    freqs
        Shape (n_qpts, n_modes) float ndarray. Frequencies per q-point
        and mode.
    mode_widths_hartree
        Shape (n_qpts, n_modes) float ndarray in hartree units. The broadening
        width for each mode at each q-point
    weights
        Shape (n_qpts,) float ndarray. The weight for each q-point.
    mode_weights
        Shape (n_qpts, n_modes) float ndarray. The weight of each mode at
        each q-point.
    adaptive_error
        Scalar float. Acceptable error for gaussian approximations, defined
        as the absolute difference between the areas of the true and
        approximate gaussians.

    Returns
    -------
    dos
        Float ndarray of shape (dos_bins - 1,) containing density of states
        ydata
    """
    freqs = np.ravel(freqs)
    mode_widths = np.ravel(mode_widths_hartree)
    combined_weights = np.ravel(mode_weights * weights[:, np.newaxis])

    # determine spacing value for mode_width samples given desired error level
    # coefficients determined from a polynomial fit to plot of
    # error vs spacing value
    spacing = np.polyval([ 612.7, -122.7, 15.40, 1.0831], adaptive_error)

    bin_width = dos_bins_hartree[1]-dos_bins_hartree[0]

    n_kernels = int(
        np.ceil(np.log(max(mode_widths)/min(mode_widths))/np.log(spacing)))
    mode_width_samples = spacing**np.arange(n_kernels+1)*min(mode_widths)
    # Determine frequency range for gaussian kernel, the choice of
    # 3*max(sigma) is arbitrary but tuned for acceptable error/peformance
    freq_range = 3*max(mode_widths)
    kernel_npts_oneside = np.ceil(freq_range/bin_width)
    kernels = gaussian(np.arange(-kernel_npts_oneside,
                                 kernel_npts_oneside+1, 1)*bin_width,
                       mode_width_samples[:, np.newaxis])*bin_width
    kernels_idx = np.searchsorted(mode_width_samples, mode_widths)

    lower_coeffs = find_coeffs(spacing)
    dos = np.zeros(len(dos_bins_hartree)-1)

    # start loop over kernels from 1 as points with insert-position 0
    # lie outside of bin range but first include points
    # where mode_widths = min(mode_width_samples)
    masked_block = (mode_widths == min(mode_width_samples))
    sigma_factors = mode_widths[masked_block]/mode_width_samples[0]
    lower_weights = (np.polyval(lower_coeffs, sigma_factors))*combined_weights[masked_block]
    hist, _ = np.histogram(freqs[masked_block], bins=dos_bins_hartree,
                           weights=lower_weights/bin_width)
    dos += convolve(hist, kernels[0], mode="same", method="fft")

    for i in range(1, len(mode_width_samples)):
        masked_block = (kernels_idx == i)
        sigma_factors = mode_widths[masked_block]/mode_width_samples[i-1]
        lower_mix = np.polyval(lower_coeffs, sigma_factors)
        lower_weights = lower_mix * combined_weights[masked_block]

        if i == 1:
            hist, _ = np.histogram(freqs[masked_block], bins=dos_bins_hartree,
                                   weights=lower_weights/bin_width)
        else:
            mixing_weights = np.concatenate((upper_weights_prev,
                                             lower_weights))
            hist_freqs = np.concatenate((freqs_prev, freqs[masked_block]))
            hist, _ = np.histogram(hist_freqs, bins=dos_bins_hartree,
                                   weights=mixing_weights/bin_width)

        freqs_prev = freqs[masked_block]
        upper_weights_prev = combined_weights[masked_block] - lower_weights

        dos += convolve(hist, kernels[i-1], mode="same", method="fft")

        if i == len(mode_width_samples)-1:
            hist, _ = np.histogram(freqs[masked_block], bins=dos_bins_hartree,
                                   weights=upper_weights_prev/bin_width)

            dos += convolve(hist, kernels[i], mode="same", method="fft")

    return dos


def gaussian(xvals: np.ndarray,
             sigma: np.ndarray) -> np.ndarray:
    """
    Evaluates the Gaussian function.

    Parameters
    ----------
    xvals
        Float ndarray. Points at which the Gaussian function should be
        evaluated
    sigma
        Float ndarray. Specifies the standard deviation for the gaussian

    Returns
    -------
    gauss_eval
        Float ndarray containing the values of the evaluated gaussian function
    """
    # evaluate gaussian function with defined sigma and center at x
    gauss_eval = np.exp(-0.5 * (xvals / sigma)**2) \
                    / (sigma * np.sqrt(2 * np.pi))
    return gauss_eval


def find_coeffs(spacing: float) -> np.ndarray:
    """"
    Function that, for a given spacing value, gives the coefficients of the
    polynomial which decsribes the relationship between sigma and the
    linear combination weights determined by optimised interpolation

    Parameters
    ----------
    spacing
        Scalar float. The spacing value between sigma samples at which
        the gaussian kernel is exactly calculated.

    Returns
    -------
    coeffs
        Array containing the polynomial coefficients, with the highest
        power first
    """
    sigma_values = np.linspace(1, spacing, num=10)
    x_range = np.linspace(-10, 10, num=101)
    actual_gaussians = gaussian(x_range, sigma_values[:,np.newaxis])
    lower_mix = np.zeros(len(sigma_values))
    ref_gaussians = actual_gaussians[[0, -1]].T

    # For each sigma value, use non-negative least sqaures fitting to
    # find the linear combination weights that best reproduce the
    # actual gaussian.
    for i in range(len(sigma_values)):
        actual_gaussian = actual_gaussians[i]
        res = nnls(ref_gaussians, actual_gaussian)[0]
        lower_mix[i] = res[0]

    coeffs = np.polyfit(sigma_values, lower_mix, 3)
    return coeffs
