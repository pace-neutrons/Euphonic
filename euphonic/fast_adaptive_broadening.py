"""
Functions for fast adaptive broadening of density of states spectra
"""
from typing import Optional
from scipy.optimize import curve_fit
from scipy.signal import convolve
from scipy.stats import norm
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
    spacing = np.polyval([656.1, -131.8, 15.98, 1.0803], adaptive_error)

    bin_width = dos_bins_hartree[1]-dos_bins_hartree[0]
    mode_widths = np.maximum(mode_widths, bin_width / 2)

    n_kernels = int(
        np.ceil(np.log(max(mode_widths)/min(mode_widths))/np.log(spacing)))
    mode_width_samples = spacing**np.arange(n_kernels+1)*min(mode_widths)
    # Determine frequency range for gaussian kernel, the choice of
    # 3*max(sigma) is arbitrary but tuned for acceptable error/peformance
    freq_range = 3*max(mode_widths)
    kernel_npts_oneside = np.ceil(freq_range/bin_width)
    kernels = norm.pdf(x=np.arange(-kernel_npts_oneside,
                       kernel_npts_oneside+1, 1)*bin_width, loc=0,
                       scale=mode_width_samples[:, np.newaxis])*bin_width
    kernels_idx = np.searchsorted(mode_width_samples, mode_widths)

    lower_coeffs = find_coeffs(spacing)
    scaled_data_matrix = np.zeros((len(dos_bins_hartree)-1, len(kernels)))
    # start loop from 1 as points with insert-position 0 lie outside of bin range
    for i in range(1, len(mode_width_samples)):
        masked_block = (kernels_idx == i)
        sigma_factors = mode_widths[masked_block]/mode_width_samples[i-1]
        lower_mix = np.polyval(lower_coeffs, sigma_factors)
        upper_mix = 1-lower_mix

        lower_hist, _ = np.histogram(
            freqs[masked_block], bins=dos_bins_hartree,
            weights=lower_mix*combined_weights[masked_block]/bin_width)
        upper_hist, _ = np.histogram(
            freqs[masked_block], bins=dos_bins_hartree,
            weights=upper_mix*combined_weights[masked_block]/bin_width)

        scaled_data_matrix[:, i-1] += lower_hist
        scaled_data_matrix[:, i] += upper_hist

    dos = np.sum([convolve(scaled_data_matrix[:, i], kernels[i],
                 mode="same") for i in range(0, n_kernels)], 0)
    return dos


def gaussian(xvals: np.ndarray,
             sigma: np.ndarray,
             centre: Optional[int] = 0) -> np.ndarray:
    """
    Evaluates the Gaussian function.

    Parameters
    ----------
    xvals
        Float ndarray. Points at which the Gaussian function should be
        evaluated
    sigma
        Float ndarray. Specifies the standard deviation for the gaussian
    centre
        Optional integer value that sets the centre of the gaussian

    Returns
    -------
    gauss_eval
        Float ndarray containing the values of the evaluated gaussian function
    """
    # evaluate gaussian function with defined sigma and center at x
    gauss_eval = np.exp(-0.5 * ((xvals - centre) / sigma)**2) \
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
    sigma_values = np.linspace(1, spacing, 10)
    x_range = np.linspace(-10, 10, 101)

    def gaussian_mix(xvals, weight):
        # Return a linear combination of two Gaussians with weights
        return (weight * gaussian(xvals, sigma=1)
                + (1-weight) * gaussian(xvals, sigma=spacing))

    lower_mix = np.zeros(len(sigma_values))

    for i, s_val in enumerate(sigma_values):
        actual_gaussian = gaussian(x_range, s_val)
        mixl, _ = curve_fit(gaussian_mix, x_range,
                            ydata=actual_gaussian, p0=[0.5], bounds=(0, 1))
        lower_mix[i] = mixl[0]

    coeffs = np.polyfit(sigma_values, lower_mix, 3)
    return coeffs
