"""
Functions for broadening spectra
"""
from typing import List
import warnings

import numpy as np
from pint import Unit
from scipy.optimize import nnls
from scipy.stats import norm
from scipy.signal import convolve

from euphonic import ureg, Quantity


def polynomial_broadening(bins: Quantity,
                          x: Quantity,
                          width_polynomial: List[float],
                          weights: np.ndarray,
                          width_unit: Unit = None,
                          width_lower_limit: float = None,
                          adaptive_error: float = 1e-2) -> Quantity:
    """Use fast approximate method to apply x-dependent Gaussian broadening

    Typically this is an energy-dependent instrumental resolution function.

    Parameters
    ----------
    bins
        Energy bins for output spectrum
    x
        Data positions (to be binned)
    width_polynomial
        Coefficients of polynomial fit to energy/width relationship. Order
        should be consistent with numpy.polyfit / numpy.polyval (i.e. from
        largest exponent to smallest.) Width is expressed as standard deviation
        (not FHWM); the conversion factor between these conventions is
        :math:`FWHM = \sqrt{8 \ln 2} \sigma`
    weights
        Weight for each data point corresponding to x. Note that these should
        be "counts" rather than binned spectral weights; this function will
        bin the data and apply bin-width weighting.
    width_unit
        x-axis units for width_polynomial. (By default, use same as bins.)
    width_lower_limit
        A lower bound is set for broadening width in WIDTH_UNIT. If set to None
        (default) the bin width will be used. To disable any lower limit, set
        to 0 or lower.
    adaptive_error
        Acceptable error for gaussian approximations, defined
        as the absolute difference between the areas of the true and
        approximate gaussians.
        
    """
    if width_unit is None:
        width_unit = bins.units
    if width_lower_limit is None:
        width_lower_limit = (bins[1] - bins[0]).to(width_unit).magnitude

    widths = np.polyval(width_polynomial, x.to(width_unit).magnitude)
    widths = np.maximum(widths, width_lower_limit) * width_unit
    
    return width_interpolated_broadening(bins, x, widths, weights,
                                         adaptive_error=adaptive_error)


def width_interpolated_broadening(bins: Quantity,
                                  x: Quantity,
                                  widths: Quantity,
                                  weights: np.ndarray,
                                  adaptive_error: float) -> Quantity:
    """
    Uses a fast, approximate method to broaden a spectrum
    with a variable-width kernel. Exact Gaussians are calculated
    at logrithmically spaced values across the range of widths.
    A small set of spectra that have been scaled using the weights
    from linear combinations of the exact Gaussians are convolved
    using Fast Fourier Transforms (FFT) and then summed to give the
    approximate broadened spectra.

    Parameters
    ----------
    bins
        The energy bin edges to use for calculating
        the spectrum
    x
        Broadening samples
    widths
        The broadening width for each peak, must be the same shape as x.
    weights
        The weight for each peak, must be the same shape as x.
    adaptive_error
        Scalar float. Acceptable error for gaussian approximations, defined
        as the absolute difference between the areas of the true and
        approximate gaussians.

    Returns
    -------
    spectrum
        Quantity of shape (bins - 1,) containing broadened spectrum
        ydata
    """
    conv = 1*ureg('hartree').to(bins.units)
    return _width_interpolated_broadening(
                                    bins.to('hartree').magnitude,
                                    x.to('hartree').magnitude,
                                    widths.to('hartree').magnitude,
                                    weights,
                                    adaptive_error)/conv

def _width_interpolated_broadening(
                            bins: np.ndarray,
                            x: np.ndarray,
                            widths: np.ndarray,
                            weights: np.ndarray,
                            adaptive_error: float) -> np.ndarray:
    """
    Broadens a spectrum using a variable-width kernel, taking the
    same arguments as `variable_width` but expects arrays with
    consistent units rather than Quantities. Also returns an array
    rather than a Quantity.
    """
    x = np.ravel(x)
    widths = np.ravel(widths)
    weights = np.ravel(weights)

    # determine spacing value for mode_width samples given desired error level
    # coefficients determined from a polynomial fit to plot of
    # error vs spacing value
    spacing = np.polyval([ 612.7, -122.7, 15.40, 1.0831], adaptive_error)

    # bins should be regularly spaced, check that this is the case and
    # raise a warning if not
    bin_widths = np.diff(bins)
    if not np.all(np.isclose(bin_widths, bin_widths[0])):
        warnings.warn('Not all bin widths are equal, so broadening by '
                      'convolution will give incorrect results.',
                      stacklevel=3)
    bin_width = bin_widths[0]

    n_kernels = int(
        np.ceil(np.log(max(widths)/min(widths))/np.log(spacing)))
    width_samples = spacing**np.arange(n_kernels+1)*min(widths)

    # Evaluate kernels on regular grid of length equal to number of bins,
    #  avoids the need for zero padding in convolution step
    if (len(bins) % 2) == 0:
        kernels = norm.pdf(
            np.arange(-len(bins)/2+1, len(bins)/2)*bin_width,
            scale=width_samples[:,np.newaxis])*bin_width
    else:
        kernels = norm.pdf(
            np.arange(-int(len(bins)/2), int(len(bins)/2)+1)*bin_width,
            scale=width_samples[:,np.newaxis])*bin_width

    kernels_idx = np.searchsorted(width_samples, widths, side="right")

    lower_coeffs = find_coeffs(spacing)
    spectrum = np.zeros(len(bins)-1)

    for i in range(1, len(width_samples)+1):
        masked_block = (kernels_idx == i)
        width_factors = widths[masked_block]/width_samples[i-1]
        lower_mix = np.polyval(lower_coeffs, width_factors)
        lower_weights = lower_mix * weights[masked_block]

        if i == 1:
            hist, _ = np.histogram(x[masked_block], bins=bins,
                                   weights=lower_weights/bin_width)
        else:
            mixing_weights = np.concatenate((upper_weights_prev,
                                             lower_weights))
            hist_x = np.concatenate((x_prev, x[masked_block]))
            hist, _ = np.histogram(hist_x, bins=bins,
                                   weights=mixing_weights/bin_width)

        x_prev = x[masked_block]
        upper_weights_prev = weights[masked_block] - lower_weights

        spectrum += convolve(hist, kernels[i-1], mode="same", method="fft")

    return spectrum

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
    actual_gaussians = norm.pdf(x_range, scale=sigma_values[:,np.newaxis])
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
