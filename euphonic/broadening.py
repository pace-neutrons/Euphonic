"""
Functions for broadening spectra
"""
from typing import Callable, Literal, Union
import warnings

import numpy as np
from numpy.polynomial import Chebyshev
from pint import Quantity
from scipy.optimize import nnls
from scipy.stats import norm
from scipy.signal import convolve

from euphonic import ureg


ErrorFit = Literal['cheby-log', 'cubic']
KernelShape = Literal['gauss', 'lorentz']


def variable_width_broadening(
    bins: Quantity,
    x: Quantity,
    width_function: Callable[[Quantity], Quantity],
    weights: Union[np.ndarray, Quantity],
    width_lower_limit: Quantity = None,
    width_convention: Literal['fwhm', 'std'] = 'fwhm',
    adaptive_error: float = 1e-2,
    shape: KernelShape = 'gauss',
    fit: ErrorFit = 'cheby-log'
    ) -> Quantity:
    r"""Apply x-dependent Gaussian broadening to 1-D data series

    Typically this is an energy-dependent instrumental resolution function.
    Data is binned and broadened to output array with reciprocal units.

    A fast interpolation-based method is used to reduce the number of Gaussian
    evaluations.

    Parameters
    ----------
    bins
        Data bins for output spectrum
    x
        Data positions (to be binned)
    width_function
        A function handle which takes an n-dimensional array of x values as
        input and returns corresponding width values for broadening. These
        should be Quantity arrays and dimensionally-consistent with x.
    weights
        Weight for each data point corresponding to x. Note that these should
        be "counts" rather than binned spectral weights; this function will
        bin the data and apply bin-width weighting.
    width_lower_limit
        A lower bound is set for broadening width in WIDTH_UNIT. If set to None
        (default) the bin width will be used. To disable any lower limit, set
        to 0 or lower.
    width_convention
        Indicate if polynomial function yields standard deviation (sigma) or
        full-width half-maximum.
    adaptive_error
        Acceptable error for gaussian approximations, defined
        as the absolute difference between the areas of the true and
        approximate gaussians.
    shape
        Select broadening kernel function.
    fit
        Select parametrisation of kernel width spacing to adaptive_error.
        'cheby-log' is recommended: for shape 'gauss', 'cubic' is also
        available.
    """

    if width_convention.lower() == 'fwhm' and shape == 'gauss':
        sigma_function = (lambda x: width_function(x)
                          / np.sqrt(8 * np.log(2)))
    elif width_convention.lower() == 'std' and shape == 'lorentz':
        raise ValueError('Standard deviation unavailable for Lorentzian '
                         'function: please use FWHM.')
    elif width_convention.lower() in ('std', 'fwhm'):
        sigma_function = width_function
    else:
        raise ValueError('width_convention must be "std" or "fwhm".')

    widths = sigma_function(x)

    if width_lower_limit is None:
        width_lower_limit = np.diff(bins).max()

    widths = np.maximum(widths, width_lower_limit)

    if isinstance(weights, np.ndarray):
        weights = weights * ureg('dimensionless')
        assert isinstance(weights, Quantity)

    weights_unit = weights.units

    return width_interpolated_broadening(bins, x, widths,
                                         weights.magnitude,
                                         adaptive_error=adaptive_error,
                                         shape=shape,
                                         fit=fit) * weights_unit


def width_interpolated_broadening(
    bins: Quantity,
    x: Quantity,
    widths: Quantity,
    weights: np.ndarray,
    adaptive_error: float,
    shape: KernelShape = 'gauss',
    fit: ErrorFit = 'cheby-log'
    ) -> Quantity:
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
    shape
        Select kernel shape. Widths will correspond to sigma or gamma
        parameters respectively.
    fit
        Select parametrisation of kernel width spacing to adaptive_error.
        'cheby-log' is recommended: for shape 'gauss', 'cubic' is also
        available.

    Returns
    -------
    spectrum
        Quantity of shape (bins - 1,) containing broadened spectrum
        ydata
    """

    conv = 1*ureg('hartree').to(bins.units)
    return _width_interpolated_broadening(bins.to('hartree').magnitude,
                                          x.to('hartree').magnitude,
                                          widths.to('hartree').magnitude,
                                          weights,
                                          adaptive_error,
                                          shape=shape,
                                          fit=fit) / conv


def _lorentzian(x: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    return gamma / (2 * np.pi * (x**2 + (gamma / 2)**2))


def _get_spacing(error,
                 shape: KernelShape = 'gauss',
                 fit: ErrorFit = 'cheby-log'):
    """
    Determine suitable spacing value for mode_width given accepted error level

    Coefficients have been fitted to plots of error vs spacing value
    """

    if fit == 'cubic' and shape == 'gauss':
        return np.polyval([612.7, -122.7, 15.40, 1.0831], error)

    elif fit == 'cheby-log':
        if shape == 'lorentz':
            cheby = Chebyshev(
                [1.26039672, 0.39900457, 0.20392176, 0.08602507,
                 0.03337662, 0.00878684, 0.00619626],
                window=[1., 1.],
                domain=[-4.99146317, -1.34655197])
            safe_domain = [-4, -1.35]

        else:  # gauss
            cheby = Chebyshev(
                [1.25885858, 0.39803148, 0.20311735, 0.08654827,
                 0.03447873, 0.00894006, 0.00715706],
                window=[-1., 1.],
                domain=[-4.64180022, -1.00029948])
            safe_domain = [-4, -1.]

        log_error = np.log10(error)
        if log_error < safe_domain[0] or log_error > safe_domain[1]:
            raise ValueError("Target error is out of fit range; value must lie"
                             f" in range {np.power(10, safe_domain)}.")
        return cheby(log_error)

    else:
        raise ValueError(f'Fit "{fit}" is not available for shape "{shape}". '
                         f'The "cheby-log" fit is recommended for "gauss" and '
                         f'"Lorentz" shapes.')


def _width_interpolated_broadening(
    bins: np.ndarray,
    x: np.ndarray,
    widths: np.ndarray,
    weights: np.ndarray,
    adaptive_error: float,
    shape: KernelShape = 'gauss',
    fit: ErrorFit = 'cheby-log') -> np.ndarray:
    """
    Broadens a spectrum using a variable-width kernel, taking the
    same arguments as `variable_width` but expects arrays with
    consistent units rather than Quantities. Also returns an array
    rather than a Quantity.
    """
    x = np.ravel(x)
    widths = np.ravel(widths)
    weights = np.ravel(weights)
    spacing = _get_spacing(adaptive_error, shape=shape, fit=fit)

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
        x_values = np.arange(-len(bins)/2+1, len(bins)/2)*bin_width
    else:
        x_values = np.arange(-int(len(bins)/2), int(len(bins)/2)+1)*bin_width

    if shape == 'gauss':
        kernels = norm.pdf(x_values, scale=width_samples[:, np.newaxis]
                           ) * bin_width
    elif shape == 'lorentz':
        kernels = _lorentzian(x_values, gamma=width_samples[:, np.newaxis]
                              ) * bin_width

    kernels_idx = np.searchsorted(width_samples, widths, side="right")

    if len(kernels) == 1:
        # Only one kernel to consider, use 100% in all cases
        lower_coeffs = [1.]
    else:
        lower_coeffs = find_coeffs(spacing, shape=shape)

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


def find_coeffs(spacing: float,
                shape: KernelShape = 'gauss') -> np.ndarray:
    """"
    Function that, for a given spacing value, gives the coefficients of the
    polynomial which describes the relationship between kernel width and the
    linear combination weights determined by optimised interpolation

    Parameters
    ----------
    spacing
        Scalar float. The spacing value between sigma (or gamma) samples at
        which the kernel is exactly calculated.
    shape
        'gauss' or 'lorentz', selecting the type of broadening kernel.

    Returns
    -------
    coeffs
        Array containing the polynomial coefficients, with the highest
        power first
    """
    width_values = np.linspace(1, spacing, num=10)
    x_range = np.linspace(-10, 10, num=101)
    if shape == 'gauss':
        actual_kernels = norm.pdf(x_range, scale=width_values[:, np.newaxis])
    elif shape == 'lorentz':
        actual_kernels = _lorentzian(x_range,
                                     gamma=width_values[:, np.newaxis])
    lower_mix = np.zeros(len(width_values))
    ref_kernels = actual_kernels[[0, -1]].T

    # For each width value, use non-negative least squares fitting to
    # find the linear combination weights that best reproduce the
    # actual kernel.
    for i in range(len(width_values)):
        actual_kernel = actual_kernels[i]
        res = nnls(ref_kernels, actual_kernel)[0]
        lower_mix[i] = res[0]

    coeffs = np.polyfit(width_values, lower_mix, 3)
    return coeffs
