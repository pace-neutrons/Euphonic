import warnings
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:
    warnings.warn(('Cannot import Matplotliib for plotting (maybe Matplotlib '
                   'is not installed?). To install Euphonic\'s optional '
                   'Matplotlib dependency, try:\n\npip install '
                   'euphonic[matplotlib]\n'))
    raise
import numpy as np
from scipy import signal
from euphonic import ureg, Spectrum1D
from euphonic.util import (gaussian_2d, is_gamma, get_qpoint_labels,
                           _calc_abscissa)


def plot_dispersion(phonons, title='', btol=10.0, **line_kwargs):
    """
    Creates a Matplotlib figure displaying phonon dispersion from a
    QpointPhononModes object

    Parameters
    ----------
    phonons : QpointPhononModes object
    btol : float, optional
        Determines the limit for plotting sections of reciprocal space on
        different subplots, as a fraction of the median distance between
        q-points. Default: 10.0
    **line_kwargs : Line2D properties, optional
        Used in the axes.plot command to specify properties like linewidth,
        linestyle
    """
    qpts = phonons.qpts

    abscissa = _calc_abscissa(phonons.crystal, qpts)
    spectra = []
    x_tick_labels = get_qpoint_labels(phonons.crystal, qpts)
    # If there is LO-TO splitting, plot in sections
#    qpts = phonons.qpts[ibreak[i]:ibreak[i + 1]]
#    gamma_i = np.where(is_gamma(qpts))[0] + ibreak[i]
#    diff = np.diff(gamma_i)
#    adjacent_gamma_i = np.where(diff == 1)[0]
#    idx = np.concatenate(([0], gamma_i[adjacent_gamma_i] + 1, [len(qpts)]))
    for i in range(len(phonons._frequencies[0])):
        spectra.append(Spectrum1D(abscissa, phonons.frequencies[:, i], x_tick_labels=x_tick_labels))

    return plot_1d(spectra, btol=btol)

def plot_1d(spectra, title='', x_label='', y_label='', y_min=None, btol=None,
            **line_kwargs):
    """
    Creates a Matplotlib figure for a Spectrum1D object, or multiple Spectrum1D
    objects to be plotted on the same axes

    Parameters
    ----------
    spectra : Spectrum1D Object or list of Spectrum1D Objects
        Containing the 1D data to plot. Note only the x_tick_labels in the first
        specrum in the list will be used
    title : string, default ''
        Plot title
    x_label : string, default ''
        X-axis label
    y_label : string, default ''
        Y-axis label
    y_min : float, default None
        Minimum value on the y-axis. Can be useful to set y-axis minimum to 0
        for energy, for example.
    btol : float, optional, default None
        If there are large gaps on the x-axis (e.g sections of reciprocal space)
        data can be plotted in sections on different subplots. btol is the limit
        for plotting on different subplots, as a fraction of the median distance
        between points. Note that if multiple Spectrum1D objects have been
        provided, the axes will only be determined by the first spectrum in the
        list
    **line_kwargs : matplotlib.line.Line2D properties, optional
        Used in the axes.plot command to specify properties like linewidth,
        linestyle
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure
    """
    if not isinstance(spectra, list):
        spectra = [spectra]

    ibreak, gridspec_kw = _get_gridspec_kw(spectra[0].x_data.magnitude, btol)
    n_subplots = len(ibreak) - 1
    fig, subplots = plt.subplots(1, n_subplots, sharey=True,
                                 gridspec_kw=gridspec_kw)
    if not isinstance(subplots, np.ndarray):  # if n_subplots = 1
        subplots = np.array([subplots])

    subplots[0].set_ylabel(y_label)
    subplots[0].set_xlabel(x_label)
    for i, ax in enumerate(subplots):
        _set_x_tick_labels(ax, spectra[0].x_tick_labels, spectra[0].x_data)
        ax.set_xlim(left=spectra[0].x_data[ibreak[i]].magnitude,
                    right=spectra[0].x_data[ibreak[i + 1] - 1].magnitude)
        for spectrum in spectra:
            plot_x = spectrum._get_bin_centres('x').magnitude
            ax.plot(plot_x, spectrum.y_data.magnitude, lw=1.0, **line_kwargs)

    if y_min is not None:
        ax.set_ylim(bottom=y_min)  # Need to set limits after plotting the data

    fig.suptitle(title)
    plt.tight_layout()

    return fig


def plot_2d(spectrum, vmin=None, vmax=None, ratio=None, x_width=0, y_width=0,
             cmap='viridis', title='', x_label='', y_label=''):
    """
    Creates a Matplotlib figure for a Spectrum2D object

    Parameters
    ----------
    spectrum : Spectrum2D object
        Containing the 2D data to plot
    vmin : float, optional
        Minimum of data range for colormap. See Matplotlib imshow docs
        Default: None
    vmax : float, optional
        Maximum of data range for colormap. See Matplotlib imshow docs
        Default: None
    ratio : float, optional
        Ratio of the size of the y and x axes. e.g. if ratio is 2, the y-axis
        will be twice as long as the x-axis
        Default: None
    y_width : float Quantity, optional, default 0
        The FWHM of the Gaussian resolution function in y
    x_width : float Quantity, optional, default 0
        The FWHM of the Gaussian resolution function in x
    cmap : string, optional, default 'viridis'
        Which colormap to use, see Matplotlib docs
    title : string, optional
        The figure title. Default: ''
    x_label : string, default ''
        X-axis label
    y_label : string, default ''
        Y-axis label

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure
    ims : (n_qpts,) 'matplotlib.image.AxesImage' ndarray
        A Numpy.ndarray of AxesImage objects, one for each x-bin, for easier
        access to some attributes/functions
    """

    x_bins = spectrum._get_bin_edges('x').magnitude
    y_bins = spectrum._get_bin_edges('y').magnitude
    # Apply broadening
    if y_width or x_width:
        # If no width has been set, make widths small enough to have
        # effectively no broadening
        if x_width:
            x_width = x_width.to(spectrum.x_data_unit).magnitude
        else:
            x_width = (x_bins[1] - x_bins[0])/10
        if y_width:
            y_width = y_width.to(spectrum.y_data_unit).magnitude
        else:
            y_width = (y_bins[1] - y_bins[0])/10
        z_data = signal.fftconvolve(spectrum.z_data.magnitude, np.transpose(
            gaussian_2d(x_bins, y_bins, x_width, y_width)), 'same')
    else:
        z_data = spectrum.z_data.magnitude

    if ratio:
        y_max = x_bins[-1]/ratio
    else:
        y_max = 1.0
    # Get 'correct' tick labels that can be applied after the plot has been
    # scaled
    # Create temporary figure to get automatic tick labels
    fig_tmp, ax_tmp = plt.subplots(1,1)
    ax_tmp.imshow(np.transpose(z_data[0, np.newaxis]), origin='lower',
                  extent=[x_bins[0], x_bins[1], y_bins[0], y_bins[-1]])
    y_ticks = ax_tmp.get_yticks()
    plt.close(fig_tmp)
    y_tick_labels = [str(tick) for tick in y_ticks]
    # Locations on rescaled axis
    y_tick_locs = (y_ticks - y_bins[0])/(y_bins[-1] - y_bins[0])*y_max

    if vmin is None:
        vmin = np.amin(z_data)
    if vmax is None:
        vmax = np.amax(z_data)

    fig, ax = plt.subplots(1, 1)
    n_x_data = len(x_bins) - 1
    ims = np.empty((n_x_data), dtype=mpl.image.AxesImage)
    for i in range(n_x_data):
        ims[i] = ax.imshow(np.transpose(z_data[i, np.newaxis]),
                           interpolation='none', origin='lower',
                           extent=[x_bins[i], x_bins[i+1], 0, y_max],
                           vmin=vmin, vmax=vmax, cmap=cmap)

    _set_x_tick_labels(ax, spectrum.x_tick_labels, spectrum.x_data)
    ax.set_yticks(y_tick_locs)
    ax.set_yticklabels(y_tick_labels)

    ax.set_ylim(0, y_max)
    ax.set_xlim(x_bins[0], x_bins[-1])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.suptitle(title)

    return fig, ims

def _set_x_tick_labels(ax, x_tick_labels, x_data):
    if x_tick_labels is not None:
        locs, labels = [list(x) for x in zip(*x_tick_labels)]
        ax.set_xticks(x_data.magnitude[locs])
        ax.xaxis.grid(True, which='major')
        # Rotate long tick labels
        if len(max(labels, key=len)) >= 11:
            ax.set_xticklabels(labels, rotation=90)
        else:
            ax.set_xticklabels(labels)


def _get_gridspec_kw(x_data, btol=None):
    """
    Creates a dictionary of gridspec_kw to be passed to
    matplotlib.pyplot.subplots

    Parameters
    ----------
    x_data : (n_x_data,) float ndarray
        The x_data points
    btol : float, optional, default None
        Determines the limit for plotting sections of data on different
        subplots, as a fraction of the median difference between x_data points.
        If None all data will be on the same subplot

    Returns
    -------
    ibreak : (n_subplots + 1,) int ndarray
        Index limits of the x_data to plot on each subplot
    gridspec_kw : dict
        Contains key 'width_ratios' which is a list of subplot widths - required
        so the x-scale is the same for each subplot
    """
    # Determine Coordinates that are far enough apart to be
    # in separate subplots, and determine index limits
    diff = np.diff(x_data)
    median = np.median(diff)
    if btol is not None:
        breakpoints = np.where(diff/median > btol)[0]
    else:
        breakpoints = np.array([], dtype=np.int32)
    ibreak = np.concatenate(([0], breakpoints + 1, [len(x_data)]))

    # Calculate width ratios so that the x-scale is the same for each subplot
    subplot_widths = [x_data[ibreak[i + 1] - 1] - x_data[ibreak[i]]
                         for i in range(len(ibreak) - 1)]
    gridspec_kw = dict(width_ratios=[w/subplot_widths[0]
                                  for w in subplot_widths])
    return ibreak, gridspec_kw