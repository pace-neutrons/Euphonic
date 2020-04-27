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
from euphonic.util import gaussian_2d


def plot_1d(spectra, title='', x_label='', y_label='', y_min=None,
            **line_kwargs):
    """
    Creates a Matplotlib figure for a Spectrum1D object, or multiple Spectrum1D
    objects to be plotted on the same axes

    Parameters
    ----------
    spectra : Spectrum1D Object or list of Spectrum1D Objects
        Containing the 1D data to plot
    title : string, default ''
        Plot title
    x_label : string, default ''
        X-axis label
    y_label : string, default ''
        Y-axis label
    y_min : float, default None
        Minimum value on the y-axis. Can be useful to set y-axis minimum to 0
        for energy, for example.
    **line_kwargs : matplotlib.line.Line2D properties, optional
        Used in the axes.plot command to specify properties like linewidth,
        linestyle
    Returns
    -------
    fig : matplotlib.figure.Figure or None
        If matplotlib.pyplot can be imported, returns a Figure containing the
        subplot containing the plotted density of states, otherwise returns
        None
    """
    if not isinstance(spectra, list):
        spectra = [spectra]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.minorticks_on()

    for spectrum in spectra:
        plot_x = spectrum.x_data.magnitude
        if len(spectrum.x_data) == len(spectrum.y_data) + 1:
            # Calculate bin centres
            plot_x = plot_x[:-1] + 0.5*np.diff(plot_x)
        ax.plot(plot_x, spectrum.y_data.magnitude, lw=1.0, **line_kwargs)

    if y_min is not None:
        ax.set_ylim(bottom=y_min)  # Need to set limits after plotting the data
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
        A Figure with a single subplot
    ims : (n_qpts,) 'matplotlib.image.AxesImage' ndarray
        A Numpy.ndarray of AxesImage objects, one for each x-bin, for easier
        access to some attributes/functions
    """

    y_bins = spectrum.y_data.magnitude
    x_bins = spectrum.x_data.magnitude
    if len(spectrum.x_data) == len(spectrum.z_data):
        # Calculate xbin edges
        x_bins = np.concatenate(
            ([x_bins[0]], (x_bins[1:] + x_bins[:-1])/2, [x_bins[-1]]))
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

    ax.set_yticks(y_tick_locs)
    ax.set_yticklabels(y_tick_labels)
    if spectrum.x_tick_labels is not None:
        locs, labels = [list(x) for x in zip(*spectrum.x_tick_labels)]
        ax.set_xticks(spectrum.x_data.magnitude[locs])
        ax.xaxis.grid(True, which='major')
        # Rotate long tick labels
        if len(max(labels, key=len)) >= 11:
            ax.set_xticklabels(labels, rotation=90)
        else:
            ax.set_xticklabels(labels)

    ax.set_ylim(0, y_max)
    ax.set_xlim(x_bins[0], x_bins[-1])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.suptitle(title)

    return fig, ims
