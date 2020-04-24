import numpy as np


def plot(plot_obj, *args, **kwargs):
    """
    plot_obj : Spectrum1D or Spectrum2D object
    """
    return _plot_1d(plot_obj, *args, **kwargs)

def _plot_1d(spectrum, title='', x_label='', y_label='', **line_kwargs):
    """
    Creates a Matplotlib figure for a Spectrum1D object

    Parameters
    ----------
    spectrum : Spectrum1D Object
        Containing the DOS to plot
    title : string, default ''
        Plot title
    x_label : string, default ''
        X-axis label
    y_label : string, default ''
        Y-axis label
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
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn(('Cannot import Matplotlib to plot dos (maybe '
                       'Matplotlib is not installed?). To install Euphonic\'s'
                       ' optional Matplotlib dependency, try:\n\npip install'
                       ' euphonic[matplotlib]\n'))
        raise

    # Create figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.minorticks_on()

    # Calculate bin centres
    x_bins = spectrum.x_bins.magnitude
    bin_centres = x_bins[:-1] + 0.5*np.diff(x_bins)

    # Plot dos
    ax.plot(bin_centres, spectrum.y_data.magnitude, lw=1.0, **line_kwargs)
    ax.set_ylim(bottom=0)  # Need to set limits after plotting the data
    plt.tight_layout()

    return fig