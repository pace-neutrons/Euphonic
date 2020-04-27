import numpy as np
import warnings
try:
    import matplotlib.pyplot as plt
except ImportError:
    warnings.warn(('Cannot import Matplotliib for plotting (maybe Matplotlib '
                   'is not installed?). To install Euphonic\'s optional '
                   'Matplotlib dependency, try:\n\npip install '
                   'euphonic[matplotlib]\n'))
    raise


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

    # Create figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.minorticks_on()

    plot_x = spectrum.x_data.magnitude
    if len(spectrum.x_data) == len(spectrum.y_data) + 1:
        # Calculate bin centres
        plot_x = plot_x[:-1] + 0.5*np.diff(plot_x)

    # Plot dos
    ax.plot(plot_x, spectrum.y_data.magnitude, lw=1.0, **line_kwargs)
    ax.set_ylim(bottom=0)  # Need to set limits after plotting the data
    plt.tight_layout()

    return fig