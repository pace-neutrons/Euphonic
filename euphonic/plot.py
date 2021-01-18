from typing import List, Optional, Sequence, Tuple, Union
import warnings

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.colors import Colormap, Normalize
    from matplotlib.image import NonUniformImage

except ImportError:
    warnings.warn((
        'Cannot import Matplotliib for plotting (maybe Matplotlib is '
        'not installed?). To install Euphonic\'s optional Matplotlib '
        'dependency, try:\n\npip install euphonic[matplotlib]\n'))
    raise

import numpy as np

from euphonic import ureg, Quantity
from euphonic.spectra import Spectrum1D, Spectrum1DCollection, Spectrum2D
from euphonic.util import is_gamma, get_qpoint_labels


def plot_dispersion(phonons: 'QpointPhononModes',
                    btol: float = 10.0, **kwargs) -> Figure:
    """
    Creates a Matplotlib figure displaying phonon dispersion from a
    QpointPhononModes object

    Parameters
    ----------
    phonons : QpointPhononModes
        Containing the q-points/frequencies to plot
    btol : float, optional
        Determines the limit for plotting sections of reciprocal space
        on different subplots, as a fraction of the median distance
        between q-points
    **kwargs
        Get passed to plot_1d
    """
    bands = phonons.get_dispersion()

    return plot_1d(bands.split(btol=btol), **kwargs)


def _plot_1d_core(spectra: Union[Spectrum1D, Spectrum1DCollection],
                  ax: Axes,
                  **mplargs) -> None:
    """Plot a (collection of) 1D spectrum lines to matplotlib axis

    Where there are two identical x-values in a row, plotting will restart
    to avoid creating a vertical line
    """

    if isinstance(spectra, Spectrum1D):
        return _plot_1d_core(Spectrum1DCollection.from_spectra([spectra]),
                             ax=ax, **mplargs)

    try:
        assert isinstance(spectra, Spectrum1DCollection)
    except AssertionError:
        raise TypeError("spectra should be a Spectrum1D or "
                        "Spectrum1DCollection")

    # Find where there are two identical x_data points in a row
    breakpoints = (np.where(spectra.x_data.magnitude[:-1]
                            == spectra.x_data.magnitude[1:])[0]
                   + 1).tolist()
    breakpoints = [0] + breakpoints + [None]

    for spectrum in spectra:
        # Plot each line in segments, keeping colour consistent
        for x0, x1 in zip(breakpoints[:-1], breakpoints[1:]):
            if x0 == 0:
                color = None
            else:
                color = p[-1].get_color()

            p = ax.plot(spectrum._get_bin_centres('x').magnitude[x0:x1],
                        spectrum.y_data.magnitude[x0:x1],
                        color=color, **mplargs)

    ax.set_xlim(left=min(spectra.x_data.magnitude),
                right=max(spectra.x_data.magnitude))

    _set_x_tick_labels(ax, spectra.x_tick_labels, spectra.x_data)


def plot_1d(spectra: Union[Spectrum1D,
                           Spectrum1DCollection,
                           Sequence[Spectrum1D],
                           Sequence[Spectrum1DCollection]],
            title: str = '',
            x_label: str = '',
            y_label: str = '',
            y_min: float = None,
            y_max: float = None,
            labels: Optional[List[str]] = None,
            **line_kwargs) -> Figure:
    """
    Creates a Matplotlib figure for a Spectrum1D object, or multiple
    Spectrum1D objects to be plotted on the same axes

    Parameters
    ----------
    spectra
        1D data to plot. Spectrum1D objects contain a single line, while
        Spectrum1DCollection is suitable for plotting multiple lines
        simultaneously (e.g. band structures).

        Data split across several regions should be provided as a sequence of
        spectrum objects::

            [Spectrum1D, Spectrum1D, ...]

        or::

            [Spectrum1DCollection, Spectrum1DCollection, ...]

        Where each segment will be plotted on a separate subplot. (This
        approach is mainly used to handle discontinuities in Brillouin-zone
        band structures, so the subplot widths will be based on the x-axis
        ranges.)

        A singular Spectrum1D or Spectrum1DCollection will be automatically
        split into segments if btol was set.

    title
        Plot title
    x_label
        X-axis label
    y_label
        Y-axis label
    y_min
        Minimum value on the y-axis. Can be useful to set y-axis minimum
        to 0 for energy, for example.
    y_max
        Maximum value on the y-axis.
    labels
        Legend labels for spectra, in the same order as spectra
    **line_kwargs
        matplotlib.line.Line2D properties, optional
        Used in the axes.plot command to specify properties like
        linewidth, linestyle

    Returns
    -------
    fig : matplotlib.figure.Figure

    """
    if isinstance(spectra, (Spectrum1D, Spectrum1DCollection)):
        spectra = [spectra]
    else:
        # Check units are consistent
        for spectrum in spectra[1:]:
            if spectrum.x_data_unit != spectra[0].x_data_unit:
                raise ValueError("Something went wrong: x data units are not "
                                 "consistent between spectrum subplots.")
            if spectrum.y_data_unit != spectra[0].y_data_unit:
                raise ValueError("Something went wrong: y data units are not "
                                 "consistent between spectrum subplots.")

    gridspec_kw = _get_gridspec_kw(spectra)
    fig, subplots = plt.subplots(1, len(spectra), sharey=True,
                                 gridspec_kw=gridspec_kw)
    if not isinstance(subplots, np.ndarray):  # if only one subplot
        subplots = np.array([subplots])

    for i, (spectrum, ax) in enumerate(zip(spectra, subplots)):
        _plot_1d_core(spectrum, ax, **line_kwargs)
        ax.set_ylim(bottom=y_min, top=y_max)

        if i == 0 and labels:
            ax.legend(labels)

    # Add an invisible large axis for common labels
    ax = fig.add_subplot(111, frameon=False)
    ax.tick_params(labelcolor="none", bottom=False, left=False)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    fig.suptitle(title)
    return fig


def _plot_2d_core(spectrum: Spectrum2D, ax: Axes,
                  cmap: Union[str, Colormap] = 'viridis',
                  interpolation: str = 'nearest',
                  norm: Optional[Normalize] = None,
                  ) -> NonUniformImage:
    """Plot Spectrum2D object to Axes

    Parameters
    ----------
    spectrum
        2D data object for plotting as NonUniformImage. The x_tick_labels
        attribute will be used to mark labelled points.
    ax
        Matplotlib axes to which image will be drawn
    cmap
        Matplotlib colormap or registered colormap name
    interpolation
        Interpolation method: 'nearest' or 'bilinear' for a pixellated or
        smooth result
    norm
        Matplotlib normalization object; set this in order to ensure separate
        plots are on the same colour scale.

    """
    x_unit = spectrum.x_data_unit
    y_unit = spectrum.y_data_unit
    z_unit = spectrum.z_data_unit

    x_bins = spectrum._get_bin_edges('x').to(x_unit).magnitude
    y_bins = spectrum._get_bin_edges('y').to(y_unit).magnitude

    image = NonUniformImage(ax, interpolation=interpolation,
                            extent=(min(x_bins), max(x_bins),
                                    min(y_bins), max(y_bins)),
                            cmap=cmap)
    if norm is not None:
        image.set_norm(norm)

    image.set_data(spectrum._get_bin_centres('x').to(x_unit).magnitude,
                   spectrum._get_bin_centres('y').to(y_unit).magnitude,
                   spectrum.z_data.to(z_unit).magnitude.T)
    ax.images.append(image)
    ax.set_xlim(min(x_bins), max(x_bins))
    ax.set_ylim(min(y_bins), max(y_bins))

    _set_x_tick_labels(ax, spectrum.x_tick_labels, spectrum.x_data)

    return image


def plot_2d(spectra: Union[Spectrum2D, Sequence[Spectrum2D]],
            vmin: Optional[float] = None,
            vmax: Optional[float] = None,
            cmap: Union[str, Colormap] = 'viridis',
            title: str = '', x_label: str = '', y_label: str = '') -> Figure:
    """
    Creates a Matplotlib figure for a Spectrum2D object

    Parameters
    ----------
    spectra
        Containing the 2D data to plot. If a sequence of Spectrum2D is given,
        they will be plotted from right-to-left as separate subplots. This is
        recommended for band structure/dispersion plots with discontinuous
        regions.
    vmin
        Minimum of data range for colormap. See Matplotlib imshow docs
    vmax
        Maximum of data range for colormap. See Matplotlib imshow docs
    cmap
        Which colormap to use, see Matplotlib docs
    title
        Set a title for the Figure.
    x_label
        X-axis label
    y_label
        Y-axis label

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Figure instance
    """

    # Wrap a bare spectrum in list so treatment is consistent with sequences
    if isinstance(spectra, Spectrum2D):
        spectra = [spectra]

    x_unit = spectra[0].x_data.units

    def _get_q_range(data: Quantity) -> float:
        dimensionless_data = data.to(x_unit).magnitude
        return dimensionless_data[-1] - dimensionless_data[0]
    widths = [_get_q_range(spectrum.x_data) for spectrum in spectra]

    fig, axes = plt.subplots(ncols=len(spectra), nrows=1,
                             gridspec_kw={'width_ratios': widths},
                             sharey=True)

    if not isinstance(axes, np.ndarray):  # Ensure axes are always iterable
        axes = [axes]

    intensity_unit = spectra[0].z_data.units

    def _get_minmax_intensity(spectrum: Spectrum2D) -> float:
        dimensionless_data = spectrum.z_data.to(intensity_unit).magnitude
        assert isinstance(dimensionless_data, np.ndarray)
        return np.min(dimensionless_data), np.max(dimensionless_data)
    min_z_list, max_z_list = zip(*map(_get_minmax_intensity, spectra))
    if vmin is None:
        vmin = min(min_z_list)
    if vmax is None:
        vmax = max(max_z_list)

    norm = Normalize(vmin=vmin, vmax=vmax)

    for spectrum, ax in zip(spectra, axes):
        _plot_2d_core(spectrum, ax, cmap=cmap, norm=norm)

    # Add an invisible large axis for common labels
    ax = fig.add_subplot(111, frameon=False)
    ax.tick_params(labelcolor="none", bottom=False, left=False)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    fig.suptitle(title)
    fig.tight_layout()

    return fig


def _set_x_tick_labels(ax: Axes,
                       x_tick_labels: List[Tuple[int, str]],
                       x_data: Quantity) -> None:
    if x_tick_labels is not None:
        locs, labels = [list(x) for x in zip(*x_tick_labels)]
        x_values = x_data.magnitude  # type: np.ndarray
        ax.set_xticks(x_values[locs])
        ax.xaxis.grid(True, which='major')
        # Rotate long tick labels
        if len(max(labels, key=len)) >= 11:
            ax.set_xticklabels(labels, rotation=90)
        else:
            ax.set_xticklabels(labels)


def _get_gridspec_kw(spectra: Sequence[Union[Spectrum1D,
                                             Spectrum1DCollection]]):
    """
    Creates a dictionary of gridspec_kw to be passed to
    matplotlib.pyplot.subplots

    Parameters
    ----------
    spectra
        series of spectral data containers corresponding to subplots

    Returns
    -------
    gridspec_kw : dict
        Contains key 'width_ratios' which is a list of subplot widths.
        Required so the x-scale is the same for each subplot
    """
    # Get width ratios so that the x-scale is the same for each subplot
    subplot_widths = [max(spectrum.x_data.magnitude)
                      - min(spectrum.x_data.magnitude)
                      for spectrum in spectra]
    gridspec_kw = dict(width_ratios=[w / subplot_widths[0]
                                     for w in subplot_widths])
    return gridspec_kw
