from collections.abc import Sequence
from itertools import pairwise
from typing import TYPE_CHECKING, Optional

import numpy as np

from euphonic.spectra import Spectrum1D, Spectrum1DCollection, Spectrum2D
from euphonic.ureg import Quantity
from euphonic.util import dedent_and_fill, zips

if TYPE_CHECKING:
    from matplotlib.lines import Line2D

try:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap, Normalize
    from matplotlib.figure import Figure
    from matplotlib.image import NonUniformImage
    import matplotlib.pyplot as plt

except ModuleNotFoundError as err:
    err_msg = dedent_and_fill("""
        Cannot import Matplotlib for plotting (maybe Matplotlib is
        not installed?). To install Euphonic's optional Matplotlib
        dependency, try:

            pip install euphonic[matplotlib]
    """)
    raise ModuleNotFoundError(err_msg) from err


def plot_1d_to_axis(spectra: Spectrum1D | Spectrum1DCollection,
                    ax: Axes, labels: Optional[Sequence[str]] = None,
                    **mplargs) -> None:
    """Plot a (collection of) 1D spectrum lines to matplotlib axis

    Where there are two identical x-values in a row, plotting will restart
    to avoid creating a vertical line

    Parameters
    ----------
    spectra
        Spectrum1D or Spectrum1DCollection to plot
    ax
        Matplotlib axes to which spectra will be drawn
    labels
        A sequence of labels corresponding to the sequence of lines in
        spectra, used to label each line. If this is None, the
        label(s) contained in spectra.metadata['label'] (Spectrum1D) or
        spectra.metadata['line_data'][i]['label']
        (Spectrum1DCollection) will be used. To disable labelling for a
        specific line, pass an empty string.
    **mplargs
        Keyword arguments passed to Axes.plot
    """

    if isinstance(spectra, Spectrum1D):
        return plot_1d_to_axis(Spectrum1DCollection.from_spectra([spectra]),
                               ax=ax, labels=labels, **mplargs)

    try:
        assert isinstance(spectra, Spectrum1DCollection)
    except AssertionError:
        msg = 'spectra should be a Spectrum1D or Spectrum1DCollection'
        raise TypeError(msg) from None

    if isinstance(labels, str):
        labels = [labels]
    if labels is not None and len(labels) != len(spectra):
        msg = (
            f'The length of labels (got {len(labels)}) should be the '
            f'same as the number of lines to plot (got {len(spectra)})'
        )
        raise ValueError(msg)

    # Find where there are two identical x_data points in a row
    breakpoints = (np.where(spectra.x_data[:-1] == spectra.x_data[1:])[0]
                   + 1).tolist()
    breakpoints = [0, *breakpoints, None]

    if labels is None:
        labels = [spec.metadata.get('label', None) for spec in spectra]

    for label, spectrum in zips(labels, spectra):
        p: list[Line2D] | None = None
        # Plot each line in segments
        for x0, x1 in pairwise(breakpoints):
            # Keep colour consistent across segments
            if p is None:  # i.e. on first iteration
                color = None
            else:
                # Only add legend label to the first segment
                label = None  # noqa: PLW2901 (redefined-loop-name)
                color = p[-1].get_color()
            # Allow user kwargs to take priority
            plot_kwargs = {'color': color, 'label': label, **mplargs}
            p = ax.plot(spectrum.get_bin_centres().magnitude[x0:x1],
                        spectrum.y_data.magnitude[x0:x1],
                        **plot_kwargs)

    # Update legend if it exists, in case new labels have been added
    legend = ax.get_legend()
    if legend is not None:
        ax.legend()

    ax.set_xlim(left=min(spectra.x_data.magnitude),
                right=max(spectra.x_data.magnitude))

    _set_x_tick_labels(ax, spectra.x_tick_labels, spectra.x_data)
    return None


OneDSpectrumOrSpectra = (Spectrum1D
                         | Spectrum1DCollection
                         | Sequence[Spectrum1D]
                         | Sequence[Spectrum1DCollection])

def plot_1d(spectra: OneDSpectrumOrSpectra,
            title: str | None = None,
            xlabel: str = '',
            ylabel: str = '',
            ymin: Optional[float] = None,
            ymax: Optional[float] = None,
            labels: Optional[Sequence[str]] = None,
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

    title
        Plot title
    xlabel
        X-axis label
    ylabel
        Y-axis label
    ymin
        Minimum value on the y-axis. Can be useful to set y-axis minimum
        to 0 for energy, for example.
    ymax
        Maximum value on the y-axis.
    labels
        A sequence of labels corresponding to the sequence of lines in
        spectra, used to label each line. If this is None, the
        label(s) contained in spectra.metadata['label'] (Spectrum1D) or
        spectra.metadata['line_data'][i]['label']
        (Spectrum1DCollection) will be used. To disable labelling for a
        specific line, pass an empty string.
    **line_kwargs
        matplotlib.line.Line2D properties, optional
        Used in the axes.plot command to specify properties like
        linewidth, linestyle

    Returns
    -------
    fig : matplotlib.figure.Figure

    """
    if isinstance(spectra, (Spectrum1D, Spectrum1DCollection)):
        spectra = (spectra,)
    else:
        # Check units are consistent
        for spectrum in spectra[1:]:
            if spectrum.x_data_unit != spectra[0].x_data_unit:
                msg = (
                    'Something went wrong: x data units are not '
                    'consistent between spectrum subplots.'
                )
                raise ValueError(msg)
            if spectrum.y_data_unit != spectra[0].y_data_unit:
                msg = (
                    'Something went wrong: y data units are not '
                    'consistent between spectrum subplots.'
                )
                raise ValueError(msg)

    gridspec_kw = _get_gridspec_kw(spectra)
    fig, subplots = plt.subplots(1, len(spectra), sharey=True,
                                 gridspec_kw=gridspec_kw, squeeze=False)

    for i, (spectrum, ax) in enumerate(zips(spectra, subplots.flatten())):
        plot_1d_to_axis(spectrum, ax, labels, **line_kwargs)
        # To avoid an ugly empty legend, only use if there are labels to plot
        if i == 0:
            leg_handles, leg_labels = ax.get_legend_handles_labels()
            if len(leg_labels) > 0:
                ax.legend()
        ax.set_ylim(bottom=ymin, top=ymax)

    # Add an invisible large axis for common labels
    ax = fig.add_subplot(111, frameon=False)
    ax.grid(False)
    ax.tick_params(labelcolor='none', bottom=False, left=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    fig.suptitle('' if title is None else title)
    return fig


def plot_2d_to_axis(spectrum: Spectrum2D, ax: Axes,
                    cmap: str | Colormap | None = None,
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

    x_bins = spectrum.get_bin_edges('x').to(x_unit).magnitude
    y_bins = spectrum.get_bin_edges('y').to(y_unit).magnitude

    image = NonUniformImage(ax, interpolation=interpolation,
                            extent=(min(x_bins), max(x_bins),
                                    min(y_bins), max(y_bins)),
                            cmap=cmap)
    if norm is not None:
        image.set_norm(norm)

    image.set_data(spectrum.get_bin_centres('x').to(x_unit).magnitude,
                   spectrum.get_bin_centres('y').to(y_unit).magnitude,
                   spectrum.z_data.to(z_unit).magnitude.T)
    ax.add_image(image)
    ax.set_xlim(min(x_bins), max(x_bins))
    ax.set_ylim(min(y_bins), max(y_bins))

    _set_x_tick_labels(ax, spectrum.x_tick_labels, spectrum.x_data)

    return image


def plot_2d(spectra: Spectrum2D | Sequence[Spectrum2D],
            vmin: Optional[float] = None,
            vmax: Optional[float] = None,
            cmap: Optional[str | Colormap] = None,
            title: str | None = None,
            xlabel: str = '',
            ylabel: str = '') -> Figure:
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
    xlabel
        X-axis label
    ylabel
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
                             sharey=True, squeeze=False)

    intensity_unit = spectra[0].z_data.units

    def _get_minmax_intensity(spectrum: Spectrum2D) -> tuple[float, float]:
        dimensionless_data = spectrum.z_data.to(intensity_unit).magnitude
        assert isinstance(dimensionless_data, np.ndarray)
        return np.nanmin(dimensionless_data), np.nanmax(dimensionless_data)
    min_z_list, max_z_list = zips(*map(_get_minmax_intensity, spectra))
    if vmin is None:
        vmin = min(min_z_list)
    if vmax is None:
        vmax = max(max_z_list)

    norm = Normalize(vmin=vmin, vmax=vmax)

    for spectrum, ax in zips(spectra, axes.flatten()):
        plot_2d_to_axis(spectrum, ax, cmap=cmap, norm=norm)

    # Add an invisible large axis for common labels
    ax = fig.add_subplot(111, frameon=False)
    ax.grid(False)
    ax.tick_params(labelcolor='none', bottom=False, left=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    fig.suptitle('' if title is None else title)

    return fig


def _set_x_tick_labels(ax: Axes,
                       x_tick_labels: Optional[Sequence[tuple[int, str]]],
                       x_data: Quantity) -> None:
    if x_tick_labels is not None:
        locs, labels = [list(x) for x in zips(*x_tick_labels)]
        x_values = x_data.magnitude  # type: np.ndarray
        ax.set_xticks(x_values[locs])

        # Rotate long tick labels
        if len(max(labels, key=len)) >= 11:
            ax.set_xticklabels(labels, rotation=90)
        else:
            ax.set_xticklabels(labels)


def _get_gridspec_kw(spectra: Sequence[Spectrum1D | Spectrum1DCollection]):
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
    return {'width_ratios': [w / subplot_widths[0]
                                     for w in subplot_widths]}
