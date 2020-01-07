import math
import sys
import warnings
import numpy as np
import seekpath
from scipy import signal
from euphonic.util import direction_changed, reciprocal_lattice, gaussian_2d


def calc_abscissa(qpts, recip_latt):
    """
    Calculates the distance between q-points (for the plot x-coordinate)
    """

    # Get distance between q-points in each dimension
    # Note: length is nqpts - 1
    delta = np.diff(qpts, axis=0)

    # Determine how close delta is to being an integer. As q = q + G where G
    # is a reciprocal lattice vector based on {1,0,0},{0,1,0},{0,0,1}, this
    # is used to determine whether 2 q-points are equivalent ie. they differ
    # by a multiple of the reciprocal lattice vector
    delta_rem = np.sum(np.abs(delta - np.rint(delta)), axis=1)

    # Create a boolean array that determines whether to calculate the distance
    # between q-points,taking into account q-point equivalence. If delta is
    # more than the tolerance, but delta_rem is less than the tolerance, the
    # q-points differ by G so are equivalent and the distance shouldn't be
    # calculated
    TOL = 0.001
    calc_modq = np.logical_not(np.logical_and(
        np.sum(np.abs(delta), axis=1) > TOL,
        delta_rem < TOL))

    # Multiply each delta by the reciprocal lattice to get delta in Cartesian
    deltaq = np.einsum('ji,kj->ki', recip_latt, delta)

    # Get distance between q-points for all valid pairs of q-points
    modq = np.zeros(np.size(delta, axis=0))
    modq[calc_modq] = np.sqrt(np.sum(np.square(deltaq), axis=1))

    # Prepend initial x axis value of 0
    abscissa = np.insert(modq, 0, 0.)

    # Do cumulative some to get position along x axis
    abscissa = np.cumsum(abscissa)

    return abscissa


def recip_space_labels(data):
    """
    Gets high symmetry point labels (e.g. GAMMA, X, L) for the q-points at
    which the path through reciprocal space changes direction

    Parameters
    ----------
    data: Data object
        Data object containing the cell vectors, q-points and optionally ion
        types and coordinates (used for determining space group)

    Returns
    -------
    labels : (n_qpts_direction_changed,) string ndarray
        List of the labels for each q-point at which the path through
        reciprocal space changes direction
    qpts_with_labels : (n_qpts_direction_changed,) int ndarray
        List of the indices of the q-points at which the path through
        reciprocal space changes direction
    """

    # First and last q-points should always be labelled
    if len(data.qpts) <= 2:
        qpt_has_label = np.ones(len(data.qpts), dtype=bool)
    else:
        qpt_has_label = np.concatenate(([True], direction_changed(data.qpts),
                                        [True]))
    qpts_with_labels = np.where(qpt_has_label)[0]

    # Get dict of high symmetry point labels to their coordinates for this
    # space group. If space group can't be determined use a generic dictionary
    # of fractional points
    sym_label_to_coords = {}
    if hasattr(data, 'ion_r'):
        _, ion_num = np.unique(data.ion_type, return_inverse=True)
        cell_vec = (data.cell_vec.to('angstrom')).magnitude
        cell = (cell_vec, data.ion_r, ion_num)
        sym_label_to_coords = seekpath.get_path(cell)["point_coords"]
    else:
        sym_label_to_coords = generic_qpt_labels()
    # Get labels for each q-point
    labels = np.array([])
    for qpt in data.qpts[qpts_with_labels]:
        labels = np.append(labels, get_qpt_label(qpt, sym_label_to_coords))

    return labels, qpts_with_labels


def generic_qpt_labels():
    """
    Returns a dictionary relating fractional q-point label strings to their
    coordinates e.g. '1/4 1/2 1/4' = [0.25, 0.5, 0.25]. Used for labelling
    q-points when the space group can't be calculated
    """
    label_strings = ['0', '1/4', '3/4', '1/2', '1/3', '2/3', '3/8', '5/8']
    label_coords = [0., 0.25, 0.75, 0.5, 1./3., 2./3., 0.375, 0.625]

    generic_labels = {}
    for i, s1 in enumerate(label_strings):
        for j, s2 in enumerate(label_strings):
            for k, s3 in enumerate(label_strings):
                key = s1 + ' ' + s2 + ' ' + s3
                value = [label_coords[i], label_coords[j], label_coords[k]]
                generic_labels[key] = value
    return generic_labels


def get_qpt_label(qpt, point_labels):
    """
    Gets a label for a particular q-point, based on the high symmetry points
    of a particular space group. Used for labelling the dispersion plot x-axis

    Parameters
    ----------
    qpt : (3,) float ndarray
        3 dimensional coordinates of a q-point
    point_labels : dictionary
        A dictionary with N entries, relating high symmetry point lables (e.g.
        'GAMMA', 'X'), to their 3-dimensional coordinates (e.g. [0.0, 0.0,
        0.0]) where N = number of high symmetry points for a particular space
        group

    Returns
    -------
    label : string
        The label for this q-point. If the q-point isn't a high symmetry point
        label is just an empty string
    """

    # Normalise qpt to [0,1]
    qpt_norm = [x - math.floor(x) for x in qpt]

    # Split dict into keys and values so labels can be looked up by comparing
    # q-point coordinates with the dict values
    labels = list(point_labels)
    # Ensure symmetry points in label_keys and label_values are in the same
    # order (not guaranteed if using .values() function)
    label_coords = [point_labels[x] for x in labels]

    # Check for matching symmetry point coordinates (roll q-point coordinates
    # if no match is found)
    TOL = 1e-6
    matching_label_index = np.where((np.isclose(
        label_coords, qpt_norm, atol=TOL)).all(axis=1))[0]
    if matching_label_index.size == 0:
        matching_label_index = np.where((np.isclose(
            label_coords, np.roll(qpt_norm, 1), atol=TOL)).all(axis=1))[0]
    if matching_label_index.size == 0:
        matching_label_index = np.where((np.isclose(
            label_coords, np.roll(qpt_norm, 2), atol=TOL)).all(axis=1))[0]

    label = ''
    if matching_label_index.size > 0:
        label = labels[matching_label_index[0]]

    return label


def plot_sqw_map(data, vmin=None, vmax=None, ratio=None, ewidth=0, qwidth=0,
                 cmap='viridis', title=''):
    """
    Plots an q-E scattering plot using imshow

    Parameters
    ----------
    data : PhononData or InterpolationData object
        Data object for which calculate_sqw_map has been called, containing
        sqw_map and sqw_ebins attributes for plotting
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
    ewidth : float, optional, default 0
        The FWHM of the Gaussian energy resolution function in the same units
        as freqs
    qwidth : float, optional, default 0
        The FWHM of the Gaussian q-vector resolution function
    cmap : string, optional, default 'viridis'
        Which colormap to use, see Matplotlib docs
    title : string, optional
        The figure title. Default: ''

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        If matplotlib.pyplot can be imported, is a Figure with a single
        subplot, otherwise is None
    ims : (n_qpts,) 'matplotlib.image.AxesImage' ndarray or None
        If matplotlib.pyplot can be imported, is an array of AxesImage objects,
        one for each q-point, for easier access to some attributes/functions.
        Otherwise is None
    """
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn(('Cannot import Matplotlib to plot S(q,w) (maybe '
                       'Matplotlib is not installed?). To install Euphonic\'s'
                       ' optional Matplotlib dependency, try:\n\npip install'
                       ' euphonic[matplotlib]\n'))
        raise

    ebins = data.sqw_ebins.magnitude
    # Apply broadening
    if ewidth or qwidth:
        qbin_width = np.linalg.norm(
            np.mean(np.absolute(np.diff(data.qpts, axis=0)), axis=0))
        qbins = np.linspace(
            0, qbin_width*data.n_qpts + qbin_width, data.n_qpts + 1)
        # If no width has been set, make widths small enough to have
        # effectively no broadening
        if not qwidth:
            qwidth = (qbins[1] - qbins[0])/10
        if not ewidth:
            ewidth = (ebins[1] - ebins[0])/10
        sqw_map = signal.fftconvolve(data.sqw_map, np.transpose(
            gaussian_2d(qbins, ebins, qwidth, ewidth)), 'same')
    else:
        sqw_map = data.sqw_map

    # Calculate qbin edges
    cell_vec = (data.cell_vec.to('angstrom').magnitude)
    recip = reciprocal_lattice(cell_vec)
    abscissa = calc_abscissa(data.qpts, recip)
    qmid = (abscissa[1:] + abscissa[:-1])/2
    qwidths = qmid + qmid[0]
    qbins = np.concatenate(([0], qwidths, [2*qwidths[-1] - qwidths[-2]]))

    if ratio:
        ymax = qbins[-1]/ratio
    else:
        ymax = 1.0
    if vmin is None:
        vmin = np.amin(sqw_map)
    if vmax is None:
        vmax = np.amax(sqw_map)

    fig, ax = plt.subplots(1, 1)
    ims = np.empty((data.n_qpts), dtype=mpl.image.AxesImage)
    for i in range(data.n_qpts):
        ims[i] = ax.imshow(np.transpose(sqw_map[i, np.newaxis]),
                           interpolation='none', origin='lower',
                           extent=[qbins[i], qbins[i+1], 0, ymax],
                           vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_ylim(0, ymax)
    ax.set_xlim(qbins[0], qbins[-1])

    # Calculate energy tick labels
    ytick_spacing_opts = [100, 50, 20, 10, 5, 2, 1, 0.1]
    min_n_yticks = 5
    ytick_spacing = ytick_spacing_opts[np.argmax((
        ebins[-1] - ebins[0])/ytick_spacing_opts > min_n_yticks)]
    ytick_min = np.ceil(ebins[0]/ytick_spacing)*ytick_spacing
    ytick_max = np.ceil(ebins[-1]/ytick_spacing)*ytick_spacing
    ylabels = np.arange(ytick_min, ytick_max, ytick_spacing)
    yticks = (ylabels - ebins[0])/(ebins[-1] - ebins[0])*ymax
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    units_str = data._e_units
    inverse_unit_index = units_str.find('/')
    if inverse_unit_index > -1:
        units_str = units_str[inverse_unit_index+1:]
        ax.set_ylabel('Energy (' + units_str + r'$^{-1}$)')
    else:
        ax.set_ylabel('Energy (' + units_str + ')')

    # Calculate q-space ticks and labels
    xlabels, qpts_with_labels = recip_space_labels(data)
    for i, label in enumerate(xlabels):
        if label == 'GAMMA':
            xlabels[i] = r'$\Gamma$'
    if np.all(xlabels == ''):
        xlabels = np.around(data.qpts[qpts_with_labels, :], decimals=2)
    xticks = (qbins[qpts_with_labels] + qbins[qpts_with_labels + 1])/2
    # Set high symmetry point x-axis ticks/labels
    ax.set_xticks(xticks)
    ax.xaxis.grid(True, which='major')
    # Convert xlabels to list from Numpy array to avoid elementwise
    # comparison FutureWarning when calling set_xticklabels
    if not isinstance(xlabels, list):
        xlabels = xlabels.tolist()
    # Rotate long tick labels
    if len(max(xlabels, key=len)) >= 11:
        ax.set_xticklabels(xlabels, rotation=90)
    else:
        ax.set_xticklabels(xlabels)

    fig.suptitle(title)

    return fig, ims


def output_grace(data, seedname='out', up=True, down=True):
    """
    Creates a .agr Grace file of the band structure

    Parameters
    ----------
    data: Data object
        Data object containing the frequencies and other data required for
        plotting (qpts, n_ions, cell_vecs)
    seedname : string, optional
        Determines the figure title and output file name, seedname.agr.
        Default: 'out'
    up : boolean, optional
        Whether to plot spin up frequencies (if applicable). Default: True
    down : boolean, optional
        Whether to plot spin down frequencies (if applicable). Default: True
    """
    try:
        from PyGrace.grace import Grace
    except ImportError:
        warnings.warn(('PyGrace is not installed, attempting to write .agr '
                       'Grace file anyway. If using Python 2, you can install'
                       ' PyGrace from https://github.com/pygrace/pygrace'),
                      stacklevel=2)

    # Do calculations required for axis, tick labels etc.
    # Calculate distance along x axis
    cell_vec = (data.cell_vec.to('angstrom').magnitude)
    recip_latt = reciprocal_lattice(cell_vec)
    abscissa = calc_abscissa(data.qpts, recip_latt)
    # Calculate x-axis (recip space) ticks and labels
    xlabels, qpts_with_labels = recip_space_labels(data)

    units_str = data._e_units
    inverse_unit_index = units_str.find('/')
    if inverse_unit_index > -1:
        units_str = units_str[inverse_unit_index+1:]
        yaxis_label = '\\f{{Symbol}}e\\f{{}} ({0}\S-1\\N)'.format(units_str)
    else:
        yaxis_label = '\\f{{Symbol}}e\\f{{}} ({0})'.format(units_str)
    # Format tick labels
    for i, label in enumerate(xlabels):
        if label == 'GAMMA':  # Format gamma symbol
            label = '\\f{Symbol}G\\f{}'

    if 'PyGrace' in sys.modules:
        grace = Grace()
        grace.background_fill = 'on'
        graph = grace.add_graph()
        graph.yaxis.label.text = yaxis_label

        graph.xaxis.tick.set_spec_ticks(
            abscissa[qpts_with_labels].tolist(), [], xlabels.tolist())
        graph.xaxis.tick.major_grid = 'on'
        if len(max(xlabels, key=len)) >= 11:
            graph.xaxis.ticklabel.configure(angle=315, char_size=1.0)

        for i in range(data.n_branches):
            if up:
                ds = graph.add_dataset(
                    zip(abscissa, data.freqs[:, i].magnitude))
                ds.line.configure(linewidth=2.0, color=i % 16)
                ds.symbol.shape = 0
            if down and hasattr(data, 'freq_down') and len(data.freq_down) > 0:
                ds = graph.add_dataset(
                    zip(abscissa, data.freq_down[:, i].magnitude))
                ds.line.configure(linewidth=2.0, color=i % 16)
                ds.symbol.shape = 0

        if hasattr(data, 'fermi'):
            for i, ef in enumerate(data.fermi.magnitude):
                ds = graph.add_dataset(zip([0, abscissa[-1]], [ef, ef]))
                ds.line.configure(linewidth=2.0, color=1, linestyle=3)

        graph.set_world_to_limits()
        grace.write_file(seedname + '.agr')

    else:
        with open(seedname + '.agr', 'w') as f:
            f.write('@with g0\n')
            f.write('@title "{0}"\n'.format(seedname))
            f.write('@view 0.150000, 0.250000, 0.700000, 0.850000\n')
            f.write('@world xmin 0\n')
            f.write('@world xmax {0:.3f}\n'.format(abscissa[-1] + 0.002))
            f.write('@world ymin 0\n')
            f.write('@default linewidth 2.0\n')
            f.write('@default char size 1.5\n')
            f.write('@autoscale onread yaxes\n')

            f.write('@yaxis  bar linewidth 2.0\n')
            f.write('@yaxis label "{0}"\n'.format(yaxis_label))
            f.write('@yaxis label char size 1.5\n')
            f.write('@yaxis ticklabel char size 1.5\n')

            f.write('@xaxis  bar linewidth 2.0\n')
            f.write('@xaxis label char size 1.5\n')
            f.write('@xaxis tick major linewidth 1.6\n')
            f.write('@xaxis tick major grid on\n')
            f.write('@xaxis tick spec type both\n')

            f.write('@xaxis tick spec {0:d}\n'.format(len(xlabels)))
            # Rotate long tick labels
            if len(max(xlabels, key=len)) <= 11:
                f.write('@xaxis ticklabel char size 1.5\n')
            else:
                f.write('@xaxis ticklabel char size 1.0\n')
                f.write('@xaxis ticklabel angle 315\n')

            for i, label in enumerate(xlabels):
                f.write('@xaxis tick major {0:d},{1:8.3f}\n'
                        .format(i, abscissa[qpts_with_labels[i]]))
                f.write('@xaxis ticklabel {0:d},"{1}"\n'.format(i, label))

            # Write frequencies
            for i in range(data.n_branches):
                n_sets = 0
                if up:
                    f.write('@target G0.S{0:d}\n'.format(n_sets))
                    f.write('@type xy\n')
                    for j, freq in enumerate(data.freqs[:, i].magnitude):
                        f.write('{0: .15e} {1: .15e}\n'
                                .format(abscissa[j], freq))
                    n_sets += 1
                if (down and hasattr(data, 'freq_down') and
                        len(data.freq_down) > 0):
                    f.write('@target G0.S{0:d}\n'.format(n_sets))
                    f.write('@type xy\n')
                    for j, freq in enumerate(data.freq_down[:, i].magnitude):
                        f.write('{0: .15e} {1: .15e}\n'
                                .format(abscissa[j], freq))
                    n_sets += 1
                f.write('&\n')

            # Write Fermi level
            if hasattr(data, 'fermi'):
                for i, ef in enumerate(data.fermi.magnitude):
                    f.write('@ G0.S{0:d} line linestyle 3\n'.
                            format(data.n_branches + i))
                    f.write('@ G0.S{0:d} line color 1\n'
                            .format(data.n_branches + i))
                    f.write('@target G0.S{0:d}\n'.format(data.n_branches + i))
                    f.write('@type xy\n')
                    f.write('{0: .15e} {1: .15e}\n'.format(0, ef))
                    f.write('{0: .15e} {1: .15e}\n'.format(abscissa[-1], ef))
                    f.write('&\n')


def plot_dispersion(data, title='', btol=10.0, up=True, down=True,
                    **line_kwargs):
    """
    Creates a Matplotlib figure of the band structure

    Parameters
    ----------
    data: Data object
        Data object containing the frequencies and other data required for
        plotting (qpts, n_ions, cell_vecs)
    title : string, optional
        The figure title. Default: ''
    btol : float, optional
        Determines the limit for plotting sections of reciprocal space on
        different subplots, as a fraction of the median distance between
        q-points. Default: 10.0
    up : boolean, optional
        Whether to plot spin up frequencies (if applicable). Default: True
    down : boolean, optional
        Whether to plot spin down frequencies (if applicable). Default: True
    **line_kwargs : Line2D properties, optional
        Used in the axes.plot command to specify properties like linewidth,
        linestyle
    Returns
    -------
    fig : matplotlib.figure.Figure or None
        If matplotlib.pyplot can be imported, returns a Figure containing
        subplot(s) for the plotted band structure, otherwise returns None.
        If there is a large gap between some q-points there can be multiple
        subplots.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn(('Cannot import Matplotlib to plot dispersion (maybe '
                       'Matplotlib is not installed?). To install Euphonic\'s'
                       ' optional Matplotlib dependency, try:\n\npip install'
                       ' euphonic[matplotlib]\n'))
        raise

    cell_vec = (data.cell_vec.to('angstrom').magnitude)
    recip_latt = reciprocal_lattice(cell_vec)

    abscissa = calc_abscissa(data.qpts, recip_latt)
    # Determine reciprocal space coordinates that are far enough apart to be
    # in separate subplots, and determine index limits
    diff = np.diff(abscissa)
    median = np.median(diff)
    breakpoints = np.where(diff/median > btol)[0]
    imin = np.concatenate(([0], breakpoints + 1))
    imax = np.concatenate((breakpoints, [len(abscissa) - 1]))

    # Calculate width ratios so that the x-scale is the same for each subplot
    subplot_widths = [abscissa[imax[i]] - abscissa[imin[i]]
                      for i in range(len(imax))]
    gridspec = dict(width_ratios=[w/subplot_widths[0]
                                  for w in subplot_widths])
    # Create figure with correct number of subplots
    n_subplots = len(breakpoints) + 1
    fig, subplots = plt.subplots(1, n_subplots, sharey=True,
                                 gridspec_kw=gridspec)
    if n_subplots == 1:
        # Ensure subplots is always an array
        subplots = np.array([subplots])

    # Y-axis formatting, only need to format y-axis for first subplot as they
    # share the y-axis
    # Replace 1/cm with cm^-1
    units_str = data._e_units
    inverse_unit_index = units_str.find('/')
    if inverse_unit_index > -1:
        units_str = units_str[inverse_unit_index+1:]
        subplots[0].set_ylabel('Energy (' + units_str + r'$^{-1}$)')
    else:
        subplots[0].set_ylabel('Energy (' + units_str + ')')
    subplots[0].ticklabel_format(style='sci', scilimits=(-2, 2), axis='y')

    # Configure each subplot
    # Calculate x-axis (recip space) ticks and labels
    xlabels, qpts_with_labels = recip_space_labels(data)
    for i, label in enumerate(xlabels):
        if label == 'GAMMA':
            xlabels[i] = r'$\Gamma$'
    if np.all(xlabels == ''):
        xlabels = np.around(data.qpts[qpts_with_labels, :], decimals=2)
    xticks = abscissa[qpts_with_labels]
    for i, ax in enumerate(subplots):
        # X-axis formatting
        # Set high symmetry point x-axis ticks/labels
        ax.set_xticks(xticks)
        ax.xaxis.grid(True, which='major')
        # Convert xlabels to list from Numpy array to avoid elementwise
        # comparison FutureWarning when calling set_xticklabels
        if not isinstance(xlabels, list):
            xlabels = xlabels.tolist()
        # Rotate long tick labels
        if len(max(xlabels, key=len)) >= 11:
            ax.set_xticklabels(xlabels, rotation=90)
        else:
            ax.set_xticklabels(xlabels)
        ax.set_xlim(left=abscissa[imin[i]], right=abscissa[imax[i]])

        # Plot frequencies and Fermi energy
        if up:
            if hasattr(data, 'split_i') and data.split_i.size > 0:
                split_i = data.split_i
            else:
                split_i = None
            # If there is LO-TO splitting, plot in sections
            if split_i is not None:
                section_i = np.where(np.logical_and(
                    split_i > imin[i], split_i < imax[i]))[0]
                n_sections = section_i.size + 1
            else:
                n_sections = 1

            # Put frequencies in order if reorder_freqs() has been called
            if hasattr(data, '_mode_map'):
                freqs = data.freqs.magnitude[
                    np.arange(len(data.qpts))[:, np.newaxis], data._mode_map]
                if split_i is not None:
                    split_freqs = data.split_freqs.magnitude[
                        np.arange(len(split_i))[:, np.newaxis],
                        data._mode_map[split_i]]
            else:
                freqs = data.freqs.magnitude
                if split_i is not None:
                    split_freqs = data.split_freqs.magnitude

            if n_sections > 1:
                section_edges = np.concatenate(
                    ([imin[i]], split_i[section_i], [imax[i]]))
                for n in range(n_sections):
                    plot_freqs = np.copy(
                        freqs[section_edges[n]:section_edges[n+1] + 1])
                    if n == 0:
                        if (imin[i] in split_i):
                            # First point in this subplot is split gamma point
                            # Replace freqs with split freqs at gamma point
                            split_idx = np.where(split_i == imin[i])[0][0]
                            plot_freqs[0] = split_freqs[split_idx]
                    else:
                        # Replace freqs with split freqs at gamma point
                        plot_freqs[0] = split_freqs[section_i[n-1]]
                    ax.plot(abscissa[section_edges[n]:section_edges[n+1] + 1],
                            plot_freqs, lw=1.0, **line_kwargs)
            else:
                ax.plot(abscissa[imin[i]:imax[i] + 1],
                        freqs[imin[i]:imax[i] + 1], lw=1.0, **line_kwargs)
        if down and hasattr(data, 'freq_down') and len(data.freq_down) > 0:
            freq_down = data.freq_down.magnitude
            ax.plot(abscissa[imin[i]:imax[i] + 1],
                    freq_down[imin[i]:imax[i] + 1], lw=1.0, **line_kwargs)
        if hasattr(data, 'fermi'):
            for i, ef in enumerate(data.fermi.magnitude):
                if i == 0:
                    ax.axhline(y=ef, ls='dashed', c='k',
                               label=r'$\epsilon_F$')
                else:
                    ax.axhline(y=ef, ls='dashed', c='k')

    # Only set legend for last subplot, they all have the same legend labels
    if hasattr(data, 'fermi'):
        subplots[-1].legend()

    # Make sure axis/figure titles aren't cut off. Rect is used to leave some
    # space at the top of the figure for suptitle
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig
