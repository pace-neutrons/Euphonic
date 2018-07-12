import math
import numpy as np
import matplotlib.pyplot as plt
import seekpath
from casteppy.dispersion.dispersion import direction_changed


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


def recip_space_labels(qpts, cell_vec, ion_pos, ion_type):
    """
    Gets high symmetry point labels (e.g. GAMMA, X, L) for the q-points at
    which the path through reciprocal space changes direction

    Parameters
    ----------
    qpts : list of floats
        N x 3 list of the q-point coordinates, where N = number of q-points
    cell_vec : list of floats
        3 x 3 list of the unit cell vectors
    ion_pos : list of floats
        n_ions x 3 list of the fractional position of each ion within the
        unit cell
    ion_type : list of strings
        n_ions length list of the chemical symbols of each ion in the unit
        cell. Ions are in the same order as in ion_pos

    Returns
    -------
    labels : list of strings
        List of the labels for each q-point at which the path through
        reciprocal space changes direction
    qpts_with_labels : list of integers
        List of the indices of the q-points at which the path through
        reciprocal space changes direction
    """

    # First and last q-points should always be labelled
    qpt_has_label = np.concatenate(([True], direction_changed(qpts), [True]))
    qpts_with_labels = np.where(qpt_has_label)[0]

    # Get dict of high symmetry point labels to their coordinates for this
    # space group. If space group can't be determined use a generic dictionary
    # of fractional points
    sym_label_to_coords = {}
    if len(ion_pos) > 0:
        _, ion_num = np.unique(ion_type, return_inverse=True)
        cell = (cell_vec, ion_pos, ion_num)
        sym_label_to_coords = seekpath.get_path(cell)["point_coords"]
    else:
        sym_label_to_coords = generic_qpt_labels()

    # Get labels for each q-point
    labels = np.array([])

    for qpt in qpts[qpts_with_labels]:
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
    qpt : list of floats
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

    label = '';
    if matching_label_index.size > 0:
        label = labels[matching_label_index[0]]

    return label


def plot_dispersion(abscissa, freq_up, freq_down, units, title='', xticks=None,
                    xlabels=None, fermi=[], btol=10.0):
    """
    Creates a Matplotlib figure of the band structure

    Parameters
    ----------
    abscissa: list of floats
        M length list of the position of each q-point along the x-axis based
        on the distance between each q-point (can be calculated by the
        calc_abscissa function), where M = number of q-points
    freq_up : list of floats
        M x N list of spin up band frequencies, where M = number of q-points
        and N = number of bands, can be empty if only spin down frequencies are
        present
    freq_down : list of floats
        M x N list of spin down band frequencies, where M = number of q-points
        and N = number of bands, can be empty if only spin up frequencies are
        present
    units : string
        String specifying the frequency units. Used for axis labels
    title : string
        The figure title. Default: ''
    xticks : list of floats
        List of floats specifying the x-axis tick label locations. Usually
        they are located where the q-point direction changes, this can be
        calculated using abscissa[qpts_with_labels], where abscissa has been
        calculated from the calc_abscissa function, and qpts_with_labels from
        the recip_space_labels function. Default: None
    xlabels : list of strings
        List of strings specifying the x-axis tick labels. Should be the same
        length as xlabels, and can be calculated using the recip_space_labels
        function. Default: None
    fermi : list of floats
        1 or 2 length list specifying the fermi energy/energies. Default: []
    btol : float
        Determines the limit for plotting sections of reciprocal space on
        different subplots, as a fraction of the median distance between
        q-points

    Returns
    -------
    fig : Matplotlib Figure
        Figure containing subplot(s) for the plotted band structure. If there
        is a large gap between some q-points there will be multiple subplots
    """

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
    #subplot_widths = abscissa[imax] - abscissa[imin]
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
    inverse_unit_index = units.find('/')
    if inverse_unit_index > -1:
        units = units[inverse_unit_index+1:]
        subplots[0].set_ylabel('Energy (' + units + r'$^{-1}$)')
    else:
        subplots[0].set_ylabel('Energy (' + units + ')')
    subplots[0].ticklabel_format(style='sci', scilimits=(-2, 2), axis='y')

    # Configure each subplot
    for i, ax in enumerate(subplots):
        # X-axis formatting
        # Set high symmetry point x-axis ticks/labels
        if xticks is not None:
            ax.set_xticks(xticks)
            ax.xaxis.grid(True, which='major')
            # Rotate long tick labels
            if len(max(xlabels, key=len)) >= 11:
                ax.set_xticklabels(xlabels, rotation=90)
            else:
                ax.set_xticklabels(xlabels)
        ax.set_xlim(left=abscissa[imin[i]], right=abscissa[imax[i]])

        # Plot frequencies and Fermi energy
        if len(freq_up) > 0:
            ax.plot(abscissa[imin[i]:imax[i] + 1],
                    freq_up[imin[i]:imax[i] + 1], lw=1.0)
        if len(freq_down) > 0:
            ax.plot(abscissa[imin[i]:imax[i] + 1],
                    freq_down[imin[i]:imax[i] + 1], lw=1.0)
        if len(fermi) > 0:
            for i, ef in enumerate(fermi):
                if i == 0:
                    ax.axhline(y=ef, ls='dashed', c='k',
                               label=r'$\epsilon_F$')
                else:
                    ax.axhline(y=ef, ls='dashed', c='k')

    # Only set legend for last subplot, they all have the same legend labels
    if len(fermi) > 0:
        subplots[-1].legend()

    # Make sure axis/figure titles aren't cut off. Rect is used to leave some
    # space at the top of the figure for suptitle
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig
