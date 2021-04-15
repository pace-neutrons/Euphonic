import json
import math
import sys
import warnings
import os.path
from typing import Dict

import numpy as np
import seekpath
from seekpath.hpkot import SymmetryDetectionError
from importlib_resources import open_text  # Backport for Python 3.6
from pint import UndefinedUnitError

from euphonic import ureg, Quantity
from euphonic.crystal import Crystal
import euphonic.data


def direction_changed(qpts, tolerance=5e-6):
    """
    Takes a N length list of q-points and returns an N - 2 length list of
    booleans indicating whether the direction has changed between each pair
    of q-points
    """

    # Get vectors between q-points
    delta = np.diff(qpts, axis=0)

    # Dot each vector with the next to determine the relative direction
    dot = np.einsum('ij,ij->i', delta[1:, :], delta[:-1, :])
    # Get magnitude of each vector
    modq = np.linalg.norm(delta, axis=1)
    # Determine how much the direction has changed (dot) relative to the
    # vector sizes (modq)
    direction_changed = (np.abs(np.abs(dot) - modq[1:]*modq[:-1]) > tolerance)

    return direction_changed


def is_gamma(qpt):
    """
    Determines whether the given point(s) are gamma points

    Parameters
    ----------
    qpts: (3,) or (N, 3) float ndarray
        The q-point or q-points

    Returns
    -------
    isgamma: bool or (N,) bool ndarray
        Whether the input q-points(s) are gamma points. Returns a scalar
        if only 1 q-point is provided
    """
    tol = 1e-15
    isgamma = np.sum(np.absolute(qpt - np.rint(qpt)), axis=-1) < tol
    return isgamma


def mp_grid(grid):
    """
    Returns the q-points on a MxNxL Monkhorst-Pack grid specified by
    grid

    Parameters
    ----------
    grid : (3,) int ndarray
        Length 3 array specifying the number of points in each direction

    Returns
    -------
    qgrid : (M*N*L, 3) float ndarray
        Q-points on an MP grid
    """

    # Monkhorst-Pack grid: ur = (2r-qr-1)/2qr where r=1,2..,qr
    qh = np.true_divide(
        2*(np.arange(grid[0]) + 1) - grid[0] - 1, 2*grid[0])
    qh = np.repeat(qh, grid[1]*grid[2])
    qk = np.true_divide(
        2*(np.arange(grid[1]) + 1) - grid[1] - 1, 2*grid[1])
    qk = np.tile(np.repeat(qk, grid[2]), grid[0])
    ql = np.true_divide(
        2*(np.arange(grid[2]) + 1) - grid[2] - 1, 2*grid[2])
    ql = np.tile(ql, grid[0]*grid[1])
    return np.column_stack((qh, qk, ql))


def get_all_origins(max_xyz, min_xyz=[0, 0, 0], step=1):
    """
    Given the max/min number of cells in each direction, get a list of
    all possible cell origins

    Parameters
    ----------
    max_xyz : (3,) int ndarray
        The number of cells to count to in each direction
    min_xyz : (3,) int ndarray, optional
        The cell number to count from in each direction
    step : integer, optional
        The step between cells

    Returns
    -------
    origins : (prod(max_xyz - min_xyz)/step, 3) int ndarray
        The cell origins
    """
    diff = np.absolute(np.subtract(max_xyz, min_xyz))
    nx = np.repeat(range(min_xyz[0], max_xyz[0], step), diff[1]*diff[2])
    ny = np.repeat(np.tile(range(min_xyz[1], max_xyz[1], step), diff[0]),
                   diff[2])
    nz = np.tile(range(min_xyz[2], max_xyz[2], step), diff[0]*diff[1])

    return np.column_stack((nx, ny, nz))


def get_qpoint_labels(qpts, cell=None):
    """
    Gets q-point labels (e.g. GAMMA, X, L) for the q-points at which the
    path through reciprocal space changes direction

    Parameters
    ----------
    qpts : (n_qpts, 3) float ndarray
        The q-points to get labels for
    cell : (list, list, list), optional
        The cell structure as defined by spglib. Can be obtained by
        Crystal.to_spglib_cell. If not provided, the labels will be
        generic e.g. '1/3 1/2 0' rather than high-symmetry point labels

    Returns
    -------
    x_tick_labels : list (int, string) tuples or None
        Tick labels and the q-point indices that they apply to
    """
    xlabels, qpts_with_labels = _recip_space_labels(qpts, cell=cell)
    for i, label in enumerate(xlabels):
        if label == 'GAMMA':
            xlabels[i] = r'$\Gamma$'
    if np.all(xlabels == ''):
        xlabels = [str(x) for x in np.around(
            qpts[qpts_with_labels, :], decimals=2)]
    qpts_with_labels = [int(x) for x in qpts_with_labels.tolist()]
    return list(zip(qpts_with_labels, xlabels))


def get_reference_data(collection: str = 'Sears1992',
                       physical_property: str = 'coherent_scattering_length'
                       ) -> Dict[str, Quantity]:
    """
    Get physical data as a dict of (possibly-complex) floats from reference
    data.

    Each "collection" refers to a JSON file which may contain any set of
    properties, indexed by physical_property.

    Properties are stored in JSON files, encoding a single dictionary with the
    structure::

      {"metadata1": "metadata1 text", "metadata2": ...,
       "physical_properties": {"property1": {"__units__": "unit_str",
                                             "H": H_property1_value,
                                             "He": He_property1_value,
                                             "Li": {"__complex__": true,
                                                    "real": Li_property1_real,
                                                    "imag": Li_property1_imag},
                                             "Nh": None,
                                             ...},
                               "property2": ...}}

    Parameters
    ----------
    collection
        Identifier of data file; this may be an inbuilt data set ("Sears1992"
        or "BlueBook") or a path to a JSON file (e.g. "./my_custom_data.json").

    physical_property
        The name of the property for which data should be extracted. This must
        match an entry of "physical_properties" in the data file.

    Returns
    -------
    Dict[str, Quantity]
        Requested data as a dict with string keys and (possibly-complex)
        float Quantity values. String or None items of the original data file
        will be omitted.

    """

    _reference_data_files = {'Sears1992': 'sears-1992.json',
                             'BlueBook': 'bluebook.json'}

    def custom_decode(dct):
        if '__complex__' in dct:
            return complex(dct['real'], dct['imag'])
        return dct

    if collection in _reference_data_files:
        filename = _reference_data_files[collection]
        with open_text(euphonic.data, filename) as fd:
            file_data = json.load(fd, object_hook=custom_decode)

    elif os.path.isfile(collection):
        filename = collection
        with open(filename) as fd:
            file_data = json.load(fd, object_hook=custom_decode)
    else:
        raise ValueError(
            f'No data files known for collection "{collection}". '
            f'Available collections: '
            + ', '.join(list(_reference_data_files)))

    if 'physical_property' not in file_data:
        raise AttributeError('Data file does not contain required key '
                             '"physical_property".')

    data = file_data['physical_property'].get(physical_property)
    if data is None:
        raise ValueError(
            f'No such collection "{collection}" with property '
            f'"{physical_property}". Available properties for this collection'
            ': ' + ', '.join(list(file_data["physical_property"].keys())))

    unit_str = data.get('__units__')
    if unit_str is None:
        raise ValueError(f'Reference data file "{filename}" does not '
                         'specify dimensions with "__units__" metadata.')

    try:
        unit = ureg[unit_str]
    except UndefinedUnitError:
        raise ValueError(
            f'Units "{unit_str}" from data file "{filename}" '
            'are not supported by the Euphonic unit register.')

    return {key: value * unit
            for key, value in data.items()
            if isinstance(value, (float, complex))}


def mode_gradients_to_widths(mode_gradients: Quantity, cell_vectors: Quantity
                             ) -> Quantity:
    """
    Converts mode gradients (units energy/(length^-1)) to an estimate of the
    mode widths (units energy) by using the cell volume and number of q-points
    to estimate the q-spacing. Note that the  number of q-points is determined
    by the size of mode_gradients, so is not likely to give accurate widths if
    the q-points have been symmetry reduced.

    Parameters
    ----------
    mode_gradients
        Shape (n_qpts, n_modes) float Quantity. The mode gradients.
    cell_vectors
        Shape (3, 3) float Quantity. The cell vectors
    """
    cell_volume = Crystal.from_cell_vectors(cell_vectors)._cell_volume()
    modg = mode_gradients.to('hartree*bohr').magnitude
    q_spacing = 2/(np.cbrt(len(mode_gradients)*cell_volume))
    mode_widths = q_spacing*modg
    return mode_widths*ureg('hartree').to(
        mode_gradients.units/cell_vectors.units)


def _calc_abscissa(reciprocal_cell, qpts):
    """
    Calculates the distance between q-points (e.g. to use as a plot
    x-coordinate)

    Parameters
    ----------
    reciprocal_cell : (3, 3) float Quantity
        The reciprocal cell, can be calculated with
        Crystal.reciprocal_cell
    qpts : (n_qpts, 3) float ndarray
        The q-points to get the distance between, in reciprocal lattice
        units

    abscissa : (n_qpts) float Quantity
        The distance between q-points in 1/crystal.cell_vectors_unit
    """
    recip = reciprocal_cell.to('1/bohr').magnitude
    # Get distance between q-points in each dimension
    # Note: length is nqpts - 1
    delta = np.diff(qpts, axis=0)

    # Determine how close delta is to being an integer. As q = q + G
    # where G is a reciprocal lattice vector based on {1,0,0},{0,1,0},
    # {0,0,1}, this is used to determine whether 2 q-points are
    # equivalent ie. they differ by a multiple of the reciprocal lattice
    # vector
    delta_rem = np.sum(np.abs(delta - np.rint(delta)), axis=1)

    # Create a boolean array that determines whether to calculate the
    # distance between q-points,taking into account q-point equivalence.
    # If delta is more than the tolerance, but delta_rem is less than
    # the tolerance, the q-points differ by G so are equivalent and the
    # distance shouldn't be calculated
    TOL = 0.001
    calc_modq = np.logical_not(np.logical_and(
        np.sum(np.abs(delta), axis=1) > TOL,
        delta_rem < TOL))

    # Multiply each delta by the recip lattice to get delta in Cartesian
    deltaq = np.einsum('ji,kj->ki', recip, delta)

    # Get distance between q-points for all valid pairs of q-points
    modq = np.zeros(np.size(delta, axis=0))
    modq[calc_modq] = np.sqrt(np.sum(np.square(deltaq[calc_modq]), axis=1))

    # Prepend initial x axis value of 0
    abscissa = np.insert(modq, 0, 0.)

    # Do cumulative sum to get position along x axis
    abscissa = np.cumsum(abscissa)
    return abscissa*ureg('1/bohr').to(reciprocal_cell.units)


def _recip_space_labels(qpts, cell=None):
    """
    Gets q-points point labels (e.g. GAMMA, X, L) for the q-points at
    which the path through reciprocal space changes direction

    Parameters
    ----------
    qpts : (n_qpts, 3) float ndarray
        The q-points to get labels for
    cell : (list, list, list), optional
        The cell structure as defined by spglib. Can be obtained by
        Crystal.to_spglib_cell. If not provided, the labels will be
        generic e.g. '1/3 1/2 0' rather than high-symmetry point labels

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
    if len(qpts) <= 2:
        qpt_has_label = np.ones(len(qpts), dtype=bool)
    else:
        qpt_has_label = np.concatenate((
            [True],
            np.logical_or(direction_changed(qpts), is_gamma(qpts[1:-1])),
            [True]))
    qpts_with_labels = np.where(qpt_has_label)[0]

    if cell is None:
        sym_label_to_coords = _generic_qpt_labels()
    else:
        try:
            sym_label_to_coords = seekpath.get_path(cell)["point_coords"]
        except SymmetryDetectionError:
            warnings.warn(('Could not determine cell symmetry, using generic '
                           'q-point labels'), stacklevel=2)
            sym_label_to_coords = _generic_qpt_labels()

    # Get labels for each q-point
    labels = np.array([])
    for qpt in qpts[qpts_with_labels]:
        labels = np.append(labels, _get_qpt_label(qpt, sym_label_to_coords))

    return labels, qpts_with_labels


def _generic_qpt_labels():
    """
    Returns a dictionary relating fractional q-point label strings to
    their coordinates e.g. '1/4 1/2 1/4' = [0.25, 0.5, 0.25]. Used for
    labelling q-points when the space group can't be calculated
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


def _get_qpt_label(qpt, point_labels):
    """
    Gets a label for a particular q-point, based on the high symmetry
    points of a particular space group. Used for labelling the
    dispersion plot x-axis

    Parameters
    ----------
    qpt : (3,) float ndarray
        3 dimensional coordinates of a q-point
    point_labels : dictionary
        A dictionary with N entries, relating high symmetry point labels
        (e.g. 'GAMMA', 'X'), to their 3-dimensional coordinates (e.g.
        [0.0, 0.0, 0.0]) where N = number of high symmetry points for a
        particular space group

    Returns
    -------
    label : string
        The label for this q-point. If the q-point isn't a high symmetry
        point label is just an empty string
    """

    # Normalise qpt to [0,1]
    qpt_norm = [x - math.floor(x) for x in qpt]

    # Split dict into keys and values so labels can be looked up by
    # comparing q-point coordinates with the dict values
    labels = list(point_labels)
    # Ensure symmetry points in label_keys and label_values are in the
    # same order (not guaranteed if using .values() function)
    label_coords = [point_labels[x] for x in labels]

    # Check for matching symmetry point coordinates (roll q-point
    # coordinates if no match is found)
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


def _get_supercell_relative_idx(cell_origins, sc_matrix):
    """"
    For each cell_origins[i] -> cell_origins[j] vector in the supercell,
    gets the index n of the equivalent cell_origins[n] vector, where
    supercell_relative_idx[i, j] = n. Is required for converting from a
    compact to full force constants matrix

    Parameters
    ----------
    cell_origins : (n_cells, 3) int ndarray
        The vector to the origin of each cell in the supercell, in unit
        cell fractional coordinates
    sc_matrix : (3, 3) int ndarray
        The matrix for converting from the unit cell to the supercell

    Returns
    -------
    supercell_relative_idx : (n_cells, n_cells) in ndarray
        The index n of the equivalent vector in cell_origins
    """
    n_cells = len(cell_origins)
    ax = np.newaxis

    # Get cell origins in supercell fractional coordinates
    inv_sc_matrix = np.linalg.inv(np.transpose(sc_matrix))
    cell_origins_sc = np.einsum('ij,kj->ik', cell_origins, inv_sc_matrix)
    sc_relative_index = np.zeros((n_cells, n_cells), dtype=np.int32)
    for nc in range(n_cells):
        # Get vectors from cell origin for a particular cell to all
        # other cell origins
        inter_cell_vectors = (cell_origins_sc
                              - np.tile(cell_origins_sc[nc], (n_cells, 1)))
        # Compare cell-cell vectors with vectors from cell 0's cell
        # origin to all other cell origins and determine which are
        # equivalent
        # Do calculation in chunks, so loop can be broken if all
        # equivalent vectors have been found
        N = 100
        dist_min = np.full((n_cells), sys.float_info.max)
        for i in range(int((n_cells - 1)/N) + 1):
            ci = i*N
            cf = min((i + 1)*N, n_cells)
            dist = (inter_cell_vectors[:, ax, :]
                    - cell_origins_sc[ax, ci:cf, :])
            dist_frac = dist - np.rint(dist)
            dist_frac_sum = np.sum(np.abs(dist_frac), axis=2)
            scri_current = np.argmin(dist_frac_sum, axis=1)
            dist_min_current = dist_frac_sum[
                range(n_cells), scri_current]
            replace = dist_min_current < dist_min
            sc_relative_index[nc, replace] = ci + scri_current[replace]
            dist_min[replace] = dist_min_current[replace]
            if np.all(dist_min <= 16*sys.float_info.epsilon):
                break
        if np.any(dist_min > 16*sys.float_info.epsilon):
            raise Exception('Couldn\'t find supercell relative index')
    return sc_relative_index
