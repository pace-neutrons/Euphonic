import math
import numpy as np
import seekpath
from euphonic import ureg


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


def get_qpoint_labels(crystal, qpts):
    """
    Gets q-points point labels (e.g. GAMMA, X, L) for the q-points at
    which the path through reciprocal space changes direction

    Parameters
    ----------
    crystal : Crystal
        The crystal to get high-symmetry labels for
    qpts : (n_qpts, 3) float ndarray
        The q-points to get labels for

    Returns
    -------
    x_tick_labels : list (int, string) tuples or None
        Tick labels and the q-point indices that they apply to
    """
    xlabels, qpts_with_labels = _recip_space_labels(crystal, qpts)
    for i, label in enumerate(xlabels):
        if label == 'GAMMA':
            xlabels[i] = r'$\Gamma$'
    if np.all(xlabels == ''):
        xlabels = [str(x) for x in np.around(
            qpts[qpts_with_labels, :], decimals=2)]
    qpts_with_labels = [int(x) for x in qpts_with_labels.tolist()]
    return list(zip(qpts_with_labels, xlabels))


def _calc_abscissa(crystal, qpts):
    """
    Calculates the distance between q-points (to use as a plot
    x-coordinate)

    Parameters
    ----------
    crystal : Crystal
        The crystal
    qpts : (n_qpts, 3) float ndarray
        The q-points to get the distance between, in reciprocal lattice
        units

    abscissa : (n_qpts) float Quantity
        The distance between q-points in 1/crystal.cell_vectors_unit
    """
    recip = crystal.reciprocal_cell().to('1/INTERNAL_LENGTH_UNIT').magnitude
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

    # Do cumulative some to get position along x axis
    abscissa = np.cumsum(abscissa)
    return abscissa*ureg('1/INTERNAL_LENGTH_UNIT').to(
        1/ureg(crystal.cell_vectors_unit))


def _recip_space_labels(crystal, qpts, symmetry_labels=True):
    """
    Gets q-points point labels (e.g. GAMMA, X, L) for the q-points at
    which the path through reciprocal space changes direction

    Parameters
    ----------
    crystal : Crystal
        The crystal to get high-symmetry labels for
    qpts : (n_qpts, 3) float ndarray
        The q-points to get labels for
    symmetry_labels : boolean, optional
        Whether to use high-symmetry point labels (e.g. GAMMA, X, L).
        Otherwise just uses generic labels (e.g. '0 0 0')

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

    # Get dict of high symmetry point labels to their coordinates for
    # this space group. If space group can't be determined use a generic
    # dictionary of fractional points
    sym_label_to_coords = {}
    if symmetry_labels:
        _, atom_num = np.unique(crystal.atom_type, return_inverse=True)
        cell_vectors = (crystal.cell_vectors.to('angstrom')).magnitude
        cell = (cell_vectors, crystal.atom_r, atom_num)
        sym_label_to_coords = seekpath.get_path(cell)["point_coords"]
    else:
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


def _bose_factor(x, T):
    """
    Calculate the Bose factor

    Parameters
    ----------
    x : (n_qpts, 3*n_ions) float ndarray
        Phonon frequencies in Hartree
    T : float
        Temperature in K

    Returns
    -------
    bose : (n_qpts, 3*n_ions) float ndarray
        Bose factor
    """
    kB = (1*ureg.k).to('E_h/K').magnitude
    bose = np.zeros(x.shape)
    bose[x > 0] = 1
    if T > 0:
        bose = bose + 1/(np.exp(np.absolute(x)/(kB*T)) - 1)
    return bose


def _gaussian(x, sigma):
    return np.exp(-np.square(x)/(2*sigma**2))/(math.sqrt(2*math.pi)*sigma)


def _lorentzian(x, gamma):
    return gamma/(2*math.pi*(np.square(x) + (gamma/2)**2))


def _get_dist_bins(bins, fwhm, extent):
    # Ensure nbins is always odd, and each bin has the same approx width
    # as original x/ybins
    bin_width = np.mean(np.diff(bins))
    nbins = int(np.ceil(2*extent*fwhm/bin_width)/2)*2 + 1
    width = extent*fwhm
    # Prevent xbins from being too large. If user accidentally selects a
    # very large broadening, xwidth and therefore xbins could be
    # extremely large. But for most cases the original nxbins should be
    # smaller
    if nbins > len(bins):
        nbins = int(len(bins)/2)*2 + 1
        width = (bins[-1] - bins[0])/2
    return np.linspace(-width, width, nbins)


def _distribution_1d(xbins, xwidth, shape='gauss', extent=3.0):
    x = _get_dist_bins(xbins, xwidth, extent)
    if shape == 'gauss':
        # Gauss FWHM = 2*sigma*sqrt(2*ln2)
        xsigma = xwidth/(2*math.sqrt(2*math.log(2)))
        dist = _gaussian(x, xsigma)
    elif shape == 'lorentz':
        dist = _lorentzian(x, xwidth)
    else:
        raise Exception(
            f'Distribution shape \'{shape}\' not recognised')
    dist = dist/np.sum(dist) # Naively normalise
    return dist


def _distribution_2d(xbins, ybins, xwidth, ywidth, shape='gauss', extent=3.0):
    x = _get_dist_bins(xbins, xwidth, extent)
    y = _get_dist_bins(ybins, ywidth, extent)

    if shape == 'gauss':
        # Gauss FWHM = 2*sigma*sqrt(2*ln2)
        xsigma = xwidth/(2*math.sqrt(2*math.log(2)))
        ysigma = ywidth/(2*math.sqrt(2*math.log(2)))
        xdist = _gaussian(x, xsigma)
        ydist = _gaussian(y, ysigma)
    elif shape == 'lorentz':
        xdist = _lorentzian(x, xwidth)
        ydist = _lorentzian(y, ywidth)
    else:
        raise Exception(
            f'Distribution shape \'{shape}\' not recognised')
    xgrid = np.tile(xdist, (len(ydist), 1))
    ygrid = np.transpose(np.tile(ydist, (len(xdist), 1)))
    dist = xgrid*ygrid
    dist = dist/np.sum(dist) # Naively normalise

    return dist


def _check_constructor_inputs(objs, types, shapes, names):
    """
    Make sure all the inputs are all the expected type, and if they are
    an array, the correct shape

    Parameters
    ----------
    objs : list of objects
        The objects to check
    types : list of types or lists of types
        The expected class of each input. If multiple types are
        accepted, the expected class can be a list of types. e.g.
        types=[[list, np.ndarray], int]
    shapes : list of tuples
        The expected shape of each object (if the object has a shape
        attribute). If the shape of some dimensions don't matter,
        provide -1 for those dimensions, or if none of the dimensions
        matter, provide an empty tuple (). If multiple shapes are
        accepted, the expected shapes can be a list of tuples. e.g.
        shapes=[[(n, 3), (n + 1, 3)], 3]
    names : list of strings
        The name of each array

    Raises
    ------
    TypeError
        If one of the items in objs isn't the correct type
    ValueError
        If an array shape don't match the expected shape
    """
    for obj, typ, shape, name in zip(objs, types, shapes, names):
        if not isinstance(typ, list):
            typ = [typ]
        if not any(isinstance(obj, t) for t in typ):
            raise TypeError((f'The type of {name} {type(obj)} doesn\'t '
                             f'match the expected type(s) {typ}'))
        if hasattr(obj, 'shape') and shape:
            if not isinstance(shape, list):
                shape = [shape]
            if not any(obj.shape == _replace_dim(s, obj.shape) for s in shape):
                raise ValueError((
                    f'The shape of {name} {obj.shape} doesn\'t match '
                    f'the expected shape(s) {shape}'))


def _replace_dim(expected_shape, obj_shape):
    # Allow -1 dimensions to be any size
    idx = np.where(np.array(expected_shape) == -1)[0]
    if len(idx) == 0 or len(expected_shape) != len(obj_shape):
        return expected_shape
    else:
        expected_shape = np.array(expected_shape)
        expected_shape[idx] = np.array(obj_shape)[idx]
        return tuple(expected_shape)


def _ensure_contiguous_attrs(obj, required_attrs, opt_attrs=[]):
    """
    Make sure all listed attributes of obj are C Contiguous and of the
    correct type (int32, float64, complex128). This should only be used
    internally, and called before any calls to Euphonic C extension
    functions

    Parameters
    ----------
    obj : Object
        The object that will have it's attributes checked
    required_attrs : list of strings
        The attributes of obj to be checked. They should all be Numpy
        arrays
    opt_attrs : list of strings, optional
        The attributes of obj to be checked, but if they don't exist
        will not throw an error. e.g. Depending on the material
        ForceConstants objects may or may not have 'born' defined
    """
    for attr_name in required_attrs:
        attr = getattr(obj, attr_name)
        attr = attr.astype(_get_dtype(attr), order='C', copy=False)
        setattr(obj, attr_name, attr)

    for attr_name in opt_attrs:
        try:
            attr = getattr(obj, attr_name)
            attr = attr.astype(_get_dtype(attr), order='C', copy=False)
            setattr(obj, attr_name, attr)
        except AttributeError:
            pass


def _ensure_contiguous_args(*args):
    """
    Make sure all arguments are C Contiguous and of the correct type
    (int32, float64, complex128). This should only be used internally,
    and called before any calls to Euphonic C extension functions
    Example use: arr1, arr2 = _ensure_contiguous_args(arr1, arr2)

    Parameters
    ----------
    *args : any number of ndarrays
        The Numpy arrays to be checked

    Returns
    -------
    args_contiguous : the same number of ndarrays as args
        The same as the provided args, but all contiguous.
    """
    args = list(args)
    for i in range(len(args)):
        args[i] = args[i].astype(_get_dtype(args[i]), order='C', copy=False)

    return args


def _get_dtype(arr):
   """
   Get the Numpy dtype that should be used for the input array

   Parameters
   ----------
   arr : ndarray
       The Numpy array to get the type of

   Returns
   -------
   dtype : Numpy dtype
       The type the array should be
   """
   if np.issubdtype(arr.dtype, np.integer):
       return np.int32
   elif np.issubdtype(arr.dtype, np.floating):
       return np.float64
   elif np.issubdtype(arr.dtype, np.complexfloating):
       return np.complex128
   return None
