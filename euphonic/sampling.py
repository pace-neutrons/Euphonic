"""Functions for sampling properties in high-dimensional space.

These are implemented in a generic way for application in euphonic.powder
"""

from itertools import product
from typing import Iterator, Tuple

import numpy as np
from scipy.optimize import fmin

_golden_ratio = (1 + np.sqrt(5)) / 2

#  Keep in mind that at this stage the implementations are intended for testing
#  and method development; clarity is prioritised over numerical efficiency


def golden_square(npts: int, offset: bool = True, jitter: bool = False
                  ) -> Iterator[Tuple[float, float]]:
    """Yield a series of well-distributed points in 2-D unit square

    These are obtained by the golden ratio method

    (x_i, y_i) = (i / npts, (i / Phi) % 1)

    Parameters
    ----------
    npts
        Number of points to distribute

    offset
        Offset x-coordinates away from origin. (This is recommended for
        spherical sampling by Marques et al. (2013))

    jitter
        Randomly displace points by a distance of up to 1/(2 sqrt(N))

    Returns
    -------
    Iterator[Tuple[float, float]

        Sequence of (x, y) pairs in range 0-1
    """

    if offset:
        x_offset = 1 / (2 * npts)
    else:
        x_offset = 0

    for i in range(npts):
        if jitter:
            displacement = (np.random.random(2) - 0.5) / np.sqrt(npts)
            delta_x, delta_y = displacement
        else:
            delta_x, delta_y = (0, 0)

        yield ((i / npts + delta_x + x_offset) % 1,
               ((i / _golden_ratio) + delta_y) % 1)


def regular_square(n_rows: int, n_cols: int,
                   offset: bool = True, jitter: bool = False
                   ) -> Iterator[Tuple[float, float]]:
    """Yield a regular grid of (x, y) points in 2-D unit square

    Parameters
    ----------

    n_rows
        number of rows
    n_cols
        number of columns
    offset
        offset points to avoid origin
    jitter
        randomly displace each point within its "cell"

    Returns
    -------
    Iterator[tuple[float, float]]
        sequence of (x, y) pairs in range 0-1
    """

    x_spacing, y_spacing = (1 / n_cols), (1 / n_rows)
    x_sequence = np.arange(n_cols) * x_spacing
    y_sequence = np.arange(n_rows) * y_spacing

    if offset:
        x_offset, y_offset = (x_spacing / 2), (y_spacing / 2)
    else:
        x_offset, y_offset = 0, 0

    for x, y in product(x_sequence, y_sequence):
        if jitter:
            displacement = (np.random.random(2) - 0.5) * [x_spacing, y_spacing]
            delta_x, delta_y = displacement
        else:
            delta_x, delta_y = 0, 0

        yield (x + x_offset + delta_x,
               y + y_offset + delta_y)


def _spherical_polar_to_cartesian(phi, theta):
    return (np.cos(phi) * np.sin(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(theta))


def _square_to_spherical_polar(x: float, y: float) -> Tuple[float, float]:
    """Map from Cartesian square to spherical polar (phi, theta)"""
    theta = np.arccos(2 * x - 1)
    phi = 2 * np.pi * y
    return (phi, theta)


def golden_sphere(npts: int, cartesian: bool = True, jitter: bool = False,
                  ) -> Iterator[Tuple[float, float, float]]:
    """Yield a series of 3D points on unit sphere surface

    These resemble spherical Fibonacci point sets, using the golden
    ratio to obtain an appropriate point set for any positive inter NPTS.
    The method is outlined for "Quasi-Monte Carlo" sampling by
    Marques et al. (2013) DOI:10.1111/cgf.12190

    Parameters
    ----------

    npts
        Number of points at sphere surface

    cartesian
        Yield points in Cartesian coordinates. If False, the 3-tuple
        points are given in spherical coordinates.

    jitter
        Randomly displace points about their positions on surface

    Returns
    -------
    Iterator[Tuple[float, float, float]]

        Sequence of (x, y, z) coordinates (if cartesian=True) or
        (r, phi, theta) spherical coordinates.
    """
    for x, y in golden_square(npts, jitter=jitter):
        phi, theta = _square_to_spherical_polar(x, y)

        if cartesian:
            yield _spherical_polar_to_cartesian(phi, theta)
        else:
            yield (1, phi, theta)


def sphere_from_square_grid(n_rows: int, n_cols: int,
                            cartesian: bool = True, jitter: bool = False
                            ) -> Iterator[Tuple[float, float, float]]:
    """Yield a series of 3D points on a unit sphere surface

    The points are projected from a uniform 2-D grid

    Parameters
    ----------
    n_rows
        Number of rows in the Cartesian generating grid
    n_cols
        Number of columns in the Cartesian generating grid
    cartesian
        Yield points in Cartesian coordinates. If False, instead yield points
        in spherical coordinates.
    jitter
        Randomly displace each point within its own "cell" of the grid

    Returns
    -------
    Iterator[Tuple[float, float, float]]

        Sequence of (x, y, z) coordinates (if cartesian=True) or
        (r, phi, theta) spherical coordinates.
    """
    for x, y in regular_square(n_rows, n_cols, jitter=jitter):
        phi, theta = _square_to_spherical_polar(x, y)

        if cartesian:
            yield _spherical_polar_to_cartesian(phi, theta)
        else:
            yield (1, phi, theta)


def spherical_polar_grid(n_phi: int, n_theta: int,
                         cartesian: bool = True, jitter: bool = False,
                         ) -> Iterator[Tuple[float, float, float]]:
    """Yield a series of 3D points on a unit sphere surface

    The points form a grid in spherical coordinates

    Parameters
    ----------

    n_phi
        Number of longitude-like radial subdivisions at poles

    n_theta
        Number of lattitude-like rings of points dividing polar axis

    cartesian
        Yield points in Cartesian coordinates. If False, instead yield points
        in spherical coordinates.

    jitter
        Randomly displace each point within its own "cell" of the grid

    Returns
    -------
    Iterator[Tuple[float, float, float]]

        Sequence of (x, y, z) coordinates (if cartesian=True) or
        (r, phi, theta) spherical coordinates.
    """
    phi_sequence = np.linspace(-np.pi, np.pi, n_phi + 1)[:-1]
    phi_spacing = phi_sequence[1] - phi_sequence[0]

    # Offset theta samples to avoid redundancy at pole
    theta_sequence = (np.linspace(0, np.pi, n_theta + 1)[:-1]
                      + np.pi / (2 * n_theta))
    theta_spacing = theta_sequence[1] - theta_sequence[0]

    for phi, theta in product(phi_sequence, theta_sequence):
        if jitter:
            displacement = ((np.random.random(2) - 0.5)
                            * [phi_spacing, theta_spacing])
            phi, theta = [phi, theta] + displacement

        if cartesian:
            yield _spherical_polar_to_cartesian(phi, theta)
        else:
            yield (1, phi, theta)


def spherical_polar_improved(npts: int,
                             cartesian: bool = True, jitter: bool = False,
                             ) -> Iterator[Tuple[float, float, float]]:
    """Yield a series of 3D points on a unit sphere surface

    The points form rings of common theta in polar coordinates. However,
    the number of samples in each ring is scaled to achieve as uniform a
    sampling density as possible.

    Analytically we find that for evenly-divided theta, the average
    length of a "latitude" line is 4.

    To obtain parity between nearest-neighbours (NN) along each axis, we
    solve (NN distance along theta) = (NN distance along phi):

    4 n_theta / npts = pi / n_theta

    Exact solutions with integer npts will be an irrational number, so
    n_theta is rounded down. The requested npts is then distributed
    between the constant-theta rings, according to their circumference.

    Parameters
    ----------

    npts
        Number of points at sphere surface

    cartesian
        Yield points in Cartesian coordinates. If False, instead yield
        points in spherical coordinates.

    jitter 
        Randomly displace each point within its own "cell" of the
        irregular grid

    Returns
    -------
    Iterator[Tuple[float, float, float]]

        Sequence of (x, y, z) coordinates (if cartesian=True) or
        (r, phi, theta) spherical coordinates.

    Raises
    ------
    ValueError
        If the number of points is not supported by this method
        """
    if npts < 6:
        raise ValueError("This sampling scheme has a minimum of 6 points")

    # round from the solution of
    # for neighbouring points
    n_theta = int(np.sqrt(np.pi / 4 * npts))

    # Offset theta samples to avoid redundancy at pole
    theta_sequence = (np.linspace(0, np.pi, n_theta + 1)[:-1]
                      + np.pi / (2 * n_theta))
    theta_spacing = theta_sequence[1] - theta_sequence[0]

    # get the lengths of all theta bins and drop points in
    theta_circumferences = 2 * np.pi * np.abs(np.sin(theta_sequence))
    bin_edges = np.concatenate([[0], np.cumsum(theta_circumferences)])

    counts, _ = np.histogram(np.linspace(0, theta_circumferences.sum(), npts),
                             bins=bin_edges, density=False)

    for row_theta, count in zip(theta_sequence, counts):
        phi_sequence = np.linspace(-np.pi, np.pi, count + 1)[:-1]
        phi_spacing = phi_sequence[1] - phi_sequence[0]
        for phi in phi_sequence:
            if jitter:
                displacement = (np.random.random(2) - 0.5) * [phi_spacing,
                                                              theta_spacing]
                phi, theta = [phi, row_theta] + displacement
            else:
                theta = row_theta

            if cartesian:
                yield _spherical_polar_to_cartesian(phi, theta)
            else:
                yield (1, phi, theta)


def random_sphere(npts, cartesian: bool = True
                  ) -> Iterator[Tuple[float, float, float]]:
    """Yield a series of 3D points on a unit sphere surface

    Points are distributed randomly in polar coordinates: phi is
    evenly-distributed from 0 to 2pi while theta is scaled with an
    arccos function from an even distribution over the range (-1, 1), to
    account for warping at the poles.

    Parameters
    ----------

    npts
        Number of points at sphere surface

    cartesian
        Yield points in Cartesian coordinates. If False, instead yield
        points in spherical coordinates.

    Returns
    -------
    Iterator[Tuple[float, float, float]]

        Sequence of (x, y, z) coordinates (if cartesian=True) or
        (r, phi, theta) spherical coordinates.
        """

    points = np.random.random((npts, 2))

    for x, y in points:
        phi, theta = _square_to_spherical_polar(x, y)

        if cartesian:
            yield _spherical_polar_to_cartesian(phi, theta)
        else:
            yield (1, phi, theta)

def recurrence_sequence(npts: int, order=3) -> Iterator[tuple]:
    """Yield a series of well-distributed points in square or cube

    This implements the R_d method of Martin Roberts (2018) published at
    http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/

    The Krcadinac (2005) method defines a set of irrational
    numbers beginning with the golden ratio. (x^{d+1} = x + 1). The number
    corresponding to the desired dimensionality is transformed to irrational
    factors alpha = (1/x_d, (1/x_d)^2, (1/x_d)^3, ...)

    For each of these factors a recurrence sequence t = (n alpha) % 1 yields a
    sequence of n low-discrepancy quasirandom numbers ranging 0 <= t < 1.
    These are packed into n-tuples to form the output n-dimensional samples.

    Parameters
    ----------

    npts
        Number of points sampled in unit of n-dimensional space
    order
        Number of dimensions in sampled space

    Returns
    -------

    n-tuples of floats ranging 0-1, where n=order

    """

    phi = fmin(lambda x: (x + 1. - x**(order + 1))**2,
               1.,
               xtol=np.finfo(float).eps, disp=False)[0]
    alpha = phi**np.arange(-1, -order - 1, -1)

    for n in range(npts):
        yield tuple((n * alpha) % 1)
