"""Functions for sampling properties in high-dimensional space"""
from itertools import product
import numpy as np
from typing import Iterator, Tuple

_golden_ratio = (1 + np.sqrt(5)) / 2


def golden_square(npts: int) -> Iterator[Tuple[float, float]]:
    """Yield a series of well-distributed points in 2-D unit square

    These are obtained by the golden ratio method

      (x_i, y_i) = (i / npts, (i / Phi) % 1)

    Returns:
        Sequence of (x, y) pairs in range 0-1
    """

    for i in range(npts):
        yield (i / npts, (i / _golden_ratio) % 1)


def _spherical_polar_to_cartesian(phi, theta):
    return (np.cos(phi) * np.sin(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(theta))


def golden_sphere(npts: int, cartesian: bool = True
                  ) -> Iterator[Tuple[float, float, float]]:
    """Yield a series of 3D points on unit sphere surface

    These resemble spherical Fibonacci point sets, using the golden
    ratio to obtain an appropriate point set for any positive inter NPTS.
    The method is outlined for "Quasi-Monte Carlo" sampling by
    Marques et al. (2013) DOI:10.1111/cgf.12190

    Args:
        npts: Number of points at sphere surface
        cartesian: Yield points in Cartesian coordinates. If False, the 3-tuple
            points are given in spherical coordinates.

    Returns:
        Sequence of (x, y, z) coordinates (if cartesian=True) or
        (r, phi, theta) spherical coordinates.
    """
    for i in range(npts):
        phi = np.pi * 2 * i / _golden_ratio
        theta = np.arccos(1 - (2 * i + 1) / npts)

        if cartesian:
            yield _spherical_polar_to_cartesian(phi, theta)
        else:
            yield (1, phi, theta)


def spherical_polar_grid(n_phi, n_theta, cartesian: bool = True,
                         ) -> Iterator[Tuple[float, float, float]]:
    """Yield a series of 3D points on a unit sphere surface

    The points form a grid in spherical coordinates

    Args:
        n_phi: number of longitude-like radial subdivisions at poles
        n_theta: number of lattitude-like rings of points dividing polar axis
        cartesian: Yield points in Cartesian coordinates. If False, instead
            yield points in spherical coordinates.

    Returns:
        Sequence of (x, y, z) coordinates (if cartesian=True) or
        (r, phi, theta) spherical coordinates.
    """
    phi_sequence = np.linspace(-np.pi, np.pi, n_phi + 1)[:-1]

    # Offset theta samples to avoid redundancy at pole
    theta_sequence = (np.linspace(0, np.pi, n_theta + 1)[:-1]
                      + np.pi / (2 * n_theta))

    for phi, theta in product(phi_sequence, theta_sequence):
        if cartesian:
            yield _spherical_polar_to_cartesian(phi, theta)
        else:
            yield (1, phi, theta)


def spherical_polar_improved(npts, cartesian: bool = True
                             ) -> Iterator[Tuple[float, float, float]]:
    """Yield a series of 3D points on a unit sphere surface

    The points form rings of common theta in polar coordinates. However, the
    number of samples in each ring is scaled to achieve as uniform a sampling
    density as possible.

    Analytically we find that for evenly-divided theta, the average length of a
    "latitude" line is 4.

    To obtain parity between nearest-neighbours (NN) along each axis,
    we solve (NN distance along theta) = (NN distance along phi):

    4 n_theta / npts = pi / n_theta

    Exact solutions with integer npts will be an irrational number, so n_theta
    is rounded down. The requested npts is then distributed between the
    constant-theta rings, according to their circumference.

    Args:
        npts: number of points at sphere surface
        cartesian: Yield points in Cartesian coordinates. If False, instead
            yield points in spherical coordinates.

    Returns:
        Sequence of (x, y, z) coordinates (if cartesian=True) or
        (r, phi, theta) spherical coordinates.
    """

    # round from the solution of
    # for neighbouring points
    n_theta = int(np.sqrt(np.pi / 4 * npts))

    # Offset theta samples to avoid redundancy at pole
    theta_sequence = (np.linspace(0, np.pi, n_theta + 1)[:-1]
                      + np.pi / (2 * n_theta))

    # get the lengths of all theta bins and drop points in
    theta_circumferences = 2 * np.pi * np.abs(np.sin(theta_sequence))
    bin_edges = np.concatenate([[0], np.cumsum(theta_circumferences)])

    counts, _ = np.histogram(np.linspace(0, theta_circumferences.sum(), npts),
                             bins=bin_edges, density=False)

    for theta, count in zip(theta_sequence, counts):
        phi_sequence = np.linspace(-np.pi, np.pi, count + 1)[:-1]
        for phi in phi_sequence:
            if cartesian:
                yield _spherical_polar_to_cartesian(phi, theta)
            else:
                yield (1, phi, theta)
