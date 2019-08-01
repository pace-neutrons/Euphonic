import math
import numpy as np


def reciprocal_lattice(unit_cell):
    """
    Calculates the reciprocal lattice from a unit cell
    """

    a = np.array(unit_cell[0])
    b = np.array(unit_cell[1])
    c = np.array(unit_cell[2])

    bxc = np.cross(b, c)
    cxa = np.cross(c, a)
    axb = np.cross(a, b)

    adotbxc = np.vdot(a, bxc)  # Unit cell volume
    norm = 2*math.pi/adotbxc  # Normalisation factor

    astar = norm*bxc
    bstar = norm*cxa
    cstar = norm*axb

    return np.array([astar, bstar, cstar])


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
    Determines whether the given q-point is a gamma point
    """
    tol = 1e-15

    return np.sum(np.absolute(qpt - np.rint(qpt))) < tol
