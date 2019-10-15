import math
import numpy as np
from euphonic import ureg


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
    Determines whether the given point(s) are gamma points

    Parameters
    ----------
    qpts: (3,) or (N, 3) float ndarray
        The q-point or q-points

    Returns
    -------
    isgamma: bool or (N,) bool ndarray
        Whether the input q-points(s) are gamma points. Returns a scalar if
        only 1 q-point is provided
    """
    tol = 1e-15
    isgamma = np.sum(np.absolute(qpt - np.rint(qpt)), axis=-1) < tol
    return isgamma


def gaussian(x, sigma):
    return np.exp(-np.square(x)/(2*sigma**2))/(math.sqrt(2*math.pi)*sigma)


def lorentzian(x, gamma):
    return gamma/(2*math.pi*(np.square(x) + (gamma/2)**2))


def gaussian_2d(xbins, ybins, xwidth, ywidth, extent=6.0):
    """
    Calculate a 2D Gaussian probability density, with independent standard
    deviations in x and y

    Parameters
    ----------
    xbins : (xbins,) float ndarray
        Bin edges in x
    ybins : (ybins,) float ndarray
        Bin edges in y
    xwidth : float
        The FWHM in x of the Gaussian function
    ywidth : float
        The FWHM in y of the Gaussian function
    extent : float
        How far out to calculate the Gaussian, in standard deviations

    Returns
    -------
    gauss : (nxbins, nybins) float ndarray
        Gaussian probability density
    """
    xbin_width = np.mean(np.diff(xbins))
    ybin_width = np.mean(np.diff(ybins))

    # Gauss FWHM = 2*sigma*sqrt(2*ln2)
    xsigma = xwidth/(2*math.sqrt(2*math.log(2)))
    ysigma = ywidth/(2*math.sqrt(2*math.log(2)))

    # Ensure nbins is always odd, and each bin has the same approx width as
    # original x/ybins
    nxbins = int(np.ceil(2*extent*xsigma/xbin_width)/2)*2 + 1
    nybins = int(np.ceil(2*extent*ysigma/ybin_width)/2)*2 + 1
    x = np.linspace(-extent*xsigma, extent*xsigma, nxbins)
    y = np.linspace(-extent*ysigma, extent*ysigma, nybins)

    xgrid = np.tile(x, (len(y), 1))
    ygrid = np.transpose(np.tile(y, (len(x), 1)))

    gauss = gaussian(xgrid, xsigma)*gaussian(ygrid, ysigma)

    return gauss


def mp_grid(grid):
    """
    Returns the q-points on a MxNxL Monkhorst-Pack grid specified by grid

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


def bose_factor(x, T):
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
