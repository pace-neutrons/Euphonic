import math
import numpy as np
from scipy import signal
from casteppy import ureg
from casteppy.data.phonon import PhononData
from casteppy.util import reciprocal_lattice
from casteppy import ureg


def structure_factor(data, scattering_lengths, T=5.0, scale=1.0, calc_bose=True,
                     dw_seedname=None, dw_grid=None):
    """
    Calculate the one phonon inelastic scattering for a list of q-points
    See M. Dove Structure and Dynamics Pg. 226

    Parameters
    ----------
    data : Data object
        Data object containing frequencies, q-points, ion positions and types,
        and eigenvectors required for the calculation
    scattering_lengths : dictionary
        Dictionary of spin and isotope averaged coherent scattering legnths
        for each element in the structure e.g. {'O': 5.803, 'Zn': 5.680}
    T : float, optional, default 5.0
        The temperature to use when calculating the Bose factor
    scale : float, optional, default 1.0
        Apply a multiplicative factor to the final structure factor.
    calc_bose : boolean, optional, default True
        Whether to calculate and apply the Bose factor
    dw_grid : ndarray, optional, default None
        If set, will calculate the Debye-Waller factor on a Monkhorst-Pack
        grid, e.g. [nh, nk, nl]. Only applciable to InterpolationData objects,
        i.e. when using a .castep_bin file
        dtype = 'float'
        shape = (3,)
    dw_seedname : string, optional, default None
        If set, will calculate the Debye-Waller factor over the q-points in
        the .phonon file with this seedname.

    Returns
    -------
    sf : ndarray
        The neutron dynamic structure factor for each q-point and phonon branch
        dtype = 'float'
        shape = (n_qpts, n_branches)
    """

    sl = [scattering_lengths[x] for x in data.ion_type]

    # Convert any Pint quantities to pure magnitudes for performance
    cell_vec = (data.cell_vec.to('angstrom')).magnitude
    freqs = (data.freqs.to('meV')).magnitude
    ion_mass = (data.ion_mass.to('amu')).magnitude
    sl = (sl*ureg('fm').to('fm')).magnitude

    # Calculate normalisation factor
    norm_factor = sl/np.sqrt(ion_mass)

    # Calculate the exponential factor for all ions and q-points
    # ion_r in fractional coords, so Qdotr = 2pi*qh*rx + 2pi*qk*ry...
    exp_factor = np.exp(1J*2*math.pi*np.einsum('ij,kj->ik',
                                               data.qpts, data.ion_r))

    # Eigenvectors are in Cartesian so need to convert hkl to Cartesian by
    # computing dot product with hkl and reciprocal lattice.
    recip = reciprocal_lattice(cell_vec)
    Q = np.einsum('ij,jk->ik', data.qpts, recip)

    # Calculate dot product of Q and eigenvectors for all branches, ions and
    # q-points
    eigenv_dot_q = np.einsum('ijkl,il->ijk', np.conj(data.eigenvecs), Q)

    # Calculate Debye-Waller factors
    if dw_grid:
        dw = dw_coeff(data, T, grid=dw_grid).to('angstrom^2').magnitude
    elif dw_seedname:
        dw_data = PhononData(dw_seedname)
        dw = dw_coeff(dw_data, T).to('angstrom^2').magnitude

    if dw_grid or dw_seedname:
        dw_factor = np.exp(-np.einsum('jkl,ik,il->ij', dw, Q, Q))
    else:
        dw_factor = np.ones((data.n_qpts, data.n_ions))

    # Multiply dw_factor, Q.eigenvector, exp factor and normalisation factor
    term = np.einsum('ik,ijk,ik,k->ij',
                     dw_factor, eigenv_dot_q, exp_factor, norm_factor)

    # Take mod squared and divide by frequency to get intensity
    sf = np.absolute(term*np.conj(term))/np.absolute(freqs)

    # Multiply by Bose factor
    if calc_bose:
        sf = sf*bose_factor(freqs, T)

    sf = np.real(sf*scale)

    return sf


def dw_coeff(data, temperature, grid=None):
    """
    Calculate the 3 x 3 Debye-Waller coefficients for each ion

    Parameters
    ----------
    data : Data object
        InterpolationData or PhononData object
    temperature : float
        Temperature in Kelvin
    grid : ndarray, optional
        If set, will calculate the Debye-Waller factor on a Monkhorst-Pack
        grid, e.g. [nh, nk, nl]. Only applciable to InterpolationData objects.
        If not set the Debye-Waller factor will be calculated over the
        q-points already present in the input Data object
        dtype = 'float'
        shape = (3,)

    Returns
    -------
    dw : ndarray
        The DW coefficients for each ion
        dtype = 'float'
        shape = (n_ions, 3, 3) 
    """
    # Get values as magnitudes for performance
    kB = (1*ureg.k).to('E_h/K').magnitude
    ion_mass = data.ion_mass.to('e_mass').magnitude

    if grid:
        # Generate q-point list
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
        qgrid = np.column_stack((qh, qk, ql))
        # Calculate frequencies and eigenvectors on MP grid
        freqs, evecs = data.calculate_fine_phonons(qgrid, set_attrs=False)
        weights = np.full(len(freqs), 1.0/len(freqs))
    else:
        freqs = data.freqs
        evecs = data.eigenvecs
        weights = data.weights

    freqs = freqs.to('E_h', 'spectroscopy').magnitude
    x = freqs/(2*kB*temperature)
    freq_term = 1/((2*math.pi)**2*freqs*np.tanh(x))

    evec_i = np.repeat(evecs[:, :, :, :, np.newaxis], 3, axis=4)
    evec_j = np.repeat(evecs[:, :, :, np.newaxis, :], 3, axis=3)
    evec_term = np.real(evec_i*np.conj(evec_j))

    mass_term = 1/(2*ion_mass)

    dw = (np.einsum('k,ijklm,ij,i->klm',
                    mass_term, evec_term, freq_term, weights))
    dw = dw/np.sum(weights)*ureg('bohr^2')

    return dw


def sqw_map(data, ebins, scattering_lengths, T=5.0, scale=1.0, dw_grid=None,
            dw_seedname=None, ewidth=0, qwidth=0):
    """
    Calculate S(Q, w) for each q-point contained in data and each bin defined
    in ebins, and sets the sqw_map and sqw_ebins attributes of the data object

    Parameters
    ----------
    data : Data object
        Data object containing frequencies, q-points, ion positions and types,
        and eigenvectors required for the calculation
    ebins : ndarray
        The energy bin edges in meV
    scattering_lengths : dictionary
        Dictionary of spin and isotope averaged coherent scattering legnths
        for each element in the structure e.g. {'O': 5.803, 'Zn': 5.680}
    T : float, optional, default 5.0
        The temperature in Kelvin
    scale : float, optional, default 1.0
        Apply a multiplicative factor to the structure factor.
    dw_grid : ndarray, optional, default None
        If set, will calculate the Debye-Waller factor on a Monkhorst-Pack
        grid, e.g. [nh, nk, nl]. Only applciable to InterpolationData objects,
        i.e. when using a .castep_bin file
        dtype = 'float'
        shape = (3,)
    dw_seedname : string, optional, default None
        If set, will calculate the Debye-Waller factor over the q-points in
        the .phonon file with this seedname.
    ewidth : float, optional, default 1.5
        The FWHM of the Gaussian energy resolution function in meV
    qwidth : float, optional, default 0.1
        The FWHM of the Gaussian q-vector resolution function

    Returns
    -------
    sqw_map : ndarray
        The intensity for each q-point and energy bin
        dtype = 'float'
        shape = (n_qpts, ebins - 1)
    """

    # Create initial sqw_map with an extra an energy bin either side, for any
    # branches that fall outside the energy bin range
    sqw_map = np.zeros((data.n_qpts, len(ebins) + 1))

    freqs = (data.freqs.to('meV')).magnitude
    sf = structure_factor(data, scattering_lengths, T=T, scale=scale,
                          calc_bose=False)
    if T > 0:
        p_intensity = sf*bose_factor(freqs, T)
        n_intensity = sf*bose_factor(-freqs, T)
    else:
        p_intensity = sf
        n_intensity = sf

    p_bin = np.digitize(freqs, ebins)
    n_bin = np.digitize(-freqs, ebins)

    # Sum intensities into bins
    first_index = np.transpose(
        np.tile(range(data.n_qpts), (data.n_branches, 1)))
    np.add.at(sqw_map, (first_index, p_bin), p_intensity)
    np.add.at(sqw_map, (first_index, n_bin), n_intensity)
    sqw_map = sqw_map[:, 1:-1] # Exclude values outside ebin range

    if ewidth or qwidth:
        qbin_width = np.linalg.norm(
            np.mean(np.diff(data.qpts, axis=0), axis=0))
        qbins = np.linspace(
            0, qbin_width*data.n_qpts + qbin_width, data.n_qpts + 1)
        # If no width has been set, make widths small enough to have
        # effectively no broadening
        if not qwidth:
            qwidth = (qbins[1] - qbins[0])/10
        if not ewidth:
            ewidth = (ebins[1] - ebins[0])/10
        sqw_map = signal.fftconvolve(sqw_map, np.transpose(
            gaussian_2d(qbins, ebins, qwidth, ewidth)), 'same')

    data.sqw_ebins = ebins
    data.sqw_map = sqw_map

    return sqw_map


def gaussian_2d(xbins, ybins, xwidth, ywidth, extent=6.0):
    """
    Calculate a 2D Gaussian probability density, with independent standard
    deviations in x and y

    Parameters
    ----------
    xbins : ndarray
        Bin edges in x
        dtype = 'float'
        shape = (xbins,)
    ybins : ndarray
        Bin edges in y
        dtype = 'float'
        shape = (ybins,)
    xwidth : float
        The FWHM in x of the Gaussian function
    ywidth : float
        The FWHM in y of the Gaussian function
    extent : float
        How far out to calculate the Gaussian, in standard deviations

    Returns
    -------
    gauss : ndarray
        Gaussian probability density
        dtype = 'float'
        shape = (nxbins, nybins)
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

    gauss = np.exp(-0.5*(np.square(xgrid/xsigma) + np.square(ygrid/ysigma)))

    return gauss


def voigt(ebins, width, mix):
    """
    Calcuate a pseudo-Voigt resolution function, a linear combination of
    Lorentzian and Gaussian functions

    Parameters
    ----------
    bins : ndarray
        Bin edges to calculate the Voigt function for
        dtype = 'float'
        shape = (ebins,)
    width : float
        The FWHM of the Gauss/Lorentzian functions
    mix : float
        The Lorentzian/Gaussian mix. 0 is fully
        Gaussian, 1 is fully Lorentzian

    Returns
    -------
    voigt : ndarray
        Voigt function for the centre of each bin
        dtype = 'float'
        shape = (bins - 1,)
    """

    ebin_width = np.mean(np.diff(ebins))
    ebin_interval = ebins[-1] - ebins[0] - ebin_width
    x = np.linspace(-ebin_interval/2, ebin_interval/2, len(ebins) - 1)
    # Gauss FWHM = 2*sigma*sqrt(2*ln2)
    sigma = width/(2*math.sqrt(2*math.log(2)))
    gauss = np.exp(-0.5*(np.square(x/sigma)))

    lorentz = 1/(1 + np.square(x/(0.5*width)))

    return height*(mix*lorentz + (1 - mix)*gauss)


def bose_factor(x, T):
    """
    Calculate the Bose factor

    Parameters
    ----------
    x : ndarray
        Phonon frequencies in meV
        dtype = 'float'
        shape = (n_qpts, 3*n_ions)
    T : float
        Temperature in K

    Returns
    -------
    bose : ndarray
        Bose factor
        dtype = 'float'
        shape = (n_qpts, 3*n_ions)
    """
    kB = (1*ureg.k).to('meV/K').magnitude
    bose = np.zeros(x.shape)
    bose[x > 0] = 1
    bose = bose + 1/(np.exp(np.absolute(x)/(kB*T)) - 1)
    return bose


def read_scattering_lengths(filename):
    """
    Read scattering lengths from a file, and return a dictionary mapping
    element symbols to scattering lengths
    """
    sl_dict = {}
    with open(filename, 'r') as f:
        f.readline() # Skip header
        for line in f:
            linesplt = line.split()
            sl_dict[linesplt[0]] = complex(linesplt[-1].strip('()')).real

    return sl_dict
