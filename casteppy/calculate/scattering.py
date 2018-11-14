import math
import numpy as np
from scipy import signal
from casteppy.util import reciprocal_lattice


def structure_factor(data, scattering_lengths, T=5.0, scale=1.0):
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
    T : float, optional
        The temperature to use when calculating the Bose factor. Default: 5K
    scale : float, optional
        Apply a multiplicative factor to the final structure factor.
        Default: 1.0

    Returns
    -------
    sf : ndarray
        The neutron dynamic structure factor for each q-point and phonon branch
        dtype = 'float'
        shape = (n_qpts, n_branches)
    """

    # Convert any Pint quantities to pure magnitudes for performance
    cell_vec = (data.cell_vec.to('angstrom')).magnitude
    freqs = (data.freqs.to('meV')).magnitude

    # Get scattering lengths and calculate normalisation factor
    sl = [scattering_lengths[x] for x in data.ion_type]
    norm_factor = sl/np.sqrt(data.ion_mass)

    # Calculate the exponential factor for all ions and q-points
    # ion_r in fractional coords, so Qdotr = 2pi*qh*rx + 2pi*qk*ry...
    exp_factor = np.exp(1J*2*math.pi*np.einsum('ij,jk->ik',
                                               data.qpts, np.transpose(data.ion_r)))

    # Eigenvectors are in Cartesian so need to convert hkl to Cartesian by
    # computing dot product with hkl and reciprocal lattice.
    recip = reciprocal_lattice(cell_vec)
    Q = np.einsum('ij,jk->ik', data.qpts, recip)

    # Calculate dot product of Q and eigenvectors for all branches, ions and
    # q-points
    eigenv_dot_q = np.einsum('ijkl,il->ijk', np.conj(data.eigenvecs), Q)

    # Multiply normalisation factor, Q dot eigenvector and exp factor
    term = np.einsum('ijk,ik,k->ij', eigenv_dot_q, exp_factor, norm_factor)

    # Take mod squared and divide by frequency to get intensity
    sf = np.absolute(term*np.conj(term))/np.absolute(freqs)

    # Multiply by bose factor if temperature is defined
    if T is not None:
        sf = sf*bose_factor(freqs, T)

    sf = np.real(sf*scale)

    return sf


def sqw_map(data, ebins, scattering_lengths, T=5.0, scale=1.0, emix=-1, qmix=-1, ewidth=1.5, qwidth=0.1):
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
    T : float, optional
        The temperature to use when calculating the Bose factor. Default: 5K
    scale : float, optional
        Apply a multiplicative factor to the structure factor.
        Default: 1.0
    emix : float, optional
        If between 0 and 1, describes the shape of the energy resolution
        function. 0 is fully Gaussian, 1 is fully Lorentzian
        Default: -1
    ewidth : float, optional
        The FWHM of the energy resolution function in meV
        Default: 1.5
    qmix : float, optional
        If between 0 and 1, describes the shape of the q-vector resolution
        function. 0 is fully Gaussian, 1 is fully Lorentzian
        Default: -1
    qwidth : float, optional
        The FWHM of the q-vector resolution function
        Default: 0.1

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
    sf = structure_factor(data, scattering_lengths, T=None, scale=scale)
    p_intensity = np.real(sf*bose_factor(freqs, T))
    n_intensity = np.real(sf*bose_factor(-freqs, T))

    p_bin = np.digitize(freqs, ebins)
    n_bin = np.digitize(-freqs, ebins)

    # Sum intensities into bins
    first_index = np.transpose(np.tile(range(data.n_qpts), (data.n_branches, 1)))
    np.add.at(sqw_map, (first_index, p_bin), p_intensity)
    np.add.at(sqw_map, (first_index, n_bin), n_intensity)
    sqw_map = sqw_map[:, 1:-1] # Exclude values outside ebin range

    if emix >= 0 and emix <= 1:
        eres = voigt(ebins, ewidth, emix)
        eres_2d = np.zeros(sqw_map.shape)
        n_qpts = sqw_map.shape[0]
        if n_qpts % 2 == 0:
            eres_2d[n_qpts/2 - 1] = eres
            eres_2d[n_qpts/2] = eres
        else:
            eres_2d[int(n_qpts/2)] = eres
        sqw_map = signal.fftconvolve(eres_2d, sqw_map, 'same')

    if qmix >= 0 and qmix <= 1:
        qbin_width = np.linalg.norm(np.mean(np.diff(data.qpts, axis=0), axis=0))
        qbins = np.linspace(0, qbin_width*data.n_qpts + qbin_width, data.n_qpts + 1)
        qres = voigt(qbins, qwidth, qmix)
        qres_2d = np.zeros(sqw_map.shape)
        n_ebins = sqw_map.shape[1]
        if n_ebins % 2 == 0:
            qres_2d[:, n_ebins/2 - 1] = qres
            qres_2d[:, n_ebins/2] = qres
        else:
            qres_2d[:, int(n_ebins/2)] = qres
        sqw_map = signal.fftconvolve(qres_2d, sqw_map, 'same')

    data.sqw_ebins = ebins
    data.sqw_map = sqw_map

    return sqw_map


def voigt(ebins, width, mix, height=1.0):
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
        Phonon frequencies
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
    kB = 8.6173324e-2
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
