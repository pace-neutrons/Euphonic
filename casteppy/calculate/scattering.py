import math
import numpy as np
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


def sqw_map(data, ebins, scattering_lengths, T=5.0, scale=1.0):
    """
    Calculate S(Q, w) for each q-point contained in data and each bin defined
    in ebins

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

    return sqw_map[:, 1:-1]


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
