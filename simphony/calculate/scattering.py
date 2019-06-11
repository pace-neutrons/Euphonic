import numpy as np
from scipy import signal
from simphony.util import gaussian_2d, bose_factor
from simphony import ureg


def sqw_map(data, ebins, scattering_lengths, T=5.0, scale=1.0, calc_bose=True,
            dw_grid=None, dw_seedname=None, ewidth=0, qwidth=0):
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
        for each element in the structure in fm e.g. {'O': 5.803, 'Zn': 5.680}
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

    # Convert units (use magnitudes for performance)
    freqs = (data.freqs.to('E_h', 'spectroscopy')).magnitude
    ebins = (ebins*ureg('meV').to('E_h')).magnitude
    ewidth = (ewidth*ureg('meV').to('E_h')).magnitude

    # Create initial sqw_map with an extra an energy bin either side, for any
    # branches that fall outside the energy bin range
    sqw_map = np.zeros((data.n_qpts, len(ebins) + 1))

    if dw_seedname:
        sf = data.structure_factor(scattering_lengths, T=T, scale=scale,
                                   calc_bose=False, dw_arg=dw_seedname)
    else:
        sf = data.structure_factor(scattering_lengths, T=T, scale=scale,
                                   calc_bose=False, dw_arg=dw_grid)

    if calc_bose:
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
    sqw_map = sqw_map[:, 1:-1]  # Exclude values outside ebin range

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

    data.sqw_ebins = ebins*ureg('E_h').to(data.freqs.units, 'spectroscopy')
    data.sqw_map = sqw_map

    return sqw_map
