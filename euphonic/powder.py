"""Functions for averaging spectra in spherical q bins"""

import numpy as np
from typing import Optional, Union

from euphonic import (Crystal, DebyeWaller, ForceConstants,
                      QpointFrequencies, QpointPhononModes, Spectrum1D)
from euphonic import ureg, Quantity
from euphonic.util import mp_grid, get_reference_data


def sample_sphere_dos(fc: ForceConstants,
                      mod_q: Quantity,
                      sampling: str = 'golden',
                      npts: int = 1000, jitter: bool = False,
                      energy_bins: Quantity = None,
                      **calc_modes_args
                      ) -> Spectrum1D:
    """Sample the phonon DOS, averaging over a sphere of constant |q|

    Parameters
    ----------

    fc
        Force constant data for system

    mod_q
        radius of sphere from which vector q samples are taken (in units
        of inverse length; usually 1/angstrom).

    sampling
        Sphere-sampling scheme. (Case-insensitive) options are:

            - 'golden':
                Fibonnaci-like sampling that steps regularly along one
                spherical coordinate while making irrational steps in
                the other

            - 'sphere-projected-grid':
                Regular 2-D square mesh projected onto sphere. npts will
                be distributed as evenly as possible (i.e. using twice
                as many 'longitude' as 'lattitude' lines), rounding up
                if necessary.

            - 'spherical-polar-grid':
                Mesh over regular subdivisions in spherical polar
                coordinates.  npts will be rounded up as necessary in
                the same scheme as for sphere-projected-grid. 'Latitude'
                lines are evenly-spaced in z

            - 'spherical-polar-improved':
                npts distributed as regularly as possible using
                spherical polar coordinates: 'latitude' lines are
                evenly-spaced in z and points are distributed among
                these rings to obtain most even spacing possible.

            - 'random-sphere':
                Points are distributed randomly in unit square and
                projected onto sphere.

    npts
        Number of samples. Note that some sampling methods have
        constraints on valid values and will round up as appropriate.

    jitter
        For non-random sampling schemes, apply an additional random
        displacement to each point.

    energy_bins
        Preferred energy bin edges. If not provided, will setup
        1000 bins (1001 bin edges) from 0 to 1.05 * [max energy]

    **calc_modes_args
        other keyword arguments (e.g. 'use_c') will be passed to
        ForceConstants.calculate_qpoint_phonon_modes()

    Returns
    -------
    Spectrum1D

    """

    qpts_cart = _get_qpts_sphere(npts, sampling=sampling, jitter=jitter
                                 ) * mod_q
    qpts_frac = _qpts_cart_to_frac(qpts_cart, fc.crystal)

    phonons = fc.calculate_qpoint_frequencies(qpts_frac, **calc_modes_args
                                              )  # type: QpointFrequencies

    if energy_bins is None:
        energy_bins = _get_default_bins(phonons)

    return phonons.calculate_dos(energy_bins)


def sample_sphere_structure_factor(
    fc: ForceConstants,
    mod_q: Quantity,
    dw: DebyeWaller = None,
    dw_spacing: Quantity = 0.025 * ureg('1/angstrom'),
    temperature: Optional[Quantity] = 273. * ureg['K'],
    sampling: str = 'golden',
    npts: int = 1000, jitter: bool = False,
    energy_bins: Quantity = None,
    scattering_lengths: Union[dict, str] = 'Sears1992',
    **calc_modes_args
) -> Spectrum1D:
    """Sample structure factor, averaging over a sphere of constant |q|

    (Specifically, this is the one-phonon inelastic-scattering structure
    factor as implemented in
    QpointPhononModes.calculate_structure_factor().)

    Parameters
    ----------

    fc
        Force constant data for system

    mod_q
        scalar radius of sphere from which vector q samples are taken

    dw
        Debye-Waller exponent used for evaluation of scattering
        function. If not provided, this is generated automatically over
        Monkhorst-Pack q-point mesh determined by ``dw_spacing``.

    dw_spacing
        Maximum distance between q-points in automatic q-point mesh (if used)
        for Debye-Waller calculation.

    temperature
        Temperature for Debye-Waller calculation. If both temperature and dw
        are set to None, Debye-Waller factor will be omitted.

    sampling
        Sphere-sampling scheme. (Case-insensitive) options are:
            - 'golden':
                Fibonnaci-like sampling that steps regularly along one
                spherical coordinate while making irrational steps in
                the other

            - 'sphere-projected-grid':
                Regular 2-D square mesh projected onto sphere. npts will
                be distributed as evenly as possible (i.e. using twice
                as many 'longitude' as 'lattitude' lines), rounding up
                if necessary.

            - 'spherical-polar-grid':
                Mesh over regular subdivisions in spherical polar
                coordinates.  npts will be rounded up as necessary in
                the same scheme as for sphere-projected-grid. 'Latitude'
                lines are evenly-spaced in z

            - 'spherical-polar-improved':
                npts distributed as regularly as possible using
                spherical polar coordinates: 'latitude' lines are
                evenly-spaced in z and points are distributed among
                these rings to obtain most even spacing possible.

            - 'random-sphere':
                Points are distributed randomly in unit square and
                projected onto sphere.

    npts
        Number of samples. Note that some sampling methods have
            constraints on valid values and will round up as
            appropriate.

    jitter
        For non-random sampling schemes, apply an additional random
            displacement to each point.

    energy_bins
        Preferred energy bin edges. If not provided, will setup 1000
        bins (1001 bin edges) from 0 to 1.05 * [max energy]

    scattering_lengths
        Dict of neutron scattering lengths labelled by element. If a
        string is provided, this selects coherent scattering lengths
        from reference data by setting the 'label' argument of the
        euphonic.util.get_reference_data() function.

    **calc_modes_args
        other keyword arguments (e.g. 'use_c') will be passed to
        ForceConstants.calculate_qpoint_phonon_modes()

    Returns
    -------
    Spectrum1D

    """

    if isinstance(scattering_lengths, str):
        scattering_lengths = get_reference_data(
            physical_property='coherent_scattering_length',
            collection=scattering_lengths)  # type: dict

    if temperature is not None:
        if (dw is None):
            dw_qpts = mp_grid(fc.crystal.get_mp_grid_spec(dw_spacing))
            dw_phonons = fc.calculate_qpoint_phonon_modes(dw_qpts,
                                                          **calc_modes_args)
            dw = dw_phonons.calculate_debye_waller(temperature
                                                   )  # type: DebyeWaller
        else:
            if not np.isclose(dw.temperature.to('K').magnitude,
                              temperature.to('K').magnitude):
                raise ValueError('Temperature argument is not consistent with '
                                 'temperature stored in DebyeWaller object.')

    qpts_cart = _get_qpts_sphere(npts, sampling=sampling, jitter=jitter
                                 ) * mod_q

    qpts_frac = _qpts_cart_to_frac(qpts_cart, fc.crystal)

    phonons = fc.calculate_qpoint_phonon_modes(qpts_frac, **calc_modes_args
                                               )  # type: QpointPhononModes

    if energy_bins is None:
        energy_bins = _get_default_bins(phonons)

    s = phonons.calculate_structure_factor(
        scattering_lengths=scattering_lengths, dw=dw)

    return s.calculate_1d_average(energy_bins)


def _get_default_bins(phonons: QpointPhononModes,
                      nbins: int = 1000) -> Quantity:
    """Get a default set of energy bin edges for set of phonon modes"""
    max_energy = np.max(phonons.frequencies) * 1.05
    return np.linspace(0, max_energy.magnitude, (nbins + 1)) * max_energy.units


def _qpts_cart_to_frac(qpts: Quantity,
                       crystal: Crystal) -> np.ndarray:
    """Convert set of q-points from Cartesian to fractional coordinates

    Parameters
    ----------

    qpts
        Array of q-points in Cartesian coordinates.
    crystal
        Crystal structure determining reciprocal lattice

    Returns
    -------
    np.ndarray
        Dimensionless array of q-points in fractional coordinates
    """
    lattice = crystal.reciprocal_cell()

    return np.linalg.solve(lattice.to(ureg('1/bohr')).magnitude.T,
                           qpts.to(ureg('1/bohr')).magnitude.T
                           ).T


def _get_qpts_sphere(npts: int,
                     sampling: str = 'golden',
                     jitter: bool = False) -> np.ndarray:
    """Get q-point coordinates according to specified sampling scheme

    Note that the return value is dimensionless; the sphere radius is
    unity.  To obtain Cartesian coordinates with units, multiply by a
    float Quantity.
    """

    from euphonic.sampling import (golden_sphere, sphere_from_square_grid,
                                   spherical_polar_grid,
                                   spherical_polar_improved,
                                   random_sphere)

    if sampling == 'golden':
        return np.asarray(list(golden_sphere(npts, jitter=jitter)))
    elif sampling == 'sphere-projected-grid':
        n_cols = _check_gridpts(npts)
        return np.asarray(list(sphere_from_square_grid(n_cols * 2, n_cols,
                                                       jitter=jitter)))
    elif sampling == 'spherical-polar-grid':
        n_cols = _check_gridpts(npts)
        return np.asarray(list(spherical_polar_grid(n_cols * 2, n_cols,
                                                    jitter=jitter)))
    elif sampling == 'spherical-polar-improved':
        return np.asarray(list(spherical_polar_improved(npts, jitter=jitter)))
    elif sampling == 'random-sphere':
        return np.asarray(list(random_sphere(npts)))
    else:
        raise ValueError(f'Sampling method "{sampling}" is unknown.')


def _check_gridpts(npts: int) -> int:
    """Check requested npts can make a Nx2N grid, round up if not"""
    n_cols = int(np.ceil(np.sqrt(npts / 2)))
    actual_npts = n_cols**2 * 2
    if actual_npts != npts:
        print("Requested npts ∉ {2x^2, x ∈ Z, x > 1}; "
              f"rounding up to {npts}.")
    return n_cols
