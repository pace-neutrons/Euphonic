"""Functions for averaging spectra in spherical q bins"""
import numpy as np
from euphonic import ForceConstants, QpointPhononModes, Spectrum1D, ureg

sampling_choices = {'golden', }


def sample_sphere_dos(fc: ForceConstants, q: float,
                      sampling: str = 'golden',
                      npts: int = 1000, jitter: bool = False,
                      energy_bins: np.ndarray = None
                      ) -> Spectrum1D:
    """Sample a spectral property, averaging over a sphere of constant |q|

    Args:
        fc: Force constant data for system

        q: scalar radius of sphere from which vector q samples are taken

        spectrum: Spectral property to sample. (Case-insensitive) options are
            "DOS" and "S".

        sampling: Sphere-sampling scheme. (Case-insensitive) options
            are:

            - 'golden': Fibonnaci-like sampling that steps regularly along one
                  spherical coordinate while making irrational steps in the
                  other

        npts: Number of samples. Note that some sampling methods have
            constraints on valid values and will round up as appropriate.

        jitter: For non-random sampling schemes, apply an additional random
            displacement to each point.

        energy_bins: Preferred energy bin edges. If not provided, will setup
            1000 bins (1001 bin edges) from 0 to 1.05 * [max energy]

    Returns:
        Spherical average spectrum

    """

    qpts = _get_qpts_sphere(npts, sampling=sampling, jitter=jitter)

    phonons = fc.calculate_qpoint_phonon_modes(qpts)  # type: QpointPhononModes

    if energy_bins is None:
        energy_bins = _get_default_bins(phonons)

    return phonons.calculate_dos(energy_bins)

def _get_default_bins(phonons: QpointPhononModes,
                      nbins: int = 1000) -> np.ndarray:
    """Get a default set of energy bin edges for a set of phonon frequencies"""
    max_energy = np.max(phonons.frequencies) * 1.05
    return np.linspace(0, max_energy.magnitude, (nbins + 1)) * max_energy.units

def _get_qpts_sphere(npts: int,
                     sampling: str = 'golden',
                     jitter: bool = False) -> np.ndarray:
    """Get q-point coordinates according to specified sampling scheme"""

    from euphonic.sampling import (golden_sphere, sphere_from_square_grid,
                                   spherical_polar_grid,
                                   spherical_polar_improved,
                                   random_sphere)

    if sampling == 'golden':
        return np.asarray(list(golden_sphere(npts, jitter=jitter)))
    elif sampling == 'sphere-from-square-grid':
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
    """Check requested npts can be converted to Nx2N grid, round up if not"""
    n_cols = int(np.ceil(np.sqrt(npts / 2)))
    actual_npts = n_cols**2 * 2
    if actual_npts != npts:
        print("Requested npts ∉ {2x^2, x ∈ Z, x > 1}; "
              f"rounding up to {npts}.")
    return n_cols
