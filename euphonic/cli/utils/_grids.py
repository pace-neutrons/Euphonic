"""Get sensible bins/grids from minimal user input"""

from collections.abc import Sequence

import numpy as np
from pint import UndefinedUnitError

from euphonic import (
    Crystal,
    QpointFrequencies,
    QpointPhononModes,
    Quantity,
    ureg,
)
from euphonic.util import (
    format_error,
)


def _get_q_distance(length_unit_string: str, q_distance: float) -> Quantity:
    """
    Parse user arguments to obtain reciprocal-length spacing Quantity
    """
    try:
        length_units = ureg(length_unit_string)
    except UndefinedUnitError as err:
        msg = format_error(
            'Length unit not known',
            reason='Euphonic uses Pint for units.',
            fix=("Try 'angstrom' or 'bohr'. "
                 "Metric prefixes are also allowed, e.g 'nm'."),
        )
        raise ValueError(msg) from err
    recip_length_units = 1 / length_units
    return q_distance * recip_length_units


def _get_energy_bins(
        modes: QpointPhononModes | QpointFrequencies,
        n_ebins: int, emin: float | None = None,
        emax: float | None = None,
        headroom: float = 1.05) -> Quantity:
    """
    Gets recommended energy bins, in same units as modes.frequencies.
    emin and emax are assumed to be in the same units as
    modes.frequencies, if not provided the min/max values of
    modes.frequencies are used to find the bin limits
    """
    if emin is None:
        # Subtract small amount from min frequency - otherwise due to unit
        # conversions binning of this frequency can vary with different
        # architectures/lib versions, making it difficult to test
        emin_room = 1e-5*ureg('meV').to(modes.frequencies.units).magnitude
        emin = min(np.min(modes.frequencies.magnitude - emin_room), 0.)
    if emax is None:
        emax = np.max(modes.frequencies.magnitude) * headroom
    if emin >= emax:
        msg = format_error(
            'Maximum energy should be greater than minimum.',
            fix='Check --e-min and --e-max arguments.',
        )
        raise ValueError(msg)
    return np.linspace(emin, emax, n_ebins) * modes.frequencies.units


def _grid_spec_from_args(crystal: Crystal,
                           grid: list[int] | None = None,
                           grid_spacing: Quantity = 0.1 * ureg('1/angstrom'),
                           ) -> tuple[int, int, int]:
    """Get Monkorst-Pack mesh divisions from user arguments"""
    if grid:
        return tuple(grid)

    return crystal.get_mp_grid_spec(spacing=grid_spacing)
