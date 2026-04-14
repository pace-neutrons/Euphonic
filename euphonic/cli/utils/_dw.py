from collections.abc import Sequence

from euphonic import (
    DebyeWaller,
    ForceConstants,
    Quantity,
    ureg,
)
from euphonic.util import (
    mp_grid,
)

from ._grids import _grid_spec_from_args


def _get_debye_waller(temperature: Quantity,
                      fc: ForceConstants,
                      grid: Sequence[int] | None = None,
                      grid_spacing: Quantity = 0.1 * ureg('1/angstrom'),
                      **calc_modes_kwargs,
                      ) -> DebyeWaller:
    """Generate Debye-Waller data from force constants and grid specification
    """
    mp_grid_spec = _grid_spec_from_args(fc.crystal, grid=grid,
                                        grid_spacing=grid_spacing)
    print('Calculating Debye-Waller factor on {} q-point grid'
          .format(' x '.join(map(str, mp_grid_spec))))
    dw_phonons = fc.calculate_qpoint_phonon_modes(
        mp_grid(mp_grid_spec), **calc_modes_kwargs)
    return dw_phonons.calculate_debye_waller(temperature)
