from argparse import (
    Namespace,
)
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from euphonic import (
    Crystal,
    DebyeWaller,
    ForceConstants,
    QpointFrequencies,
    QpointPhononModes,
    Quantity,
    Spectrum1D,
    Spectrum1DCollection,
    ureg,
)
from euphonic.util import (
    dedent_and_fill,
    format_error,
    mp_grid,
    spglib_new_errors,
)

from ._band_structure import (
    _bands_from_force_constants,
    _convert_labels_to_fractions,
    _get_break_points,
    _get_tick_labels,
    _insert_gamma,
)
from ._cli_parser import _get_cli_parser
from ._grids import _get_energy_bins, _get_q_distance, _grid_spec_from_args
from ._loaders import load_data_from_file
from ._pdos import _arrange_pdos_groups, _get_pdos_weighting

__all__ = [
    'Crystal',
    'QpointFrequencies',
    'QpointPhononModes',
    'Spectrum1D',
    'Spectrum1DCollection',
    '_arrange_pdos_groups',
    '_bands_from_force_constants',
    '_convert_labels_to_fractions',
    '_get_break_points',
    '_get_cli_parser',
    '_get_energy_bins',
    '_get_pdos_weighting',
    '_get_q_distance',
    '_get_tick_labels',
    '_grid_spec_from_args',
    '_insert_gamma',
    'dedent_and_fill',
    'format_error',
    'load_data_from_file',
    'spglib_new_errors',
]


def matplotlib_save_or_show(save_filename: Path | str | None = None) -> None:
    """
    Save or show the current matplotlib plot.
    Show if save_filename is not None which by default it is.

    Parameters
    ----------
    save_filename
        The file to save the plot in
    """
    import matplotlib.pyplot as plt
    if save_filename is not None:
        plt.savefig(save_filename)
        print(f'Saved plot to {Path(save_filename).resolve()}')
    else:
        plt.show()


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


def _plot_label_kwargs(args: Namespace, default_xlabel: str = '',
                       default_ylabel: str = '') -> dict[str, str]:
    """Collect title/label arguments that can be passed to plot_nd
    """
    plot_kwargs = {'title': args.title,
                   'xlabel': default_xlabel,
                   'ylabel': default_ylabel}
    if args.ylabel is not None:
        plot_kwargs['ylabel'] = args.ylabel
    if args.xlabel is not None:
        plot_kwargs['xlabel'] = args.xlabel
    return plot_kwargs


def _calc_modes_kwargs(args: Namespace) -> dict[str, Any]:
    """
    Collect arguments that can be passed to
    ForceConstants.calculate_qpoint_phonon_modes()
    """
    return {'asr': args.asr, 'dipole_parameter': args.dipole_parameter,
            'use_c': args.use_c, 'n_threads': args.n_threads}

def _brille_calc_modes_kwargs(args: Namespace) -> dict[str, Any]:
    """
    Collect arguments that can be passed to
    BrilleInterpolator.calculate_qpoint_phonon_modes()
    """
    if args.n_threads is None:
        # Nothing specified, allow defaults
        return {}

    return {'useparallel': args.n_threads > 1, 'threads': args.n_threads}


MplStyle = str | dict[str, str]


def _compose_style(
        *, user_args: Namespace, base: list[MplStyle] | None,
        ) -> list[MplStyle]:
    """Combine user-specified style options with default stylesheets

    Args:
        user_args: from _get_cli_parser().parse_args()
        base: Euphonic default styles for this plot

    N.B. matplotlib applies styles from left to right, so the right-most
    elements of the list take the highest priority. This function builds a
    list in the order:

    [base style(s), user style(s), CLI arguments]
    """

    style = base if not user_args.no_base_style and base is not None else []

    if user_args.style:
        style += user_args.style

    # Explicit args take priority over any other
    explicit_args = {}
    for user_arg, mpl_property in {'cmap': 'image.cmap',
                                   'fontsize': 'font.size',
                                   'font': 'font.sans-serif',
                                   'linewidth': 'lines.linewidth',
                                   'figsize': 'figure.figsize'}.items():
        if getattr(user_args, user_arg, None):
            explicit_args.update({mpl_property: getattr(user_args, user_arg)})

    if 'font.sans-serif' in explicit_args:
        explicit_args.update({'font.family': 'sans-serif'})

    if 'figure.figsize' in explicit_args:
        dimensioned_figsize = [dim * ureg(user_args.figsize_unit)
                               for dim in explicit_args['figure.figsize']]
        explicit_args['figure.figsize'] = [dim.to('inches').magnitude
                                           for dim in dimensioned_figsize]

    style.append(explicit_args)
    return style


def _get_title(filename: str, title: str | None = None) -> str:
    """Get a plot title: either user-provided string, or from filename"""
    return title if title is not None else Path(filename).stem
