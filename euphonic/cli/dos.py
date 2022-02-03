from argparse import ArgumentParser
from typing import List, Optional

import matplotlib.style

from euphonic import ureg, ForceConstants, QpointFrequencies
from euphonic.util import mp_grid, mode_gradients_to_widths
from euphonic.plot import plot_1d
from euphonic.styles import base_style
from .utils import (load_data_from_file, get_args, matplotlib_save_or_show,
                    _calc_modes_kwargs, _compose_style,
                    _get_cli_parser, _get_energy_bins,
                    _grid_spec_from_args, _get_pdos_weighting,
                    _arrange_pdos_groups, _plot_label_kwargs)


def main(params: Optional[List[str]] = None) -> None:
    parser = get_parser()
    args = get_args(parser, params)

    frequencies_only = (args.weighting == 'dos' and args.pdos is None)
    data = load_data_from_file(args.filename, verbose=True,
                               frequencies_only=frequencies_only)

    if not frequencies_only and type(data) is QpointFrequencies:
        raise TypeError('Eigenvectors are required to use "--pdos" or '
                        'any "--weighting" option other than plain DOS')
    if args.adaptive:
        if not isinstance(data, ForceConstants):
            raise TypeError(
                'Force constants are required to use --adaptive option')
        if args.shape != 'gauss':
            raise ValueError('Currently only Gaussian shape is supported '
                             'with adaptive broadening')

    mode_widths = None
    if isinstance(data, ForceConstants):

        recip_length_unit = ureg(f'1 / {args.length_unit}')
        grid_spec = _grid_spec_from_args(data.crystal, grid=args.grid,
                                         grid_spacing=(args.grid_spacing
                                                       * recip_length_unit))
        print("Calculating phonon modes "
              "on {} q-point grid...".format(
                  ' x '.join([str(x) for x in grid_spec])))
        if args.adaptive:
            cmkwargs = _calc_modes_kwargs(args)
            cmkwargs['return_mode_gradients'] = True
            modes, mode_grads = data.calculate_qpoint_phonon_modes(
                mp_grid(grid_spec), **cmkwargs)
            mode_widths = mode_gradients_to_widths(mode_grads,
                                                   modes.crystal.cell_vectors)
            if args.energy_broadening:
                mode_widths *= args.energy_broadening
        else:
            modes = data.calculate_qpoint_phonon_modes(
                mp_grid(grid_spec), **_calc_modes_kwargs(args))

    else:
        modes = data
    modes.frequencies_unit = args.energy_unit
    ebins = _get_energy_bins(
        modes, args.ebins + 1, emin=args.e_min, emax=args.e_max)

    kwargs = {"mode_widths": mode_widths,
              "adaptive_method": args.adaptive_method,
              "adaptive_error": args.adaptive_error}

    if args.weighting == 'dos' and args.pdos is None:
        dos = modes.calculate_dos(ebins, **kwargs)
    else:
        kwargs["weighting"] = _get_pdos_weighting(args.weighting)
        pdos = modes.calculate_pdos(ebins, **kwargs)
        dos = _arrange_pdos_groups(pdos, args.pdos)

    if args.energy_broadening and not args.adaptive:
        dos = dos.broaden(args.energy_broadening*ebins.units, shape=args.shape)

    plot_label_kwargs = _plot_label_kwargs(
        args, default_xlabel=f"Energy / {dos.x_data.units:~P}")

    if args.save_json:
        dos.to_json_file(args.save_json)
    style = _compose_style(user_args=args, base=[base_style])
    with matplotlib.style.context(style):
        _ = plot_1d(dos, ymin=0, **plot_label_kwargs)
        matplotlib_save_or_show(save_filename=args.save_to)


def get_parser() -> ArgumentParser:
    parser, _ = _get_cli_parser(features={'read-fc', 'read-modes', 'mp-grid',
                                          'plotting', 'ebins',
                                          'adaptive-broadening',
                                          'pdos-weighting'})
    parser.description = (
        'Plots a DOS from the file provided. If a force '
        'constants file is provided, a DOS is generated on the Monkhorst-Pack '
        'grid specified by the grid (or grid-spacing) argument.')

    return parser
