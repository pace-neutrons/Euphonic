from typing import List

import euphonic
from euphonic.util import mp_grid
from euphonic.plot import plot_1d
from .utils import (load_data_from_file, get_args, matplotlib_save_or_show,
                    _calc_modes_kwargs,
                    _get_cli_parser, _get_energy_bins_and_units,
                    _grid_spec_from_args)


def main(params: List[str] = None):
    parser = get_parser()
    args = get_args(parser, params)

    data = load_data_from_file(args.filename)
    mode_widths = None
    if isinstance(data, euphonic.ForceConstants):

        recip_length_unit = euphonic.ureg(f'1 / {args.length_unit}')
        grid_spec = _grid_spec_from_args(data.crystal, grid=args.grid,
                                         grid_spacing=(args.grid_spacing
                                                       * recip_length_unit))

        print("Force Constants data was loaded. Calculating phonon modes "
              "on {} q-point grid...".format(
                  ' x '.join([str(x) for x in grid_spec])))
        if args.adaptive:
            if args.shape != 'gauss':
                raise ValueError('Currently only Gaussian shape is supported '
                                 'with adaptive broadening')
            cmkwargs = _calc_modes_kwargs(args)
            cmkwargs['return_mode_widths'] = True
            modes, mode_widths = data.calculate_qpoint_frequencies(
                mp_grid(grid_spec), **cmkwargs)
            if args.energy_broadening:
                mode_widths *= args.energy_broadening
        else:
            modes = data.calculate_qpoint_frequencies(mp_grid(grid_spec),
                                                      **_calc_modes_kwargs(args))

    elif isinstance(data, euphonic.QpointPhononModes):
        print("Phonon band data was loaded.")
        modes = data
    modes.frequencies_unit = args.energy_unit
    ebins, energy_unit = _get_energy_bins_and_units(
        args.energy_unit, modes, args.ebins, emin=args.e_min, emax=args.e_max)
    dos = modes.calculate_dos(ebins, mode_widths=mode_widths)

    if args.energy_broadening and not args.adaptive:
        dos = dos.broaden(args.energy_broadening*energy_unit, shape=args.shape)

    if args.x_label is None:
        x_label = f"Energy / {dos.x_data.units:~P}"
    else:
        x_label = args.x_label
    if args.y_label is None:
        y_label = ""
    else:
        y_label = args.y_label

    fig = plot_1d(dos, title=args.title, x_label=x_label, y_label=y_label,
                  y_min=0, lw=1.0)
    matplotlib_save_or_show(save_filename=args.save_to)


def get_parser():
    parser, _ = _get_cli_parser(features={'read-fc', 'read-modes', 'mp-grid',
                                          'plotting', 'ebins',
                                          'adaptive-broadening'})
    parser.description = (
        'Plots a DOS from the file provided. If a force '
        'constants file is provided, a DOS is generated on the Monkhorst-Pack '
        'grid specified by the grid (or grid-spacing) argument.')

    return parser
