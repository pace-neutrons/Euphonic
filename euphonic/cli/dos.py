from typing import List

import euphonic
from euphonic.util import mp_grid
from euphonic.plot import plot_1d
from .utils import (load_data_from_file, get_args, matplotlib_save_or_show,
                    _get_cli_parser, _get_energy_bins_and_units)


def main(params: List[str] = None):
    parser = get_parser()
    args = get_args(parser, params)

    data = load_data_from_file(args.filename)
    if isinstance(data, euphonic.ForceConstants):
        print((f"Force Constants data was loaded. Calculating points "
               f"on {args.grid} grid..."))
        modes = data.calculate_qpoint_phonon_modes(mp_grid(args.grid))
    elif isinstance(data, euphonic.QpointPhononModes):
        print("Phonon band data was loaded.")
        modes = data
    modes.frequencies_unit = args.energy_unit
    ebins, energy_unit = _get_energy_bins_and_units(
        args.energy_unit, modes, args.ebins, emin=args.e_min, emax=args.e_max)
    dos = modes.calculate_dos(ebins)

    if args.energy_broadening:
        dos = dos.broaden(args.energy_broadening*energy_unit, shape=args.shape)

    if args.x_label is None:
        x_label = f"Energy / {dos.x_data.units:~P}"
    else:
        x_label = args.x_label
    fig = plot_1d(dos, title=args.title, x_label=x_label, y_label=args.y_label,
                  y_min=0, lw=1.0)
    matplotlib_save_or_show(save_filename=args.save_to)


def get_parser():
    parser = _get_cli_parser(n_ebins=True)
    parser.description = (
        'Plots a DOS from the file provided. If a force '
        'constants file is provided, a DOS is generated on the '
        'grid specified by the grid argument')
    interp_group = parser.add_argument_group(
        'Interpolation arguments',
        ('Arguments specific to DOS that is generated from Force '
         'Constants data'))
    interp_group.add_argument(
        '--grid', type=int, nargs=3, default=[6, 6, 6],
        help=('Defines a Monkhorst-Pack grid to calculate the DOS'))
    return parser
