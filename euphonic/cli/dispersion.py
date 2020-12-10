from typing import List

import euphonic
from euphonic.plot import plot_1d
from .utils import (load_data_from_file, get_args, _bands_from_force_constants,
                    _get_q_distance, matplotlib_save_or_show, _get_cli_parser)


def main(params: List[str] = None):
    parser = get_parser()
    args = get_args(parser, params)

    data = load_data_from_file(args.filename)

    if isinstance(data, euphonic.ForceConstants):
        print("Force Constants data was loaded. Getting band path...")
        q_distance = _get_q_distance(args.length_unit, args.q_distance)
        (modes, x_tick_labels, split_args) = _bands_from_force_constants(
            data, q_distance=q_distance, asr=args.asr)
    elif isinstance(data, euphonic.QpointPhononModes):
        print("Phonon band data was loaded.")
        modes = data
        split_args = {'btol': args.btol}
        x_tick_labels = None
    modes.frequencies_unit = args.energy_unit

    print("Mapping modes to 1D band-structure")
    if args.reorder:
        modes.reorder_frequencies()

    spectrum = modes.get_dispersion()

    if args.y_label is None:
        y_label = f"Energy / {spectrum.y_data.units:~P}"
    else:
        y_label = args.y_label

    if x_tick_labels:
        spectrum.x_tick_labels = x_tick_labels

    spectra = spectrum.split(**split_args)

    _ = plot_1d(spectra,
                title=args.title,
                x_label=args.x_label,
                y_label=y_label,
                y_min=args.e_min, y_max=args.e_max,
                lw=1.0)
    matplotlib_save_or_show(save_filename=args.save_to)


def get_parser():
    parser = _get_cli_parser(qe_band_plot=True)
    parser.description = (
        'Plots a band structure from the file provided. If a force '
        'constants file is provided, a band structure path is '
        'generated using Seekpath')
    parser.add_argument(
        '--reorder',
        action='store_true',
        help=('Try to determine branch crossings from eigenvectors and'
              ' rearrange frequencies accordingly'))
    return parser
