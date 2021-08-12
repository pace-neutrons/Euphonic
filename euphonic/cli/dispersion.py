from argparse import ArgumentParser
from typing import List, Optional

import euphonic
from euphonic.plot import plot_1d
from euphonic import Spectrum1D
from .utils import (load_data_from_file, get_args, _bands_from_force_constants,
                    _get_q_distance, matplotlib_save_or_show, _get_cli_parser,
                    _calc_modes_kwargs)


def main(params: Optional[List[str]] = None) -> None:
    args = get_args(get_parser(), params)
    data = load_data_from_file(args.filename)

    if isinstance(data, euphonic.ForceConstants):
        frequencies_only = not args.reorder  # Need eigenvectors to reorder

        print("Force Constants data was loaded. Getting band path...")

        q_distance = _get_q_distance(args.length_unit, args.q_spacing)
        (modes, x_tick_labels, split_args) = _bands_from_force_constants(
            data, q_distance=q_distance, frequencies_only=frequencies_only,
            **_calc_modes_kwargs(args))
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
    if args.x_label is None:
        x_label = ""
    else:
        x_label = args.x_label

    if x_tick_labels:
        spectrum.x_tick_labels = x_tick_labels

    spectra = spectrum.split(**split_args)  # type: List[Spectrum1D]

    _ = plot_1d(spectra,
                title=args.title,
                x_label=x_label,
                y_label=y_label,
                y_min=args.e_min, y_max=args.e_max,
                lw=1.0)
    matplotlib_save_or_show(save_filename=args.save_to)


def get_parser() -> ArgumentParser:
    parser, _ = _get_cli_parser(features={'read-fc', 'read-modes', 'plotting',
                                          'q-e', 'btol'})
    parser.description = (
        'Plots a band structure from the file provided. If a force '
        'constants file is provided, a band structure path is '
        'generated using Seekpath')
    bands_group = parser.add_argument_group(
        'Band arguments',
        'Options related to plotting 1D bands ("spaghetti plots").')
    bands_group.add_argument(
        '--reorder',
        action='store_true',
        help=('Try to determine branch crossings from eigenvectors and'
              ' rearrange frequencies accordingly'))
    return parser
