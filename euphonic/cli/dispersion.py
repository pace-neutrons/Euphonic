from argparse import ArgumentParser
from typing import List, Optional

import matplotlib.style

from euphonic.plot import plot_1d
from euphonic.styles import base_style
from euphonic.writers.phonon_website import write_phonon_website_json
from euphonic import Spectrum1D, ForceConstants, QpointFrequencies
from .utils import (load_data_from_file, get_args, _bands_from_force_constants,
                    _compose_style,
                    _get_q_distance, matplotlib_save_or_show, _get_cli_parser,
                    _get_title,
                    _calc_modes_kwargs, _plot_label_kwargs)


def main(params: Optional[List[str]] = None) -> None:
    args = get_args(get_parser(), params)

    # Need eigenvectors to reorder bands or write website JSON
    frequencies_only = args.save_web_json is None and not args.reorder

    data = load_data_from_file(args.filename, verbose=True,
                               frequencies_only=frequencies_only)
    if not frequencies_only and type(data) is QpointFrequencies:
        raise TypeError(
            'Eigenvectors are required to use "--reorder" option')

    if isinstance(data, ForceConstants):
        print("Getting band path...")
        q_distance = _get_q_distance(args.length_unit, args.q_spacing)
        (bands, x_tick_labels, split_args) = _bands_from_force_constants(
            data, q_distance=q_distance, frequencies_only=frequencies_only,
            **_calc_modes_kwargs(args))
    else:
        bands = data
        split_args = {'btol': args.btol}
        x_tick_labels = None

    if args.save_web_json is not None:
        write_phonon_website_json(modes=bands,
                                  name=_get_title(args.filename, args.title),
                                  output_file=args.save_web_json,
                                  x_tick_labels=x_tick_labels)

    bands.frequencies_unit = args.energy_unit

    print("Mapping modes to 1D band-structure")
    if args.reorder:
        bands.reorder_frequencies()

    spectrum = bands.get_dispersion()

    plot_label_kwargs = _plot_label_kwargs(
        args, default_ylabel=f"Energy / {spectrum.y_data.units:~P}")

    if x_tick_labels:
        spectrum.x_tick_labels = x_tick_labels

    spectra = spectrum.split(**split_args)  # type: List[Spectrum1D]

    style = _compose_style(user_args=args,
                           base=[base_style])

    if args.save_json:
        spectrum.to_json_file(args.save_json)

    with matplotlib.style.context(style):
        _ = plot_1d(spectra,
                    ymin=args.e_min,
                    ymax=args.e_max,
                    **plot_label_kwargs)
        matplotlib_save_or_show(save_filename=args.save_to)


def get_parser() -> ArgumentParser:
    parser, groups = _get_cli_parser(features={'read-fc', 'read-modes', 'plotting',
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
    groups["plotting"].add_argument(
        '--save-web-json',
        dest='save_web_json',
        default=None,
        help='Write JSON file for use with phonon website',
    )

    return parser
