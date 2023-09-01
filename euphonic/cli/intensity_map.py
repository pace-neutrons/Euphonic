from argparse import ArgumentParser
from typing import List, Optional

import matplotlib.style
import numpy as np

import euphonic
from euphonic import ureg, Spectrum2D, QpointFrequencies, ForceConstants
import euphonic.plot
from euphonic.util import get_qpoint_labels
from euphonic.styles import base_style
from .utils import (_bands_from_force_constants, _calc_modes_kwargs,
                    _compose_style, _plot_label_kwargs,
                    get_args, _get_debye_waller,
                    _get_energy_bins, _get_q_distance,
                    _get_cli_parser, load_data_from_file,
                    matplotlib_save_or_show)


def main(params: Optional[List[str]] = None) -> None:
    args = get_args(get_parser(), params)
    calc_modes_kwargs = _calc_modes_kwargs(args)

    frequencies_only = (args.weighting != 'coherent')
    data = load_data_from_file(args.filename, verbose=True,
                               frequencies_only=frequencies_only)
    if not frequencies_only and type(data) is QpointFrequencies:
        raise TypeError('Eigenvectors are required to use '
                        '"--weighting coherent" option')
    if args.weighting.lower() == 'coherent' and args.temperature is not None:
        if not isinstance(data, ForceConstants):
            raise TypeError('Force constants data is required to generate '
                            'the Debye-Waller factor. Leave "--temperature" '
                            'unset if plotting precalculated phonon modes.')

    q_spacing = _get_q_distance(args.length_unit, args.q_spacing)
    recip_length_unit = q_spacing.units

    if isinstance(data, ForceConstants):
        print("Getting band path...")
        (modes, x_tick_labels, split_args) = _bands_from_force_constants(
            data, q_distance=q_spacing, insert_gamma=False,
            frequencies_only=frequencies_only,
            **calc_modes_kwargs)
    else:
        modes = data
        split_args = {'btol': args.btol}
        x_tick_labels = get_qpoint_labels(modes.qpts,
                                          cell=modes.crystal.to_spglib_cell())
    modes.frequencies_unit = args.energy_unit
    ebins = _get_energy_bins(modes, args.ebins + 1, emin=args.e_min,
                             emax=args.e_max)

    print("Computing intensities and generating 2D maps")

    if args.weighting.lower() == 'coherent':
        if args.temperature is not None:
            temperature = args.temperature * ureg('K')
            dw = _get_debye_waller(temperature, data,
                                   grid=args.grid,
                                   grid_spacing=(args.grid_spacing
                                                 * recip_length_unit),
                                   **calc_modes_kwargs)
        else:
            dw = None

        spectrum = (modes.calculate_structure_factor(dw=dw)
                    .calculate_sqw_map(ebins))

    elif args.weighting.lower() == 'dos':
        spectrum = modes.calculate_dos_map(ebins)

    if args.q_broadening or args.energy_broadening:
        spectrum = spectrum.broaden(
            x_width=(args.q_broadening * recip_length_unit
                     if args.q_broadening else None),
            y_width=(args.energy_broadening[0] * ebins.units
                     if args.energy_broadening else None),
            shape=args.shape, method='convolve')

    print("Plotting figure")
    plot_label_kwargs = _plot_label_kwargs(
        args, default_ylabel=f"Energy / {spectrum.y_data.units:~P}")

    if x_tick_labels:
        spectrum.x_tick_labels = x_tick_labels

    if args.scale is not None:
        spectrum *= args.scale

    spectra = spectrum.split(**split_args)  # type: List[Spectrum2D]
    if len(spectra) > 1:
        print(f"Found {len(spectra)} regions in q-point path")

    if args.save_json:
        spectrum.to_json_file(args.save_json)
    style = _compose_style(user_args=args, base=[base_style])
    with matplotlib.style.context(style):

        euphonic.plot.plot_2d(spectra,
                              vmin=args.vmin,
                              vmax=args.vmax,
                              **plot_label_kwargs)
        matplotlib_save_or_show(save_filename=args.save_to)


def get_parser() -> ArgumentParser:
    parser, sections = _get_cli_parser(
        features={'read-fc', 'read-modes', 'q-e', 'map', 'btol', 'ebins',
                  'ins-weighting', 'plotting', 'scaling'})
    parser.description = (
        'Plots a 2D intensity map from the file provided. If a force '
        'constants file is provided, a band structure path is '
        'generated using Seekpath')

    sections['q'].description = (
        '"GRID" options relate to Monkhorst-Pack sampling for the '
        'Debye-Waller factor, and only apply when --weighting=coherent '
        'and --temperature is set. "Q" options relate to the x-axis of '
        'spectrum data.')

    return parser
