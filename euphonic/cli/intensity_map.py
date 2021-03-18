import argparse
from typing import List

import numpy as np

import euphonic
from euphonic import ureg
import euphonic.plot
from euphonic.util import get_qpoint_labels
from .utils import (_bands_from_force_constants, _calc_modes_kwargs,
                    get_args, _get_debye_waller,
                    _get_energy_bins_and_units, _get_q_distance,
                    _get_cli_parser, load_data_from_file,
                    matplotlib_save_or_show)


def main(params: List[str] = None) -> None:
    args = get_args(get_parser(), params)
    calc_modes_kwargs = _calc_modes_kwargs(args)

    data = load_data_from_file(args.filename)

    q_spacing = _get_q_distance(args.length_unit, args.q_spacing)
    recip_length_unit = q_spacing.units

    frequencies_only = (args.weights != 'coherent')

    if isinstance(data, euphonic.ForceConstants):
        print("Force Constants data was loaded. Getting band path...")
        (modes, x_tick_labels, split_args) = _bands_from_force_constants(
            data, q_distance=q_spacing, insert_gamma=False,
            frequencies_only=frequencies_only,
            **calc_modes_kwargs)
    elif isinstance(data, euphonic.QpointPhononModes):
        print("Phonon band data was loaded.")
        modes = data
        split_args = {'btol': args.btol}
        x_tick_labels = get_qpoint_labels(modes.qpts,
                                          cell=modes.crystal.to_spglib_cell())
    modes.frequencies_unit = args.energy_unit
    ebins, energy_unit = _get_energy_bins_and_units(
        args.energy_unit, modes, args.ebins, emin=args.e_min, emax=args.e_max)

    print("Computing intensities and generating 2D maps")

    if args.weights.lower() == 'coherent':
        if args.temperature is not None:
            if not isinstance(data, euphonic.ForceConstants):
                raise TypeError("Cannot generate Debye-Waller factor without "
                                "force constants data. Leave --temperature "
                                "unset if plotting precalculated phonon "
                                "modes.")

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

    elif args.weights.lower() == 'dos':
        spectrum = calculate_dos_map(modes, ebins)

    if args.q_broadening or args.energy_broadening:
        spectrum = spectrum.broaden(
            x_width=(args.q_broadening * recip_length_unit
                     if args.q_broadening else None),
            y_width=(args.energy_broadening * energy_unit
                     if args.energy_broadening else None),
            shape=args.shape)

    print("Plotting figure")
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

    spectra = spectrum.split(**split_args)
    if len(spectra) > 1:
        print(f"Found {len(spectra)} regions in q-point path")

    euphonic.plot.plot_2d(spectra,
                          cmap=args.cmap,
                          vmin=args.v_min, vmax=args.v_max,
                          x_label=x_label,
                          y_label=y_label,
                          title=args.title)
    matplotlib_save_or_show(save_filename=args.save_to)


def calculate_dos_map(modes: euphonic.QpointPhononModes,
                      ebins: euphonic.Quantity) -> euphonic.Spectrum2D:
    from euphonic.util import _calc_abscissa
    q_bins = _calc_abscissa(modes.crystal.reciprocal_cell(), modes.qpts)

    bin_indices = np.digitize(modes.frequencies.magnitude, ebins.magnitude)
    intensity_map = np.zeros((modes.n_qpts, len(ebins) + 1))
    first_index = np.tile(range(modes.n_qpts),
                          (3 * modes.crystal.n_atoms, 1)).transpose()
    np.add.at(intensity_map, (first_index, bin_indices), 1)

    return euphonic.Spectrum2D(q_bins, ebins,
                               intensity_map[:, :-1] * ureg('dimensionless'))


def get_parser() -> argparse.ArgumentParser:
    parser, sections = _get_cli_parser(features={'read-fc', 'read-modes',
                                                 'q-e', 'map', 'btol', 'ebins',
                                                 'weights', 'plotting'})
    parser.description = (
        'Plots a 2D intensity map from the file provided. If a force '
        'constants file is provided, a band structure path is '
        'generated using Seekpath')

    sections['q'].description = (
        '"GRID" options relate to Monkhorst-Pack sampling for the Debye-Waller'
        ' factor, and only apply when --weights=coherent and --temperature is '
        'set. "Q" options relate to the x-axis of spectrum data.')

    return parser
