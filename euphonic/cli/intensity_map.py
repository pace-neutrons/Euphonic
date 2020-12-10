import argparse
from typing import List, Optional, Tuple

import numpy as np

import euphonic
from euphonic import ureg
import euphonic.plot
from euphonic.util import get_qpoint_labels
from .utils import (_bands_from_force_constants,
                    get_args, _get_energy_bins_and_units, _get_q_distance,
                    _get_cli_parser, load_data_from_file,
                    matplotlib_save_or_show)


_spectrum_choices = ('dos', 'coherent')


def main(params: List[str] = None) -> None:
    args = get_args(get_parser(), params)

    data = load_data_from_file(args.filename)

    q_distance = _get_q_distance(args.length_unit, args.q_distance)
    recip_length_unit = q_distance.units

    if isinstance(data, euphonic.ForceConstants):
        print("Force Constants data was loaded. Getting band path...")
        (modes, x_tick_labels, split_args) = _bands_from_force_constants(
            data, q_distance=q_distance, asr=args.asr, insert_gamma=False)
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
        spectrum = modes.calculate_structure_factor().calculate_sqw_map(ebins)

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

    if x_tick_labels:
        spectrum.x_tick_labels = x_tick_labels

    spectra = spectrum.split(**split_args)
    if len(spectra) > 1:
        print(f"Found {len(spectra)} regions in q-point path")

    euphonic.plot.plot_2d(spectra,
                          cmap=args.cmap,
                          vmin=args.v_min, vmax=args.v_max,
                          x_label=args.x_label,
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
    parser = _get_cli_parser(qe_band_plot=True, n_ebins=True)
    parser.description = (
        'Plots a 2D intensity map from the file provided. If a force '
        'constants file is provided, a band structure path is '
        'generated using Seekpath')
    parser.add_argument('--weights', '-w', default='dos',
                        choices=_spectrum_choices,
                        help=('Spectral weights to plot: phonon DOS or '
                              'coherent inelastic neutron scattering.'))
    parser.add_argument('--v-min', type=float, default=None, dest='v_min',
                        help='Minimum of data range for colormap.')
    parser.add_argument('--v-max', type=float, default=None, dest='v_max',
                        help='Maximum of data range for colormap.')
    parser.add_argument('--cmap', type=str, default='viridis',
                        help='Matplotlib colormap')
    parser.add_argument('--q-broadening', '--qb', type=float, default=None,
                        dest='q_broadening',
                        help='Width of Gaussian broadening on q axis in recip '
                             'LENGTH_UNIT. (No broadening if unspecified.)')
    return parser
