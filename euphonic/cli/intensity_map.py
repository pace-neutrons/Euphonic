import argparse
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

import euphonic
from euphonic import ureg
import euphonic.plot
from euphonic.util import get_qpoint_labels
from .utils import (_bands_from_force_constants,
                    get_args, _get_energy_unit, _get_q_distance,
                    load_data_from_file)


_spectrum_choices = ('dos', 'coherent')


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'filename', type=str,
        help=('Phonon data file. This should contain force constants or band '
              'data. Force constants formats: .yaml, force_constants.hdf5 '
              '(Phonopy); .castep_bin , .check (Castep); .json (Euphonic). [A '
              'band structure path will be obtained using Seekpath.  ] Band '
              'data formats: {band,qpoints,mesh}.{hdf5,yaml} (Phonopy); '
              '.phonon (Castep); .json (Euphonic)'
              ))
    parser.add_argument('--weights', '-w', default='dos',
                        choices=_spectrum_choices,
                        help=('Spectral weights to plot: phonon DOS or '
                              'coherent inelastic neutron scattering.'))
    parser.add_argument('--ebins', type=int, default=200,
                        help='Number of energy bins on y-axis')
    parser.add_argument('--e-min', type=float, default=None, dest='e_min',
                        help='Energy range minimum in ENERGY_UNIT')
    parser.add_argument('--e-max', type=float, default=None, dest='e_max',
                        help='Energy range maximum in ENERGY_UNIT')
    parser.add_argument('--v-min', type=float, default=None, dest='v_min',
                        help='Minimum of data range for colormap.')
    parser.add_argument('--v-max', type=float, default=None, dest='v_max',
                        help='Maximum of data range for colormap.')
    parser.add_argument('--length-unit', type=str, default='angstrom',
                        dest='length_unit',
                        help=('Length units; these will be inverted to obtain '
                              'units of distance between q-points (e.g. "bohr"'
                              ' for bohr^-1).'))
    parser.add_argument('--energy-unit', '-u', dest='energy_unit',
                        type=str, default='meV', help='Energy units')
    parser.add_argument('--x-label', type=str, default=None,
                        dest='x_label')
    parser.add_argument('--y-label', type=str, default=None,
                        dest='y_label')
    parser.add_argument('--title', type=str, default=None)
    parser.add_argument('--cmap', type=str, default='viridis',
                        help='Matplotlib colormap')
    parser.add_argument('--q-broadening', type=float, default=None,
                        dest='gaussian_x',
                        help='Width of Gaussian broadening on q axis in recip '
                             'LENGTH_UNIT. (No broadening if unspecified.)')
    parser.add_argument('--energy-broadening', type=float, default=None,
                        dest='gaussian_y',
                        help='Width of Gaussian broadening on energy axis in '
                        'ENERGY_UNIT. (No broadening if unspecified.)')
    disp_group = parser.add_argument_group(
        'Existing-path arguments',
        'Arguments specific to plotting along pre-calculated path')
    disp_group.add_argument(
        '--btol',
        default=10.0,
        type=float,
        help=('The tolerance for plotting sections of reciprocal space on'
              ' different subplots, as a fraction of the median distance'
              ' between q-points'))
    interp_group = parser.add_argument_group(
        'Interpolation arguments',
        ('Arguments specific to band structures that are generated from '
         'Force Constants data'))
    interp_group.add_argument('--q-distance', type=float, default=0.025,
                              dest='q_distance',
                              help=('Target distance between q-point samples '
                                    'in 1/LENGTH_UNIT'))
    return parser


def _get_energy_range(frequencies: np.ndarray,
                      emin: Optional[float] = None,
                      emax: Optional[float] = None,
                      headroom: float = 1.05) -> Tuple[float, float]:
    if emin is None:
        emin = min(np.min(frequencies), 0.)
    if emax is None:
        emax = np.max(frequencies) * headroom

    if emin >= emax:
        raise ValueError("Maximum energy should be greater than minimum. "
                         "Check --e-min and --e-max arguments.")

    return (emin, emax)


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


def main(params: List[str] = None) -> None:
    args = get_args(get_parser(), params)

    data = load_data_from_file(args.filename)
    data.frequencies_unit = args.energy_unit

    energy_unit = _get_energy_unit(args.energy_unit)
    q_distance = _get_q_distance(args.length_unit, args.q_distance)
    recip_length_unit = q_distance.units

    if isinstance(data, euphonic.ForceConstants):
        print("Force Constants data was loaded. Getting band path...")
        (modes, x_tick_labels, split_args) = _bands_from_force_constants(
            data, q_distance=q_distance, insert_gamma=False)
    elif isinstance(data, euphonic.QpointPhononModes):
        print("Phonon band data was loaded.")
        modes = data
        split_args = {'btol': args.btol}
        x_tick_labels = get_qpoint_labels(modes.qpts,
                                          cell=modes.crystal.to_spglib_cell())
    else:
        raise TypeError("Input data must be phonon modes or force constants.")

    print("Computing intensities and generating 2D maps")
    emin, emax = _get_energy_range(modes.frequencies.to(energy_unit).magnitude,
                                   emin=args.e_min, emax=args.e_max)

    ebins = np.linspace(emin, emax, args.ebins) * energy_unit

    if args.weights.lower() == 'coherent':
        spectrum = modes.calculate_structure_factor().calculate_sqw_map(ebins)

    elif args.weights.lower() == 'dos':
        spectrum = calculate_dos_map(modes, ebins)

    else:
        raise ValueError(f'Could not compute "{args.weights}" spectrum. Valid '
                         'choices: ' + ', '.join(_spectrum_choices))

    if args.gaussian_x or args.gaussian_y:
        spectrum = spectrum.broaden(
            x_width=(args.gaussian_x * recip_length_unit
                     if args.gaussian_x else None),
            y_width=(args.gaussian_y * energy_unit
                     if args.gaussian_y else None))

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
    plt.show()
