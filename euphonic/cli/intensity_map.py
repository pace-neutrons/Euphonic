import argparse
import pathlib
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seekpath

import euphonic
from euphonic import ureg
import euphonic.plot
from .utils import (get_args, _get_break_points, _get_energy_unit,
                    _get_q_distance, _get_tick_labels)


_spectrum_choices = ('dos', 'coherent')


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help=('File with force constants data. Supported '
                              'formats: .yaml (Phonopy); .castep_bin, .check '
                              '(Castep); .json (Euphonic)'))
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
    parser.add_argument('--q-distance', type=float, default=0.025,
                        dest='q_distance',
                        help=('Target distance between q-point samples in '
                              '1/LENGTH_UNIT'))
    parser.add_argument('--length-unit', type=str, default='angstrom',
                        dest='length_unit',
                        help=('Length units; these will be inverted to obtain'
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


def force_constants_from_file(filename):
    path = pathlib.Path(filename)
    if path.suffix == '.yaml':
        force_constants = euphonic.ForceConstants.from_phonopy(
            path=path.parent, summary_name=path.name)
    elif path.suffix in ('.castep_bin', '.check'):
        force_constants = euphonic.ForceConstants.from_castep(filename)
    elif path.suffix == '.json':
        force_constants = euphonic.ForceConstants.from_json_file(filename)
    else:
        raise ValueError("File not recognised. Should have extension "
                         ".yaml (phonopy), .castep_bin or .check "
                         "(castep) or .json (JSON from Euphonic).")

    return force_constants


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
    filename = args.file

    q_distance = _get_q_distance(args.length_unit, args.q_distance)
    energy_unit = _get_energy_unit(args.energy_unit)

    print(f"Reading force constants from {filename}")
    force_constants = force_constants_from_file(filename)

    print(f"Getting band path")
    structure = force_constants.crystal.to_spglib_cell()
    bandpath = seekpath.get_explicit_k_path(
        structure, reference_distance=q_distance.to('1 / angstrom').magnitude)

    print("Computing phonon modes: {n_modes} modes across {n_qpts} q-points"
          .format(n_modes=(force_constants.crystal.n_atoms * 3),
                  n_qpts=len(bandpath["explicit_kpoints_rel"])))

    qpts = bandpath["explicit_kpoints_rel"]
    modes = force_constants.calculate_qpoint_phonon_modes(qpts,
                                                          reduce_qpts=False)

    print("Computing structure factors and generating 2D maps")
    emin, emax = _get_energy_range(modes.frequencies.to(energy_unit).magnitude,
                                   emin=args.e_min, emax=args.e_max)

    ebins = np.linspace(emin, emax, args.ebins) * energy_unit

    if args.weights.lower() == 'coherent':
        spectrum = modes.calculate_structure_factor().calculate_sqw_map(ebins)

    elif args.weights.lower() == 'dos':
        spectrum = calculate_dos_map(modes, ebins)
        
    else:
        raise ValueError(f'Could not compute "{weights}" spectrum. Valid '
                         'choices: ' + ', '.join(_spectrum_choices))

    if args.gaussian_x or args.gaussian_y:
        spectrum = spectrum.broaden(
            x_width=(args.gaussian_x * q_distance.units
                     if args.gaussian_x else None),
            y_width=(args.gaussian_y * energy_unit
                     if args.gaussian_y else None))

    print("Plotting figure")
    if args.y_label is None:
        y_label = f"Energy / {spectrum.y_data.units:~P}"
    else:
        y_label = args.y_label

    spectrum.x_tick_labels = _get_tick_labels(bandpath)
    break_points = _get_break_points(bandpath)
    if break_points:
        print(f"Found {len(break_points) + 1} regions in q-point path")
        spectrum = spectrum.split(indices=break_points)

    euphonic.plot.plot_2d(spectrum,
                          cmap=args.cmap,
                          x_label=args.x_label,
                          y_label=y_label,
                          title=args.title)
    plt.show()
