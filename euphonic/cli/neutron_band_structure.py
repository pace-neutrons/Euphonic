import argparse
import pathlib
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from pint import UndefinedUnitError
import seekpath

import euphonic
from euphonic import ureg
import euphonic.plot


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help=('File with force constants data. Supported '
                              'formats: .yaml (Phonopy); .castep_bin, .check '
                              '(Castep); .json (Euphonic)'))
    parser.add_argument('--ebins', type=int, default=200,
                        help='Number of energy bins on y-axis')
    parser.add_argument('--e-min', type=float, default=None, dest='e_min',
                        help='Energy range minimum in ENERGY_UNITS')
    parser.add_argument('--e-max', type=float, default=None, dest='e_max',
                        help='Energy range maximum in ENERGY_UNITS')
    parser.add_argument('--q-distance', type=float, default=0.025,
                        dest='q_distance',
                        help=('Target distance between q-point samples in '
                              '1/LENGTH_UNITS'))
    parser.add_argument('--length-units', type=str, default='angstrom',
                        dest='length_units',
                        help=('Length units; these will be inverted to obtain'
                              'units of distance between q-points (e.g. "bohr"'
                              ' for bohr^-1).'))
    parser.add_argument('--energy-units', dest='energy_units',
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
                             'LENGTH_UNITS. (No broadening if unspecified.)')
    parser.add_argument('--energy-broadening', type=float, default=None,
                        dest='gaussian_y',
                        help='Width of Gaussian broadening on energy axis in '
                        'ENERGY_UNITS. (No broadening if unspecified.)')
    parser.add_argument('--lines', action='store_true',
                        help='Plot a conventional phonon band structure '
                             'instead of coherent neutron spectrum.')
    return parser


def _get_energy_range(frequencies: np.ndarray,
                      emin: Optional[float] = None,
                      emax: Optional[float] = None,
                      headroom = 1.05) -> Tuple[float, float]:
    if emin is None:
        emin = min(np.min(frequencies), 0.)
    if emax is None:
        emax = np.max(frequencies) * headroom

    if emin >= emax:
        raise ValueError("Maximum energy should be greater than minimum. "
                         "Check --e-min and --e-max arguments.")

    return (emin, emax)


def _get_break_points(bandpath: dict) -> List[int]:
    """Get information about band path labels and break points

    Parameters
    ----------
    bandpath
        Bandpath dictionary from Seekpath

    Returns
    -------
    list[int]

        Indices at which the spectrum should be split into subplots

    """
    # Find break points between continuous spectra: wherever there are two
    # adjacent labels
    special_point_bools = np.fromiter(
        map(bool, bandpath["explicit_kpoints_labels"]), dtype=bool)

    # [T F F T T F T] -> [F F T T F T] AND [T F F T T F] = [F F F T F F] -> 3,
    break_points = np.where(np.logical_and(special_point_bools[:-1],
                                           special_point_bools[1:]))[0]
    return (break_points + 1).tolist()


def _get_tick_labels(bandpath: dict) -> List[Tuple[int, str]]:
    """Convert x-axis labels from seekpath format to euphonic format

    i.e.::

        ['L', '', '', 'X', '', 'GAMMA']   -->

        [(0, 'L'), (3, 'X'), (5, '$\\Gamma$')]
    """

    label_indices = np.where(bandpath["explicit_kpoints_labels"])[0]
    labels = [bandpath["explicit_kpoints_labels"][i] for i in label_indices]

    for i, label in enumerate(labels):
        if label == 'GAMMA':
            labels[i] = r'$\Gamma$'

    return list(zip(label_indices, labels))


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


def main():
    args = get_parser().parse_args()
    filename = args.file

    try:
        length_units = ureg(args.length_units)
    except UndefinedUnitError:
        raise ValueError("Length unit not known. Euphonic uses Pint for units."
                         " Try 'angstrom' or 'bohr'. Metric prefixes "
                         "are also allowed, e.g 'nm'.")
    recip_length_units = 1 / length_units
    q_distance = args.q_distance * recip_length_units

    try:
        energy_units = ureg(args.energy_units)
    except UndefinedUnitError:
        raise ValueError("Energy unit not known. Euphonic uses Pint for units."
                         " Try 'eV' or 'hartree'. Metric prefixes are also "
                         "allowed, e.g 'meV' or 'fJ'.")

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

    if args.lines:
        print("Mapping modes to 1D band-structure")
        spectrum = modes.get_dispersion()

    else:
        print("Computing structure factors and generating 2D maps")
        emin, emax = _get_energy_range(
            modes.frequencies.to(energy_units).magnitude,
            emin=args.e_min, emax=args.e_max)

        ebins = np.linspace(emin, emax, args.ebins) * energy_units

        structure_factor = modes.calculate_structure_factor()
        spectrum = structure_factor.calculate_sqw_map(ebins)

        if args.gaussian_x or args.gaussian_y:
            spectrum = spectrum.broaden(
                x_width=(args.gaussian_x * ureg['1 / angstrom']
                         if args.gaussian_x else None),
                y_width=(args.gaussian_y * ureg['meV']
                         if args.gaussian_y else None))

    print("Plotting figure")
    if args.y_label is None:
        if args.lines:
            y_label = f"Energy / {spectrum.y_data.units:~P}"
        else:
            y_label = f"Energy / {spectrum.y_data.units:~P}"
    else:
        y_label = args.y_label

    spectrum.x_tick_labels = _get_tick_labels(bandpath)
    break_points = _get_break_points(bandpath)
    if break_points:
        print(f"Found {len(break_points) + 1} regions in q-point path")
        spectrum = spectrum.split(indices=break_points)

    if args.lines:
        euphonic.plot.plot_1d(spectrum, x_label=args.x_label, y_label=y_label)

    else:
        euphonic.plot.plot_2d(spectrum,
                              cmap=args.cmap,
                              x_label=args.x_label,
                              y_label=y_label,
                              title=args.title)
    plt.show()
