#! /usr/bin/env python3
import argparse
import os

import euphonic
from euphonic import ureg
import euphonic.plot
import matplotlib.pyplot as plt
import numpy as np
import seekpath


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str)
    parser.add_argument('--ebins', type=int, default=200)
    parser.add_argument('--seekpath-labels', action='store_true',
                        dest='seekpath_labels',
                        help='Use the exact labels reported by Seekpath when '
                             'constructing the band structure path. Otherwise,'
                             ' use the labels detected by Euphonic.')
    parser.add_argument('--q-distance', type=float, default=0.025,
                        dest='q_distance',
                        help='Target distance between q-point samples')
    return parser


def main():
    args = get_parser().parse_args()
    filename = args.file
    summary_name = os.path.basename(filename)
    path = os.path.dirname(filename)

    print(f"Reading force constants from {filename}")
    force_constants = euphonic.ForceConstants.from_phonopy(
        path=path, summary_name=summary_name)

    print(f"Getting band path")
    # Seekpath needs a set of integer identities, while Crystal stores strings
    # so we need to convert e.g. ['H', 'C', 'Cl', 'C'] -> [0, 1, 2, 1]
    all_atom_types = list(set(force_constants.crystal.atom_type.tolist()))
    atom_id_list = [all_atom_types.index(symbol)
                    for symbol in force_constants.crystal.atom_type]

    structure = (force_constants.crystal.cell_vectors.to('angstrom').magnitude,
                 force_constants.crystal.atom_r,
                 atom_id_list)

    bandpath = seekpath.get_explicit_k_path(structure,
                                            reference_distance=args.q_distance)

    # Find break points between continuous spectra: wherever there are two
    # adjacent labels
    special_point_bools = np.fromiter(
        map(bool, bandpath["explicit_kpoints_labels"]), dtype=bool)
    # End points are always special even if unlabeled
    special_point_bools[0] = special_point_bools[-1] = True

    # [T F F T T F T] -> [F F T T F T] AND [T F F T T F] = [F F F T F F] -> 3,
    break_points = (np.logical_and(special_point_bools[:-1],
                                   special_point_bools[1:])
                    .nonzero()[0].tolist())
    if break_points:
        print(f"Found {len(break_points)} regions in q-point path")

    regions = []
    start_point = 0
    for break_point in break_points:
        regions.append((start_point, break_point + 1))
        start_point = break_point + 1
    regions.append((start_point, len(special_point_bools)))

    print("Computing phonon modes")
    region_modes = []
    for (start, end) in regions:
        qpts = bandpath["explicit_kpoints_rel"][start:end]
        region_modes.append(force_constants
                            .calculate_qpoint_phonon_modes(qpts,
                                                           reduce_qpts=False))

    frequency_units = region_modes[0].frequencies.units
    emin = min(np.min(modes.frequencies.magnitude)
               for modes in region_modes) * frequency_units
    emax = max(np.max(modes.frequencies.magnitude)
               for modes in region_modes) * frequency_units
    ebins = np.linspace(emin.to('meV').magnitude, emax.to('meV').magnitude,
                        args.ebins) * ureg['meV']

    print(f"Computing structure factor")
    structure_factors = []
    for modes in region_modes:
        structure_factors.append(modes.calculate_structure_factor())

    print(f"Compiling structure factor to 2D map")
    special_point_indices = special_point_bools.nonzero()[0]

    for (region_start, region_end), structure_factor in zip(regions,
                                                            structure_factors):
        sqw = structure_factor.calculate_sqw_map(ebins)
        if args.seekpath_labels:
            label_indices = special_point_indices[
                np.logical_and(special_point_indices >= region_start,
                               special_point_indices < region_end)]

            labels = [bandpath["explicit_kpoints_labels"][i]
                      for i in label_indices]
            for i, label in enumerate(labels):
                if label == 'GAMMA':
                    labels[i] = r'$\Gamma$'

            label_indices_shifted = [i - region_start for i in label_indices]
            sqw.x_tick_labels = list(zip(label_indices_shifted, labels))

        print(f"Plotting figure")
        euphonic.plot.plot_2d(sqw)

        plt.show()


if __name__ == '__main__':
    main()
