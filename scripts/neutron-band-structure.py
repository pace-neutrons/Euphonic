#! /usr/bin/env python3
import argparse
import os

import ase
import euphonic
from euphonic import ureg
import euphonic.plot
import matplotlib.pyplot as plt
import numpy as np


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str)
    parser.add_argument('--qpoints', type=int, default=200)
    parser.add_argument('--ebins', type=int, default=200)
    parser.add_argument('--ase-labels', action='store_true', dest='ase_labels',
                        help="Get high-symmetry labels from ASE bandpath. "
                             "Otherwise these are detected by Euphonic.")
    return parser


def get_special_point_indices(bandpath):
    all_x, special_x, labels = bandpath.get_linear_kpoint_axis()

    initial_list = all_x.searchsorted(special_x).tolist()
    results = []
    left_label = None
    for (x_index, label) in zip(initial_list, labels):
        if initial_list.count(x_index) == 1:
            results.append((x_index, label))
        elif initial_list.count(x_index) == 2:
            # Index is repeated at jumps between BS segments
            if left_label is None:
                left_label = label
            else:
                right_label = label
                results.append((x_index, f"{left_label}|{right_label}"))
                left_label = None
        else:
            raise Exception()

    return results

def main():
    args = get_parser().parse_args()
    filename = args.file
    summary_name = os.path.basename(filename)
    path = os.path.dirname(filename)

    print(f"Reading force constants from {filename}")
    force_constants = euphonic.ForceConstants.from_phonopy(
        path=path, summary_name=summary_name)

    print(f"Getting band path with {args.qpoints} q-points")
    atoms = ase.Atoms(force_constants.crystal.atom_type,
                      cell=force_constants.crystal.cell_vectors.to('angstrom').magnitude,
                      scaled_positions=force_constants.crystal.atom_r)
    bandpath = atoms.cell.bandpath(npoints=args.qpoints)

    print(f"Computing phonon modes")
    modes = force_constants.calculate_qpoint_phonon_modes(bandpath.kpts, reduce_qpts=False)
    emin = np.min(modes.frequencies.magnitude) * modes.frequencies.units
    emax = np.max(modes.frequencies.magnitude) * modes.frequencies.units
    ebins = np.linspace(emin.to('meV').magnitude, emax.to('meV').magnitude,
                        args.ebins) * ureg['meV']

    print(f"Computing structure factor")
    structure_factor = modes.calculate_structure_factor()

    print(f"Compiling structure factor to 2D map")
    sqw = structure_factor.calculate_sqw_map(ebins)
    if args.ase_labels:
        sqw.x_tick_labels = get_special_point_indices(bandpath)

    print(f"Plotting figure")    
    euphonic.plot.plot_2d(sqw)

    plt.show()

if __name__ == '__main__':
    main()
