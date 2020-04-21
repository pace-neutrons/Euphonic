#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Parse a *.phonon or *.bands CASTEP output file for electronic/vibrational
frequency data and save or display a matplotlib plot of the electronic
or vibrational band structure or dispersion.
"""

import argparse
import warnings

from typing import List

from euphonic.plot.dispersion import output_grace, plot_dispersion

from euphonic.script_utils import load_data_from_file, get_args_and_set_up_and_down, matplotlib_save_or_show


def main(params: List[str] = None):
    parser = get_parser()
    args = get_args_and_set_up_and_down(parser, params)

    data, seedname, file = load_data_from_file(args.filename)
    data.convert_e_units(args.units)

    # Reorder frequencies if requested
    if args.reorder:
        if not file.endswith(".bands"):
            data.reorder_frequencies()
        else:
            warnings.warn("Cannot reorder bands data")

    # Plot
    if args.grace:
        output_grace(data, seedname, up=args.up, down=args.down)
    else:
        fig = plot_dispersion(data, args.filename, btol=args.btol, up=args.up,
                              down=args.down)
        if fig is not None:
            matplotlib_save_or_show(save_filename=args.s)


def get_parser():
    parser = argparse.ArgumentParser(
        description=('Extract phonon or bandstructure data from a .phonon or'
                     ' .bands file and plot the band structure with matplolib'))
    parser.add_argument(
        'filename',
        help='The .phonon or .bands file to extract the data from')
    parser.add_argument(
        '-units',
        default='eV',
        help=('Convert frequencies to specified units for plotting (e.g 1/cm,'
              ' Ry)'))
    parser.add_argument(
        '-s',
        default=None,
        help='Save resulting plot to a file with this name')
    parser.add_argument(
        '-grace',
        action='store_true',
        help='Output a .agr Grace file')

    spin_group = parser.add_mutually_exclusive_group()
    spin_group.add_argument(
        '-up',
        action='store_true',
        help='Extract and plot only spin up from *.bands')
    spin_group.add_argument(
        '-down',
        action='store_true',
        help='Extract and plot only spin down from *.bands')

    disp_group = parser.add_argument_group(
        'Dispersion arguments',
        'Arguments specific to plotting the band structure')
    disp_group.add_argument(
        '-reorder',
        action='store_true',
        help=('Try to determine branch crossings from eigenvectors and'
              ' rearrange frequencies accordingly (only applicable if using a'
              ' .phonon file)'))
    disp_group.add_argument(
        '-btol',
        default=10.0,
        type=float,
        help=('The tolerance for plotting sections of reciprocal space on'
              ' different subplots, as a fraction of the median distance'
              ' between q-points'))
    return parser


if __name__ == '__main__':
    main()
