#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Parse a *.phonon or *.bands CASTEP output file for electronic/vibrational
frequency data and save or display a matplotlib plot of the electronic
or vibrational band structure or dispersion.
"""

import argparse
import os
from euphonic.data.bands import BandsData
from euphonic.data.phonon import PhononData
from euphonic.plot.dispersion import output_grace, plot_dispersion


def main():
    parser = get_parser()
    args = parser.parse_args()
    # If neither -up nor -down specified, plot both
    if not args.up and not args.down:
        args.up = True
        args.down = True

    # Read data
    path, file = os.path.split(args.filename)
    seedname = file[:file.rfind('.')]
    if file.endswith('.bands'):
        data = BandsData.from_castep(seedname, path=path)
    else:
        data = PhononData.from_castep(seedname, path=path)

    data.convert_e_units(args.units)

    # Reorder frequencies if requested
    if args.reorder:
        data.reorder_freqs()

    # Plot
    if args.grace:
        output_grace(data, seedname, up=args.up, down=args.down)
    else:
        fig = plot_dispersion(data, args.filename, btol=args.btol, up=args.up,
                              down=args.down)
        if fig is not None:
            import matplotlib.pyplot as plt
            # Save or show Matplotlib figure
            if args.s:
                plt.savefig(args.s)
            else:
                plt.show()


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
        action='store_true',
        help='Save resulting plot to a file')
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
