"""
Parse a *.phonon or *.bands CASTEP output file for electronic/vibrational
frequency data and save or display a matplotlib plot of the electronic
or vibrational band structure or dispersion.
"""

import argparse
import os
import matplotlib.pyplot as plt
from casteppy import ureg
from casteppy.data.bands import BandsData
from casteppy.data.phonon import PhononData
from casteppy.plot.dispersion import plot_dispersion


def main():
    args = parse_arguments()

    # Read data
    path, file = os.path.split(args.filename)
    seedname = file[:file.rfind('.')]
    if file.endswith('.bands'):
        data = BandsData(seedname, path)
    else:
        data = PhononData(seedname, path, read_eigenvecs=args.reorder, read_ir=False)

    data.convert_e_units(args.units)

    # Calculate and plot dispersion
    # Reorder frequencies if requested
    if args.reorder:
        data.reorder_freqs()
    if args.up:
        fig = plot_dispersion(data, args.filename, btol=args.btol, down=False)
    elif args.down:
        fig = plot_dispersion(data, args.filename, btol=args.btol, up=False)
    else:
        fig = plot_dispersion(data, args.filename, btol=args.btol)

    # Save or show figure
    if args.s:
        plt.savefig(args.s)
    else:
        plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="""Extract phonon or bandstructure data from a .phonon or
                       .bands file and plot the band structure with
                       matplotlib""")
    parser.add_argument(
        'filename',
        help="""The .phonon or .bands file to extract the data from""")
    parser.add_argument(
        '-units',
        default='eV',
        help="""Convert frequencies to specified units for plotting (e.g
                1/cm, Ry). Default: eV""")
    parser.add_argument(
        '-s',
        default=None,
        help='Save resulting plot to a file')

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
        help="""Try to determine branch crossings from eigenvectors and
                rearrange frequencies accordingly (only applicable if
                using a .phonon file)""")
    disp_group.add_argument(
        '-btol',
        default=10.0,
        type=float,
        help="""The tolerance for plotting sections of reciprocal space on
                different subplots, as a fraction of the median distance
                between q-points. Default: 10.0""")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
