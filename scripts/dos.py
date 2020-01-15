#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Parse a *.phonon or *.band CASTEP output file for electronic/vibrational
frequency data and display or save a matplotlib plot of the electronic or
vibrational band structure or dispersion.
"""

import argparse
import os
import numpy as np
from euphonic import ureg
from euphonic.data.bands import BandsData
from euphonic.data.phonon import PhononData
from euphonic.plot.dos import plot_dos, output_grace


def main():
    parser = get_parser()
    args = parser.parse_args()
    # If neither -up nor -down specified, plot both
    if not args.up and not args.down:
        args.up = True
        args.down = True

    path, file = os.path.split(args.filename)
    seedname = file[:file.rfind('.')]
    if file.endswith('.bands'):
        data = BandsData.from_castep(seedname, path=path)
    else:
        data = PhononData.from_castep(seedname, path=path)

    data.convert_e_units(args.units)

    # Calculate and plot DOS
    # Set default DOS bin and broadening width based on whether it's
    # electronic or vibrational
    if args.b is None:
        if file.endswith('.bands'):
            bwidth = 0.05*ureg.eV
        else:
            bwidth = 1.0*(1/ureg.cm)
        bwidth.ito(args.units, 'spectroscopy')
    else:
        bwidth = args.b*ureg[args.units]
    if args.w is None:
        if file.endswith('.bands'):
            gwidth = 0.1*ureg.eV
        else:
            gwidth = 10.0*(1/ureg.cm)
        gwidth.ito(args.units, 'spectroscopy')
    else:
        gwidth = args.w*ureg[args.units]
    if isinstance(data, BandsData):
        all_freqs = np.append(data.freqs.magnitude, data.freq_down.magnitude)
    else:
        all_freqs = data.freqs.magnitude
    bwidth = bwidth.magnitude
    dos_bins = np.arange(all_freqs.min(), all_freqs.max() + bwidth, bwidth)
    data.calculate_dos(dos_bins, gwidth, lorentz=args.lorentz)

    if args.grace:
        output_grace(data, seedname, mirror=args.mirror, up=args.up,
                     down=args.down)
    else:
        fig = plot_dos(data, args.filename, mirror=args.mirror, up=args.up,
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
                     ' .bands file and plot the density of states with'
                     ' matplotlib'))
    parser.add_argument(
        'filename',
        help='The .phonon or .bands file to extract the data from')
    parser.add_argument(
        '-units',
        default='eV',
        help=('Convert frequencies to specified units for plotting (e.g'
              ' 1/cm, Ry)'))
    parser.add_argument(
        '-s',
        default=None,
        help='Save resulting plot to a file')
    parser.add_argument(
        '-grace',
        action='store_true',
        help='Output a .agr Grace file')

    spin_group = parser.add_mutually_exclusive_group()
    spin_group.add_argument(
        '-up',
        action='store_true',
        help='Extract and plot only spin up from .bands')
    spin_group.add_argument(
        '-down',
        action='store_true',
        help='Extract and plot only spin down from .bands')

    dos_group = parser.add_argument_group(
        'DOS arguments',
        'Arguments specific to plotting the density of states')
    dos_group.add_argument(
        '-w',
        default=None,
        type=float,
        help=('Set Gaussian/Lorentzian FWHM for broadening (in units specified'
              ' by -units argument or default eV). Default: 0.1 eV for'
              ' electronic DOS, 10.0/cm for vibrational DOS'))
    dos_group.add_argument(
        '-b',
        default=None,
        type=float,
        help=('Set histogram resolution for binning (in units specified by'
              ' -units argument or default eV). Default: 0.05 eV for'
              ' electronic DOS, 1.0/cm for vibrational DOS'))
    dos_group.add_argument(
        '-lorentz',
        action='store_true',
        help='Use Lorentzian broadening instead of Gaussian')
    dos_group.add_argument(
        '-mirror',
        action='store_true',
        help='Plot spin down electronic DOS mirrored in the x axis')

    return parser


if __name__ == '__main__':
    main()
