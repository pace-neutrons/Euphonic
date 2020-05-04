#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Parse a *.phonon or *.band CASTEP output file for electronic/vibrational
frequency data and display or save a matplotlib plot of the electronic or
vibrational band structure or dispersion.
"""

import argparse
import numpy as np
from euphonic import ureg
from euphonic.plot import plot_1d
from typing import List

from euphonic.script_utils import load_data_from_file, get_args_and_set_up_and_down, matplotlib_save_or_show


def main(params: List[str] = None):
    parser = get_parser()
    args = get_args_and_set_up_and_down(parser, params)

    data, seedname, file = load_data_from_file(args.filename)
    data.frequencies_unit = args.units

    # Calculate and plot DOS
    if args.b is None:
        bwidth = 0.1*ureg('meV')
    else:
        bwidth = args.b*ureg(data.frequencies_unit)

    if args.w is None:
        gwidth = 1.0*ureg('meV')
    else:
        gwidth = args.w*ureg(data.frequencies_unit)

    freqs = data.frequencies.magnitude
    dos_bins = np.arange(freqs.min(),
                         freqs.max() + bwidth.magnitude,
                         bwidth.magnitude)*ureg(data.frequencies_unit)
    dos = data.calculate_dos(dos_bins, gwidth, lorentz=args.lorentz)
    if args.lorentz:
        shape='lorentz'
    else:
        shape='gauss'
    dos = dos.broaden(x_width=gwidth, shape=shape)

    fig = plot_1d(dos, args.filename, mirror=args.mirror, up=args.up,
                  down=args.down)
    matplotlib_save_or_show(save_filename=args.s)


def get_parser():
    parser = argparse.ArgumentParser(
        description=('Extract bandstructure data from a .phonon file '
                     'and plot the density of states with matplotlib'))
    parser.add_argument(
        'filename',
        help='The .phonon or .bands file to extract the data from')
    parser.add_argument(
        '-units',
        default='meV',
        help=('Convert frequencies to specified units for plotting (e.g'
              ' 1/cm, Ry)'))
    parser.add_argument(
        '-s',
        default=None,
        help='Save resulting plot to a file with this name')
    dos_group = parser.add_argument_group(
        'DOS arguments',
        'Arguments specific to plotting the density of states')
    dos_group.add_argument(
        '-w',
        default=None,
        type=float,
        help=('Set Gaussian/Lorentzian FWHM for broadening (in units specified'
              ' by -units argument or default meV). Default: 1 meV'))
    dos_group.add_argument(
        '-b',
        default=None,
        type=float,
        help=('Set histogram resolution for binning (in units specified by'
              ' -units argument or default eV). Default: 0.1 meV'))
    dos_group.add_argument(
        '-lorentz',
        action='store_true',
        help='Use Lorentzian broadening instead of Gaussian')
    return parser


if __name__ == '__main__':
    main()
