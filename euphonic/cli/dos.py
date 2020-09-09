# -*- coding: UTF-8 -*-
"""
Parse a *.phonon CASTEP output file for vibrational frequency data and
display or save a matplotlib plot of the density of states
"""

import argparse
import numpy as np
from euphonic import ureg
from euphonic.plot import plot_1d
from typing import List
from .utils import (load_data_from_file, get_args,
                    matplotlib_save_or_show)


def main(params: List[str] = None):
    parser = get_parser()
    args = get_args(parser, params)

    data = load_data_from_file(args.filename)
    data.frequencies_unit = args.unit

    # Calculate and plot DOS
    if args.b is None:
        bwidth = 0.1*ureg('meV').to(args.unit)
    else:
        bwidth = args.b*ureg(data.frequencies_unit)

    if args.w is None:
        gwidth = 1.0*ureg('meV').to(args.unit)
    else:
        gwidth = args.w*ureg(data.frequencies_unit)

    freqs = data.frequencies.magnitude
    dos_bins = np.arange(freqs.min(),
                         freqs.max() + bwidth.magnitude,
                         bwidth.magnitude)*ureg(data.frequencies_unit)
    dos = data.calculate_dos(dos_bins)
    if args.lorentz:
        shape='lorentz'
    else:
        shape='gauss'
    dos = dos.broaden(x_width=gwidth, shape=shape)

    fig = plot_1d(dos, x_label=f'Energy ({dos.x_data.units:~P})',
                  y_min=0, lw=1.0)
    matplotlib_save_or_show(save_filename=args.s)


def get_parser():
    parser = argparse.ArgumentParser(
        description=('Extract bandstructure data from a .phonon file '
                     'and plot the density of states with matplotlib'))
    parser.add_argument(
        'filename',
        help='The .phonon file to extract the data from')
    parser.add_argument(
        '-unit',
        default='meV',
        help=('Convert frequencies to specified unit for plotting (e.g'
              ' 1/cm)'))
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
              ' by -unit argument or default meV). Default: 1 meV'))
    dos_group.add_argument(
        '-b',
        default=None,
        type=float,
        help=('Set histogram resolution for binning (in units specified by'
              ' -unit argument or default meV). Default: 0.1 meV'))
    dos_group.add_argument(
        '-lorentz',
        action='store_true',
        help='Use Lorentzian broadening instead of Gaussian')
    return parser
