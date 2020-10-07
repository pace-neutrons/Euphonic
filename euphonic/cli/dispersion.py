# -*- coding: UTF-8 -*-
"""
Parse a *.phonon CASTEP output file for vibrational frequency data and
save or display a matplotlib plot of the vibrational dispersion
"""

import argparse
from typing import List
from euphonic.plot import plot_dispersion
from .utils import (load_data_from_file, get_args,
                    matplotlib_save_or_show)


def main(params: List[str] = None):
    parser = get_parser()
    args = get_args(parser, params)

    data = load_data_from_file(args.filename)
    data.frequencies_unit = args.unit

    # Reorder frequencies if requested
    if args.reorder:
        data.reorder_frequencies()

    fig = plot_dispersion(data, btol=args.btol,
                          y_label=f'Energy ({data.frequencies.units:~P})',
                          y_min=0, lw=1.0)
    matplotlib_save_or_show(save_filename=args.s)


def get_parser():
    parser = argparse.ArgumentParser(
        description=('Extract band structure data from a .phonon file '
                     'and plot it with matplotlib'))
    parser.add_argument(
        'filename',
        help='The .phonon file to extract the data from')
    parser.add_argument(
        '-unit',
        default='meV',
        help=('Convert frequencies to specified units for plotting (e.g 1/cm'))
    parser.add_argument(
        '-s',
        default=None,
        help='Save resulting plot to a file with this name')
    disp_group = parser.add_argument_group(
        'Dispersion arguments',
        'Arguments specific to plotting the band structure')
    disp_group.add_argument(
        '-reorder',
        action='store_true',
        help=('Try to determine branch crossings from eigenvectors and'
              ' rearrange frequencies accordingly'))
    disp_group.add_argument(
        '-btol',
        default=10.0,
        type=float,
        help=('The tolerance for plotting sections of reciprocal space on'
              ' different subplots, as a fraction of the median distance'
              ' between q-points'))
    return parser
