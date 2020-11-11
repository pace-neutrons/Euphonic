# -*- coding: utf-8 -*-
"""
Parse a *.phonon CASTEP output file for vibrational frequency data and
save or display a matplotlib plot of the vibrational dispersion
"""

import argparse
from typing import List

import euphonic
from euphonic.plot import plot_1d
from .utils import (load_data_from_file, get_args, _bands_from_force_constants,
                    _get_q_distance, matplotlib_save_or_show)


def main(params: List[str] = None):
    parser = get_parser()
    args = get_args(parser, params)

    data = load_data_from_file(args.filename)
    data.frequencies_unit = args.energy_unit

    if isinstance(data, euphonic.ForceConstants):
        print("Force Constants data was loaded. Getting band path...")
        q_distance = _get_q_distance(args.length_unit, args.q_distance)
        (modes, x_tick_labels, split_args) = _bands_from_force_constants(
            data, q_distance=q_distance)
    elif isinstance(data, euphonic.QpointPhononModes):
        print("Phonon band data was loaded.")
        modes = data
        split_args = {'btol': args.btol}
        x_tick_labels = None
    else:
        raise TypeError("Input data must be phonon modes or force constants.")

    print("Mapping modes to 1D band-structure")
    if args.reorder:
        modes.reorder_frequencies()

    spectrum = modes.get_dispersion()
    if x_tick_labels is not None:
        spectrum.x_tick_labels = x_tick_labels
    spectra = spectrum.split(**split_args)

    _ = plot_1d(spectra,
                title=args.title,
                y_label=f'Energy ({spectrum.y_data.units:~P})',
                y_min=args.e_min, y_max=args.e_max,
                lw=1.0)
    matplotlib_save_or_show(save_filename=args.save_to)


def get_parser():
    parser = argparse.ArgumentParser(
        description=('Extract band structure data from a .phonon file '
                     'and plot it with matplotlib'))
    parser.add_argument(
        'filename',
        help=('Phonon data file. This should contain force constants or band '
              'data. Force constants formats: .yaml, force_constants.hdf5 '
              '(Phonopy); .castep_bin , .check (Castep); .json (Euphonic). [A '
              'band structure path will be obtained using Seekpath.  ] Band '
              'data formats: {band,qpoints,mesh}.{hdf5,yaml} (Phonopy); '
              '.phonon (Castep); .json (Euphonic)'
              ))
    parser.add_argument(
        '-s', '--save-to', dest='save_to', default=None,
        help='Save resulting plot to a file with this name')
    parser.add_argument(
        '-u', '--energy-unit', dest='energy_unit', default='meV',
        help=('Convert frequencies to specified units for plotting (e.g 1/cm'))
    parser.add_argument('--length-unit', type=str, default='angstrom',
                        dest='length_unit',
                        help=('Length units; these will be inverted to obtain '
                              'units of distance between q-points (e.g. "bohr"'
                              ' for bohr^-1).'))
    parser.add_argument('--e-min', type=float, default=None, dest='e_min',
                        help='Energy range minimum in ENERGY_UNIT')
    parser.add_argument('--e-max', type=float, default=None, dest='e_max',
                        help='Energy range maximum in ENERGY_UNIT')
    parser.add_argument('--title', type=str, default='', help='Plot title')
    parser.add_argument(
        '--reorder',
        action='store_true',
        help=('Try to determine branch crossings from eigenvectors and'
              ' rearrange frequencies accordingly'))
    disp_group = parser.add_argument_group(
        'Dispersion arguments',
        'Arguments specific to plotting a pre-calculated band structure')
    disp_group.add_argument(
        '--btol',
        default=10.0,
        type=float,
        help=('The tolerance for plotting sections of reciprocal space on'
              ' different subplots, as a fraction of the median distance'
              ' between q-points'))

    interp_group = parser.add_argument_group(
        'Interpolation arguments',
        ('Arguments specific to band structures that are generated from '
         'Force Constants data'))
    interp_group.add_argument('--q-distance', type=float, default=0.025,
                              dest='q_distance',
                              help=('Target distance between q-point samples '
                                    'in 1/LENGTH_UNIT'))
    return parser
