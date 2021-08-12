# -*- coding: UTF-8 -*-
from argparse import ArgumentParser
from typing import List, Optional

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import euphonic.sampling
from .utils import get_args, matplotlib_save_or_show

choices_2d = {'golden-square', 'regular-square'}
choices_3d = {'golden-sphere', 'sphere-from-square-grid',
              'spherical-polar-grid', 'spherical-polar-improved',
              'random-sphere'}


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('npts', type=int)
    parser.add_argument('sampling', type=str,
                        choices=(choices_2d | choices_3d))
    parser.add_argument('--jitter', action='store_true',
                        help='Apply local random displacements to points')
    parser.add_argument('-s', '--save-plot', default=None, type=str,
                        dest='save_plot', metavar='OUTPUT_FILE',
                        help='Save resulting plot to given filename')
    return parser


def main(params: Optional[List[str]] = None) -> None:
    parser = get_parser()
    args = get_args(parser, params)

    if args.sampling in choices_2d:
        fig, ax = plt.subplots()
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    if args.sampling == 'golden-square':
        ax.scatter(*zip(*euphonic.sampling.golden_square(args.npts,
                                                         jitter=args.jitter)),
                   marker='o')
    elif args.sampling == 'regular-square':
        n_rows = int(np.ceil(np.sqrt(args.npts)))
        npts = n_rows**2

        if npts != args.npts:
            print("Requested npts ∉ {x^2, x ∈ Z, x > 1}; "
                  f"rounding up to {npts}.")
        ax.scatter(*zip(*euphonic.sampling.regular_square(n_rows, n_rows,
                                                          jitter=args.jitter)),
                   marker='o')

    elif args.sampling == 'golden-sphere':
        ax.scatter(*zip(*euphonic.sampling.golden_sphere(args.npts,
                                                         jitter=args.jitter)),
                   marker='x')
    elif args.sampling == 'spherical-polar-grid':
        n_theta = int(np.ceil(np.sqrt(args.npts / 2)))
        npts = n_theta**2 * 2

        if npts != args.npts:
            print("Requested npts ∉ {2x^2, x ∈ Z, x > 1}; "
                  f"rounding up to {npts}.")

        ax.scatter(*zip(
            *euphonic.sampling.spherical_polar_grid(n_theta * 2, n_theta,
                                                    jitter=args.jitter)),
                   marker='x')

    elif args.sampling == 'sphere-from-square-grid':
        n_theta = int(np.ceil(np.sqrt(args.npts / 2)))
        npts = n_theta**2 * 2

        if npts != args.npts:
            print("Requested npts ∉ {2x^2, x ∈ Z, x > 1}; "
                  f"rounding up to {npts}.")

        ax.scatter(*zip(
            *euphonic.sampling.sphere_from_square_grid(n_theta * 2, n_theta,
                                                       jitter=args.jitter)),
                   marker='x')

    elif args.sampling == 'spherical-polar-improved':
        ax.scatter(
            *zip(*euphonic.sampling.spherical_polar_improved(
                args.npts, jitter=args.jitter)),
            marker='x')
    elif args.sampling == 'random-sphere':
        ax.scatter(
            *zip(*euphonic.sampling.random_sphere(args.npts)),
            marker='x')
    else:
        raise ValueError("Sampling type f{args.sampling} is not implemented.")

    matplotlib_save_or_show(save_filename=args.save_plot)
