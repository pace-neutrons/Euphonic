from argparse import ArgumentParser
from typing import Optional

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

import euphonic.sampling
from euphonic.util import zips

from .utils import get_args, matplotlib_save_or_show

choices_2d = {'golden-square', 'regular-square', 'recurrence-square'}
choices_3d = {'golden-sphere', 'sphere-from-square-grid',
              'spherical-polar-grid', 'spherical-polar-improved',
              'random-sphere', 'recurrence-cube'}


def _get_rng() -> euphonic.util.RNG:
    # A convenient hook for monkey-patching in different generators
    return euphonic.util.rng


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


def main(params: Optional[list[str]] = None) -> None:
    parser = get_parser()
    args = get_args(parser, params)

    if args.sampling in choices_2d:
        fig, ax = plt.subplots()
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    if args.sampling == 'golden-square':
        ax.scatter(*zips(*euphonic.sampling.golden_square(args.npts,
                                                          jitter=args.jitter,
                                                          rng=_get_rng())),
                   marker='o')
    elif args.sampling == 'regular-square':
        n_rows = int(np.ceil(np.sqrt(args.npts)))
        npts = n_rows**2

        if npts != args.npts:
            print('Requested npts ∉ {x^2, x ∈ Z, x > 1}; '
                  f'rounding up to {npts}.')
        ax.scatter(*zips(*euphonic.sampling.regular_square(n_rows, n_rows,
                                                           jitter=args.jitter,
                                                           rng=_get_rng())),
                   marker='o')

    elif args.sampling == 'golden-sphere':
        ax.scatter(*zips(*euphonic.sampling.golden_sphere(args.npts,
                                                          jitter=args.jitter,
                                                          rng=_get_rng())),
                   marker='x')

    elif args.sampling == 'recurrence-square':
        ax.scatter(*zips(*euphonic.sampling.recurrence_sequence(args.npts,
                                                                order=2)),
                   )

    elif args.sampling == 'spherical-polar-grid':
        n_theta = int(np.ceil(np.sqrt(args.npts / 2)))
        npts = n_theta**2 * 2

        if npts != args.npts:
            print('Requested npts ∉ {2x^2, x ∈ Z, x > 1}; '
                  f'rounding up to {npts}.')

        ax.scatter(*zips(*euphonic.sampling.spherical_polar_grid(
            n_theta * 2, n_theta, jitter=args.jitter, rng=_get_rng())),
                   marker='x')

    elif args.sampling == 'sphere-from-square-grid':
        n_theta = int(np.ceil(np.sqrt(args.npts / 2)))
        npts = n_theta**2 * 2

        if npts != args.npts:
            print('Requested npts ∉ {2x^2, x ∈ Z, x > 1}; '
                  f'rounding up to {npts}.')

        ax.scatter(*zips(*euphonic.sampling.sphere_from_square_grid(
                n_theta * 2, n_theta, jitter=args.jitter, rng=_get_rng())),
            marker='x')

    elif args.sampling == 'spherical-polar-improved':
        ax.scatter(
            *zips(*euphonic.sampling.spherical_polar_improved(
                args.npts, jitter=args.jitter, rng=_get_rng())),
            marker='x',
        )

    elif args.sampling == 'recurrence-cube':
        ax.scatter(
            *zips(
                *euphonic.sampling.recurrence_sequence(args.npts, order=3)),
            marker='x',
        )

    else:  # random-sphere
        ax.scatter(
            *zips(*euphonic.sampling.random_sphere(args.npts, rng=_get_rng())),
            marker='x',
        )

    matplotlib_save_or_show(save_filename=args.save_plot)
