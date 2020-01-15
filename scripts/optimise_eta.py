#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Run and time the Ewald sum part of an interpolation calculation for a small
number of q-points for different values of eta (this is the balance between
the real/reciprocal space cutoffs when performing the Ewald sum for polar
materials) to determine the optimal value
"""

import argparse
import time
import numpy as np
from euphonic.data.interpolation import InterpolationData


def main():
    parser = get_parser()
    args = parser.parse_args()

    etas = np.arange(args.min, args.max + args.step/100, args.step)
    t_init = np.zeros(len(etas), dtype=np.float64)
    t_tot = np.zeros(len(etas), dtype=np.float64)

    idata = InterpolationData.from_castep(args.seedname)
    sfmt = '{:20s}'
    tfmt = '{: 3.2f}'
    etafmt = '{: 2.2f}'
    for i, eta in enumerate(etas):
        print(('Results for eta ' + etafmt).format(eta))
        # Time Ewald sum initialisation
        start = time.time()
        idata._dipole_correction_init(eta_scale=eta)
        end = time.time()
        t_init[i] = end - start
        print((sfmt + ': ' + tfmt + ' s').format('Ewald init time', t_init[i]))

        # Time per qpt
        start = time.time()
        for n in range(args.n):
            idata._calculate_dipole_correction(np.array([0.5, 0.5, 0.5]))
        end = time.time()
        t_tot[i] = end - start
        print((sfmt + ': ' + tfmt + ' ms\n').format('Ewald time/qpt', t_tot[i]*1000/args.n))

    opt = np.argmin(t_tot)
    print('******************************')
    print(('Suggested optimum eta is ' + etafmt).format(etas[opt]))
    print((sfmt + ': ' + tfmt + ' s').format('init time', t_init[opt]))
    print((sfmt + ': ' + tfmt + ' ms\n').format('time/qpt', t_tot[opt]*1000/args.n))


def get_parser():
    parser = argparse.ArgumentParser(
        description=('Run and time an interpolation calculation for a small '
                     'number of q-points for different values of eta to '
                     'determine eta\'s optimum value for this material'))
    parser.add_argument(
        'seedname',
        help='The seedname of the .castep_bin file to read from')
    parser.add_argument(
        '-n',
        default=100,
        type=int,
        help=('The number of times to loop over q-points. A higher value will '
              'get a more reliable timing, but will take longer')
    )
    parser.add_argument(
        '--min',
        default=0.25,
        type=float,
        help='The minimum value of eta to test'
    )
    parser.add_argument(
        '--max',
        default=1.5,
        type=float,
        help='The maximum value of eta to test'
    )
    parser.add_argument(
        '--step',
        default=0.25,
        type=float,
        help='The difference between each eta to test'
    )

    return parser


if __name__ == '__main__':
    main()
