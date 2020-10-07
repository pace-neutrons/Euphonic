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
from euphonic import ForceConstants


def main():
    parser = get_parser()
    args = parser.parse_args()
    params = vars(args)
    params.update({'print_to_terminal': True})
    calculate_optimum_eta(**params)


def calculate_optimum_eta(filename: str, eta_min: float = 0.25, eta_max: float = 1.5, eta_step: float = 0.25,
                          n: int = 100, print_to_terminal: bool = False):
    """
    Calculate the optimum eta and other etas from the filename castep_bin file

    Parameters
    ----------
    filename : str
        The path and name of the .castep_bin/.check to read from
    eta_min : float, optional, Default: 0.25
        The minimum value of eta to test
    eta_max : float, optional, Default: 1.5
        The maximum value of eta to test
    eta_step : float, optional, Default: 0.25
        The difference between each eta to test
    n : int, optional, Default: 100
        The number of times to loop over q-points. A higher value will get a more reliable timing, but will take longer
    print_to_terminal : bool, optional, Default: False
        Whether to print the outcome to terminal or not

    Returns
    -------
    Tuple[float, float, float, np.array, np.array, np.array]:
        A tuple of the optimal eta, the time it took to initialise the optimal eta,
        the time per qpoint for the optimal eta, other etas, their initialisation times and their times per qpoint.
    """
    etas = np.arange(eta_min, eta_max + eta_step / 100, eta_step)
    t_init = np.zeros(len(etas), dtype=np.float64)
    t_tot = np.zeros(len(etas), dtype=np.float64)

    idata = ForceConstants.from_castep(filename)
    sfmt = '{:20s}'
    tfmt = '{: 3.2f}'
    etafmt = '{: 2.2f}'
    for i, eta in enumerate(etas):
        if print_to_terminal:
            print(('Results for eta ' + etafmt).format(eta))
        # Time Ewald sum initialisation
        start = time.time()
        idata._dipole_correction_init(eta_scale=eta)
        end = time.time()
        t_init[i] = end - start
        if print_to_terminal:
            print((sfmt + ': ' + tfmt + ' s').format('Ewald init time', t_init[i]))

        # Time per qpt
        start = time.time()
        for n in range(n):
            idata._calculate_dipole_correction(np.array([0.5, 0.5, 0.5]))
        end = time.time()
        t_tot[i] = end - start
        if print_to_terminal:
            print((sfmt + ': ' + tfmt + ' ms\n').format('Ewald time/qpt', t_tot[i]*1000/n))

    opt = np.argmin(t_tot)
    if print_to_terminal:
        print('******************************')
        print(('Suggested optimum eta is ' + etafmt).format(etas[opt]))
        print((sfmt + ': ' + tfmt + ' s').format('init time', t_init[opt]))
        print((sfmt + ': ' + tfmt + ' ms\n').format('time/qpt', t_tot[opt]*1000/n))

    return etas[opt], t_init[opt], t_tot[opt], etas, t_init, t_tot


def get_parser():
    parser = argparse.ArgumentParser(
        description=('Run and time an interpolation calculation for a small '
                     'number of q-points for different values of eta to '
                     'determine eta\'s optimum value for this material'))
    parser.add_argument(
        'filename',
        help='The .castep_bin file to extract the data from')
    parser.add_argument(
        '-n',
        default=100,
        type=int,
        help=('The number of times to loop over q-points. A higher value will '
              'get a more reliable timing, but will take longer')
    )
    parser.add_argument(
        '--eta_min',
        default=0.25,
        type=float,
        help='The minimum value of eta to test'
    )
    parser.add_argument(
        '--eta_max',
        default=1.5,
        type=float,
        help='The maximum value of eta to test'
    )
    parser.add_argument(
        '--eta_step',
        default=0.25,
        type=float,
        help='The difference between each eta to test'
    )

    return parser
