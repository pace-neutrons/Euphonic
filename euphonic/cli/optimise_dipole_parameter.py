# -*- coding: UTF-8 -*-
"""
Run and time the Ewald sum part of an interpolation calculation for a
small number of q-points for different values of dipole_parameter (this
is the balance between the real/reciprocal space cutoffs when
performing the Ewald sum for polar materials) to determine the optimal
value
"""

import argparse
import time
from typing import List, Tuple

import numpy as np

from euphonic.cli.utils import (_get_cli_parser, get_args,
                                force_constants_from_file)


def main(params: List[str] = None) -> None:
    args = get_args(get_parser(), params)
    params = vars(args)
    params.update({'print_to_terminal': True})
    calculate_optimum_dipole_parameter(**params)


def calculate_optimum_dipole_parameter(
        filename: str,
        dipole_parameter_min: float = 0.25,
        dipole_parameter_max: float = 1.5,
        dipole_parameter_step: float = 0.25,
        n: int = 100,
        print_to_terminal: bool = False,
        **calc_modes_kwargs
        ) -> Tuple[float, float, float, np.array, np.array, np.array]:
    """
    Calculate the optimum dipole_parameter and other dipole_parameters
    from the filename castep_bin file

    Parameters
    ----------
    filename : str
        The filename to read from
    dipole_parameter_min
        The minimum value of dipole_parameter to test
    dipole_parameter_max
        The maximum value of dipole_parameter to test
    dipole_parameter_step
        The difference between each dipole_parameter to test
    n
        The number of times to loop over q-points. A higher value will
        get a more reliable timing, but will take longer
    print_to_terminal
        Whether to print the outcome to terminal or not
    **calc_modes_kwargs
        Will be passed through to calculate_qpoint_phonon_modes

    Returns
    -------
    results
        A tuple of:
         - the optimal dipole_parameter
         - the time it took to initialise the optimal dipole_parameter,
         - the time per qpoint for the optimal dipole_parameter
         - other tested dipole_parameters
         - other tested diple_parameters initialisation times
         - other tested diple_parameters times per q-point
    """
    dipole_parameters = np.arange(
        dipole_parameter_min,
        dipole_parameter_max + dipole_parameter_step / 100,
        dipole_parameter_step)
    t_init = np.zeros(len(dipole_parameters), dtype=np.float64)
    t_tot = np.zeros(len(dipole_parameters), dtype=np.float64)

    fc = force_constants_from_file(filename)
    sfmt = '{:20s}'
    tfmt = '{: 3.2f}'
    dparamfmt = '{: 2.2f}'
    for i, dipole_parameter in enumerate(dipole_parameters):
        if print_to_terminal:
            print(('Results for dipole_parameter ' + dparamfmt).format(
                dipole_parameter))
        # Time Ewald sum initialisation
        start = time.time()
        fc._dipole_correction_init(dipole_parameter=dipole_parameter)
        end = time.time()
        t_init[i] = end - start
        if print_to_terminal:
            print((sfmt + ': ' + tfmt + ' s').format(
                'Ewald init time', t_init[i]))

        # Time per qpt
        start = time.time()
        for n in range(n):
            fc._calculate_dipole_correction(np.array([0.5, 0.5, 0.5]))
        end = time.time()
        t_tot[i] = end - start
        if print_to_terminal:
            print((sfmt + ': ' + tfmt + ' ms\n').format(
                'Ewald time/qpt', t_tot[i]*1000/n))

    opt = np.argmin(t_tot)
    if print_to_terminal:
        print('******************************')
        print(('Suggested optimum dipole_parameter is ' + dparamfmt).format(
            dipole_parameters[opt]))
        print((sfmt + ': ' + tfmt + ' s').format('init time', t_init[opt]))
        print((sfmt + ': ' + tfmt + ' ms\n').format(
            'time/qpt', t_tot[opt]*1000/n))

    return (dipole_parameters[opt], t_init[opt], t_tot[opt],
            dipole_parameters, t_init, t_tot)


def get_parser() -> argparse.ArgumentParser:
    parser, sections = _get_cli_parser(
        features={'read-fc', 'dipole-parameter-optimisation'})
    parser.description=(
        'Run and time an interpolation calculation for a small number of '
        'q-points for different values of dipole_parameter to determine '
        'dipole_parameter\'s optimum value for this material')

    return parser
