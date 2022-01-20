# -*- coding: UTF-8 -*-
"""
Run and time the Ewald sum part of an interpolation calculation for a
small number of q-points for different values of dipole_parameter (this
is the balance between the real/reciprocal space cutoffs when
performing the Ewald sum for polar materials) to determine the optimal
value
"""

from argparse import ArgumentParser
import time
from typing import List, Tuple, Optional
import warnings

import numpy as np

from euphonic import ForceConstants
from euphonic.cli.utils import (_get_cli_parser, get_args,
                                load_data_from_file)


def main(params: Optional[List[str]] = None) -> None:
    args = get_args(get_parser(), params)
    params = vars(args)
    params.update({'print_to_terminal': True})
    calculate_optimum_dipole_parameter(**params)


def calculate_optimum_dipole_parameter(
        filename: str,
        dipole_parameter_min: float = 0.25,
        dipole_parameter_max: float = 1.5,
        dipole_parameter_step: float = 0.25,
        n: int = 500,
        print_to_terminal: bool = False,
        **calc_modes_kwargs
        ) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
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
    t_per_qpt = np.zeros(len(dipole_parameters), dtype=np.float64)

    fc = load_data_from_file(filename)
    if not isinstance(fc, ForceConstants):
        raise TypeError('Force constants are required to use the '
                        'euphonic-optimise-dipole-parameter tool')
    # Only warn rather than error - although not designed for it
    # this script could still be useful for getting approximate
    # per-qpt timings for any material
    if fc.born is None:
        warnings.warn('Born charges not found for this material - '
                      'changing dipole_parameter will have no effect.')
    sfmt = '{:20s}'
    tfmt = '{: 3.2f}'
    dparamfmt = '{: 2.2f}'
    for i, dipole_parameter in enumerate(dipole_parameters):
        if print_to_terminal:
            print(('Results for dipole_parameter ' + dparamfmt).format(
                dipole_parameter))

        # Time Ewald sum initialisation
        start = time.time()
        fc.calculate_qpoint_phonon_modes(
            np.full((1, 3), 0.5), dipole_parameter=dipole_parameter,
            **calc_modes_kwargs)
        end = time.time()
        t_init[i] = end - start
        if print_to_terminal:
            print((sfmt + ': ' + tfmt + ' s').format(
                'Initialisation Time', t_init[i]))

        # Time per qpt
        qpts = np.full((n, 3), 0.5)
        start = time.time()
        # Need reduce_qpts=False because all q-points are the same,
        # if reduce_qpts=True only one q-point would be calculated
        fc.calculate_qpoint_phonon_modes(
            qpts, dipole_parameter=dipole_parameter, reduce_qpts=False,
            **calc_modes_kwargs)
        end = time.time()
        t_per_qpt[i] = (end - start)/n
        if print_to_terminal:
            print((sfmt + ': ' + tfmt + ' ms\n').format(
                'Time/qpt', t_per_qpt[i]*1000))
    opt = np.argmin(t_per_qpt)
    if print_to_terminal:
        print('******************************')
        print(('Suggested optimum dipole_parameter is ' + dparamfmt).format(
            dipole_parameters[opt]))
        print((sfmt + ': ' + tfmt + ' s').format('Initialisation Time', t_init[opt]))
        print((sfmt + ': ' + tfmt + ' ms\n').format(
            'Time/qpt', t_per_qpt[opt]*1000))

    return (dipole_parameters[opt], t_init[opt], t_per_qpt[opt],
            dipole_parameters, t_init, t_per_qpt)


def get_parser() -> ArgumentParser:
    parser, sections = _get_cli_parser(
        features={'read-fc', 'dipole-parameter-optimisation'})
    parser.description=(
        'Run and time an interpolation calculation for a small number of '
        'q-points for different values of dipole_parameter to determine '
        'dipole_parameter\'s optimum value for this material')

    return parser
