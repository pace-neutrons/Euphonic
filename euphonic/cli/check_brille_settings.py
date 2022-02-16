# -*- coding: UTF-8 -*-
from argparse import ArgumentParser
import itertools
import time
from typing import List, Tuple, Optional
import warnings

import matplotlib.pyplot as plt
import numpy as np

from euphonic import ForceConstants
from euphonic.brille import BrilleInterpolator
from euphonic.cli.utils import (_get_cli_parser, get_args,
                                _brille_calc_modes_kwargs,
                                load_data_from_file)
from euphonic.sampling import recurrence_sequence


def main(params: Optional[List[str]] = None) -> None:
    args = get_args(get_parser(), params)
    params = vars(args)
    check_brille_settings(**params)


def check_brille_settings(
        filename: str,
        npts: int = 500,
        use_brille: bool = True,
        brille_grid_type: str = 'trellis',
        brille_n_qpts: int = 5000,
        **calc_modes_kwargs
        ) -> None:

    fc = load_data_from_file(filename)
    if not isinstance(fc, ForceConstants):
        raise TypeError('Force constants are required to use the '
                        'euphonic-check-brille-settings tool')

    brille_fc = BrilleInterpolator.from_force_constants(
        fc, grid_type=brille_grid_type,
        n_grid_points=brille_n_qpts,
        interpolation_kwargs=calc_modes_kwargs)    

    qpts = np.fromiter(itertools.chain.from_iterable(
        recurrence_sequence(npts, order=3)), dtype=float).reshape(-1, 3)

    modes = fc.calculate_qpoint_phonon_modes(qpts, **calc_modes_kwargs)
    interp_modes = brille_fc.calculate_qpoint_phonon_modes(qpts)

    eps = interp_modes.frequencies.magnitude - modes.frequencies.magnitude
    plt.plot(modes.frequencies.magnitude.flatten(), eps.flatten(), 'x')
    plt.xlabel('Frequency')
    plt.ylabel('Epsilon')
    plt.show()


def get_parser() -> ArgumentParser:
    parser, sections = _get_cli_parser(
        features={'read-fc', 'brille'})
    parser.description=(
        'Estimate linear interpolation error by comparing with Fourier '
        'interpolation')

    sections['brille'].add_argument('--npts', type=int, default=1000,
                                    help=('Number of quasirandom samples '
                                          'used in error estimate.'))

    return parser
