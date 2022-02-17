# -*- coding: UTF-8 -*-
from argparse import ArgumentParser, SUPPRESS
import itertools
from pathlib import Path
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

    qpts = np.fromiter(itertools.chain.from_iterable(
        recurrence_sequence(npts, order=3)), dtype=float).reshape(-1, 3)
    modes = fc.calculate_qpoint_phonon_modes(qpts, **calc_modes_kwargs)

    fig, ax = plt.subplots()

    for brille_n_qpts_i in brille_n_qpts:

        brille_fc = BrilleInterpolator.from_force_constants(
            fc, grid_type=brille_grid_type,
            n_grid_points=brille_n_qpts_i,
            interpolation_kwargs=calc_modes_kwargs)    

        actual_brille_n_qpts = len(brille_fc._grid.rlu)

        interp_modes = brille_fc.calculate_qpoint_phonon_modes(qpts)

        eps = interp_modes.frequencies.magnitude - modes.frequencies.magnitude

        ax.plot(modes.frequencies.magnitude.flatten(), eps.flatten(), 'x',
                label=str(actual_brille_n_qpts))

    ax.set_xlabel(f'Frequency / {modes.frequencies.units:~P}')
    ax.set_ylabel(f'Epsilon / {modes.frequencies.units:~P}')

    ax.legend(title='Brille mesh size')
    ax.set_title(Path(filename).name)
    plt.show()


def get_parser() -> ArgumentParser:
    parser, sections = _get_cli_parser(
        features={'read-fc', 'brille'},
        conflict_handler='resolve')
    parser.description=(
        'Estimate linear interpolation error by comparing with Fourier '
        'interpolation')

    sections['brille'].add_argument('--npts', type=int, default=1000,
                                    help=('Number of quasirandom samples '
                                          'used in error estimate.'))
    sections['brille'].add_argument('--use-brille', help=SUPPRESS)
    sections['brille'].add_argument(
        '--brille-n-qpts', type=int, default=[5000], nargs='+',
        help=('Approximate number of q-points to generate on the '
              'Brille grid, is passed to the "n_grid_points" kwarg '
              'of BrilleInterpolator.from_force_constants'))

    return parser
