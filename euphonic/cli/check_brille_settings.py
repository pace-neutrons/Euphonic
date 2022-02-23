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
        brille_npts: int = 5000,
        brille_npts_density: Optional[int] = None,
        **calc_modes_kwargs
        ) -> None:

    fc = load_data_from_file(filename)
    if not isinstance(fc, ForceConstants):
        raise TypeError('Force constants are required to use the '
                        'euphonic-check-brille-settings tool')

    qpts = np.fromiter(itertools.chain.from_iterable(
        recurrence_sequence(npts, order=3)), dtype=float).reshape(-1, 3)
    modes = fc.calculate_qpoint_phonon_modes(qpts, **calc_modes_kwargs)
    sf = modes.calculate_structure_factor()

    fig, ax = plt.subplots()
    fig_sf, ax_sf = plt.subplots(subplot_kw={'projection': '3d'})

    if brille_npts_density is not None:
        brille_params = brille_npts_density
        brille_param_key = 'grid_density'
    else:
        brille_params = brille_npts
        brille_param_key = 'n_grid_points'

    for brille_param_i in brille_params:

        brille_fc = BrilleInterpolator.from_force_constants(
            fc, grid_type=brille_grid_type,
            interpolation_kwargs=calc_modes_kwargs,
            **{brille_param_key: brille_param_i})

        actual_brille_npts = len(brille_fc._grid.rlu)
        bz_vol = brille_fc._grid.BrillouinZone.ir_polyhedron.volume
        actual_brille_density = int(actual_brille_npts/bz_vol)

        interp_modes = brille_fc.calculate_qpoint_phonon_modes(qpts)
        interp_sf = interp_modes.calculate_structure_factor()

        eps = interp_modes.frequencies.magnitude - modes.frequencies.magnitude
        eps_sf = (interp_sf.structure_factors.magnitude
                  - sf.structure_factors.magnitude)

        ax.plot(modes.frequencies.magnitude.flatten(), eps.flatten(), 'x',
                label=f'{actual_brille_npts} ({actual_brille_density})')
        ax_sf.plot(modes.frequencies.magnitude.flatten(),
                   sf.structure_factors.magnitude.flatten(),
                   eps_sf.flatten(), 'x',
                   label=f'{actual_brille_npts} ({actual_brille_density})')

    ax.set_xlabel(f'Frequency / {modes.frequencies.units:~P}')
    ax.set_ylabel(f'Epsilon / {modes.frequencies.units:~P}')
    ax.legend(title='Brille mesh size (density)')
    ax.set_title(Path(filename).name)

    ax.set_xlabel(f'Frequency / {modes.frequencies.units:~P}')
    ax_sf.set_ylabel(f'Structure factors / {sf.structure_factors.units:~P}')
    ax_sf.set_zlabel(f'Epsilon / {sf.structure_factors.units:~P}')
    ax_sf.legend(title='Brille mesh size (density)')
    ax_sf.set_title(Path(filename).name)

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
        '--brille-npts', type=int, default=[5000], nargs='+',
        help=('Approximate number of q-points to generate on the '
              'Brille grid, is passed to the "n_grid_points" kwarg '
              'of BrilleInterpolator.from_force_constants'))
    sections['brille'].add_argument(
        '--brille-npts-density', type=int, default=None, nargs='+',
        help=('Approximate density of q-points to generate on the '
              'Brille grid, is passed to the "grid_density" kwarg '
              'of BrilleInterpolator.from_force_constants'))

    return parser
