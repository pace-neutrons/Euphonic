# -*- coding: UTF-8 -*-
from argparse import ArgumentParser, SUPPRESS
import itertools
from pathlib import Path
import time
from typing import List, Tuple, Optional
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np

from euphonic import ForceConstants, ureg
from euphonic.brille import BrilleInterpolator
from euphonic.cli.utils import (_get_cli_parser, get_args,
                                _brille_calc_modes_kwargs,
                                load_data_from_file,
                                _get_energy_bins)
from euphonic.plot import plot_1d_to_axis
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
        n: int = 0,
        eb: float = 0,
        ebins: int = 500,
        **calc_modes_kwargs
        ) -> None:

    fc = load_data_from_file(filename)
    if not isinstance(fc, ForceConstants):
        raise TypeError('Force constants are required to use the '
                        'euphonic-check-brille-settings tool')

    qpts = np.fromiter(itertools.chain.from_iterable(
        recurrence_sequence(npts, order=3)), dtype=float).reshape(-1, 3)
    modes = fc.calculate_qpoint_phonon_modes(qpts, **calc_modes_kwargs)
    freqs_plot = modes.frequencies.magnitude
    sf_obj = modes.calculate_structure_factor()
    sf = sf_obj.structure_factors.magnitude

    fig, ax = plt.subplots()
    fig_sf, ax_sf = plt.subplots(subplot_kw={'projection': '3d'})
    ebins = _get_energy_bins(modes, ebins)
    sqw1d = sf_obj.calculate_1d_average(ebins)
    sqw1d = sqw1d.broaden(eb*ureg('meV'))
    ebins_plot = sqw1d.get_bin_centres().magnitude
    # First ax for values, second for residual
    fig_sqw1d, ax_sqw1d = plt.subplots(2)
    fig_sqw1d.suptitle('1D Average over all Q-points')
    ax_sqw1d[0].plot(ebins_plot, sqw1d.y_data.magnitude, label='Euphonic', color='k')

    ax_sqwn = []
    if n > 0:
        sqw = sf_obj.calculate_sqw_map(ebins)
        sqw = sqw.broaden(y_width=eb*ureg('meV'))

        for idx in range(n):
            fign, axn = plt.subplots(2)
            fign.suptitle(f'Q={qpts[idx]}')
            axn[0].plot(ebins_plot, sqw.z_data[idx].magnitude,
                        label='Euphonic', color='k')
            ax_sqwn.append(axn)

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
        interp_sf_obj = interp_modes.calculate_structure_factor()
        interp_sf = interp_sf_obj.structure_factors.magnitude

        eps = interp_modes.frequencies.magnitude - modes.frequencies.magnitude
        eps_sf = (interp_sf - sf)

        label = f'{actual_brille_npts} ({actual_brille_density})'
        ax.plot(modes.frequencies.magnitude.flatten(), eps.flatten(), 'x',
                label=label)
        ax_sf.plot(freqs_plot.flatten(), sf.flatten(), eps_sf.flatten(), 'x',
                   label=label)

        interp_sqw1d = interp_sf_obj.calculate_1d_average(ebins)
        interp_sqw1d = interp_sqw1d.broaden(eb*ureg('meV'))
        eps_sqw1d = interp_sqw1d.y_data.magnitude - sqw1d.y_data.magnitude
        ax_sqw1d[0].plot(ebins_plot, interp_sqw1d.y_data.magnitude,
                         label=label)
        ax_sqw1d[1].plot(ebins_plot, eps_sqw1d, label=label)
        if n is not None:
            interp_sqwn = interp_sf_obj.calculate_sqw_map(ebins)
            interp_sqwn = interp_sqwn.broaden(y_width=eb*ureg('meV'))
            eps_sqwn = interp_sqwn.z_data.magnitude - sqw.z_data.magnitude
            for idx, ax_sqw in enumerate(ax_sqwn):
                ax_sqw[0].plot(ebins_plot, interp_sqwn.z_data[idx].magnitude,
                               label=label)
                ax_sqw[1].plot(ebins_plot, eps_sqwn[idx],
                               label=label)

    ax.set_xlabel(f'Frequency / {modes.frequencies.units:~P}')
    ax.set_ylabel(f'Epsilon / {modes.frequencies.units:~P}')
    ax.legend(title='Brille mesh size (density)')
    ax.set_title(Path(filename).name)

    ax_sf.set_xlabel(f'Frequency / {modes.frequencies.units:~P}')
    ax_sf.set_ylabel(
        f'Structure factors / {sf_obj.structure_factors.units:~P}')
    ax_sf.set_zlabel(f'Epsilon / {sf_obj.structure_factors.units:~P}')
    ax_sf.legend(title='Brille mesh size (density)')
    ax_sf.set_title(Path(filename).name)

    ax_sqw1d[0].legend()
    for axi in ax_sqwn:
        axi[0].legend()

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
    sections['brille'].add_argument('-n', type=int, default=0)
    sections['brille'].add_argument('--eb', type=float, default=0)
    sections['brille'].add_argument('--ebins', type=float, default=500)

    return parser
