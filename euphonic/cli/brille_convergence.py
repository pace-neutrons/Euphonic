# -*- coding: UTF-8 -*-
from argparse import ArgumentParser, Namespace, SUPPRESS
import itertools
from pathlib import Path
from typing import List, Tuple, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from euphonic import ForceConstants, Spectrum1D, ureg
from euphonic.brille import BrilleInterpolator
from euphonic.cli.utils import (_get_cli_parser, get_args,
                                _brille_calc_modes_kwargs,
                                load_data_from_file,
                                _get_energy_bins)
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
        energy_broadening: Optional[float] = None,
        ebins: int = 500,
        e_min: Optional[float] = None,
        e_max: Optional[float] = None,
        energy_unit: str = 'meV',
        shape: str = 'gauss',
        **calc_modes_kwargs
        ) -> None:

    fc = load_data_from_file(filename)
    if not isinstance(fc, ForceConstants):
        raise TypeError('Force constants are required to use the '
                        'euphonic-check-brille-settings tool')

    brille_calc_modes_kwargs = _brille_calc_modes_kwargs(
        Namespace(**calc_modes_kwargs))
    qpts = np.fromiter(itertools.chain.from_iterable(
        recurrence_sequence(npts, order=3)), dtype=float).reshape(-1, 3)

    # Calculate Euphonic frequencies, structure factors and intensities
    modes = fc.calculate_qpoint_phonon_modes(qpts, **calc_modes_kwargs)
    modes.frequencies_unit = energy_unit
    sf = modes.calculate_structure_factor()
    ebins = _get_energy_bins(modes, ebins + 1, emin=e_min, emax=e_max)
    sqw_avg = sf.calculate_1d_average(ebins)
    if energy_broadening is not None:
        sqw_avg = sqw_avg.broaden(energy_broadening*ebins.units, shape=shape)

    # Set up axis labels
    frequency_label = f'Frequency / {modes.frequencies.units:~P}'
    residual_label = 'Epsilon / '
    fname = f'{Path(filename).name}'

    # Set up plot for frequency residuals
    _, ax_freq = plt.subplots()
    ax_freq.set_xlabel(frequency_label)
    ax_freq.set_ylabel(residual_label + f'{modes.frequencies.units:~P}')
    ax_freq.set_title(f'Frequency residual ({fname})')

    # Set up plot for structure factor residuals
    _, ax_sf = plt.subplots(subplot_kw={'projection': '3d'})
    ax_sf.set_xlabel(frequency_label)
    ax_sf.set_ylabel(f'Structure factors / {sf.structure_factors.units:~P}')
    ax_sf.set_zlabel(residual_label + f'{sf.structure_factors.units:~P}')
    ax_sf.set_title(f'Structure factor residual ({fname})')

    # Set up plot for 1D average intensity & residuals
    ax_avg, eu_handle = create_intensity_plot(
        sqw_avg, f'1D Average Intensity over all Q-points ({fname})',
        frequency_label, residual_label)

    # Set up plot for q-point specific intensity & residuals
    ax_sqwn = []
    sqw1d_n = []
    if n > 0:
        sqw = sf.calculate_sqw_map(ebins)
        if energy_broadening is not None:
            sqw = sqw.broaden(y_width=energy_broadening*ebins.units,
                              shape=shape)
        for idx in range(n):
            # Use idx + 1 to avoid Q=[0., 0., 0.] at idx=0
            sqw1d = Spectrum1D(sqw.y_data, sqw.z_data[idx + 1])
            sqw1d_n.append(sqw1d)
            title = f'Intensity at Q={np.round(qpts[idx + 1], 4)} ({fname})'
            ax, _ = create_intensity_plot(sqw1d, title, frequency_label,
                                          residual_label)
            ax_sqwn.append(ax)

    if brille_npts_density is not None:
        brille_params = brille_npts_density
        brille_param_key = 'grid_density'
    else:
        brille_params = brille_npts
        brille_param_key = 'grid_npts'

    handles = []
    for brille_param_i in brille_params:

        # Calculate Brille frequencies, structure factors and intensities
        brille_fc = BrilleInterpolator.from_force_constants(
            fc, grid_type=brille_grid_type,
            interpolation_kwargs=calc_modes_kwargs,
            **{brille_param_key: brille_param_i})
        brille_modes = brille_fc.calculate_qpoint_phonon_modes(
            qpts, **brille_calc_modes_kwargs)
        brille_modes.frequencies_unit = energy_unit
        brille_sf = brille_modes.calculate_structure_factor()
        brille_sqw_avg = brille_sf.calculate_1d_average(ebins)
        if energy_broadening is not None:
            brille_sqw_avg = brille_sqw_avg.broaden(
                energy_broadening*ebins.units, shape=shape)

        # Create plot labels
        actual_brille_npts = len(brille_fc._grid.rlu)
        bz_vol = brille_fc._grid.BrillouinZone.ir_polyhedron.volume
        actual_brille_density = int(actual_brille_npts/bz_vol)
        label = f'{actual_brille_npts} ({actual_brille_density})'

        # Plot frequency residuals
        eps = (brille_modes.frequencies
               - modes.frequencies).magnitude.flatten()
        ax_freq.plot(modes.frequencies.magnitude.flatten(), eps, 'x',
                     label=label)

        # Plot structure factor residuals
        eps_sf = (brille_sf.structure_factors
                  - sf.structure_factors).magnitude.flatten()
        ax_sf.plot(modes.frequencies.magnitude.flatten(),
                   sf.structure_factors.magnitude.flatten(), eps_sf, 'x',
                   label=label)

        # Plot 1D average intensity & residuals
        handle = plot_to_intensity_axes(ax_avg, sqw_avg, brille_sqw_avg, label)
        handles.append(handle)

        # Plot q-point specific intensity & residuals
        if n > 0:
            brille_sqw = brille_sf.calculate_sqw_map(ebins)
            if energy_broadening is not None:
                brille_sqw = brille_sqw.broaden(
                    y_width=energy_broadening*ebins.units, shape=shape)
            for idx, ax_sqw in enumerate(ax_sqwn):
                brille_sqw1d = Spectrum1D(brille_sqw.y_data,
                                          brille_sqw.z_data[idx + 1])
                plot_to_intensity_axes(
                    ax_sqw, sqw1d_n[idx], brille_sqw1d, label)

    # Add legends
    legend_title = (f'Brille mesh: N qpts '
                    f'(density {ureg("angstrom**3").units:~P})')
    ax_freq.legend(title=legend_title)
    ax_sf.legend(title=legend_title)
    add_intensity_legend(ax_avg, eu_handle, handles, legend_title)
    for ax_sqw in ax_sqwn:
        add_intensity_legend(ax_sqw, eu_handle, handles, legend_title)

    plt.show()


def create_intensity_plot(sqw1d: Spectrum1D, title: str,
                          frequency_label: str, residual_label: str,
                          ) -> Tuple[Tuple[Axes, Axes], Line2D]:
    fig, axes = plt.subplots(nrows=2, sharex=True)
    axes[1].set_xlabel(frequency_label)
    axes[1].set_ylabel(residual_label + f'{sqw1d.y_data.units:~P}')
    axes[0].set_ylabel(f'Intensity / {sqw1d.y_data.units:~P}')
    ebins_plot = sqw1d.get_bin_centres().magnitude
    handle, = axes[0].plot(ebins_plot, sqw1d.y_data.magnitude,
                           label='Euphonic', color='k')
    fig.suptitle(title)
    fig.tight_layout()
    return axes, handle


def plot_to_intensity_axes(axes: Tuple[Axes, Axes], sqw1d: Spectrum1D,
                           brille_sqw1d: Spectrum1D, label: str) -> Line2D:
    eps = (brille_sqw1d.y_data - sqw1d.y_data).magnitude
    ebins_plot = sqw1d.get_bin_centres().magnitude
    handle, = axes[0].plot(ebins_plot, brille_sqw1d.y_data.magnitude,
                           label=label)
    axes[1].plot(ebins_plot, eps)
    return handle


def add_intensity_legend(axes: Tuple[Axes, Axes], euphonic_handle: Line2D,
                         brille_handles: Sequence[Line2D], title: str
                         ) -> None:
    euph_legend = axes[0].legend(handles=[euphonic_handle], loc='upper left')
    axes[0].add_artist(euph_legend)
    axes[0].legend(handles=brille_handles, loc='upper right', title=title)


def get_parser() -> ArgumentParser:
    parser, sections = _get_cli_parser(
        features={'read-fc', 'brille', 'ebins'},
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
    sections['brille'].add_argument(
        '-n', type=int, default=0,
        help=('Number of single q-point intensity slices to plot'))

    return parser
