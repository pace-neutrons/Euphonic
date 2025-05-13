from argparse import ArgumentParser
from typing import Optional

import matplotlib.style
from numpy import sqrt
from numpy.polynomial import Polynomial

from euphonic import ForceConstants, QpointPhononModes, ureg
from euphonic.plot import plot_1d
from euphonic.styles import base_style
from euphonic.util import dedent_and_fill, mode_gradients_to_widths, mp_grid

from .utils import (
    _arrange_pdos_groups,
    _calc_modes_kwargs,
    _compose_style,
    _get_cli_parser,
    _get_energy_bins,
    _get_pdos_weighting,
    _grid_spec_from_args,
    _plot_label_kwargs,
    get_args,
    load_data_from_file,
    matplotlib_save_or_show,
)


def main(params: Optional[list[str]] = None) -> None:
    parser = get_parser()
    args = get_args(parser, params)

    frequencies_only = (args.weighting == 'dos' and args.pdos is None)
    data = load_data_from_file(args.filename, verbose=True,
                               frequencies_only=frequencies_only)

    if not frequencies_only and not isinstance(
            data, (QpointPhononModes, ForceConstants)):
        msg = (
            'Eigenvectors are required to use "--pdos" or '
            'any "--weighting" option other than plain DOS'
        )
        raise TypeError(msg)
    if args.adaptive and not isinstance(data, ForceConstants):
        msg = 'Force constants are required to use --adaptive option'
        raise TypeError(msg)

    if (args.energy_broadening
            and args.adaptive
            and len(args.energy_broadening) == 1):
        if args.adaptive_scale is not None:
            msg = dedent_and_fill("""
                Adaptive scale factor was specified twice; use either
                --adaptive-scale or --energy-broadening.  To add a fixed width
                to adaptive broadening, use --instrument-broadening.""")
            raise ValueError(msg)
        args.adaptive_scale = args.energy_broadening[0]

    elif args.energy_broadening:
        if args.inst_broadening is not None:
            msg = (
                'Broadening width was specified twice; use either'
                '--instrument-broadening or --energy-broadening.'
            )
            raise ValueError(msg)
        args.inst_broadening = args.energy_broadening

    if args.inst_broadening:
        energy_broadening_poly = Polynomial(args.inst_broadening)

    mode_widths = None
    if isinstance(data, ForceConstants):

        recip_length_unit = ureg(f'1 / {args.length_unit}')
        grid_spec = _grid_spec_from_args(data.crystal, grid=args.grid,
                                         grid_spacing=(args.grid_spacing
                                                       * recip_length_unit))
        print('Calculating phonon modes '
              'on {} q-point grid...'.format(
                  ' x '.join([str(x) for x in grid_spec])))
        if args.adaptive:
            cmkwargs = _calc_modes_kwargs(args)
            cmkwargs['return_mode_gradients'] = True
            modes, mode_grads = data.calculate_qpoint_phonon_modes(
                mp_grid(grid_spec), **cmkwargs)
            mode_widths = mode_gradients_to_widths(mode_grads,
                                                   modes.crystal.cell_vectors)
            if args.adaptive_scale:
                mode_widths *= args.adaptive_scale
            if args.inst_broadening and args.shape == 'gauss':
                # Combine instrumental broadening and adaptive sample
                # broadening: the convolution of a Gaussian with a Gaussian is
                # a Gaussian with sigma = sqrt(sigma1^2 + sigma2^2)
                mode_widths = sqrt(
                    mode_widths**2
                    + (energy_broadening_poly(modes.frequencies
                                              .to(args.energy_unit).magnitude,
                                              ) * ureg(args.energy_unit))**2)
        else:
            modes = data.calculate_qpoint_phonon_modes(
                mp_grid(grid_spec), **_calc_modes_kwargs(args))

    else:
        modes = data
    modes.frequencies_unit = args.energy_unit
    ebins = _get_energy_bins(
        modes, args.ebins + 1, emin=args.e_min, emax=args.e_max)

    kwargs = {'mode_widths': mode_widths,
              'adaptive_method': args.adaptive_method,
              'adaptive_error': args.adaptive_error,
              'adaptive_error_fit': args.adaptive_fit}

    if args.weighting == 'dos' and args.pdos is None:
        dos = modes.calculate_dos(ebins, **kwargs)
    else:
        kwargs['weighting'] = _get_pdos_weighting(args.weighting)
        pdos = modes.calculate_pdos(ebins, **kwargs)
        dos = _arrange_pdos_groups(pdos, args.pdos)

    if args.inst_broadening and args.shape == 'gauss' and args.adaptive:
        pass  # Gaussian broadening included with adaptive sampling

    elif (args.inst_broadening and len(energy_broadening_poly) > 1):
        # Variable-width Gaussian broadening
        def energy_broadening_func(x):
            return energy_broadening_poly(x.to(args.energy_unit).magnitude,
                                          ) * ureg(args.energy_unit)

        dos = dos.broaden(x_width=energy_broadening_func,
                          shape=args.shape,
                          method='convolve',
                          width_interpolation_error=args.adaptive_error,
                          width_fit='cheby-log')

    elif args.inst_broadening:
        # Fixed-width broadening
        dos = dos.broaden(energy_broadening_poly.coef[0] * ebins.units,
                          shape=args.shape)

    plot_label_kwargs = _plot_label_kwargs(
        args, default_xlabel=f'Energy / {dos.x_data.units:~P}')

    if args.scale is not None:
        dos *= args.scale

    if args.save_json:
        dos.to_json_file(args.save_json)
    style = _compose_style(user_args=args, base=[base_style])
    with matplotlib.style.context(style):
        _ = plot_1d(dos, ymin=0, **plot_label_kwargs)
        matplotlib_save_or_show(save_filename=args.save_to)


def get_parser() -> ArgumentParser:
    parser, _ = _get_cli_parser(features={'read-fc', 'read-modes', 'mp-grid',
                                          'plotting', 'ebins',
                                          'adaptive-broadening',
                                          'pdos-weighting',
                                          'scaling'})
    parser.description = (
        'Plots a DOS from the file provided. If a force '
        'constants file is provided, a DOS is generated on the Monkhorst-Pack '
        'grid specified by the grid (or grid-spacing) argument.')

    return parser
