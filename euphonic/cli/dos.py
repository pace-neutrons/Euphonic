from typing import List

import euphonic
from euphonic import Spectrum1DCollection
from euphonic.util import mp_grid
from euphonic.plot import plot_1d
from .utils import (load_data_from_file, get_args, matplotlib_save_or_show,
                    _calc_modes_kwargs, _modes_from_fc_and_qpts,
                    _get_cli_parser, _get_energy_bins,
                    _grid_spec_from_args)


def main(params: List[str] = None):
    parser = get_parser()
    args = get_args(parser, params)

    data = load_data_from_file(args.filename)
    mode_widths = None
    if isinstance(data, euphonic.ForceConstants):

        recip_length_unit = euphonic.ureg(f'1 / {args.length_unit}')
        grid_spec = _grid_spec_from_args(data.crystal, grid=args.grid,
                                         grid_spacing=(args.grid_spacing
                                                       * recip_length_unit))

        print("Force Constants data was loaded. Calculating phonon modes "
              "on {} q-point grid...".format(
                  ' x '.join([str(x) for x in grid_spec])))
        cmkwargs, frequencies_only = _calc_modes_kwargs(args)
        modes = _modes_from_fc_and_qpts(data, mp_grid(grid_spec),
                                        frequencies_only, **cmkwargs)
        if args.adaptive:
            if args.shape != 'gauss':
                raise ValueError('Currently only Gaussian shape is supported '
                                 'with adaptive broadening')
            mode_widths = modes[1]
            modes = modes[0]
            if args.energy_broadening:
                mode_widths *= args.energy_broadening
    elif isinstance(data, euphonic.QpointPhononModes):
        print("Phonon band data was loaded.")
        modes = data
    modes.frequencies_unit = args.energy_unit
    ebins = _get_energy_bins(
            modes, args.ebins, emin=args.e_min, emax=args.e_max)
    if len(args.weights.split('-')) > 1:
        weighting = args.weights.split('-')[0]
    else:
        weighting = None
    if weighting is None and args.pdos is None:
        dos = modes.calculate_dos(ebins, mode_widths=mode_widths)
    else:
        dos = modes.calculate_pdos(ebins, mode_widths=mode_widths,
                                   weighting=weighting)
        if args.pdos is None:
            dos = dos[0]
        elif len(args.pdos) > 0:
            labels = [x['label'] for x in dos.metadata['line_data']]
            idx = [labels.index(x) for x in args.pdos]
            dos = Spectrum1DCollection.from_spectra([dos[x] for x in idx])
    if args.energy_broadening and not args.adaptive:
        if isinstance(dos, Spectrum1DCollection):
            dos = Spectrum1DCollection.from_spectra(
                [spec.broaden(args.energy_broadening*ebins.units, shape=args.shape)
                for spec in dos])
        else:
            dos = dos.broaden(args.energy_broadening*ebins.units, shape=args.shape)

    if args.x_label is None:
        x_label = f"Energy / {dos.x_data.units:~P}"
    else:
        x_label = args.x_label
    if args.y_label is None:
        y_label = ""
    else:
        y_label = args.y_label

    fig = plot_1d(dos, title=args.title, x_label=x_label, y_label=y_label,
                  y_min=0, lw=1.0)
    matplotlib_save_or_show(save_filename=args.save_to)


def get_parser():
    parser, _ = _get_cli_parser(features={'read-fc', 'read-modes', 'mp-grid',
                                          'plotting', 'ebins',
                                          'adaptive-broadening',
                                          'dos-weights'})
    parser.description = (
        'Plots a DOS from the file provided. If a force '
        'constants file is provided, a DOS is generated on the Monkhorst-Pack '
        'grid specified by the grid (or grid-spacing) argument.')

    return parser
