#! /usr/bin/env python3

import argparse
from math import ceil
from typing import List

import numpy as np

from euphonic import ureg
from euphonic.cli.utils import (_calc_modes_kwargs, _get_cli_parser,
                                _get_debye_waller, _get_energy_bins_and_units,
                                _get_q_distance)
from euphonic.cli.utils import (force_constants_from_file, get_args,
                                matplotlib_save_or_show)
import euphonic.plot
from euphonic.powder import sample_sphere_dos, sample_sphere_structure_factor
import euphonic.util

# Dummy tqdm function if tqdm progress bars unavailable
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(sequence):
        return sequence


def get_parser() -> 'argparse.ArgumentParser':

    parser, sections = _get_cli_parser(
        features={'read-fc', 'weights', 'powder',
                  'plotting', 'ebins', 'q-e', 'map'})

    sections['q'].description = (
        '"GRID" options relate to Monkhorst-Pack sampling for the Debye-Waller'
        ' factor, and only apply when --weights=coherent and --temperature is '
        'set. "NPTS" options determine spherical groups of q-points for '
        'powder-averaging. '
        '"Q" options relate to the sphere sizes (i.e. radial distances).')

    sections['q'].add_argument('--q-min', type=float, default=0., dest='q_min',
                               help="Minimum |q| in 1/LENGTH_UNIT")
    sections['q'].add_argument('--q-max', type=float, default=3., dest='q_max',
                               help="Maximum |q| in 1/LENGTH_UNIT")
    return parser


def main(params: List[str] = None):
    args = get_args(get_parser(), params)
    calc_modes_kwargs = _calc_modes_kwargs(args)

    # Make sure we get an error if accessing NPTS inappropriately
    if args.npts_density is not None:
        args.npts = None

    fc = force_constants_from_file(args.filename)
    print("Force constants data was loaded. Setting up dimensions...")

    q_min = _get_q_distance(args.length_unit, args.q_min)
    q_max = _get_q_distance(args.length_unit, args.q_max)
    recip_length_unit = q_min.units

    n_q_bins = ceil((args.q_max - args.q_min) / args.q_spacing)
    q_bin_edges = np.linspace(q_min.magnitude, q_max.magnitude, n_q_bins + 1,
                              endpoint=True) * recip_length_unit
    q_bin_centers = (q_bin_edges[:-1] + q_bin_edges[1:]) / 2

    # Use X-point modes to estimate frequency range, set up energy bins
    # (Not Gamma in case there are only 3 branches; value would be zero!)

    energy_bins, energy_unit = _get_energy_bins_and_units(
        args.energy_unit,
        fc.calculate_qpoint_frequencies(np.array([[0., 0., 0.5]]),
                                        **calc_modes_kwargs),
        args.ebins, emin=args.e_min, emax=args.e_max,
        headroom=1.2)  # Generous headroom as we only checked one q-point

    if args.weights in ('coherent',):
        # Compute Debye-Waller factor once for re-use at each mod(q)
        # (If temperature is not set, this will be None.)
        if args.temperature is not None:
            temperature = args.temperature * ureg['K']
            dw = _get_debye_waller(temperature, fc, grid=args.grid,
                                   grid_spacing=(args.grid_spacing
                                                 * recip_length_unit),
                                   **calc_modes_kwargs)
        else:
            temperature = None
            dw = None

    print(f"Sampling {n_q_bins} |q| shells between {q_min:~P} and {q_max:~P}")

    z_data = np.empty((n_q_bins, len(energy_bins) - 1))

    for q_index in tqdm(range(n_q_bins)):
        q = q_bin_centers[q_index]

        if args.npts_density is not None:
            npts = ceil(args.npts_density * (q / recip_length_unit)**2)
            npts = max(args.npts_min,
                       min(args.npts_max, npts))
        else:
            npts = args.npts

        if args.weights == 'dos':
            spectrum_1d = sample_sphere_dos(
                fc, q,
                npts=npts, sampling=args.sampling, jitter=args.jitter,
                energy_bins=energy_bins,
                **calc_modes_kwargs)
        elif args.weights == 'coherent':
            spectrum_1d = sample_sphere_structure_factor(
                fc, q,
                dw=dw,
                temperature=temperature,
                sampling=args.sampling, jitter=args.jitter,
                npts=npts,
                energy_bins=energy_bins,
                **calc_modes_kwargs)

        z_data[q_index, :] = spectrum_1d.y_data.magnitude

    print(f"Final npts: {npts}")

    spectrum = euphonic.Spectrum2D(q_bin_edges, energy_bins,
                                   z_data * spectrum_1d.y_data.units)

    if args.q_broadening or args.energy_broadening:
        spectrum = spectrum.broaden(
            x_width=(args.q_broadening * recip_length_unit
                     if args.q_broadening else None),
            y_width=(args.energy_broadening * energy_unit
                     if args.energy_broadening else None),
            shape=args.shape)

    print(f"Plotting figure: max intensity {np.max(spectrum.z_data):~P}")
    if args.y_label is None:
        y_label = f"Energy / {spectrum.y_data.units:~P}"
    else:
        y_label = args.y_label
    if args.x_label is None:
        x_label = f"|q| / {q_min.units:~P}"
    else:
        x_label = args.x_label

    euphonic.plot.plot_2d(spectrum,
                          cmap=args.cmap,
                          vmin=args.v_min, vmax=args.v_max,
                          x_label=x_label,
                          y_label=y_label,
                          title=args.title)
    matplotlib_save_or_show(save_filename=args.save_to)


if __name__ == '__main__':
    main()
