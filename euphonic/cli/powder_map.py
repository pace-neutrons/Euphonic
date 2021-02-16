#! /usr/bin/env python3

from math import ceil
import pathlib

import numpy as np

from euphonic import ureg
from euphonic.cli.utils import (_get_cli_parser, _get_energy_bins_and_units,
                                _get_q_distance)
from euphonic.cli.utils import (force_constants_from_file,
                                matplotlib_save_or_show)
import euphonic.plot
from euphonic.powder import sample_sphere_dos, sample_sphere_structure_factor
from euphonic.spectra import Spectrum2D
import euphonic.util

# Dummy tqdm function if tqdm progress bars unavailable
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(sequence):
        return sequence


_sampling_choices = {'golden', 'sphere-projected-grid',
                     'spherical-polar-grid', 'spherical-polar-improved',
                     'random-sphere'}
_spectrum_choices = ('dos', 'coherent')


def get_parser() -> 'argparse.ArgumentParser':

    parser = _get_cli_parser(n_ebins=True)

    parser.add_argument('--weights', '-w', default='dos',
                        choices=_spectrum_choices,
                        help=('Spectral weights to plot: phonon DOS or '
                              'coherent inelastic neutron scattering.'))

    sampling_group = parser.add_argument_group(
        'Sampling arguments', 'Powder-averaging options')

    npts_group = sampling_group.add_mutually_exclusive_group()
    npts_group.add_argument('--npts', '-n', type=int, default=1000,
                            help=('Number of samples at each |q| sphere'
                                  ' (default 1000)'))
    npts_group.add_argument('--npts-density', type=int, default=None,
                            dest='npts_density',
                            help=('NPTS specified as the number of points at '
                                  'surface of 1/LENGTH_UNIT-radius sphere;'
                                  ' otherwise scaled to equivalent area '
                                  'density at sphere surface.'))
    sampling_group.add_argument(
        '--npts-min', type=int, default=100, dest='npts_min',
        help=('Minimum number of samples per sphere. This ensures adequate '
              'sampling at small q when using --npts-density.'))
    sampling_group.add_argument(
        '--npts-max', type=int, default=10000, dest='npts_max',
        help=('Maximum number of samples per sphere. This avoids diminishing '
              'returns at large q when using --npts-density.'))

    sampling_group.add_argument(
        '--sampling', type=str, default='golden', choices=_sampling_choices,
        help=('Sphere sampling scheme; "golden" is generally recommended '
              'uniform quasirandom sampling.'))
    sampling_group.add_argument(
        '--jitter', action='store_true',
        help=('Apply additional jitter to sample positions in angular '
              'direction. Recommended for sampling methods other than "golden"'
              ' and "random-sphere".'))

    q_group = parser.add_argument_group(
        'q-axis arguments', 'Arguments controlling |q| axis sampling')
    q_group.add_argument(
        '--length-unit', type=str, default='angstrom', dest='length_unit',
        help=('Length units; these will be inverted to obtain units of '
              'distance between q-points (e.g. "bohr" for bohr^-1).'))
    q_group.add_argument('--q-min', type=float, default=0., dest='q_min',
                        help="Minimum |q| in 1/LENGTH_UNIT")
    q_group.add_argument('--q-max', type=float, default=3., dest='q_max',
                        help="Maximum |q| in 1/LENGTH_UNIT")
    q_group.add_argument(
        '--q-spacing', type=float, dest='q_spacing', default=0.2,
        help=('Target distance between q-point samples in 1/LENGTH_UNIT'))

    #### QUITE REDUNDANT WITH EUPHONIC-INTENSITY-MAP ####
    q_group.add_argument(
        '--q-broadening', '--qb', type=float, default=None,
        dest='q_broadening',
        help=('fwhm of broadening on |q| axis in 1/LENGTH_UNIT. '
              '(No broadening if unspecified.)'))

    ins_group = parser.add_argument_group(
        'Inelastic neutron scattering arguments',
        'Options used when --weights=coherent')
    ins_group.add_argument('--temperature', type=float, default=273.,
                           help='Temperature in K')
    ins_group.add_argument('--dw-spacing', type=float, default=0.1,
                           dest='dw_spacing',
                           help=('q-point spacing of grid in Debye-Waller '
                                 'factor calculation'))

    cmap_group = parser.add_argument_group(
        'Spectrogram arguments', 'Colour mapping options')
    cmap_group.add_argument('--v-min', type=float, default=None, dest='v_min',
                        help='Minimum of data range for colormap.')
    cmap_group.add_argument('--v-max', type=float, default=None, dest='v_max',
                        help='Maximum of data range for colormap.')
    cmap_group.add_argument('--cmap', type=str, default='viridis',
                        help='Matplotlib colormap')
    #### END REDUNDANCY ####

    return parser

def main():
    args = get_parser().parse_args()

    # Make sure we get an error if accessing NPTS inappropriately
    if args.npts_density is not None:
        args.npts = None

    temperature = args.temperature * ureg['K']

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
        fc.calculate_qpoint_phonon_modes(np.array([[0., 0., 0.5]])),
        args.ebins, emin=args.e_min, emax=args.e_max,
        headroom=1.2)  # Generous headroom as we only checked one q-point

    if args.weights in ('coherent',):
        # Compute Debye-Waller factor once for re-use at each mod(q)
        spacing = args.dw_spacing * recip_length_unit
        lattice = fc.crystal.reciprocal_cell().to(recip_length_unit)
        mp_grid = np.linalg.norm(lattice.magnitude, axis=1) / spacing.magnitude
        # math.ceil is better than np.ceil because it returns ints
        mp_grid = [ceil(x) for x in mp_grid]
        print("Calculating Debye-Waller factor on {} q-point grid"
              .format(' x '.join(map(str, mp_grid))))
        dw_phonons = fc.calculate_qpoint_phonon_modes(
            euphonic.util.mp_grid(mp_grid))
        dw = dw_phonons.calculate_debye_waller(temperature)

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
                energy_bins=energy_bins)
        elif args.weights == 'coherent':
            spectrum_1d = sample_sphere_structure_factor(
                fc, q,
                dw=dw,
                temperature=temperature,
                sampling=args.sampling, jitter=args.jitter,
                npts=npts,
                energy_bins=energy_bins)

        z_data[q_index, :] = spectrum_1d.y_data.magnitude

    print(f"Final npts: {npts}")

    spectrum = euphonic.Spectrum2D(q_bin_edges, energy_bins,
                                   z_data * spectrum_1d.y_data.units)


    ##### THIS IS REDUNDANT WITH EUPHONIC-INTENSITY-MAP
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
    #### END REDUNDANCY ###

if __name__ == '__main__':
    main()
