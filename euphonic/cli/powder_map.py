from argparse import ArgumentParser
from math import ceil
from typing import Callable, Dict, List, Optional, Sequence, Union

import matplotlib.style
import numpy as np
from numpy.polynomial import Polynomial
from pint import Unit


from euphonic import ureg, ForceConstants, Quantity
from euphonic.cli.utils import (_calc_modes_kwargs, _brille_calc_modes_kwargs,
                                _compose_style, _get_cli_parser,
                                _get_debye_waller, _get_energy_bins,
                                _get_q_distance, _get_pdos_weighting,
                                _arrange_pdos_groups, _plot_label_kwargs)
from euphonic.cli.utils import (load_data_from_file, get_args,
                                matplotlib_save_or_show)
import euphonic.plot
from euphonic.powder import (sample_sphere_dos, sample_sphere_pdos,
                             sample_sphere_structure_factor)
from euphonic.spectra import apply_kinematic_constraints
from euphonic.styles import base_style, intensity_widget_style
import euphonic.util

# Dummy tqdm function if tqdm progress bars unavailable
try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        def tqdm(sequence):
            return sequence


def get_parser() -> ArgumentParser:

    parser, sections = _get_cli_parser(
        features={'read-fc', 'pdos-weighting', 'ins-weighting',
                  'powder', 'plotting', 'ebins', 'q-e', 'map',
                  'brille', 'kinematic', 'scaling'})

    sections['q'].description = (
        '"GRID" options relate to Monkhorst-Pack sampling for the '
        'Debye-Waller factor, and only apply when --weighting=coherent and '
        '--temperature is set. "NPTS" options determine spherical groups of '
        'q-points for powder-averaging. "Q" options relate to the sphere '
        'sizes (i.e. radial distances).')

    sections['q'].add_argument('--q-min', type=float, default=0., dest='q_min',
                               help="Minimum |q| in 1/LENGTH_UNIT")
    sections['q'].add_argument('--q-max', type=float, default=3., dest='q_max',
                               help="Maximum |q| in 1/LENGTH_UNIT")
    sections['plotting'].add_argument('--no-widgets', action='store_true',
                                      dest='disable_widgets', default=False,
                                      help=("Don't use Matplotlib widgets to "
                                            "enable interactive setting of "
                                            "colormap intensity limits"))

    kinematic = parser.add_argument_group('Kinematic constraints')
    e_lims = kinematic.add_mutually_exclusive_group()

    e_lims.add_argument('--e-incident', '--e-i', dest='e_i',
                        type=float, default=None,
                        help="Incident energy for direct-geometry constraints")
    e_lims.add_argument('--e-final', '--e-f', dest='e_f',
                        type=float, default=None,
                        help="Final energy for indirect-geometry constraints")
    kinematic.add_argument(
        '--angle-range', nargs=2, type=float, dest='angle_range',
        default=[0., 180.],
        help=("Range of scattering angles (2θ) in degrees. These lower/upper "
              "bounds are used with incident/final energy to determine "
              "accessible (|q|, ω) region."))
    return parser


def _get_broaden_kwargs(q_broadening: Optional[Sequence[float]] = None,
                        q_unit: Unit = ureg('1/angstrom').units,
                        energy_broadening: Optional[Sequence[float]] = None,
                        energy_unit: Unit = ureg('meV').units,
                        ) -> Dict[str, Union[Quantity, Callable, None]]:
    """Collect suitable width arguments for 2D broaden() method

    Internal Euphonic methods take a Quantity or a callable returning Quantity:
    CLI takes numbers and units separately so these have to be collected into
    appropriate objects.
    """

    if (q_broadening is not None) and len(q_broadening) > 1:
        q_poly = Polynomial(q_broadening)

        def q_width(x):
            return q_poly(x.to(q_unit).magnitude
                          ) * q_unit
    elif q_broadening:
        q_width = q_broadening[0] * q_unit
    else:
        q_width = None

    if (energy_broadening is not None) and len(energy_broadening) > 1:
        energy_poly = Polynomial(energy_broadening)

        def energy_width(x):
            return energy_poly(x.to(energy_unit).magnitude
                               ) * ureg(energy_unit)
    elif energy_broadening:
        energy_width = energy_broadening[0] * ureg(energy_unit)
    else:
        energy_width = None

    return dict(x_width=q_width, y_width=energy_width)


def main(params: Optional[List[str]] = None) -> None:
    args = get_args(get_parser(), params)
    calc_modes_kwargs = _calc_modes_kwargs(args)

    # Make sure we get an error if accessing NPTS inappropriately
    if args.npts_density is not None:
        args.npts = None

    fc = load_data_from_file(args.filename, verbose=True)
    if not isinstance(fc, ForceConstants):
        raise TypeError('Force constants are required to use the '
                        'euphonic-powder-map tool')
    if args.pdos is not None and args.weighting == 'coherent':
        raise ValueError('"--pdos" is only compatible with '
                         '"--weighting" options that include dos')

    if args.use_brille:
        from euphonic.brille import BrilleInterpolator
        fc = BrilleInterpolator.from_force_constants(
            fc, grid_type=args.brille_grid_type,
            grid_npts=args.brille_npts,
            grid_density=args.brille_npts_density,
            interpolation_kwargs=calc_modes_kwargs)
        calc_modes_kwargs = _brille_calc_modes_kwargs(args)

    print("Setting up dimensions...")
    q_min = _get_q_distance(args.length_unit, args.q_min)
    q_max = _get_q_distance(args.length_unit, args.q_max)
    recip_length_unit = q_min.units

    n_q_bins = ceil((args.q_max - args.q_min) / args.q_spacing)
    q_bin_edges = np.linspace(q_min.magnitude, q_max.magnitude, n_q_bins + 1,
                              endpoint=True) * recip_length_unit
    q_bin_centers = (q_bin_edges[:-1] + q_bin_edges[1:]) / 2

    # Use X-point modes to estimate frequency range, set up energy bins
    # (Not Gamma in case there are only 3 branches; value would be zero!)
    modes = fc.calculate_qpoint_frequencies(
        np.array([[0., 0., 0.5]]), **calc_modes_kwargs)
    modes.frequencies_unit = args.energy_unit

    if args.e_i is not None and args.e_max is None:
        emax = args.e_i
    else:
        emax = args.e_max

    energy_bins = _get_energy_bins(
        modes, args.ebins + 1, emin=args.e_min, emax=emax,
        headroom=1.2)  # Generous headroom as we only checked one q-point

    if args.weighting in ('coherent',):
        # Compute Debye-Waller factor once for re-use at each mod(q)
        # (If temperature is not set, this will be None.)
        if args.temperature is not None:
            temperature = args.temperature * ureg('K')
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

        if args.weighting == 'dos' and args.pdos is None:
            spectrum_1d = sample_sphere_dos(
                fc, q,
                npts=npts, sampling=args.sampling, jitter=args.jitter,
                energy_bins=energy_bins,
                **calc_modes_kwargs)
        elif 'dos' in args.weighting:
            spectrum_1d_col = sample_sphere_pdos(
                    fc, q,
                    npts=npts, sampling=args.sampling, jitter=args.jitter,
                    energy_bins=energy_bins,
                    weighting=_get_pdos_weighting(args.weighting),
                    **calc_modes_kwargs)
            spectrum_1d = _arrange_pdos_groups(spectrum_1d_col, args.pdos)
        elif args.weighting == 'coherent':
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

    if (args.energy_broadening is not None) or (args.q_broadening is not None):
        spectrum = spectrum.broaden(
            shape=args.shape,
            **_get_broaden_kwargs(q_broadening=args.q_broadening,
                                  q_unit=recip_length_unit,
                                  energy_broadening=args.energy_broadening,
                                  energy_unit=args.energy_unit))

    if not (args.e_i is None and args.e_f is None):
        print("Applying kinematic constraints")
        energy_unit = args.energy_unit
        e_i = args.e_i * ureg(energy_unit) if (args.e_i is not None) else None
        e_f = args.e_f * ureg(energy_unit) if (args.e_f is not None) else None
        spectrum = apply_kinematic_constraints(
            spectrum, e_i=e_i, e_f=e_f, angle_range=args.angle_range)

    if args.scale is not None:
        spectrum *= args.scale

    print(f"Plotting figure: max intensity "
          f"{np.nanmax(spectrum.z_data.magnitude) * spectrum.z_data.units:~P}")
    plot_label_kwargs = _plot_label_kwargs(
        args, default_xlabel=f"|q| / {q_min.units:~P}",
        default_ylabel=f"Energy / {spectrum.y_data.units:~P}")

    if args.save_json:
        spectrum.to_json_file(args.save_json)
    if args.disable_widgets:
        base = [base_style]
    else:
        base = [base_style, intensity_widget_style]
    style = _compose_style(user_args=args, base=base)
    with matplotlib.style.context(style):
        fig = euphonic.plot.plot_2d(spectrum,
                                    vmin=args.vmin,
                                    vmax=args.vmax,
                                    **plot_label_kwargs)

        if args.disable_widgets is False:
            # TextBox only available from mpl 2.1.0
            try:
                from matplotlib.widgets import TextBox
            except ImportError:
                args.disable_widgets = True

        if args.disable_widgets is False:
            min_label = f'Min Intensity ({spectrum.z_data.units:~P})'
            max_label = f'Max Intensity ({spectrum.z_data.units:~P})'
            boxw = 0.15
            boxh = 0.05
            x0 = 0.1 + len(min_label)*0.01
            y0 = 0.025
            axmin = fig.add_axes([x0, y0, boxw, boxh])
            axmax = fig.add_axes([x0, y0 + 0.075, boxw, boxh])
            image = fig.get_axes()[0].images[0]
            cmin, cmax = image.get_clim()
            pad = 0.05
            fmt_str = '.2e' if cmax < 0.1 else '.2f'
            minbox = TextBox(axmin, min_label,
                             initial=f'{cmin:{fmt_str}}', label_pad=pad)
            maxbox = TextBox(axmax, max_label,
                             initial=f'{cmax:{fmt_str}}', label_pad=pad)

            def update_min(min_val):
                image.set_clim(vmin=float(min_val))
                fig.canvas.draw()

            def update_max(max_val):
                image.set_clim(vmax=float(max_val))
                fig.canvas.draw()
            minbox.on_submit(update_min)
            maxbox.on_submit(update_max)

        matplotlib_save_or_show(save_filename=args.save_to)
