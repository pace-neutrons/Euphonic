from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    _ArgumentGroup,
)
from collections.abc import Collection

from euphonic.util import (
    dedent_and_fill,
    format_error,
)


def _get_cli_parser(features: Collection[str] = {},  # noqa: C901
                    conflict_handler: str = 'error',
                    ) -> tuple[ArgumentParser,
                               dict[str, _ArgumentGroup]]:
    """Instantiate an ArgumentParser with appropriate argument groups

    Parameters
    ----------
    features
        collection (e.g. set, list) of str for known argument groups.
        Known keys: read-fc, read-modes, ins-weighting, pdos-weighting,
        powder, mp-grid, plotting, ebins, adaptive-broadening, q-e,
        map, btol, dipole-parameter-optimisation
    conflict_handler
        With default value ('error') parser will not allow arguments
        to be re-defined. To allow this, set to 'resolve'.

    Returns
    -------
    parser
        ArgumentParser with the requested features
    sections
        Dictionary of section labels and their argument groups. This
        allows their help strings to be customised and specialised
        options to be added

    """
    _pdos_choices = ('coherent-dos', 'incoherent-dos',
                     'coherent-plus-incoherent-dos')
    _ins_choices = ('coherent',)

    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        conflict_handler=conflict_handler)

    # Set up groups; these are only displayed if used, so simplest to
    # instantiate them all now and guarantee consistent order/presence
    # regardless of control logic
    section_defs = [('file', 'File I/O arguments'),
                    ('q', 'q-point sampling arguments'),
                    ('energy', 'energy/frequency arguments'),
                    ('interpolation',
                     'Force constants interpolation arguments'),
                    ('property', 'Property-calculation arguments'),
                    ('plotting', 'Plotting arguments'),
                    ('performance', 'Performance-related arguments'),
                    ('brille',
                     'Brille interpolation related arguments. '
                     'Only applicable if Brille has been installed.'),
                    ]

    sections = {label: parser.add_argument_group(doc)
                for label, doc in section_defs}

    if {'read-fc', 'read-modes'}.intersection(features):
        if {'read-fc', 'read-modes'}.issubset(features):
            filename_doc = (
                'Phonon data file. This should contain force constants or '
                'phonon mode data. Force constants formats: .yaml, '
                'force_constants.hdf5 (Phonopy); .castep_bin , .check '
                '(Castep); .json (Euphonic). Phonon mode data formats: '
                '{band,qpoints,mesh}.{hdf5,yaml} (Phonopy); '
                '.phonon (Castep); .json (Euphonic)')
        elif 'read-fc' in features:
            filename_doc = (
                'Phonon data file. This should contain force constants data. '
                'Accepted formats: .yaml, force_constants.hdf5 (Phonopy); '
                '.castep_bin, .check (Castep); .json (Euphonic).')
        else:
            msg = format_error(
                'Invalid option requested.',
                reason='No band-data-only tools have been defined.',
                fix='Check the requested features are correct.',
            )
            raise ValueError(msg)
        sections['file'].add_argument('filename', type=str, help=filename_doc)

    if 'read-fc' in features:
        sections['interpolation'].add_argument(
            '--asr', type=str, nargs='?', default=None, const='reciprocal',
            choices=('reciprocal', 'realspace'),
            help=('Apply an acoustic-sum-rule (ASR) correction to the '
                  'data: "realspace" applies the correction to the force '
                  'constant matrix in real space. "reciprocal" applies '
                  'the correction to the dynamical matrix at each q-point.'))
        if 'dipole-parameter-optimisation' not in features:
            sections['interpolation'].add_argument(
                '--dipole-parameter', type=float, default=1.0,
                dest='dipole_parameter',
                help=('Set the cutoff in real/reciprocal space for the dipole '
                      'Ewald sum; higher values use more reciprocal terms. If '
                      'tuned correctly this can result in performance '
                      'improvements. See euphonic-optimise-dipole-parameter '
                      'program for help on choosing a good DIPOLE_PARAMETER.'))

        use_c = sections['performance'].add_mutually_exclusive_group()
        use_c.add_argument(
            '--use-c', action='store_true', dest='use_c', default=None,
            help=('Force use of compiled C extension when computing '
                  'phonon frequencies/eigenvectors (or raise error).'))
        use_c.add_argument(
            '--disable-c', action='store_false', dest='use_c', default=None,
            help=('Do not attempt to use compiled C extension when computing '
                  'phonon frequencies/eigenvectors.'))
        sections['performance'].add_argument(
            '--n-threads', type=int, default=None, dest='n_threads',
            help=('Number of parallel processes for computing phonon modes. '
                  '(Only applies when using C extension.)'))

    if 'pdos-weighting' in features:
        # Currently do not support multiple PDOS for 2D plots
        if 'q-e' not in features:
            sections['property'].add_argument(
                '--pdos', type=str, action='store', nargs='*',
                help=(dedent_and_fill("""
                    Plot PDOS. With --pdos, per-species PDOS will be plotted
                    alongside total DOS. A subset of species can also be
                    selected by adding more arguments e.g. --pdos Si O""")))
        else:
            sections['property'].add_argument(
                '--pdos', type=str, action='store', nargs=1,
                help=('Plot PDOS for a specific species e.g. --pdos Si'))

    if {'pdos-weighting', 'ins-weighting'}.intersection(
            features):
        # We can plot plain DOS with all CL tools
        _weighting_choices = ('dos',)
        desc = 'DOS'
        if 'pdos-weighting' in features:
            _weighting_choices += _pdos_choices
            desc += (', coherent neutron-weighted DOS, incoherent '
                     'neutron-weighted DOS or total (coherent + incoherent) '
                     'neutron-weighted DOS')
        # We may have both 'pdos' and 'ins' so use separate if
        if 'ins-weighting' in features:
            _weighting_choices += _ins_choices
            desc += ', coherent inelastic neutron scattering'
            sections['property'].add_argument(
                '--weighting', '-w', default='dos', choices=_weighting_choices,
                help=(f'Spectral weighting to plot: {desc}'))
        else:
            sections['property'].add_argument(
                '--weighting', '-w', default='dos', choices=_weighting_choices,
                help=f'Type of DOS to plot: {desc}')

    if 'ins-weighting' in features:
        sections['property'].add_argument(
            '--temperature', type=float, default=None,
            help=('Temperature in K; enable Debye-Waller factor calculation. '
                  '(Only applicable when --weighting=coherent).'))

    # 'ins-weighting' implies 'mp-grid' as well because mesh is needed for DW
    # factor
    if {'ins-weighting', 'mp-grid'}.intersection(features):
        grid_spec = sections['q'].add_mutually_exclusive_group()

        grid_spec.add_argument(
            '--grid', type=int, nargs=3, default=None,
            help=('Defines a Monkhorst-Pack grid.'))
        grid_spec.add_argument(
            '--grid-spacing', type=float, default=0.1, dest='grid_spacing',
            help=('q-point spacing of Monkhorst-Pack grid.'))

    if 'scaling' in features:
        sections['property'].add_argument(
            '--scale', type=float, help='Intensity scale factor', default=None)

    if 'powder' in features:
        _sampling_choices = {'golden', 'sphere-projected-grid',
                             'spherical-polar-grid',
                             'spherical-polar-improved',
                             'random-sphere'}
        npts_group = sections['q'].add_mutually_exclusive_group()
        npts_group.add_argument('--npts', '-n', type=int, default=1000,
                                help=('Number of samples at each |q| sphere.'))
        npts_group.add_argument(
            '--npts-density', type=int, default=None, dest='npts_density',
            help=('NPTS specified as the number of points at surface of '
                  '1/LENGTH_UNIT-radius sphere; otherwise scaled to equivalent'
                  ' area density at sphere surface.'))

        sections['q'].add_argument(
            '--npts-min', type=int, default=100, dest='npts_min',
            help=('Minimum number of samples per sphere. This ensures adequate'
                  ' sampling at small q when using --npts-density.'))
        sections['q'].add_argument(
            '--npts-max', type=int, default=10000, dest='npts_max',
            help=('Maximum number of samples per sphere. This avoids '
                  'diminishing returns at large q when using --npts-density.'))
        sections['q'].add_argument(
            '--sampling', type=str, default='golden',
            choices=_sampling_choices,
            help=('Spherical sampling scheme; "golden" is generally '
                  'recommended uniform quasirandom sampling.'))
        sections['q'].add_argument(
            '--jitter', action='store_true',
            help=('Apply additional jitter to sample positions in angular '
                  'direction. Recommended for sampling methods other than '
                  '"golden" and "random-sphere".'))

    if 'plotting' in features:
        sections['file'].add_argument(
            '--save-json', dest='save_json', default=None,
            help='Save spectrum to a .json file with this name')
        section = sections['plotting']
        section.add_argument(
            '-s', '--save-to', dest='save_to', default=None,
            help='Save resulting plot to a file with this name')
        section.add_argument('--title', type=str, default=None,
                             help='Plot title')
        section.add_argument('--x-label', '--xlabel', type=str, default=None,
                             dest='xlabel', help='Plot x-axis label')
        section.add_argument('--y-label', '--ylabel', type=str, default=None,
                             dest='ylabel', help='Plot y-axis label')
        section.add_argument('--style', type=str, nargs='+',
                             help='Matplotlib styles (name or file)')
        section.add_argument('--no-base-style', action='store_true',
                             dest='no_base_style',
                             help=('Remove all default formatting before '
                                   'applying other style options.'))
        section.add_argument('--font', type=str, default=None,
                             help=('Select text font. (This has to be a name '
                                   'known to Matplotlib. font-family will be '
                                   "set to sans-serif; it doesn't matter if)"
                                   'the font is actually sans-serif.'))
        section.add_argument('--font-size', '--fontsize', type=float,
                             default=None, dest='fontsize',
                             help='Set base font size in pt.')
        section.add_argument('--fig-size', '--figsize', type=float, nargs=2,
                             default=None, dest='figsize',
                             help='Figure canvas size in FIGSIZE-UNITS')
        section.add_argument('--fig-size-unit', '--figsize-unit', type=str,
                             default='cm', dest='figsize_unit',
                             help='Unit of length for --figsize')

    if ('plotting' in features) and 'map' not in features:
        section = sections['plotting']
        section.add_argument('--line-width', '--linewidth', type=float,
                             default=None, dest='linewidth',
                             help='Set line width in pt.')

    if {'ebins', 'q-e'}.intersection(features):
        section = sections['energy']
        section.add_argument('--e-min', type=float, default=None, dest='e_min',
                             help='Energy range minimum in ENERGY_UNIT')
        section.add_argument('--e-max', type=float, default=None, dest='e_max',
                             help='Energy range maximum in ENERGY_UNIT')
        section.add_argument('--energy-unit', '-u', dest='energy_unit',
                             type=str, default='meV', help='Energy units')

    if {'ebins', 'adaptive-broadening'}.intersection(features):
        if 'ebins' in features:
            section = sections['energy']
            section.add_argument('--ebins', type=int, default=200,
                                 help='Number of energy bins')

            ib_help = (
                'The FWHM of broadening on energy axis in ENERGY_UNIT (no '
                'broadening if unspecified). If multiple values are provided, '
                'these will be interpreted as polynomial coefficients to be '
                'evaluated in ENERGY_UNIT base, e.g. --energy-broadening'
                ' 1. 0.01 1e-6 --energy-unit meV will apply FWHM of '
                '(1. + 0.01 (energy / meV) + 1e-6 (energy / meV)^2) meV.')

            if 'adaptive-broadening' in features:
                section.add_argument(
                    '--adaptive', action='store_true',
                    help=('Use adaptive broadening on the energy axis to '
                          'broaden based on phonon mode widths, rather than '
                          'using fixed width broadening'))
                section.add_argument(
                    '--adaptive-method', type=str, default='reference',
                    dest='adaptive_method', choices=('reference', 'fast'),
                    help=('Adaptive broadening method. "Reference" is default '
                          'for compatibility purposes: "Fast" method is '
                          'approximate with much better performance.' ))
                section.add_argument(
                    '--adaptive-error', type=float, default=0.01,
                    dest='adaptive_error',
                    help=('Maximum absolute error for gaussian approximations '
                          'when using the fast adaptive broadening method'))
                section.add_argument(
                    '--adaptive-scale', type=float, default=None,
                    dest='adaptive_scale',
                    help='Scale factor applied to adaptive broadening width',
                    )
                section.add_argument(
                    '--adaptive-fit', type=str, choices=['cubic', 'cheby-log'],
                    default='cubic', dest='adaptive_fit',
                    help=('Select parametrisation for fast adaptive broadening'
                          '. "cheby-log" is generally recommended, "cubic" is '
                          'default retained for backward-compatibility. This '
                          'only applies when adaptive broadening is used; if '
                          'variable-width instrument broadening is used alone '
                          'then "cheby-log" will be used.' ),
                    )
                section.add_argument(
                    '--instrument-broadening', type=float, nargs='+',
                    default=None, dest='inst_broadening', help=ib_help)

                eb_help = (
                    'If using adaptive broadening and a single (i.e. scalar) '
                    'value is provided, this is an alias for --adaptive-scale.'
                    ' Otherwise, this is an alias for --instrument-broadening.'
                    )
            else:
                eb_help = ib_help
            section.add_argument(
                '--energy-broadening', '--eb', type=float, default=None,
                nargs='+', dest='energy_broadening', help=eb_help)

            section.add_argument(
                '--shape', type=str, nargs='?', default='gauss',
                choices=('gauss', 'lorentz'),
                help='The broadening shape')
        else:
            msg = format_error(
                'Missing "ebins" argument.',
                reason='"adaptive-broadening" cannot be used without "ebins".',
                fix='Include "ebins" in features.',
            )
            raise ValueError(msg)

    if {'q-e', 'mp-grid'}.intersection(features):
        sections['q'].add_argument(
            '--length-unit', type=str, default='angstrom', dest='length_unit',
            help=('Length units; these will be inverted to obtain '
                  'units of distance between q-points (e.g. "bohr"'
                  ' for bohr^-1).'))

    if 'q-e' in features:
        sections['q'].add_argument(
            '--q-spacing', type=float, dest='q_spacing', default=0.025,
            help=('Target distance between q-point samples in 1/LENGTH_UNIT'))

    if {'q-e', 'map'}.issubset(features):
        qb_nargs = '+' if 'powder' in features else 1

        qb_help = ('FWHM of broadening on q axis in 1/LENGTH_UNIT '
                   '(no broadening if unspecified).')
        if qb_nargs == '+':
            qb_help = qb_help + (
                'If multiple values are provided, these will be interpreted as'
                ' polynomial coefficients to be evaluated in 1/LENGTH_UNIT '
                'base.')

        sections['q'].add_argument(
            '--q-broadening', '--qb', type=float, default=None,
            nargs=qb_nargs, dest='q_broadening',
            help=qb_help)

    if 'map' in features:
        sections['plotting'].add_argument(
            '--v-min', '--vmin', type=float, default=None, dest='vmin',
            help='Minimum of data range for colormap.')
        sections['plotting'].add_argument(
            '--v-max', '--vmax', type=float, default=None, dest='vmax',
            help='Maximum of data range for colormap.')
        sections['plotting'].add_argument(
            '--c-map', '--cmap', type=str, default=None, dest='cmap',
            help='Matplotlib colormap')

    if 'btol' in features:
        sections['q'].add_argument(
            '--btol', default=10.0, type=float,
            help=('Distance threshold used for automatically splitting '
                  'discontinuous segments of reciprocal space onto separate '
                  'subplots. This is specified as a multiple of the median '
                  'distance between q-points.'))

    if 'brille' in features:
        sections['brille'].add_argument(
            '--use-brille', action='store_true',
            help=('Use a BrilleInterpolator object to calculate phonon '
                  'frequencies and eigenvectors instead of '
                  'ForceConstants'))
        sections['brille'].add_argument(
            '--brille-grid-type', default='trellis',
            choices=('trellis', 'mesh', 'nest'),
            help=('Type of Brille grid to use, passed to the '
                  '"grid_type" kwarg of '
                  'BrilleInterpolator.from_force_constants'))
        sections['brille'].add_argument(
            '--brille-npts', type=int, default=5000,
            help=('Approximate number of q-points to generate on the '
                  'Brille grid, is passed to the "n_grid_points" kwarg '
                  'of BrilleInterpolator.from_force_constants'))
        sections['brille'].add_argument(
            '--brille-npts-density', type=int, default=None,
            help=('Approximate density of q-points to generate on the '
                  'Brille grid, is passed to the "grid_density" kwarg '
                  'of BrilleInterpolator.from_force_constants'))

    if 'dipole-parameter-optimisation' in features:
        parser.add_argument(
            '-n',
            default=500,
            type=int,
            help=('The number of times to loop over q-points. A higher '
                  'value will get a more reliable timing, but will take '
                  'longer'),
        )
        parser.add_argument(
            '--dipole-parameter-min', '--min',
            default=0.25,
            type=float,
            help='The minimum value of dipole_parameter to test',
        )
        parser.add_argument(
            '--dipole-parameter-max', '--max',
            default=1.5,
            type=float,
            help='The maximum value of dipole_parameter to test',
        )
        parser.add_argument(
            '--dipole-parameter-step', '--step',
            default=0.25,
            type=float,
            help='The difference between each dipole_parameter to test',
        )

    return parser, sections
