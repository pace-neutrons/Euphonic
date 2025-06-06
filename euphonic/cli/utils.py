from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    Namespace,
    _ArgumentGroup,
)
from collections.abc import Collection, Sequence
from contextlib import suppress
import json
import os
import pathlib
from typing import Any, Optional
import warnings

import numpy as np
from pint import UndefinedUnitError
import seekpath

from euphonic import (
    Crystal,
    DebyeWaller,
    ForceConstants,
    QpointFrequencies,
    QpointPhononModes,
    Quantity,
    Spectrum1D,
    Spectrum1DCollection,
    ureg,
)
from euphonic.util import dedent_and_fill, mp_grid


def _load_euphonic_json(filename: str | os.PathLike,
                        frequencies_only: bool = False,
) -> QpointPhononModes | QpointFrequencies | ForceConstants:
    with open(filename) as f:
        data = json.load(f)

    if 'force_constants' in data:
        return ForceConstants.from_json_file(filename)
    if 'frequencies' in data:
        if 'eigenvectors' in data and not frequencies_only:
            return QpointPhononModes.from_json_file(filename)
        return QpointFrequencies.from_json_file(filename)

    msg = 'Could not identify Euphonic data in JSON file.'
    raise ValueError(msg)


def _load_phonopy_file(filename: str | os.PathLike,
                       frequencies_only: bool = False,
) -> QpointPhononModes | QpointFrequencies | ForceConstants:
    path = pathlib.Path(filename)
    loaded_data = None
    if not frequencies_only:
        with suppress(KeyError, RuntimeError):
            # KeyError will be raised if it is actually a force
            # constants file, RuntimeError will be raised if
            # it only contains q-point frequencies (no eigenvectors)

            loaded_data = QpointPhononModes.from_phonopy(
                path=path.parent, phonon_name=path.name)

    # Try to read QpointFrequencies if loading QpointPhononModes has
    # failed, or has been specifically requested with frequencies_only
    if frequencies_only or loaded_data is None:
        with suppress(KeyError):
            loaded_data = QpointFrequencies.from_phonopy(
                path=path.parent, phonon_name=path.name)

    if loaded_data is None:
        phonopy_kwargs: dict[str, str | os.PathLike] = {}
        phonopy_kwargs['path'] = path.parent
        if (path.parent / 'BORN').is_file():
            phonopy_kwargs['born_name'] = 'BORN'
        # Set summary_name and fc_name depending on input file
        if path.suffix == '.hdf5':
            if (path.parent / 'phonopy.yaml').is_file():
                phonopy_kwargs['summary_name'] = 'phonopy.yaml'
                phonopy_kwargs['fc_name'] = path.name
            else:
                msg = (
                    'Phonopy force_constants.hdf5 file '
                    'must be accompanied by phonopy.yaml'
                )
                raise ValueError(msg)
        elif path.suffix == '.yaml':
            phonopy_kwargs['summary_name'] = path.name
            # Assume this is a (renamed?) phonopy.yaml file
            if (path.parent / 'force_constants.hdf5').is_file():
                phonopy_kwargs['fc_name'] = 'force_constants.hdf5'
            else:
                phonopy_kwargs['fc_name'] = 'FORCE_CONSTANTS'
        loaded_data = ForceConstants.from_phonopy(**phonopy_kwargs)

    return loaded_data


def load_data_from_file(filename: str | os.PathLike,
                        frequencies_only: bool = False,
                        verbose: bool = False,
) -> QpointPhononModes | QpointFrequencies | ForceConstants:
    """
    Load phonon mode or force constants data from file

    Parameters
    ----------
    filename
        The file with a path
    frequencies_only
        If true only reads frequencies (not eigenvectors) from the
        file. Only applies if the file is not a force constants
        file.

    Returns
    -------
    file_data
    """
    castep_qpm_suffixes = ('.phonon',)
    castep_fc_suffixes = ('.castep_bin', '.check')
    phonopy_suffixes = ('.hdf5', '.yaml')

    path = pathlib.Path(filename)
    if path.suffix in castep_qpm_suffixes:
        if frequencies_only:
            data = QpointFrequencies.from_castep(path)
        else:
            data = QpointPhononModes.from_castep(path)
    elif path.suffix in castep_fc_suffixes:
        data = ForceConstants.from_castep(path)
    elif path.suffix == '.json':
        data = _load_euphonic_json(path, frequencies_only)
    elif path.suffix in phonopy_suffixes:
        data = _load_phonopy_file(path, frequencies_only)
    else:
        msg = dedent_and_fill(f"""
            File format was not recognised. CASTEP force constants data for
            import should have extension from {castep_fc_suffixes}, CASTEP
            phonon mode data for import should have extension
            '{castep_qpm_suffixes}', data from Phonopy should have extension
            from {phonopy_suffixes}, data from Euphonic should have extension
            '.json'.""")
        raise ValueError(msg)
    if verbose:
        print(f'{data.__class__.__name__} data was loaded')
    return data


def get_args(parser: ArgumentParser, params: Optional[list[str]] = None,
             ) -> Namespace:
    """
    Get the arguments from the parser. `params` should only be `None` when
    running from command line.

    Parameters
    ----------
    parser
        The parser to get the arguments from
    params
        The parameters to get arguments from, if None,
         then parse_args gets them from command line args

    Returns
    -------
    args
        Arguments object for use e.g. args.unit
    """
    return parser.parse_args(args=params)


def matplotlib_save_or_show(save_filename: str | None = None) -> None:
    """
    Save or show the current matplotlib plot.
    Show if save_filename is not None which by default it is.

    Parameters
    ----------
    save_filename
        The file to save the plot in
    """
    import matplotlib.pyplot as plt
    if save_filename is not None:
        plt.savefig(save_filename)
        print(f'Saved plot to {os.path.realpath(save_filename)}')
    else:
        plt.show()


def _get_q_distance(length_unit_string: str, q_distance: float) -> Quantity:
    """
    Parse user arguments to obtain reciprocal-length spacing Quantity
    """
    try:
        length_units = ureg(length_unit_string)
    except UndefinedUnitError as err:
        msg = (
            'Length unit not known. Euphonic uses Pint for units. Try '
            "'angstrom' or 'bohr'. Metric prefixes are also allowed, e.g 'nm'."
        )
        raise ValueError(msg) from err
    recip_length_units = 1 / length_units
    return q_distance * recip_length_units


def _get_energy_bins(
        modes: QpointPhononModes | QpointFrequencies,
        n_ebins: int, emin: Optional[float] = None,
        emax: Optional[float] = None,
        headroom: float = 1.05) -> Quantity:
    """
    Gets recommended energy bins, in same units as modes.frequencies.
    emin and emax are assumed to be in the same units as
    modes.frequencies, if not provided the min/max values of
    modes.frequencies are used to find the bin limits
    """
    if emin is None:
        # Subtract small amount from min frequency - otherwise due to unit
        # conversions binning of this frequency can vary with different
        # architectures/lib versions, making it difficult to test
        emin_room = 1e-5*ureg('meV').to(modes.frequencies.units).magnitude
        emin = min(np.min(modes.frequencies.magnitude - emin_room), 0.)
    if emax is None:
        emax = np.max(modes.frequencies.magnitude) * headroom
    if emin >= emax:
        msg = (
            'Maximum energy should be greater than minimum. '
            'Check --e-min and --e-max arguments.'
        )
        raise ValueError(msg)
    return np.linspace(emin, emax, n_ebins) * modes.frequencies.units


def _get_tick_labels(bandpath: dict) -> list[tuple[int, str]]:
    """Convert x-axis labels from seekpath format to euphonic format

    i.e.::

        ['L', '', '', 'X', '', 'GAMMA']   -->

        [(0, 'L'), (3, 'X'), (5, '$\\Gamma$')]
    """

    label_indices = np.where(bandpath['explicit_kpoints_labels'])[0]
    labels = [bandpath['explicit_kpoints_labels'][i] for i in label_indices]

    for i, label in enumerate(labels):
        if label == 'GAMMA':
            labels[i] = r'$\Gamma$'

    return list(zip(label_indices, labels, strict=True))


def _get_break_points(bandpath: dict) -> list[int]:
    """Get information about band path labels and break points

    Parameters
    ----------
    bandpath
        Bandpath dictionary from Seekpath

    Returns
    -------
    break_points
        Indices at which the spectrum should be split into subplots
    """
    # Find break points between continuous spectra: wherever there are two
    # adjacent labels
    labels = np.array(bandpath['explicit_kpoints_labels'])

    special_point_bools = np.fromiter(
        map(bool, labels), dtype=bool)

    # [T F F T T F T] -> [F F T T F T] AND [T F F T T F] = [F F F T F F] -> 3,
    adjacent_non_empty_labels = np.logical_and(special_point_bools[:-1],
                                               special_point_bools[1:])

    adjacent_different_labels = (labels[:-1] != labels[1:])

    break_points = np.where(np.logical_and(adjacent_non_empty_labels,
                                           adjacent_different_labels))[0]
    return (break_points + 1).tolist()


def _insert_gamma(bandpath: dict) -> None:
    """Modify seekpath.get_explicit_k_path() results; duplicate Gamma

    This enables LO-TO splitting to be included
    """
    import numpy as np
    gamma_indices = np.where(
        np.array(bandpath['explicit_kpoints_labels'][1:-1]) == 'GAMMA',
    )[0] + 1

    rel_kpts = bandpath['explicit_kpoints_rel'].tolist()
    labels = bandpath['explicit_kpoints_labels']
    for i in reversed(gamma_indices.tolist()):
        rel_kpts.insert(i, [0., 0., 0.])
        labels.insert(i, 'GAMMA')

    bandpath['explicit_kpoints_rel'] = np.array(rel_kpts)
    bandpath['explicit_kpoints_labels'] = labels

    # These unused properties have been invalidated: safer
    # to leave None than incorrect values
    bandpath['explicit_kpoints_abs'] = None
    bandpath['explicit_kpoints_linearcoord'] = None
    bandpath['explicit_segments'] = None


XTickLabels = list[tuple[int, str]]
SplitArgs = dict[str, Any]


def _bands_from_force_constants(data: ForceConstants,
                                q_distance: Quantity,
                                insert_gamma: bool = True,
                                frequencies_only: bool = False,
                                **calc_modes_kwargs,
) -> tuple[QpointPhononModes | QpointFrequencies, XTickLabels, SplitArgs]:
    structure = data.crystal.to_spglib_cell()
    with warnings.catch_warnings():
        # SeeK-path is raising spglib 2.5.0 deprecation warnings, we
        # don't care to see those for now
        warnings.simplefilter('ignore', category=DeprecationWarning)
        bandpath = seekpath.get_explicit_k_path(
            structure,
            reference_distance=q_distance.to('1 / angstrom').magnitude)

    if insert_gamma:
        _insert_gamma(bandpath)

    x_tick_labels = _get_tick_labels(bandpath)
    split_args = {'indices': _get_break_points(bandpath)}

    print(
        'Computing phonon modes: {n_modes} modes across {n_qpts} q-points'
        .format(n_modes=(data.crystal.n_atoms * 3),
                n_qpts=len(bandpath['explicit_kpoints_rel'])))
    qpts = bandpath['explicit_kpoints_rel']

    if frequencies_only:
        modes = data.calculate_qpoint_frequencies(qpts,
                                                  reduce_qpts=False,
                                                  **calc_modes_kwargs)
    else:
        modes = data.calculate_qpoint_phonon_modes(qpts,
                                                   reduce_qpts=False,
                                                   **calc_modes_kwargs)
    return modes, x_tick_labels, split_args


def _grid_spec_from_args(crystal: Crystal,
                           grid: Optional[Sequence[int]] = None,
                           grid_spacing: Quantity = 0.1 * ureg('1/angstrom'),
                           ) -> tuple[int, int, int]:
    """Get Monkorst-Pack mesh divisions from user arguments"""
    if grid:
        grid_spec = tuple(grid)
    else:
        grid_spec = crystal.get_mp_grid_spec(spacing=grid_spacing)
    return grid_spec


def _get_debye_waller(temperature: Quantity,
                      fc: ForceConstants,
                      grid: Optional[Sequence[int]] = None,
                      grid_spacing: Quantity = 0.1 * ureg('1/angstrom'),
                      **calc_modes_kwargs,
                      ) -> DebyeWaller:
    """Generate Debye-Waller data from force constants and grid specification
    """
    mp_grid_spec = _grid_spec_from_args(fc.crystal, grid=grid,
                                        grid_spacing=grid_spacing)
    print('Calculating Debye-Waller factor on {} q-point grid'
          .format(' x '.join(map(str, mp_grid_spec))))
    dw_phonons = fc.calculate_qpoint_phonon_modes(
        mp_grid(mp_grid_spec), **calc_modes_kwargs)
    return dw_phonons.calculate_debye_waller(temperature)


def _get_pdos_weighting(cl_arg_weighting: str) -> Optional[str]:
    """
    Convert CL --weighting to weighting for calculate_pdos
    e.g. --weighting coherent-dos to weighting=coherent
    """
    if cl_arg_weighting == 'dos':
        pdos_weighting = None
    else:
        idx = cl_arg_weighting.rfind('-')
        if idx == -1:
            msg = f'Unexpected weighting "{cl_arg_weighting}"'
            raise ValueError(msg)
        pdos_weighting = cl_arg_weighting[:idx]
    return pdos_weighting


def _arrange_pdos_groups(pdos: Spectrum1DCollection,
                         cl_arg_pdos: Sequence[str],
                         ) -> Spectrum1D | Spectrum1DCollection:
    """
    Convert PDOS returned by calculate_pdos to PDOS/DOS
    wanted as CL output according to --pdos
    """
    dos = pdos.sum()
    if cl_arg_pdos is not None:
        # Only label total DOS if there are other lines on the plot
        dos.metadata['label'] = 'Total'
        pdos = pdos.group_by('species')
        for line_metadata in pdos.metadata['line_data']:
            line_metadata['label'] = line_metadata['species']
        if len(cl_arg_pdos) > 0:
            pdos = pdos.select(species=cl_arg_pdos)
            dos = pdos
        else:
            dos = Spectrum1DCollection.from_spectra([dos, *pdos])
    return dos


def _plot_label_kwargs(args: Namespace, default_xlabel: str = '',
                       default_ylabel: str = '') -> dict[str, str]:
    """Collect title/label arguments that can be passed to plot_nd
    """
    plot_kwargs = {'title': args.title,
                   'xlabel': default_xlabel,
                   'ylabel': default_ylabel}
    if args.ylabel is not None:
        plot_kwargs['ylabel'] = args.ylabel
    if args.xlabel is not None:
        plot_kwargs['xlabel'] = args.xlabel
    return plot_kwargs


def _calc_modes_kwargs(args: Namespace) -> dict[str, Any]:
    """
    Collect arguments that can be passed to
    ForceConstants.calculate_qpoint_phonon_modes()
    """
    return {'asr': args.asr, 'dipole_parameter': args.dipole_parameter,
            'use_c': args.use_c, 'n_threads': args.n_threads}

def _brille_calc_modes_kwargs(args: Namespace) -> dict[str, Any]:
    """
    Collect arguments that can be passed to
    BrilleInterpolator.calculate_qpoint_phonon_modes()
    """
    if args.n_threads is None:
        # Nothing specified, allow defaults
        return {}

    return {'useparallel': args.n_threads > 1, 'threads': args.n_threads}


def _get_cli_parser(features: Collection[str] = {},
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
            msg = 'No band-data-only tools have been defined.'
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
            msg = '"adaptive-broadening" cannot be applied without "ebins"'
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


MplStyle = str | dict[str, str]


def _compose_style(
        *, user_args: Namespace, base: Optional[list[MplStyle]],
        ) -> list[MplStyle]:
    """Combine user-specified style options with default stylesheets

    Args:
        user_args: from _get_cli_parser().parse_args()
        base: Euphonic default styles for this plot

    N.B. matplotlib applies styles from left to right, so the right-most
    elements of the list take the highest priority. This function builds a
    list in the order:

    [base style(s), user style(s), CLI arguments]
    """

    style = base if not user_args.no_base_style and base is not None else []

    if user_args.style:
        style += user_args.style

    # Explicit args take priority over any other
    explicit_args = {}
    for user_arg, mpl_property in {'cmap': 'image.cmap',
                                   'fontsize': 'font.size',
                                   'font': 'font.sans-serif',
                                   'linewidth': 'lines.linewidth',
                                   'figsize': 'figure.figsize'}.items():
        if getattr(user_args, user_arg, None):
            explicit_args.update({mpl_property: getattr(user_args, user_arg)})

    if 'font.sans-serif' in explicit_args:
        explicit_args.update({'font.family': 'sans-serif'})

    if 'figure.figsize' in explicit_args:
        dimensioned_figsize = [dim * ureg(user_args.figsize_unit)
                               for dim in explicit_args['figure.figsize']]
        explicit_args['figure.figsize'] = [dim.to('inches').magnitude
                                           for dim in dimensioned_figsize]

    style.append(explicit_args)
    return style


def _get_title(filename: str, title: str | None = None) -> str:
    """Get a plot title: either user-provided string, or from filename"""
    return title if title is not None else pathlib.Path(filename).stem
