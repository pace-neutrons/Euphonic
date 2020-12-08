from argparse import ArgumentParser
import json
import os
import pathlib
from typing import Any, Dict, List, Tuple, Union, Optional

import numpy as np
from pint import UndefinedUnitError
import seekpath

from euphonic import ForceConstants, QpointPhononModes, Quantity, ureg
Unit = ureg.Unit


def force_constants_from_file(filename: Union[str, os.PathLike]
                              ) -> ForceConstants:
    """
    Load force constants data from file

    Parameters
    ----------
    filename
        Data file

    Returns
    -------
    ForceConstants
    """
    path = pathlib.Path(filename)
    if path.name == 'force_constants.hdf5':
        if (path.parent / 'phonopy.yaml').is_file():
            return ForceConstants.from_phonopy(path=path.parent,
                                               summary_name='phonopy.yaml',
                                               fc_name=path.name)
        raise ValueError("Phonopy force_constants.hdf5 file "
                         "must be accompanied by phonopy.yaml")
    elif path.suffix == '.yaml':
        # Assume this is a (renamed?) phonopy.yaml file
        if (path.parent / 'force_constants.hdf5').is_file():
            fc_name = 'force_constants.hdf5'
        else:
            fc_name = 'FORCE_CONSTANTS'

        return ForceConstants.from_phonopy(path=path.parent,
                                           fc_name=fc_name,
                                           summary_name=path.name)
    elif path.suffix in ('.castep_bin', '.check'):
        return ForceConstants.from_castep(filename)
    elif path.suffix == '.json':
        return ForceConstants.from_json_file(filename)
    else:
        raise ValueError("File not recognised. Filename should be "
                         "*.yaml or force_constants.hdf5 (phonopy), "
                         "*.castep_bin or *.check "
                         "(castep) or *.json (JSON from Euphonic).")


def modes_from_file(filename: Union[str, os.PathLike]
                    ) -> QpointPhononModes:
    """
    Load phonon mode data from file

    Parameters
    ----------
    filename
        Data file

    Returns
    -------
    QpointPhononmodes
    """
    path = pathlib.Path(filename)
    if path.suffix == '.phonon':
        return QpointPhononModes.from_castep(path)
    elif path.suffix == '.json':
        return QpointPhononModes.from_json_file(path)
    elif path.suffix in ('.yaml', '.hdf5'):
        return QpointPhononModes.from_phonopy(path=path.parent,
                                              phonon_name=path.name)
    else:
        raise ValueError("File not recognised. Should have extension "
                         ".yaml or .hdf5 (phonopy), "
                         ".phonon (castep) or .json (JSON from Euphonic).")


def _load_json(filename: Union[str, os.PathLike]
               ) -> Union[QpointPhononModes, ForceConstants]:
    with open(filename, 'r') as f:
        data = json.load(f)

    if 'force_constants' in data:
        return force_constants_from_file(filename)
    elif 'eigenvectors' in data:
        return modes_from_file(filename)
    else:
        raise ValueError("Could not identify Euphonic data in JSON file.")


def load_data_from_file(filename: Union[str, os.PathLike]
                        ) -> Union[QpointPhononModes, ForceConstants]:
    """
    Load phonon mode or force constants data from file

    Parameters
    ----------
    filename : str
        The file with a path

    Returns
    -------
    QpointPhononmodes or ForceConstants
    """
    qpoint_phonon_modes_suffixes = ('.phonon')
    force_constants_suffixes = ('.castep_bin', '.check')

    path = pathlib.Path(filename)
    if path.suffix in qpoint_phonon_modes_suffixes:
        return modes_from_file(path)
    elif path.suffix in force_constants_suffixes:
        return force_constants_from_file(path)
    elif path.suffix == '.json':
        return _load_json(path)
    elif path.suffix in ('.hdf5', '.yaml'):
        if path.stem in ('band', 'qpoints', 'mesh'):
            return modes_from_file(path)
        elif path.suffix == '.yaml':
            return force_constants_from_file(path)
        else:
            raise ValueError(
                "Supported Phonopy files are force_constants.hdf5, "
                "{band,qpoints,mesh}.{hdf5,yaml} or phonopy.yaml. Other .yaml "
                "files are interpreted as phonopy.yaml")
    else:
        raise ValueError(
            "File format was not recognised. Force constant data for import "
            f"should have extension from {force_constants_suffixes},"
            " phonon mode data for import should have extension "
            f"'{qpoint_phonon_modes_suffixes[0]}',"
            " data from phonpy should have extension .yaml or .hdf5,"
            " data from Euphonic should have extension '.json'.")


def get_args(parser: ArgumentParser, params: List[str] = None):
    """
    Get the arguments from the parser. params should only be none when
    running from command line.

    Parameters
    ----------
    parser : ArgumentParser
        The parser to get the arguments from
    params : List[str], optional
        The parameters to get arguments from, if None,
         then parse_args gets them from command line args

    Returns
    -------
    Arguments object for use e.g. args.unit
    """
    if params is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(params)
    return args


def matplotlib_save_or_show(save_filename: str = None):
    """
    Save or show the current matplotlib plot.
    Show if save_filename is not None which by default it is.

    Parameters
    ----------
    save_filename : str, optional
        The file to save the plot in
    """
    import matplotlib.pyplot as plt
    if save_filename is not None:
        plt.savefig(save_filename)
    else:
        plt.show()


def _get_q_distance(length_unit_string: str, q_distance: float) -> Quantity:
    """
    Parse user arguments to obtain reciprocal-length spacing Quantity
    """
    try:
        length_units = ureg(length_unit_string)
    except UndefinedUnitError:
        raise ValueError("Length unit not known. Euphonic uses Pint for units."
                         " Try 'angstrom' or 'bohr'. Metric prefixes "
                         "are also allowed, e.g 'nm'.")
    recip_length_units = 1 / length_units
    return q_distance * recip_length_units


def _get_energy_bins_and_units(
        energy_unit_string: str, modes: QpointPhononModes,
        n_ebins: int, emin: Optional[float] = None,
        emax: Optional[float] = None,
        headroom: float = 1.05) -> Tuple[Quantity, Unit]:

    try:
        energy_unit = ureg(energy_unit_string)
    except UndefinedUnitError:
        raise ValueError("Energy unit not known. Euphonic uses Pint for units."
                         " Try 'eV' or 'hartree'. Metric prefixes are also "
                         "allowed, e.g 'meV' or 'fJ'.")

    if emin is None:
        emin = min(np.min(modes.frequencies.magnitude), 0.)
    if emax is None:
        emax = np.max(modes.frequencies.magnitude) * headroom
    if emin >= emax:
        raise ValueError("Maximum energy should be greater than minimum. "
                         "Check --e-min and --e-max arguments.")
    ebins = np.linspace(emin, emax, n_ebins) * energy_unit

    return ebins, energy_unit


def _get_tick_labels(bandpath: dict) -> List[Tuple[int, str]]:
    """Convert x-axis labels from seekpath format to euphonic format

    i.e.::

        ['L', '', '', 'X', '', 'GAMMA']   -->

        [(0, 'L'), (3, 'X'), (5, '$\\Gamma$')]
    """

    label_indices = np.where(bandpath["explicit_kpoints_labels"])[0]
    labels = [bandpath["explicit_kpoints_labels"][i] for i in label_indices]

    for i, label in enumerate(labels):
        if label == 'GAMMA':
            labels[i] = r'$\Gamma$'

    return list(zip(label_indices, labels))


def _get_break_points(bandpath: dict) -> List[int]:
    """Get information about band path labels and break points

    Parameters
    ----------
    bandpath
        Bandpath dictionary from Seekpath

    Returns
    -------
    list[int]

        Indices at which the spectrum should be split into subplots

    """
    # Find break points between continuous spectra: wherever there are two
    # adjacent labels
    labels = np.array(bandpath["explicit_kpoints_labels"])

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
    gamma_indices = np.where(np.array(bandpath['explicit_kpoints_labels'][1:-1]
                                      ) == 'GAMMA')[0] + 1

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


XTickLabels = List[Tuple[int, str]]
SplitArgs = Dict[str, Any]


def _bands_from_force_constants(data: ForceConstants,
                                q_distance: Quantity,
                                insert_gamma=True,
                                asr=None
                                ) -> Tuple[QpointPhononModes,
                                           XTickLabels, SplitArgs]:
    structure = data.crystal.to_spglib_cell()
    bandpath = seekpath.get_explicit_k_path(
        structure,
        reference_distance=q_distance.to('1 / angstrom').magnitude)

    if insert_gamma:
        _insert_gamma(bandpath)

    x_tick_labels = _get_tick_labels(bandpath)
    split_args = {'indices': _get_break_points(bandpath)}

    print(
        "Computing phonon modes: {n_modes} modes across {n_qpts} q-points"
        .format(n_modes=(data.crystal.n_atoms * 3),
                n_qpts=len(bandpath["explicit_kpoints_rel"])))
    qpts = bandpath["explicit_kpoints_rel"]
    modes = data.calculate_qpoint_phonon_modes(qpts, asr=asr,
                                               reduce_qpts=False)

    return modes, x_tick_labels, split_args

def _get_cli_parser(qe_plot=False, n_ebins=False) -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        'filename', type=str,
        help=('Phonon data file. This should contain force constants or band '
              'data. Force constants formats: .yaml, force_constants.hdf5 '
              '(Phonopy); .castep_bin , .check (Castep); .json (Euphonic). '
              'Band data formats: {band,qpoints,mesh}.{hdf5,yaml} (Phonopy); '
              '.phonon (Castep); .json (Euphonic)'))
    parser.add_argument(
        '-s', '--save-to', dest='save_to', default=None,
        help='Save resulting plot to a file with this name')
    parser.add_argument('--title', type=str, default=None, help='Plot title')
    parser.add_argument('--x-label', type=str, default=None,
                        dest='x_label', help='Plot x-axis label')
    parser.add_argument('--y-label', type=str, default=None,
                        dest='y_label', help='Plot y-axis label')
    if n_ebins:
        parser.add_argument('--ebins', type=int, default=200,
                            help='Number of energy bins')
    parser.add_argument('--e-min', type=float, default=None, dest='e_min',
                        help='Energy range minimum in ENERGY_UNIT')
    parser.add_argument('--e-max', type=float, default=None, dest='e_max',
                        help='Energy range maximum in ENERGY_UNIT')
    parser.add_argument('--energy-unit', '-u', dest='energy_unit',
                        type=str, default='meV', help='Energy units')
    if qe_plot:
        parser.add_argument(
            '--length-unit', type=str, default='angstrom', dest='length_unit',
            help=('Length units; these will be inverted to obtain '
                  'units of distance between q-points (e.g. "bohr"'
                  ' for bohr^-1).'))
        interp_group = parser.add_argument_group(
            'Interpolation arguments',
            ('Arguments specific to band structures that are generated '
             'from Force Constants data'))
        interp_group.add_argument(
            '--asr', type=str, nargs='?', default=None, const='reciprocal',
            choices=('reciprocal', 'realspace'),
            help=('Apply an acoustic-sum-rule (ASR) correction to the '
                  'data: "realspace" applies the correction to the force '
                  'constant matrix in real space. "reciprocal" applies '
                  'the correction to the dynamical matrix at each q-point.'))
        interp_group.add_argument(
            '--q-distance', type=float, dest='q_distance', default=0.025,
            help=('Target distance between q-point samples in 1/LENGTH_UNIT'))
        disp_group = parser.add_argument_group(
            'Dispersion arguments',
            'Arguments specific to plotting a pre-calculated band structure')
        disp_group.add_argument(
            '--btol', default=10.0, type=float,
            help=('The tolerance for plotting sections of reciprocal '
                  'space on different subplots, as a fraction of the '
                  'median distance between q-points'))

    return parser
