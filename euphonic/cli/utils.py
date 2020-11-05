from argparse import ArgumentParser
import json
import os
import pathlib
from typing import List, Tuple, Union

import numpy as np
from pint import UndefinedUnitError
from pint.unit import Unit

from euphonic import ForceConstants, QpointPhononModes, Quantity, ureg


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
    if path.suffix == '.yaml':
        return ForceConstants.from_phonopy(
            path=path.parent, summary_name=path.name)
    elif path.suffix in ('.castep_bin', '.check'):
        return ForceConstants.from_castep(filename)
    elif path.suffix == '.json':
        return ForceConstants.from_json_file(filename)
    else:
        raise ValueError("File not recognised. Should have extension "
                         ".yaml (phonopy), .castep_bin or .check "
                         "(castep) or .json (JSON from Euphonic).")


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
        return QpointPhononModes.from_json(path)
    else:
        raise ValueError("File not recognised. Should have extension "
                         ".phonon (castep) or .json (JSON from Euphonic).")


def _load_json(filename: Union[str, os.PathLike]
               ) -> Union[QpointPhononModes, ForceConstants]:
    with open(filename, 'r') as f:
        data = json.load(f)

    if 'force_constants' in data:
        return force_constants_from_file(filename)
    elif 'eigenvectors' in data:
        return force_constants_from_file(filename)
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
    force_constants_suffixes = ('.yaml', '.hdf5', '.castep_bin', '.check')

    path = pathlib.Path(filename)
    if path.suffix in qpoint_phonon_modes_suffixes:
        return modes_from_file(path)
    elif path.suffix in force_constants_suffixes:
        return force_constants_from_file(path)
    elif path.suffix == '.json':
        return _load_json(path)
    else:
        raise ValueError(
            "File format was not recognised. Force constant data for import "
            f"should have extension from {force_constants_suffixes},"
            " phonon mode data for import should have extension "
            f"'{qpoint_phonon_modes_suffixes[0]}', data from Euphonic should"
            " have extension '.json'.")


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


def _get_energy_unit(energy_unit_string: str) -> Unit:
    try:
        energy_unit = ureg(energy_unit_string)
        return energy_unit
    except UndefinedUnitError:
        raise ValueError("Energy unit not known. Euphonic uses Pint for units."
                         " Try 'eV' or 'hartree'. Metric prefixes are also "
                         "allowed, e.g 'meV' or 'fJ'.")


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
    special_point_bools = np.fromiter(
        map(bool, bandpath["explicit_kpoints_labels"]), dtype=bool)

    # [T F F T T F T] -> [F F T T F T] AND [T F F T T F] = [F F F T F F] -> 3,
    break_points = np.where(np.logical_and(special_point_bools[:-1],
                                           special_point_bools[1:]))[0]
    return (break_points + 1).tolist()
