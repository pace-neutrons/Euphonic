import os
from euphonic import QpointPhononModes
from argparse import ArgumentParser
from typing import List, Tuple


def load_data_from_file(filename: str) -> Tuple[QpointPhononModes, str, str]:
    """
    Load castep data from filename and return it

    Parameters
    ----------
    filename : str
        The file with a path

    Returns
    -------
    QpointPhononmodes
    """
    data = QpointPhononModes.from_castep(filename)
    return data


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
