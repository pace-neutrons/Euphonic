import os
# Required for mocking
import matplotlib.pyplot
from typing import List
from ..utils import get_data_path


def get_phonon_file() -> str:
    """
    Returns
    -------
    str
        The full path to the NaH.phonon file
    """
    return os.path.join(get_data_path(), "NaH.phonon")


def get_script_data_folder() -> str:
    """
    Returns
    -------
    str
        The data folder for scripts testing data
    """
    folder = os.path.join(get_data_path(), "script_data")
    if not os.path.exists(folder):
        os.mkdir(folder)
    return folder


def get_dispersion_data_file() -> str:
    """
    Returns
    -------
    str
        A full path prefix of all dispersion regression test files
    """
    return os.path.join(get_script_data_folder(), "dispersion.json")


def get_dos_data_file() -> str:
    """
    Returns
    -------
    str
        A full path prefix of all dos regression test files
    """
    return os.path.join(get_script_data_folder(), "dos.json")


def get_current_plot_lines_xydata() -> List[List[List[float]]]:
    """
    Get the current matplotlib plot with gcf() and return the xydata of
    the lines of the plot

    Returns
    -------
    List[List[List[float]]]
        The list of lines xy data from the current matplotlib plot
    """
    return [line.get_xydata().T.tolist()
            for line in matplotlib.pyplot.gcf().axes[0].lines]


def get_dos_params() -> List[List[str]]:
    """
    Get the parameters to run and test scripts/dos.py with

    Returns
    -------
    List[str]
        The parameters to run the script with
    """
    return [[], ["-w 2.3"], ["-b 3.3"], ["-w 2.3", "-b 3.3"], ["-unit=meV"], ["-lorentz"]]


def get_dispersion_params() -> List[List[str]]:
    """
    Get the parameters to run and test scripts/dispersion.py with.

    Returns
    -------
    List[str]
        The parameters to run the script with
    """
    return [[], ["-unit=meV"], ["-btol=5.0"], ["-reorder"]]
