import os
from ..utils import get_data_path
import numpy as np
# Required for mocking
import matplotlib.pyplot


def get_phonon_file() -> str:
    """
    Returns
    -------
    str: the full path to the NaH.phonon file.
    """
    return os.path.join(get_data_path(), "NaH.phonon")


def get_dispersion_data_folder() -> str:
    """
    Returns
    -------
    str: the directory containing all the regression test data for scripts/dispersion.py
    """
    folder = os.path.join(get_data_path(), "dispersion_script")
    return folder


def get_dos_data_folder() -> str:
    """
    Returns
    -------
    str: the directory containing all the regression test data for scripts/dos.py
    """
    folder = os.path.join(get_data_path(), "dos_script")
    return folder


def _get_dos_or_dispersion_data_filepath_prefix(dos_or_dispersion_data_folder: str) -> str:
    phonon_filename = os.path.split(get_phonon_file())[-1]
    filename_prefix = phonon_filename.replace(".", "_") + "_line"
    if not os.path.exists(dos_or_dispersion_data_folder):
        os.mkdir(dos_or_dispersion_data_folder)
    filepath_prefix = os.path.join(dos_or_dispersion_data_folder, filename_prefix)
    return filepath_prefix


def get_dispersion_data_filepath_prefix() -> str:
    """
    Returns
    -------
    str: a full path prefix of all dispersion regression test files.
    """
    return _get_dos_or_dispersion_data_filepath_prefix(get_dispersion_data_folder())


def get_dos_data_filepath_prefix():
    """
    Returns
    -------
    str: a full path prefix of all dos regression test files.
    """
    return _get_dos_or_dispersion_data_filepath_prefix(get_dos_data_folder())


def _iter_regression_test_files(filepath_prefix: str, data_folder: str):
    for file in os.listdir(data_folder):
        file_full_path = os.path.join(data_folder, file)
        print(file_full_path)
        if file_full_path.startswith(filepath_prefix):
            filenum = file_full_path[len(filepath_prefix):][0]
            print(file_full_path[len(filepath_prefix):])
            yield int(filenum), file_full_path


def iter_dispersion_data_files():
    """
    A generator function to get all the files
     containing regression testing data for scripts/dispersion.py
    Yields the number of the file and the full file path as a tuple.

    Examples
    --------
         >>> for filenum, file in iter_dispersion_data_files():
         >>>   print(filenum)
         >>>
         >>> 0
         >>> 1
    """
    filepath_prefix = get_dispersion_data_filepath_prefix()
    data_folder = get_dispersion_data_folder()
    yield from _iter_regression_test_files(filepath_prefix, data_folder)


def iter_dos_data_files():
    """
    A generator function to get all the files
     containing regression testing data for scripts/dos.py
    Yields the number of the file and the full file path as a tuple.

    Examples
    --------
         >>> for filenum, file in iter_dos_data_files():
         >>>   print(filenum)
         >>>
         >>> 0
         >>> 1
    """
    filepath_prefix = get_dos_data_filepath_prefix()
    data_folder = get_dos_data_folder()
    yield from _iter_regression_test_files(filepath_prefix, data_folder)


def write_plot_to_file_for_regression_testing(filepath_prefix: str):
    """
    Get the current matplotlib plot with gcf() and write the xydata of the lines of the plot
     to separate csv files using the filepath_prefix parameter as a prefix and the
      index of the line and .csv to complete the filename.

    Parameters
    ----------
    filepath_prefix : str
        A prefix to use to write the files e.g. C:/data
    """
    for index, line in enumerate(matplotlib.pyplot.gcf().axes[0].lines):
        np.savetxt(filepath_prefix + str(index) + ".csv", line.get_xydata().T, delimiter=",")
