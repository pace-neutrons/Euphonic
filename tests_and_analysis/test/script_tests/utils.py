import os
# Required for mocking
import matplotlib.pyplot
from typing import Dict, List, Union
from ..utils import get_data_path


def get_phonon_file() -> str:
    """
    Returns
    -------
    str
        The full path to the NaH.phonon file
    """
    return os.path.join(get_data_path(),
                       'qpoint_phonon_modes', 'NaH', 'NaH.phonon')


def get_force_constants_file() -> str:
    """
    Returns
    -------
    str
        The full path to the NaH.phonon file
    """
    return os.path.join(get_data_path(),
                       'force_constants', 'graphite',
                        'graphite_force_constants.json')


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


def get_intensity_map_data_file() -> str:
    """
    Returns
    -------
    str
        A full path prefix of all intensity map regression test files
    """
    return os.path.join(get_script_data_folder(), "intensity-map.json")


def get_dos_data_file() -> str:
    """
    Returns
    -------
    str
        A full path prefix of all dos regression test files
    """
    return os.path.join(get_script_data_folder(), "dos.json")


def get_sphere_sampling_data_file() -> str:
    """
    Returns
    -------
    str
        A full path prefix of all sampling script regression test files
    """
    return os.path.join(get_script_data_folder(), "sphere_sampling.json")


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


def get_current_plot_offsets() -> List[List[float]]:
    """
    Get the positions (offsets) from an active matplotlib scatter plot

    This should work for both 2D and 3D scatter plots; in the 3D case these
    offsets are based on 2D projections.

    Returns
    -------
    List[List[float]]
        Scatter plot offsets
    """
    return matplotlib.pyplot.gca().collections[0].get_offsets().data.tolist()


def get_current_plot_image_data() -> Dict[str,
                                          Union[str, List[float], List[int]]]:
    from hashlib import md5
    import numpy as np

    fig = matplotlib.pyplot.gcf()
    for ax in fig.axes:
        if len(ax.get_images()) == 1:
            break
    else:
        raise Exception("Could not find axes with a single image")

    im = ax.get_images()[0]

    # Convert negative zero to positive zero
    data = im.get_array()
    data_slice_1 = data[:, (data.shape[1] // 2)].flatten()
    data_slice_2 = data[data.shape[0] // 2, :].flatten()

    return {'cmap': im.cmap.name,
            'extent': [float(x) for x in im.get_extent()],
            'size': [int(x) for x in im.get_size()],
            'data_1': list(map(float, data_slice_1)),
            'data_2': list(map(float, data_slice_2))}


def get_dos_params() -> List[List[str]]:
    """
    Get the parameters to run and test euphonic-dos with

    Returns
    -------
    List[str]
        The parameters to run the script with
    """
    return [[], ["-w 2.3"], ["-b 3.3"], ["-w 2.3", "-b 3.3"],
            ["-unit=meV"], ["-lorentz"]]


def get_dispersion_params() -> List[List[str]]:
    """
    Get the parameters to run and test euphonic-dispersion with.

    Returns
    -------
    List[str]
        The parameters to run the script with
    """
    return [[], ["--energy-unit=meV"], ["--btol=5.0"], ["--reorder"],
            ["--asr"], ["--asr=realspace"]]


def get_intensity_map_params() -> List[List[str]]:
    """
    Get the parameters to run and test euphonic-intensity-map with.

    Returns
    -------
    List[str]
        The parameters to run the script with
    """
    return [[], ["--energy-unit=meV"], ["--weights=coherent", "--cmap=bone"],
            ["-w", "dos", "--y-label='DOS'", "--title='DOS TITLE'"],
            ["--e-min=5", "--energy-unit=cm^-1", "--x-label='wavenumber'"],
            ["--e-min=-10", "--e-max=200", "--energy-unit=cm^-1"],
            ["--energy-broadening=2e-3", "--energy-unit=eV"],
            ["--q-distance=0.05", "--length-unit=bohr", "--q-broadening=0.1"],
            ["--asr"],
            ["--asr=realspace"],
            ]


def get_sphere_sampling_params() -> List[List[str]]:
    """
    Get the parameters to run and test euphonic-show-sampling

    Returns
    -------
    List[str]
        The parameters to run the script with
    """
    return [['27', 'golden-square'],
            ['8', 'regular-square'],
            ['9', 'regular-square'],
            ['10', 'golden-sphere'],
            ['10', 'golden-sphere', '--jitter'],
            ['15', 'spherical-polar-grid'],
            ['18', 'spherical-polar-grid', '--jitter'],
            ['17', 'sphere-from-square-grid', '--jitter'],
            ['18', 'sphere-from-square-grid'],
            ['15', 'spherical-polar-improved'],
            ['15', 'spherical-polar-improved', '--jitter'],
            ['10', 'random-sphere'],
            ['10', 'random-sphere', '--jitter']
            ]
