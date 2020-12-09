import os
# Required for mocking
import matplotlib.pyplot
from typing import Dict, List, Union
from ..utils import get_data_path


def args_to_key(cl_args: str) -> str:
    """
    From CL tool arguments, return the key that should be used to store
    its testing output
    """
    if os.path.isfile(cl_args[0]):
        cl_args[0] = ' '.join([os.path.split(os.path.dirname(cl_args[0]))[1],
                               os.path.split(cl_args[0])[1]])
    key = ' '.join(cl_args)
    return key


def get_script_test_data_path() -> str:
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
