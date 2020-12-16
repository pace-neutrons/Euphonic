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


def get_current_plot_line_data() -> Dict[str, Union[str,
                                                    List[str],
                                                    List[List[List[float]]]]]:
    fig = matplotlib.pyplot.gcf()
    data = get_fig_label_data(fig)
    data['xy_data'] = [line.get_xydata().T.tolist()
                       for line in fig.axes[0].lines]
    return data


def get_fig_label_data(fig) -> Dict[str, Union[str, List[str]]]:
    label_data = {}
    label_data['title'] = fig._suptitle.get_text()
    # Get axis label no matter which axis it's on
    label_data['x_label'] = ''.join([ax.get_xlabel() for ax in fig.axes])
    label_data['y_label'] = ''.join([ax.get_ylabel() for ax in fig.axes])
    label_data['x_ticklabels'] = [lab.get_text()
                                  for ax in fig.axes
                                  for lab in ax.get_xticklabels()]
    return label_data


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
    im_data = im.get_array()
    data_slice_1 = im_data[:, (im_data.shape[1] // 2)].flatten()
    data_slice_2 = im_data[im_data.shape[0] // 2, :].flatten()

    data = get_fig_label_data(fig)
    data['cmap'] = im.cmap.name
    data['extent'] = [float(x) for x in im.get_extent()]
    data['size'] = [int(x) for x in im.get_size()]
    data['data_1'] = list(map(float, data_slice_1))
    data['data_2'] = list(map(float, data_slice_2))
    return data