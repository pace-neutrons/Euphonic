from copy import copy
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Required for mocking
try:
    import matplotlib.pyplot  # noqa: ICN001
except ModuleNotFoundError:
    pass

from tests_and_analysis.test.utils import get_data_path


def args_to_key(cl_args: List[str]) -> str:
    """
    From CL tool arguments, return the key that should be used to store
    its testing output
    """
    cl_args = copy(cl_args)
    if os.path.isfile(cl_args[0]):
        cl_args[0] = ' '.join([os.path.split(os.path.dirname(cl_args[0]))[1],
                               os.path.split(cl_args[0])[1]])
    return ' '.join(cl_args)


def get_script_test_data_path(*subpaths: Tuple[str]) -> str:
    """
    Returns
    -------
    str
        The data folder for scripts testing data
    """
    return get_data_path('script_data', *subpaths)


def get_plot_line_data(fig: Optional['matplotlib.figure.Figure'] = None
                       ) -> Dict[str, Any]:
    if fig is None:
        fig = matplotlib.pyplot.gcf()
    data = get_fig_label_data(fig)
    data['xy_data'] = []
    for ax in fig.axes:
        if '3D' in type(ax).__name__:
            data['xy_data'].append([np.array(line.get_data_3d()).tolist()
                                    for line in ax.lines])
        else:
            data['xy_data'].append([line.get_xydata().T.tolist()
                                    for line in ax.lines])
    return data


def get_all_figs() -> List['matplotlib.figure.Figure']:
    all_figs = []
    fignums = matplotlib.pyplot.get_fignums()
    for fignum in fignums:
        all_figs.append(matplotlib.pyplot.figure(fignum))
    return all_figs


def get_all_plot_line_data(figs: List['matplotlib.figure.Figure']
                           ) -> List[Dict[str, Any]]:
    data = []
    for fig in figs:
        data.append(get_plot_line_data(fig))
    return data


def get_fig_label_data(fig) -> Dict[str, Union[str, List[str]]]:
    from mpl_toolkits.mplot3d import Axes3D

    label_data = {'x_ticklabels': [],
                  'x_label': [],
                  'y_label': [],
                  'title': fig._suptitle.get_text() \
                           if fig._suptitle is not None else None}

    # Get axis/tick labels from all axes, collect only non-empty values
    # to avoid breaking tests if the way we set up axes changes
    for ax in fig.axes:
        xlabel = ax.get_xlabel()
        if xlabel:
            label_data['x_label'].append(xlabel)
        ylabel = ax.get_ylabel()
        if ylabel:
            label_data['y_label'].append(ylabel)
        if '3D' in type(ax).__name__:
            zlabel = ax.get_zlabel()
            if 'z_label' not in  label_data:
                label_data['z_label'] = []
            if zlabel:
                label_data['z_label'].append(zlabel)
        # Collect tick labels from visible axes only,
        # we don't care about invisible axis tick labels
        if isinstance(ax, Axes3D) or ax.get_frame_on():
            xticklabels = [lab.get_text() for lab in ax.get_xticklabels()]
            label_data['x_ticklabels'].append(xticklabels)

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
    fig = matplotlib.pyplot.gcf()
    for ax in fig.axes:
        if len(ax.get_images()) == 1:
            break
    else:
        raise Exception("Could not find axes with a single image")

    data = get_fig_label_data(fig)
    data.update(get_ax_image_data(ax))

    return data


def get_ax_image_data(ax: 'matplotlib.axes.Axes'
                      ) -> Dict[str, Union[str, List[float], List[int]]]:
    im = ax.get_images()[0]
    # Convert negative zero to positive zero
    im_data = im.get_array()
    data_slice_1 = im_data[:, (im_data.shape[1] // 2)].flatten()
    data_slice_2 = im_data[im_data.shape[0] // 2, :].flatten()

    data = {}

    data['cmap'] = im.cmap.name
    data['extent'] = [float(x) for x in im.get_extent()]
    data['size'] = [int(x) for x in im.get_size()]
    data['data_1'] = list(map(float, data_slice_1.filled(np.nan)))
    data['data_2'] = list(map(float, data_slice_2.filled(np.nan)))

    return data
