import os
# Required for mocking
try:
    import matplotlib.pyplot
except ModuleNotFoundError:
    pass
from copy import copy
from typing import Dict, List, Union, Tuple
from ..utils import get_data_path


def args_to_key(cl_args: List[str]) -> str:
    """
    From CL tool arguments, return the key that should be used to store
    its testing output
    """
    cl_args = copy(cl_args)
    if os.path.isfile(cl_args[0]):
        cl_args[0] = ' '.join([os.path.split(os.path.dirname(cl_args[0]))[1],
                               os.path.split(cl_args[0])[1]])
    key = ' '.join(cl_args)
    return key


def get_script_test_data_path(*subpaths: Tuple[str]) -> str:
    """
    Returns
    -------
    str
        The data folder for scripts testing data
    """
    return get_data_path('script_data', *subpaths)


def get_current_plot_line_data() -> Dict[str, Union[str,
                                                    List[str],
                                                    List[List[List[float]]]]]:
    fig = matplotlib.pyplot.gcf()
    data = get_fig_label_data(fig)
    data['xy_data'] = [line.get_xydata().T.tolist()
                       for line in fig.axes[0].lines]
    return data


def get_fig_label_data(fig) -> Dict[str, Union[str, List[str]]]:
    label_data = {'x_ticklabels': [],
                  'title': fig._suptitle.get_text()}

    # Get axis labels from whichever axis has them
    for ax in fig.axes:
        if ax.get_frame_on():
            # Collect all tick labels from visible axes only
            ax_label_data = get_ax_label_data(ax)

            for key in 'x_label', 'y_label':
                if ax_label_data[key]:
                    label_data[key] = ax_label_data[key]

            label_data['x_ticklabels'] += ax_label_data['x_ticklabels']
        else:
            for ticks in ax.get_xticks(), ax.get_yticks():
                assert len(ticks) == 0

    return label_data


def get_ax_label_data(ax) -> Dict[str, Union[str, List[str]]]:
    label_data = {}
    label_data['x_label'] = ax.get_xlabel()
    label_data['y_label'] = ax.get_ylabel()
    label_data['x_ticklabels'] = [lab.get_text()
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
    data['data_1'] = list(map(float, data_slice_1))
    data['data_2'] = list(map(float, data_slice_2))

    return data
