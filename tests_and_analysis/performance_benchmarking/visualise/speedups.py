import json
import matplotlib.pyplot as plt


def plot_speedups_for_file(filename: str, figure_index: int) -> int:
    """
    Plot a figure for each test that has had speedups calculated for it in
    filename. There is a trace for each seedname used in the test.

    Parameters
    ----------
    filename : str
        The file to get the calculated speedups from
    figure_index : str
        The index of the first figure to plot

    Returns
    -------
    int
        The next free figure index to use
    """
    data = json.load(open(filename))
    if "speedups" in data:
        for test in data["speedups"]:
            plt.figure(figure_index)
            for seedname in data["speedups"][test]:
                plt.plot(
                    list(data["speedups"][test][seedname].keys()),
                    list(data["speedups"][test][seedname].values()),
                    label=seedname
                )
            plt.xlabel("Number of threads")
            plt.ylabel("Speedup (Ts/Tp)")
            plt.title("Speedups for {}\n {}".format(filename, test))
            # Create the legend to the right of the figure and shrink the
            # figure to account for that
            plt.legend(title="Seedname/material",
                       loc='center left',
                       bbox_to_anchor=(1, 0.5))
            plt.gcf().tight_layout()
            figure_index += 1
    return figure_index
