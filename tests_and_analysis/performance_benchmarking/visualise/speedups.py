import json
import matplotlib.pyplot as plt


def plot_speedups_for_file(filename: str):
    """
    Plot a figure for each test that has had speedups calculated for it in
    filename. There is a trace for each seedname used in the test.

    Parameters
    ----------
    filename : str
        The file to get the calculated speedups from
    """
    data = json.load(open(filename))
    if "speedups" in data:
        for test in data["speedups"]:
            fig, subplots = plt.subplots()
            for seedname in data["speedups"][test]:
                subplots.plot(
                    list(data["speedups"][test][seedname].keys()),
                    list(data["speedups"][test][seedname].values()),
                    label=seedname
                )
            subplots.set_xlabel("Number of threads")
            subplots.set_ylabel("Speedup (Ts/Tp)")
            subplots.set_title("Speedups for {}\n {}".format(filename, test))
            # Create the legend to the right of the figure and shrink the
            # figure to account for that
            subplots.legend(
                title="Seedname",
                loc='center left',
                bbox_to_anchor=(1, 0.5),
                fontsize="small"
            )
            fig.tight_layout()
