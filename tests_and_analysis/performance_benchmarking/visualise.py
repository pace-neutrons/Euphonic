import argparse
import matplotlib.pyplot as plt
from visualise.performance_over_time import plot_median_values
from visualise.speedups_over_time import plot_speedups_over_time


def get_parser() -> argparse.ArgumentParser:
    """
    Get the directory specified as an argument on the command line.

    Returns
    -------
    str
        The path of the directory
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", action="store", dest="dirname",
                        default="reports",
                        help="The directory containing the historical"
                             " benchmark data json files")
    parser.add_argument("-st", "--speedup-over-time", action="store_true",
                        dest="speedup_over_time",
                        help="Plot and show how the speedups data has changed"
                             " over time")
    parser.add_argument("-p", "--performance", action="store_true",
                        dest="performance",
                        help="Plot and show performance data")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args_parsed = parser.parse_args()
    dirname = args_parsed.dirname
    figure_index: int = 0
    if args_parsed.speedup_over_time:
        figure_index = plot_speedups_over_time(dirname, figure_index)
    if args_parsed.performance:
        plot_median_values(dirname, figure_index)
    plt.show()
