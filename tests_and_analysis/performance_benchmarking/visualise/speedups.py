import json
import matplotlib.pyplot as plt


def plot_speedups_for_file(filename: str, figure_index: int) -> int:
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
            plt.legend(title="Seedname/material")
            figure_index += 1
    return figure_index