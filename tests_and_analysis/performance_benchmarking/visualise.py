from typing import List, Dict

import matplotlib.pyplot as plt
import matplotlib.dates as dates
import os
import json
from datetime import datetime
import pandas as pd


class Figure:

    def __init__(self, machine_info: str):
        self.machine_info = machine_info
        self.tests = {}

    def add_test(self, test):
        if test not in self.tests:
            self.tests[test] = {}


class Figures:

    def __init__(self):
        self.figures = {}

    def plot(self, figure_index: int):
        for figure in self.figures:
            self.figures[figure].plot(figure_index)


class SpeedupMachineFigure(Figure):

    def add_speedup(self, test: str, n_threads: int, date: datetime.date,
                    speedup: float):
        self.add_n_threads(test, n_threads)
        self.tests[test][n_threads][date] = speedup

    def add_n_threads(self, test: str, n_threads: int):
        self.add_test(test)
        if n_threads not in self.tests[test]:
            self.tests[test][n_threads] = {}

    def plot(self, figure_index: int):
        plt.figure(figure_index)
        for test in self.tests:
            for n_threads in self.tests[test]:
                plt.plot(
                    list(self.tests[test][n_threads].keys()),
                    list(self.tests[test][n_threads].values()),
                    label=str(n_threads)
                )
        plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(dates.DayLocator())
        plt.xlabel("Date")
        plt.ylabel("Speedup value (Ts/Tn)")
        plt.title("Speedups over time\n {}".format(self.machine_info))
        plt.legend(title="Number of threads")
        plt.gcf().autofmt_xdate()


class SpeedupMachineFigures(Figures):

    def add_subplot(self, machine_info: str):
        if machine_info not in self.figures:
            self.figures[machine_info] = SpeedupMachineFigure(machine_info)

    def get_subplot(self, machine_info: str) -> SpeedupMachineFigure:
        return self.figures[machine_info]


def json_files(directory: str):
    """
    A generator for all JSON files stored in directory and it's subdirectories.

    Parameters
    ----------
    directory: str
        The top level directory to start the search from.

    Yields
    -------
    str
        A JSON file stored below in the directory structure
    """
    for subdir, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".json"):
                yield os.path.join(subdir, file)


def plot_speedups(directory: str, figure_index: int):
    # Format data into threads for plot
    subplots = SpeedupMachineFigures()
    for file in json_files(directory):
        data = json.load(open(file))
        if "speedups" in data:
            subplots.add_subplot(data["machine_info"]["cpu"]["brand"])
            for func in data["speedups"]:
                for n_threads in data["speedups"][func]:
                    subplots.get_subplot(
                        data["machine_info"]["cpu"]["brand"]
                    ).add_speedup(
                        func,
                        n_threads,
                        datetime.strptime(
                            data["datetime"].split("T")[0],
                            '%Y-%m-%d'
                        ).date(),
                        data["speedups"][func][n_threads]
                    )
    subplots.plot(figure_index)


class MedianMachineFigure(Figure):

    def add_time_taken(self, test: str, date: datetime.date, time_taken: float):
        self.add_test(test)
        self.tests[test][date] = time_taken

    def plot(self, figure_index: int):
        dataframes: List[Dict] = self.build_dataframes()
        self.plot_dataframes(dataframes, figure_index)

    def build_dataframes(self) -> List[Dict]:
        # Initialise
        last_test_name = None
        x_axis = []
        y_axes = {}
        title = None
        dataframes = []
        for test in self.tests:
            test_name = test.split("[")[0]
            test_params = test.split("[")[1][:-1]
            # Detect if we have found an all new test to plot a figure for
            if last_test_name != test_name:
                if last_test_name is not None:
                    dataframe = {'x': x_axis, 'title': title}
                    dataframe.update(y_axes)
                    dataframes.append(dataframe)
                x_axis = []
                y_axes = {}
                title = "Performance over time\n {}\n {}".format(
                    self.machine_info, test_name)
                last_test_name = test_name
            y_axis = []
            new_x_axis = []
            for key, value in self.tests[test].items():
                index = 0
                while index < len(x_axis) and key > x_axis[index]:
                    index += 1
                new_x_axis = new_x_axis[:index] + [key] + new_x_axis[index:]
                y_axis = y_axis[:index] + [value] + y_axis[index:]
            x_axis = new_x_axis
            y_axes[test_params] = y_axis
        dataframe = {'x': x_axis, 'title': title}
        dataframe.update(y_axes)
        dataframes.append(dataframe)
        return dataframes

    def plot_dataframes(self, dataframes: List[Dict], figure_index):
        for index, dataframe in enumerate(dataframes):
            print(dataframe)
            plt.figure(figure_index + index)
            panda_dataframe = pd.DataFrame(dataframe)
            for key in dataframe.keys():
                if key != "x" and key != "title":
                    plt.plot(
                        'x', key,
                        data=panda_dataframe
                    )
            plt.title(dataframe['title'])
            plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(dates.DayLocator())
            plt.xlabel("Date")
            plt.ylabel("Time taken")
            plt.legend(title="Test Params")
            plt.gcf().autofmt_xdate()


class MedianMachineFigures(Figures):

    def add_subplot(self, machine_info: str):
        if machine_info not in self.figures:
            self.figures[machine_info] = MedianMachineFigure(machine_info)

    def get_subplot(self, machine_info: str) -> MedianMachineFigure:
        return self.figures[machine_info]


def plot_median_values(directory: str, figure_index: int):
    plots = MedianMachineFigures()
    for file in json_files(directory):
        data = json.load(open(file))
        if "benchmarks" in data:
            plots.add_subplot(data["machine_info"]["cpu"]["brand"])
            for benchmark in data["benchmarks"]:
                test = benchmark["name"]
                plots.get_subplot(
                    data["machine_info"]["cpu"]["brand"]
                ).add_time_taken(
                    test,
                    datetime.strptime(
                        data["datetime"].split("T")[0],
                        '%Y-%m-%d'
                    ).date(),
                    benchmark["stats"]["median"]
                )
    plots.plot(figure_index)


if __name__ == "__main__":
    plot_speedups("reports", 0)
    plot_median_values("reports", 1)
    plt.show()
