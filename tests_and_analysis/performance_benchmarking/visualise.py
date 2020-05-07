import matplotlib.pyplot as plt
import matplotlib.dates as dates
import os
import json
from datetime import datetime
import pandas as pd


class Subplot:

    def __init__(self, machine_info: str):
        self.machine_info = machine_info
        self.test = {}

    def add_test(self, func):
        if func not in self.test:
            self.test[func] = {}


class Subplots:

    def __init__(self):
        self.subplots = {}

    def plot(self, figure_index: int):
        for subplot in self.subplots:
            self.subplots[subplot].plot(figure_index)


class SpeedupMachineSubplot(Subplot):

    def add_speedup(self, func: str, n_threads: int, date: datetime.date,
                    speedup: float):
        self.add_n_threads(func, n_threads)
        self.test[func][n_threads][date] = speedup

    def add_n_threads(self, func: str, n_threads: int):
        self.add_test(func)
        if n_threads not in self.test[func]:
            self.test[func][n_threads] = {}

    def plot(self, figure_index: int):
        plt.figure(figure_index)
        for func in self.test:
            for n_threads in self.test[func]:
                plt.plot(
                    list(self.test[func][n_threads].keys()),
                    list(self.test[func][n_threads].values()),
                    label=str(n_threads)
                )
        plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(dates.DayLocator())
        plt.xlabel("Date")
        plt.ylabel("Speedup value (Ts/Tn)")
        plt.title("Speedups over time\n {}".format(self.machine_info))
        plt.legend(title="Number of threads")
        plt.gcf().autofmt_xdate()


class SpeedupMachineSubplots(Subplots):

    def add_subplot(self, machine_info: str):
        if machine_info not in self.subplots:
            self.subplots[machine_info] = SpeedupMachineSubplot(machine_info)

    def get_subplot(self, machine_info: str) -> SpeedupMachineSubplot:
        return self.subplots[machine_info]


def json_files(directory: str):
    for subdir, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".json"):
                yield os.path.join(subdir, file)


def plot_speedups(directory: str, figure_index: int):
    # Format data into threads for plot
    subplots = SpeedupMachineSubplots()
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


class MedianMachineSubplot(Subplot):

    def add_time_taken(self, test: str, date: datetime.date, time_taken: float):
        self.add_test(test)
        self.test[test][date] = time_taken

    def plot(self, figure_index: int):
        last_func = None
        x_axis = []
        y_axes = {}
        title = None
        dataframes = []
        for test in self.test:
            if last_func != test.split("[")[0]:
                dataframe = {'x': x_axis, 'title': title}
                dataframe.update(y_axes)
                dataframes.append(dataframe)
                x_axis = []
                y_axes = {}
                title = "Performance over time\n {}\n {}".format(
                    self.machine_info, test.split("[")[0])
                last_func = test.split("[")[0]
            y_axis = []
            new_x_axis = []
            for key, value in self.test[test].items():
                index = 0
                while index < len(x_axis) and key > x_axis[index]:
                    index += 1
                new_x_axis.insert(index, key)
                y_axis.insert(index, value)
            x_axis = new_x_axis
            y_axes[test.split("[")[1][:-1]] = y_axis
        dataframe = {'x': x_axis, 'title': title}
        dataframe.update(y_axes)
        dataframes.append(dataframe)
        for index, dataframe in enumerate(dataframes[1:]):
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


class MedianMachineSubplots(Subplots):

    def add_subplot(self, machine_info: str):
        if machine_info not in self.subplots:
            self.subplots[machine_info] = MedianMachineSubplot(machine_info)

    def get_subplot(self, machine_info: str) -> MedianMachineSubplot:
        return self.subplots[machine_info]


def plot_median_values(directory: str, figure_index: int):
    plots = MedianMachineSubplots()
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
