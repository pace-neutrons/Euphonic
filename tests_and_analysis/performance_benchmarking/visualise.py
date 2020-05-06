import matplotlib.pyplot as plt
import matplotlib.dates as dates
import os
import json
from datetime import datetime


class Subplot:

    def __init__(self, machine_info: str):
        self.machine_info = machine_info
        self.funcs = {}

    def add_func(self, func):
        if func not in self.funcs:
            self.funcs[func] = {}

    def add_n_threads(self, func: str, n_threads: int):
        self.add_func(func)
        if n_threads not in self.funcs[func]:
            self.funcs[func][n_threads] = {}


class SpeedupMachineSubplot(Subplot):

    def add_speedup(self, func: str, n_threads: int, date: datetime.date, speedup: float):
        self.add_n_threads(func, n_threads)
        self.funcs[func][n_threads][date] = speedup

    def plot(self):
        print(self.funcs)
        for func in self.funcs:
            for n_threads in self.funcs[func]:
                plt.plot(
                    list(self.funcs[func][n_threads].keys()),
                    list(self.funcs[func][n_threads].values()),
                    label=str(n_threads)
                )
        plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(dates.DayLocator())
        plt.xlabel("Date")
        plt.ylabel("Speedup value (Ts/Tn)")
        plt.title("Speedups over time")
        plt.legend(title="Number of threads")
        plt.show()
        plt.gcf().autofmt_xdate()

class Subplots:

    def __init__(self):
        self.subplots = {}

    def plot(self):
        for subplot in self.subplots:
            self.subplots[subplot].plot()


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


def plot_speedups(directory: str):
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
    subplots.plot()

if __name__ == "__main__":
    plot_speedups("reports")
