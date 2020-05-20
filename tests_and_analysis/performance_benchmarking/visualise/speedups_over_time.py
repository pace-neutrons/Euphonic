from .figures import Figure, Figures, json_files
from typing import Dict
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import json


class SpeedupMachineFigure(Figure):

    def add_speedup(self, test: str, seedname: str,
                    n_threads: int, date: datetime.date,
                    speedup: float):
        """
        Add speedup data for a specific test, date of test
        and number of threads.
        If the number of threads and test hasn't been added yet, add them.

        Parameters
        ----------
        test : str
            The name of the test this data is associated with
        seedname : str
            The material the data comes from
        n_threads : int
            The number of threads the test was run with
            and the speedup calculated with
        date : datetime.date
            The date the test was run on
        speedup : float
            The speedup value calculated by:
            speed on 1 thread / speed on n_threads
        """
        self.add_n_threads(test, seedname, n_threads)
        self.tests[test][seedname][n_threads][date] = speedup

    def add_seedname(self, test: str, seedname: str):
        """
        Add an entry for the seedname in the recorded data.

        Parameters
        ----------
        test : str
            The name of the test run with the seedname
        seedname : str
            The name of the material run with the test
        """
        self.add_test(test)
        if seedname not in self.tests[test]:
            self.tests[test][seedname] = {}

    def add_n_threads(self, test: str, seedname: str, n_threads: int):
        """
        The given test has been run with the given number of
        threads (n_threads).
        Add n_threads as a key for the given test to record
        speedup data against.
        If the test is present, add it.

        Parameters
        ----------
        test : str
            The test that has been run with the given number
            of threads (n_threads)
        seedname : str
            The material the data comes from
        n_threads : int
            A number of threads the test has been run with.
        """
        self.add_seedname(test, seedname)
        if n_threads not in self.tests[test][seedname]:
            self.tests[test][seedname][n_threads] = {}

    def plot(self):
        """
        Plot the speedup performance over time across a range of threads.
        """
        # Plot a figure for each test and seedname combination
        for test in self.tests:
            for seedname in self.tests[test]:
                fig, subplots = plt.subplots()
                # Plot a line on the figure for each number of threads
                for n_threads in self.tests[test][seedname]:
                    subplots.plot(
                        list(self.tests[test][seedname][n_threads].keys()),
                        list(self.tests[test][seedname][n_threads].values()),
                        label=str(n_threads)
                    )
                # Format x axis to use dates
                subplots.xaxis.set_major_formatter(
                    dates.DateFormatter('%Y-%m-%d')
                )
                subplots.xaxis.set_major_locator(dates.DayLocator())
                # Label axes, title and legend correctly
                subplots.set_xlabel("Date")
                subplots.set_ylabel("Speedup value (Ts/Tn)")
                subplots.set_title("Speedups over time\n {}\n {}, {}".format(
                    self.machine_info, test, seedname
                ))
                subplots.legend(
                    title="Threads",
                    loc='center left',
                    bbox_to_anchor=(1, 0.5),
                    fontsize="small"
                )
                fig.tight_layout()
                fig.autofmt_xdate()


class SpeedupMachineFigures(Figures):

    def __init__(self):
        super().__init__()
        # A dictionary of figures with keys as the machine information
        # and the figure as the value
        self.figures: Dict[str, SpeedupMachineFigure] = {}

    def add_figure(self, machine_info: str):
        """
        Add a SpeedupMachineFigure to the figures to plot associated
        with the given machine information.

        Parameters
        ----------
        machine_info : str
            A string describing the machine the speedup benchmarks
            have been run on.
        """
        if machine_info not in self.figures:
            self.figures[machine_info] = SpeedupMachineFigure(machine_info)

    def get_figure(self, machine_info: str) -> SpeedupMachineFigure:
        """
        Get the figure associated with the given machine information.

        Parameters
        ----------
        machine_info : str
            The machine information associated with the figure to get

        Returns
        --------
        SpeedupMachineFigure
            The figure associated with the given machine information
        """
        return self.figures[machine_info]


def plot_speedups_over_time(directory: str):
    """
    Plot and show graphs displaying how speedups on different numbers of threads
     have changed over time using the data from the json files under the given
      directory.

    Parameters
    ----------
    directory : str
        The directory under which the json files are stored.
    """
    figures = SpeedupMachineFigures()
    for file in json_files(directory):
        data = json.load(open(file))
        if "speedups" in data:
            # Add a new figure for each different machine the tests
            # have been run on
            figures.add_figure(data["machine_info"]["cpu"]["brand"])
            for test in data["speedups"]:
                for seedname in data["speedups"][test]:
                    for n_threads in data["speedups"][test][seedname]:
                        # Add speedup data for each test and the number of
                        # threads that have been used for the test
                        figures.get_figure(
                            data["machine_info"]["cpu"]["brand"]
                        ).add_speedup(
                            test,
                            seedname,
                            n_threads,
                            datetime.strptime(
                                data["datetime"].split("T")[0],
                                '%Y-%m-%d'
                            ).date(),
                            data["speedups"][test][seedname][n_threads]
                        )
    figures.plot()
