from typing import List, Dict
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import os
import json
from datetime import datetime
import pandas as pd
from abc import abstractmethod, ABC


class Figure(ABC):

    def __init__(self, machine_info: str):
        # The details of the machine the tests have been run on
        self.machine_info: str = machine_info
        # The tests for which we are recording performance over time
        self.tests: Dict[str, Dict] = {}

    def add_test(self, test: str):
        """
        The figure has a set of tests (keys for the test dict)
         for which we are recording the performance
         over time. Add to this set of tests.

        Parameters
        ----------
        test : str
            The test to add to the set of tests we are recording.
        """
        if test not in self.tests:
            self.tests[test] = {}

    @abstractmethod
    def plot(self, figure_index: int) -> int:
        """
        Plot the test performance over time held by this figure.

        Parameters
        ----------
        figure_index : int
            The index of the figure to plot

        Returns
        -------
        int
            The next free index to plot a figure on.
        """
        raise NotImplementedError


class Figures(ABC):

    def __init__(self):
        # A dictionary of figures with keys as the machine information
        # and the figure as the value
        self.figures: Dict[str, Figure] = {}

    def plot(self, figure_index: int) -> int:
        """
        Plot the figures currently held in this object.

        Parameters
        ----------
        figure_index : int

        Returns
        -------
        int
            THe next free figure index after plotting the figures.
        """
        for figure in self.figures:
            figure_index = self.figures[figure].plot(figure_index)
        return figure_index


class SpeedupMachineFigure(Figure):

    def add_speedup(self, test: str, n_threads: int, date: datetime.date,
                    speedup: float):
        """
        Add speedup data for a specific test, date of test
         and number of threads.
        If the number of threads and test hasn't been added yet, add them.

        Parameters
        ----------
        test : str
            The name of the test this data is associated with
        n_threads : int
            The number of threads the test was run with
             and the speedup calculated with
        date : datetime.date
            The date the test was run on
        speedup : float
            The speedup value calculated by:
             speed on 1 thread / speed on n_threads
        """
        self.add_n_threads(test, n_threads)
        self.tests[test][n_threads][date] = speedup

    def add_n_threads(self, test: str, n_threads: int):
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
        n_threads : int
            A number of threads the test has been run with.
=        """
        self.add_test(test)
        if n_threads not in self.tests[test]:
            self.tests[test][n_threads] = {}

    def plot(self, figure_index: int):
        """
        Plot the speedup performance over time across a range of threads.

        Parameters
        ----------
        figure_index : int
            The index of the figure to plot.

        Returns
        -------
        int
            The next free index for a figure to use
        """
        # Plot a figure for each test
        for test in self.tests:
            plt.figure(figure_index)
            # Plot a line on the figure for each number of threads
            for n_threads in self.tests[test]:
                plt.plot(
                    list(self.tests[test][n_threads].keys()),
                    list(self.tests[test][n_threads].values()),
                    label=str(n_threads)
                )
            # Format x axis to use dates
            plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(dates.DayLocator())
            # Label axes, title and legend correctly
            plt.xlabel("Date")
            plt.ylabel("Speedup value (Ts/Tn)")
            plt.title("Speedups over time\n {}\n {}"
                      .format(self.machine_info, test))
            plt.legend(title="Number of threads")
            plt.gcf().autofmt_xdate()
            figure_index += 1
        return figure_index


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


def plot_speedups(directory: str, figure_index: int) -> int:
    """
    Plot and show graphs displaying how speedups on different numbers of threads
     have changed over time using the data from the json files under the given
      directory.

    Parameters
    ----------
    directory : str
        The directory under which the json files are stored.
    figure_index : int
        The index of the first figure to plot

    Returns
    -------
    int
        The next free figure index after these plots
    """
    figures = SpeedupMachineFigures()
    for file in json_files(directory):
        data = json.load(open(file))
        if "speedups" in data:
            # Add a new figure for each different machine the tests
            # have been run on
            figures.add_figure(data["machine_info"]["cpu"]["brand"])
            for test in data["speedups"]:
                for n_threads in data["speedups"][test]:
                    # Add speedup data for each test and the number of
                    # threads that have been used for the test
                    figures.get_figure(
                        data["machine_info"]["cpu"]["brand"]
                    ).add_speedup(
                        test,
                        n_threads,
                        datetime.strptime(
                            data["datetime"].split("T")[0],
                            '%Y-%m-%d'
                        ).date(),
                        data["speedups"][test][n_threads]
                    )
    next_free_figure_index: int = figures.plot(figure_index)
    return next_free_figure_index


class MedianMachineFigure(Figure):

    def add_time_taken(self, test: str, date: datetime.date, time_taken: float):
        """
        Add data on the time taken to run the given test on the given date.

        Parameters
        ----------
        test : str
            The test being benchmarked.
        date : datetime.date
            The date the benchmarking took place.
        time_taken : float
            The time taken in seconds
        """
        test_name = test.split("[")[0]
        test_params = test.split("[")[1][:-1]
        self.add_test_params(test_name, test_params)
        self.tests[test_name][test_params][date] = time_taken

    def add_test_params(self, test: str, params: str):
        """
        Add parameters used for a test.

        Parameters
        ----------
        test : str
            The test being benchmarked.
        params : str
            The parameter values for an execution of the test.
        """
        self.add_test(test)
        if params not in self.tests[test]:
            self.tests[test][params] = {}

    def plot(self, figure_index: int) -> int:
        """
        Plot the (possibly multiple) figures. One figure for each test with
         a line for each set of parameters of the test and a point
          on each line for each date it has been run.

        Parameters
        ----------
        figure_index : int
            The first free figure index to use to plot a figure

        Returns
        -------
        int
            The next free figure index to use after these plots.
        """
        dataframes: Dict[str, pd.DataFrame] = self.build_dataframes()
        next_free_figure_index: int = \
            self.plot_dataframes(dataframes, figure_index)
        return next_free_figure_index

    def build_dataframes(self) -> Dict[str, pd.DataFrame]:
        """
        Build a dataframe for each test. Each set of params has a new y_axis
         points in the dataframe.

        Returns
        -------
        Dict[str, pd.DataFrame]
            The key is what will be the title of the plots.
            The value is the dataframe to create a figure for.
        """
        dataframes: Dict[str, pd.DataFrame] = {}
        for test in self.tests:
            x_axis: List[str] = []
            y_axes: Dict[str, List[float]] = {}
            # Create a y_axis and overwrite the x_axis (will be the
            # same as the last) for each set of parameters used with the test
            for params in self.tests[test]:
                y_axis: List[float] = []
                new_x_axis: List[datetime.date] = []
                # Sort the entries of the time taken by the date the test
                # was executed on
                for key, value in self.tests[test][params].items():
                    # Search through the list and find the two elements between
                    # which the new date fits in order
                    index: int = 0
                    while index < len(new_x_axis) and key > new_x_axis[index]:
                        index += 1
                    # Insert the date into the list in between the two elements
                    # either side in the order
                    new_x_axis = new_x_axis[:index] + [key] + new_x_axis[index:]
                    # Maintain the order found for the x_axis in
                    # the y_axis elements
                    y_axis = y_axis[:index] + [value] + y_axis[index:]
                x_axis = new_x_axis
                y_axes[params] = y_axis
            # Create a new dataframe for each test
            panda_dataframe: pd.DataFrame = \
                self.create_dataframe(x_axis, y_axes)
            title: str = "Performance over time\n {}\n {}".format(
                self.machine_info, test)
            dataframes[title] = panda_dataframe
        return dataframes

    def create_dataframe(self, x_axis: List[datetime.date],
                         y_axes: Dict[str, List[float]]) -> pd.DataFrame:
        """
        Create a single dataframe from an x_axis and multiple y_axes.

        Parameters
        ----------
        x_axis : List[datetime.date]
            The dates to plot on the x axis of the dataframe.
        y_axes : Dict[str, List[float]]
            The key is the parameters used when recording the time taken and
             the values are the times taken at the corresponding x_axis dates.

        Returns
        -------
        The pandas dataframe containing the data.
        """
        dataframe = {'x': x_axis}
        dataframe.update(y_axes)
        return pd.DataFrame(dataframe)

    def plot_dataframes(self, dataframes: Dict[str, pd.DataFrame],
                        figure_index: int) -> int:
        """
        Plot a figure for each dataframe and use the dictionary
         key as the title. Each dataframe has an x axis with the key 'x'
          and a number of y axis traces.

        Parameters
        ----------
        dataframes : Dict[str, pd.DataFrame]
            The key is the title of the figure to be and the
             value is the dataframe to plot for the figure.
        figure_index : int
            The index of the first figure to be plotted.

        Returns
        -------
        int
            The next free figure index to use after these plots.
        """
        # A plot for each test (dataframe)
        for title, dataframe in dataframes.items():
            plt.figure(figure_index)
            # A trace for each combination of the test parameters
            for key in dataframe.keys():
                if key != "x":
                    plt.plot(
                        'x', key,
                        data=dataframe
                    )
            # Set figure display details
            plt.title(title)
            plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(dates.DayLocator())
            plt.xlabel("Date")
            plt.ylabel("Time taken (seconds)")
            plt.legend(title="Test Params")
            plt.gcf().autofmt_xdate()
            figure_index += 1
        return figure_index


class MedianMachineFigures(Figures):

    def __init__(self):
        super().__init__()
        # A dictionary of figures with keys as the machine information
        # and the figure as the value
        self.figures: Dict[str, MedianMachineFigure] = {}

    def add_figure(self, machine_info: str):
        """
        Add a MedianMachine Figure to the figures stored in this object
         with the given machine info (does not add another figure if there
          is already one with the given machine information.

        Parameters
        ----------
        machine_info : str
            The information describing the machine the tests have run on.
        """
        if machine_info not in self.figures:
            self.figures[machine_info] = MedianMachineFigure(machine_info)

    def get_figure(self, machine_info: str) -> MedianMachineFigure:
        """
        Get the figure with the given machine information.

        Parameters
        ----------
        machine_info : str
            The machine information the tests of the returned
             figure have been run on.

        Returns
        -------
        MedianMachineFigure
            The figure which contains benchmark data that has been run
             on the machine with the given information.
        """
        return self.figures[machine_info]


def plot_median_values(directory: str, figure_index: int) -> int:
    """
    Plot and show a graph for each test displaying performance changes over
     time with a trace for each combination of parameters the test has run on.
      If the tests have been run on multiple different types of machines there
       will be a separate figure for each type of machine. Data is taken from
        json files under the given directory.

    Parameters
    ----------
    directory : str
        The directory under which the json files are stored.
    figure_index : int
        The index of the first figure to plot

    Returns
    -------
    int
        The next free figure index to use after these plots
    """
    plots = MedianMachineFigures()
    for file in json_files(directory):
        data = json.load(open(file))
        if "benchmarks" in data:
            plots.add_figure(data["machine_info"]["cpu"]["brand"])
            for benchmark in data["benchmarks"]:
                test = benchmark["name"]
                plots.get_figure(
                    data["machine_info"]["cpu"]["brand"]
                ).add_time_taken(
                    test,
                    datetime.strptime(
                        data["datetime"].split("T")[0],
                        '%Y-%m-%d'
                    ).date(),
                    benchmark["stats"]["median"]
                )
    next_free_figure_index: int = plots.plot(figure_index)
    return next_free_figure_index


if __name__ == "__main__":
    median_value_starting_figure_index: int = plot_speedups("reports", 0)
    plot_median_values("reports", median_value_starting_figure_index)
    plt.show()
