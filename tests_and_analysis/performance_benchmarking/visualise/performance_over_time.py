from .figures import Figures, Figure, linestyle_tuple, json_files
from utils import get_seednames
from datetime import datetime
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import json


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

    def plot(self):
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
        self.plot_dataframes(dataframes)

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

    def plot_dataframes(self, dataframes: Dict[str, pd.DataFrame]):
        """
        Plot a figure for each dataframe and use the dictionary
        key as the title. Each dataframe has an x axis with the key 'x'
        and a number of y axis traces.

        Parameters
        ----------
        dataframes : Dict[str, pd.DataFrame]
            The key is the title of the figure to be and the
            value is the dataframe to plot for the figure.
        """
        # A plot for each test (dataframe)
        for title, dataframe in dataframes.items():
            fig, subplots = plt.subplots()
            # A trace for each combination of the test parameters
            # Vary linestyles
            for i, key in enumerate(dataframe.keys()):
                if key != "x":
                    subplots.plot(
                        'x', key,
                        data=dataframe,
                        linestyle=linestyle_tuple[i%len(get_seednames()) - 1]
                    )
            # Set figure display details
            subplots.set_title(title)
            subplots.xaxis.set_major_formatter(dates.DateFormatter('%Y-%m-%d'))
            subplots.xaxis.set_major_locator(dates.DayLocator())
            subplots.set_xlabel("Date")
            subplots.set_ylabel("Time taken (seconds)")
            subplots.legend(
                title="Params",
                loc='center left',
                bbox_to_anchor=(1, 0.5),
                fontsize="small"
            )
            fig.tight_layout()
            fig.autofmt_xdate()


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


def plot_median_values(directory: str):
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
    plots.plot()
