from abc import ABC, abstractmethod
from typing import Dict
import os

# Adapted from:
# https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/linestyles.html
linestyle_tuple = [
     'solid', 'dotted', 'dashed', 'dashdot'
]


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
