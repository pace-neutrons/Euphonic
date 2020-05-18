import argparse
import json
from typing import Dict
import os


def get_file_or_dir() -> str:
    """
    Get the filename to calculate speedups of that has
    been specified on the command line.

    Returns
    -------
    str
        The filename to calculate speedups for.
    """
    parser = argparse.ArgumentParser()
    dir_file_group = parser.add_mutually_exclusive_group()
    dir_file_group.add_argument("-f", action="store", dest="filename",
                                help="The file to calculate speedups for")
    dir_file_group.add_argument("-d", action="store", dest="dirname",
                                help="The directory containing files"
                                     " to calculate speedups for",
                                default="reports")
    args_parsed = parser.parse_args()
    if args_parsed.filename:
        return args_parsed.filename
    else:
        return args_parsed.dirname


def median_value(benchmark: Dict) -> float:
    """
    Extract the median value from the benchmark disctionary.

    Parameters
    ----------
    benchmark : Dict
        A dictionary containing a median values

    Returns
    -------
    float
        The median time taken value from the benchmark data
    """
    return benchmark["stats"]["median"]


def calculate_speedups(filename: str) -> Dict[str, Dict[str, Dict[int, float]]]:
    """
    Calculate speedups for the tests that are parameterised to
    use a number of different threads.

    Parameters
    ----------
    filename : str
        The file to calculate speedups for

    Returns
    -------
    Dict[str, Dict[str, Dict[int, float]]]
        The keys of the top level dictionary are the name of the test.
        The keys of the next level of the dictionary are the seednames
        used in the tests.
        The keys of the next level dictionary are the number of threads used.
        The values are the speedups for the given test and number of threads.
    """
    data = json.load(open(filename))
    data["benchmarks"].sort(key=median_value)
    # Extract the time taken for all the tests at the various numbers of threads
    # and format the data to easily calculate speedups
    speed_at_threads = {}
    for benchmark in data["benchmarks"]:
        # Filter out the tests that haven't used different numbers of threads
        if "use_c" in benchmark["params"] and \
                benchmark["params"]["use_c"] is True:
            # Initialise performance data structure
            test = benchmark["name"].split("[")[0]
            if test not in speed_at_threads:
                speed_at_threads[test] = {}
            seedname = benchmark["params"]["seedname"]
            if seedname not in speed_at_threads[test]:
                speed_at_threads[test][seedname] = {}
            # At the given test and number of threads extract the
            # median time taken
            speed_at_threads[test][seedname][benchmark["params"]["n_threads"]] \
                = benchmark["stats"]["median"]
    # Calculate the speedups from the formatted data
    speedups = {}
    for test in speed_at_threads:
        speedups[test] = {}
        for seedname in speed_at_threads[test]:
            speedups[test][seedname] = {}
            sequential_speed = speed_at_threads[test][seedname][1]
            for n_threads in speed_at_threads[test][seedname]:
                speedups[test][seedname][n_threads] = \
                    sequential_speed / speed_at_threads[test][seedname][n_threads]
    return speedups


def write_speedups(filename: str, speedups: Dict[str, Dict[str, Dict[int, float]]]):
    """
    Write the calculated speedups to the given json file in
    the "speedups" entry.

    Parameters
    ----------
    filename : str
        The file to write the speedups to
    speedups : Dict[str, Dict[str, Dict[int, float]]]
        The calculated speedups to write to file.
    """
    # Load in the data and update with the speedups
    data = json.load(open(filename))
    data["speedups"] = speedups
    # Format the data nicely when overwriting to the file
    json.dump(data, open(filename, "w+"), indent=4, sort_keys=True)


if __name__ == "__main__":
    path: str = get_file_or_dir()
    if os.path.isdir(path):
        for filename in os.listdir(path):
            filepath = os.path.join(path, filename)
            speedups: Dict[str, Dict[str, Dict[int, float]]] = \
                calculate_speedups(filepath)
            write_speedups(filepath, speedups)
    elif os.path.isfile(path):
        speedups: Dict[str, Dict[str, Dict[int, float]]] = \
            calculate_speedups(path)
        write_speedups(path, speedups)
