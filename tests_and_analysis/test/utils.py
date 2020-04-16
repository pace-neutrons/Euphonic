import os


def get_data_path() -> str:
    """
    Get the path to the data for the tests.

    Returns
    -------
    str: The path to the test data.
    """
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), "data")
