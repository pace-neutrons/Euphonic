import os


def get_data_path() -> str:
    """
    Returns
    -------
    str: The path to the data files for use in performance benchmarking
    """
    return os.path.join(os.path.dirname(__file__), "data").replace("\\", "\\\\")
