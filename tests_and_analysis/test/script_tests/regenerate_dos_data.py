from unittest.mock import patch
import scripts.dos
from tests_and_analysis.test.script_tests.utils import get_phonon_file, get_dos_params, \
    get_current_plot_lines_xydata, get_dos_data_file
import json


@patch("matplotlib.pyplot.show")
def regenerate_dos_data(_):

    json_data = {}

    for dos_params in get_dos_params():
        # Generate current figure for us to retrieve with gcf
        scripts.dos.main([get_phonon_file()] + dos_params)

        # Retrieve with gcf and record data
        json_data[" ".join(dos_params)] = get_current_plot_lines_xydata()

    json_filepath: str = get_dos_data_file()
    with open(json_filepath, "w+") as json_file:
        json.dump(json_data, json_file)


if __name__ == "__main__":
    regenerate_dos_data()
