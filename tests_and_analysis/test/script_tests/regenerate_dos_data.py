from unittest.mock import patch
import scripts.dos
from tests_and_analysis.test.script_tests.utils import get_phonon_file,\
    get_dos_data_filepath_prefix, write_plot_to_file_for_regression_testing


@patch("matplotlib.pyplot.show")
def regenerate_dos_data(_):
    # Generate current figure for us to retrieve with gcf
    scripts.dos.main([get_phonon_file()])

    filepath_prefix = get_dos_data_filepath_prefix()

    # Retrieve with gcf and write to file
    write_plot_to_file_for_regression_testing(filepath_prefix)


if __name__ == "__main__":
    regenerate_dos_data()
