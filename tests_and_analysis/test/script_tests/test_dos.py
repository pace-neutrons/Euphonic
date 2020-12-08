import json
import os
import sys
from unittest.mock import patch

import pytest
import numpy as np
import numpy.testing as npt
# Required for mocking
import matplotlib.pyplot

from tests_and_analysis.test.utils import get_data_path
from tests_and_analysis.test.script_tests.utils import (
    get_script_test_data_path, get_current_plot_lines_xydata)
import euphonic.cli.dos


nah_phonon_file = os.path.join(
    get_data_path(), 'qpoint_phonon_modes', 'NaH', 'NaH.phonon')
dos_output_file = os.path.join(get_script_test_data_path(), "dos.json")
dos_params = [[], ["--energy-broadening=2.3"], ["--ebins=100"],
              ["--energy-broadening=2.3", "--ebins=100"],
              ["--energy-unit=meV"], ["--shape=lorentz"]]


@pytest.mark.integration
class TestRegression:

    @pytest.fixture
    def inject_mocks(self, mocker):
        # Prevent calls to show so we can get the current figure using
        # gcf()
        mocker.patch("matplotlib.pyplot.show")
        mocker.resetall()

    def teardown_method(self):
        # Ensure figures are closed
        matplotlib.pyplot.close('all')

    @pytest.mark.parametrize("dos_args", dos_params)
    def test_plots_produce_expected_xydata(self, inject_mocks, dos_args):
        euphonic.cli.dos.main([nah_phonon_file] + dos_args)

        lines = matplotlib.pyplot.gcf().axes[0].lines

        with open(dos_output_file, 'r') as f:
            expected_lines = json.load(f)[" ".join(dos_args)]

        for index, line in enumerate(lines):
            npt.assert_allclose(line.get_xydata().T,
                                np.array(expected_lines[index]),
                                atol=5*sys.float_info.epsilon)


@patch("matplotlib.pyplot.show")
@pytest.mark.skip(reason='Only run if you want to regenerate the test data')
def test_regenerate_dos_data(_):
    json_data = {}
    for dos_param in dos_params:
        # Generate current figure for us to retrieve with gcf
        euphonic.cli.dos.main([nah_phonon_file] + dos_param)
        # Retrieve with gcf and record data
        json_data[" ".join(dos_param)] = get_current_plot_lines_xydata()
    with open(dos_output_file, "w+") as json_file:
        json.dump(json_data, json_file, indent=4)
