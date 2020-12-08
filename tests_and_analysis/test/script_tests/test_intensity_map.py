import os
import json
from unittest.mock import patch

import pytest
import numpy as np
import numpy.testing as npt
# Required for mocking
import matplotlib.pyplot

from tests_and_analysis.test.utils import get_data_path
from tests_and_analysis.test.script_tests.utils import (
    get_script_test_data_path, get_current_plot_image_data)
import euphonic.cli.intensity_map


graphite_fc_file = os.path.join(
    get_data_path(), 'force_constants', 'graphite', 'graphite.castep_bin')
intensity_map_output_file = os.path.join(get_script_test_data_path(),
                                     "intensity-map.json")
intensity_map_params = [
    [], ["--energy-unit=meV"], ["--weights=coherent", "--cmap=bone"],
    ["-w", "dos", "--y-label='DOS'", "--title='DOS TITLE'"],
    ["--e-min=5", "--energy-unit=cm^-1", "--x-label='wavenumber'"],
    ["--e-min=-10", "--e-max=200", "--energy-unit=cm^-1"],
    ["--energy-broadening=2e-3", "--energy-unit=eV"],
    ["--q-distance=0.05", "--length-unit=bohr", "--q-broadening=0.1"],
    ["--asr"],
    ["--asr=realspace"]]


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

    @pytest.mark.parametrize("intensity_map_args", intensity_map_params)
    def test_plots_produce_expected_image(
            self, inject_mocks, intensity_map_args):
        euphonic.cli.intensity_map.main([graphite_fc_file]
                                        + intensity_map_args)

        image_data = get_current_plot_image_data()

        with open(intensity_map_output_file, 'r') as expected_data_file:
            expected_image_data = json.load(
                expected_data_file)[" ".join(intensity_map_args)]
        for key, value in image_data.items():
            if key == 'extent':
                # Lower bound of y-data (energy) varies by up to ~2e-6 on
                # different systems when --asr is used, compared to
                # the upper bound of 100s of meV this is effectively zero,
                # so increase tolerance to allow for this
                npt.assert_allclose(value, expected_image_data[key], atol=2e-6)
            elif isinstance(value, list) and isinstance(value[0], float):
                # Errors of 2-4 epsilon seem to be common when using
                # broadening, so slightly increase tolerance
                npt.assert_allclose(value, expected_image_data[key],
                                    atol=1e-14)
            else:
                assert value == expected_image_data[key]


@patch("matplotlib.pyplot.show")
@pytest.mark.skip(reason='Only run if you want to regenerate the test data')
def test_regenerate_intensity_map_data(_):

    json_data = {}

    for intensity_map_param in intensity_map_params:
        # Generate current figure for us to retrieve with gcf
        euphonic.cli.intensity_map.main([graphite_fc_file]
                                        + intensity_map_param)

        # Retrieve with gcf and write to file
        json_data[" ".join(intensity_map_param)
                  ] = get_current_plot_image_data()

    with open(intensity_map_output_file, "w+") as json_file:
        json.dump(json_data, json_file, indent=4)
