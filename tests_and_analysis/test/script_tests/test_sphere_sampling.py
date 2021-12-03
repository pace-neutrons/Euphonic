import sys
import os
import json
from unittest.mock import patch

import pytest
import numpy as np
import numpy.testing as npt

from tests_and_analysis.test.script_tests.utils import (
    get_script_test_data_path, get_current_plot_offsets)

pytestmark = pytest.mark.matplotlib
# Allow tests with matplotlib marker to be collected and
# deselected if Matplotlib is not installed
try:
    import matplotlib.pyplot
    import euphonic.cli.show_sampling
except ModuleNotFoundError:
    pass

sphere_sampling_output_file = os.path.join(get_script_test_data_path(),
                                           "sphere_sampling.json")
sphere_sampling_params =  [
    ['27', 'golden-square'],
    ['8', 'regular-square'],
    ['9', 'regular-square'],
    ['10', 'golden-sphere'],
    ['10', 'golden-sphere', '--jitter'],
    ['15', 'spherical-polar-grid'],
    ['18', 'spherical-polar-grid', '--jitter'],
    ['17', 'sphere-from-square-grid', '--jitter'],
    ['18', 'sphere-from-square-grid'],
    ['15', 'spherical-polar-improved'],
    ['15', 'spherical-polar-improved', '--jitter'],
    ['10', 'random-sphere'],
    ['10', 'random-sphere', '--jitter']]


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

    @pytest.mark.parametrize("sampling_params", sphere_sampling_params)
    def test_plots_produce_expected_xydata(
            self, inject_mocks, sampling_params):
        np.random.seed(0)
        euphonic.cli.show_sampling.main(sampling_params)

        # For 3D plots, this will be a 2D projected position of visible points
        offsets = get_current_plot_offsets()

        with open(sphere_sampling_output_file, 'r') as sampling_json_file:
            expected_offsets = json.load(
                sampling_json_file)[" ".join(sampling_params)]

        npt.assert_allclose(offsets, np.array(expected_offsets),
                            atol=sys.float_info.epsilon)


@patch("matplotlib.pyplot.show")
@pytest.mark.skip(reason='Only run if you want to regenerate the test data')
def test_regenerate_sphere_sampling_data(_):

    json_data = {}

    for sampling_params in sphere_sampling_params:
        # Reset random number generator for deterministic results
        np.random.seed(0)

        # Generate current figure for us to retrieve with gcf
        euphonic.cli.show_sampling.main(sampling_params)

        # Retrieve with gcf and write to file
        json_data[" ".join(sampling_params)] = get_current_plot_offsets()

    with open(sphere_sampling_output_file, "w+") as json_file:
        json.dump(json_data, json_file, indent=4)
