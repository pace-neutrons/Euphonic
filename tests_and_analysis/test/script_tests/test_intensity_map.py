import sys
import json
import pytest
import numpy as np
import numpy.testing as npt
# Required for mocking
import matplotlib.pyplot
from .utils import (get_current_plot_image_data,
                    get_force_constants_file, get_intensity_map_params,
                    get_intensity_map_data_file)
import euphonic.cli.intensity_map


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

    @pytest.mark.parametrize("intensity_map_args", get_intensity_map_params())
    def test_plots_produce_expected_image(
            self, inject_mocks, intensity_map_args):
        euphonic.cli.intensity_map.main([get_force_constants_file()]
                                        + intensity_map_args)

        image_data = get_current_plot_image_data()

        with open(get_intensity_map_data_file()) as expected_data_file:
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
