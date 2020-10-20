import sys
import json
import pytest
import numpy as np
import numpy.testing as npt
import matplotlib
# Default mpl backend fails on system without $DISPLAY set for < 3.0.0
if int(matplotlib.__version__.split('.')[0]) < 3:
    matplotlib.use('Agg')
# Required for mocking
import matplotlib.pyplot
from .utils import (get_sphere_sampling_params,
                    get_sphere_sampling_data_file,
                    get_current_plot_offsets)
import euphonic.cli.show_sampling


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

    @pytest.mark.parametrize("sampling_params", get_sphere_sampling_params())
    def test_plots_produce_expected_xydata(
            self, inject_mocks, sampling_params):
        np.random.seed(0)
        euphonic.cli.show_sampling.main(sampling_params)

        # For 3D plots, this will be a 2D projected position of visible points
        offsets = get_current_plot_offsets()

        with open(get_sphere_sampling_data_file()) as sampling_json_file:
            expected_offsets = json.load(
                sampling_json_file)[" ".join(sampling_params)]

        npt.assert_allclose(offsets, np.array(expected_offsets),
                            atol=sys.float_info.epsilon)
    
