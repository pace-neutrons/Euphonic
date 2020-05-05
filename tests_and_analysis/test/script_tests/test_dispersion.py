import pytest
import numpy as np
import numpy.testing as npt
import json
# Required for mocking
import matplotlib.pyplot
from .utils import (get_phonon_file, get_dispersion_params,
                    get_dispersion_data_file)
import scripts.dispersion


@pytest.mark.integration
class TestRegression:

    @pytest.fixture
    def inject_mocks(self, mocker):
        # Prevent calls to show so we can get the current figure using
        # gcf()
        mocker.patch("matplotlib.pyplot.show")
        mocker.resetall()

    @pytest.mark.parametrize("dispersion_args", get_dispersion_params())
    def test_plots_produce_expected_xydata(
            self, inject_mocks, dispersion_args):
        scripts.dispersion.main([get_phonon_file()] + dispersion_args)

        lines = matplotlib.pyplot.gcf().axes[0].lines

        with open(get_dispersion_data_file()) as disp_json_file:
            expected_lines = json.load(
                disp_json_file)[" ".join(dispersion_args)]
        for index, line in enumerate(lines):
            npt.assert_allclose(
                line.get_xydata().T, np.array(expected_lines[index]))
