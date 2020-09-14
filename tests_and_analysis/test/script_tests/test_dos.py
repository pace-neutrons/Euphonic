import json
import sys
import pytest
import numpy as np
import numpy.testing as npt
# Required for mocking
import matplotlib.pyplot
from .utils import get_phonon_file, get_dos_params, get_dos_data_file
import scripts.dos


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

    @pytest.mark.parametrize("dos_args", get_dos_params())
    def test_plots_produce_expected_xydata(self, inject_mocks, dos_args):
        scripts.dos.main([get_phonon_file()] + dos_args)

        lines = matplotlib.pyplot.gcf().axes[0].lines

        with open(get_dos_data_file()) as dos_json_file:
            expected_lines = json.load(dos_json_file)[" ".join(dos_args)]

        for index, line in enumerate(lines):
            npt.assert_allclose(line.get_xydata().T,
                                np.array(expected_lines[index]),
                                atol=sys.float_info.epsilon)
