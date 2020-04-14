import pytest
import os
import numpy as np
# Required for mocking
import matplotlib.pyplot
from unittest.mock import Mock

from ..utils import mock_has_method_call
from .utils import get_phonon_file, iter_dos_data_files

import scripts.dos
import scripts.utils


@pytest.mark.unit
class TestUnit:

    @pytest.fixture
    def inject_mocks(self, mocker):
        # Mock out calls to the phonon data structure
        phonon_mock = Mock()
        phonon_mock.freqs.magnitude.min = lambda: 1.0
        phonon_mock.freqs.magnitude.max = lambda: 10.0
        mocker.patch("scripts.utils.PhononData.from_castep", return_value=phonon_mock)

        # Mock out calls to the output_grace function
        mocker.patch("scripts.dos.output_grace")

        # Mock out calls to the plot_dos function
        fig_mock = Mock()
        mocker.patch("scripts.dos.plot_dos", return_value=fig_mock)

        # Mock out calls to pint
        mocker.patch("scripts.dos.ureg.Quantity.ito")

        # Mock oug calls to np arange
        dos_bins_mock = Mock()
        mocker.patch("scripts.dos.np.arange", return_value=dos_bins_mock)

        # Mock out calls to pyplot
        mocker.patch("matplotlib.pyplot")

        mocker.resetall()

        return {"phonon_mock": phonon_mock, "fig_mock": fig_mock, "dos_bins_mock": dos_bins_mock}

    def test_when_not_called_with_up_and_down_both_are_true(self, inject_mocks):
        scripts.dos.main([get_phonon_file()])
        call_kwargs = scripts.dos.plot_dos.call_args[1]
        assert call_kwargs["up"] is True and call_kwargs["down"] is True

    def test_when_called_with_up_then_down_is_false(self, inject_mocks):
        scripts.dos.main([get_phonon_file(), "-up"])
        call_kwargs = scripts.dos.plot_dos.call_args[1]
        assert call_kwargs["up"] is True and call_kwargs["down"] is False

    def test_when_called_with_down_then_up_is_false(self, inject_mocks):
        scripts.dos.main([get_phonon_file(), "-down"])
        call_kwargs = scripts.dos.plot_dos.call_args[1]
        assert call_kwargs["up"] is False and call_kwargs["down"] is True

    def test_when_called_with_phonon_data_file_then_correct_data_is_loaded(self, inject_mocks):
        phonon_file = get_phonon_file()
        path, filename = os.path.split(phonon_file)
        scripts.dos.main([phonon_file])
        assert scripts.utils.PhononData.from_castep.call_args[0][0] == filename.split(".")[0]
        assert scripts.utils.PhononData.from_castep.call_args[1]["path"] == path

    def test_when_called_then_data_is_converted_to_units(self, inject_mocks):
        scripts.dos.main([get_phonon_file()])
        assert mock_has_method_call(inject_mocks["phonon_mock"], "convert_e_units"), \
            "No call to convert units detected"

    @pytest.mark.parametrize("w_b", [[], ["-w 2.3"], ["-b 3.3"], ["-w 2.3", "-b 3.3"]])
    def test_when_called_with_w_and_or_b_then_ito_is_called_correctly(self, inject_mocks, w_b):
        num_of_args = len(w_b)
        scripts.dos.main([get_phonon_file()] + w_b)
        assert scripts.dos.ureg.Quantity.ito.call_count == 2-num_of_args, \
            "ito should not be called for each time -w and -b is specified"

    def test_when_called_then_np_arange_called_correctly(self, inject_mocks):
        phonon_mock = inject_mocks["phonon_mock"]
        scripts.dos.main([get_phonon_file()])
        assert scripts.dos.np.arange.call_count == 1

    def test_when_called_without_grace_but_save_then_saved(self, inject_mocks):
        filename = "FileName.plot"
        scripts.dos.main([get_phonon_file(), "-s={}".format(filename)])
        assert matplotlib.pyplot.savefig.call_args[0][0] == filename, \
            "When specifying -s and a filename, should use matplotlib function to save figure with the given name"

    def test_when_called_without_grace_or_show_then_saved(self, inject_mocks):
        scripts.dos.main([get_phonon_file()])
        assert matplotlib.pyplot.show.call_count == 1, \
            "When not specifying -grace or -s, then matplotlib function to show the plot should be called"

    def test_when_called_with_grace_then_grace_and_not_saved_or_shown(self, inject_mocks):
        scripts.dos.main([get_phonon_file(), "-grace"])
        assert scripts.dos.output_grace.call_count == 1, \
            "When specifying the -grace function, the euphonic output_grace function should be called"


@pytest.mark.integration
class TestRegression:

    @pytest.fixture
    def inject_mocks(self, mocker):
        # Prevent calls to show so we can get the current figure using gcf()
        mocker.patch("matplotlib.pyplot.show")

        mocker.resetall()

    def test_plots_produce_expected_xydata(self, inject_mocks):
        scripts.dos.main([get_phonon_file()])

        lines = matplotlib.pyplot.gcf().axes[0].lines

        for filenum, file in iter_dos_data_files():
            file_contents = np.genfromtxt(file, delimiter=",")
            assert np.array_equal(file_contents, lines[filenum].get_xydata().T)
