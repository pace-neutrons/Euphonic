import pytest
import os
import numpy as np
# Required for mocking
import matplotlib.pyplot
from unittest.mock import Mock

from ..utils import mock_has_method_call
from .utils import get_phonon_file, iter_dispersion_data_files

import scripts.dispersion
import euphonic.script_utils


@pytest.mark.unit
class TestUnit:

    @pytest.fixture
    def inject_mocks(self, mocker):
        # Mock out calls to the phonon data structure
        phonon_mock = Mock()
        mocker.patch("euphonic.script_utils.PhononData.from_castep", return_value=phonon_mock)

        # Mock out calls to the output_grace function
        mocker.patch("scripts.dispersion.output_grace")

        # Mock out calls to the plot_dispersion function
        fig_mock = Mock()
        mocker.patch("scripts.dispersion.plot_dispersion", return_value=fig_mock)

        # Mock out calls to pyplot
        mocker.patch("matplotlib.pyplot")

        mocker.resetall()

        return {"phonon_mock": phonon_mock, "fig_mock": fig_mock}

    def test_when_not_called_with_up_and_down_both_are_true(self, inject_mocks):
        scripts.dispersion.main([get_phonon_file()])
        call_kwargs = scripts.dispersion.plot_dispersion.call_args[1]
        assert call_kwargs["up"] is True and call_kwargs["down"] is True

    def test_when_called_with_up_then_down_is_false(self, inject_mocks):
        scripts.dispersion.main([get_phonon_file(), "-up"])
        call_kwargs = scripts.dispersion.plot_dispersion.call_args[1]
        assert call_kwargs["up"] is True and call_kwargs["down"] is False

    def test_when_called_with_down_then_up_is_false(self, inject_mocks):
        scripts.dispersion.main([get_phonon_file(), "-down"])
        call_kwargs = scripts.dispersion.plot_dispersion.call_args[1]
        assert call_kwargs["up"] is False and call_kwargs["down"] is True

    def test_when_called_with_phonon_data_file_then_correct_data_is_loaded(self, inject_mocks):
        phonon_file = get_phonon_file()
        path, filename = os.path.split(phonon_file)
        scripts.dispersion.main([phonon_file])
        assert euphonic.script_utils.PhononData.from_castep.call_args[0][0] == filename.split(".")[0]
        assert euphonic.script_utils.PhononData.from_castep.call_args[1]["path"] == path

    def test_when_called_then_data_is_converted_to_units(self, inject_mocks):
        scripts.dispersion.main([get_phonon_file()])
        assert mock_has_method_call(inject_mocks["phonon_mock"], "convert_e_units"), \
            "No call to convert units detected"

    def test_when_called_with_reorder_freqs_then_freqs_are_reordered(self, inject_mocks):
        scripts.dispersion.main([get_phonon_file(), "-reorder"])
        assert mock_has_method_call(inject_mocks["phonon_mock"], "reorder_freqs"), \
            "If -reorder is specified, then the correct function should be called to reorder them"

    def test_when_called_without_reorder_freqs_then_freqs_are_not_reordered(self, inject_mocks):
        scripts.dispersion.main([get_phonon_file()])
        assert not mock_has_method_call(inject_mocks["phonon_mock"], "reorder_freqs"), \
            "If -reorder is not specified, then the function to reorder them should not be called"

    def test_when_called_without_grace_but_save_then_saved(self, inject_mocks):
        filename = "FileName.plot"
        scripts.dispersion.main([get_phonon_file(), "-s={}".format(filename)])
        assert matplotlib.pyplot.savefig.call_args[0][0] == filename, \
            "When specifying -s and a filename, should use matplotlib function to save figure with the given name"

    def test_when_called_without_grace_or_show_then_saved(self, inject_mocks):
        scripts.dispersion.main([get_phonon_file()])
        assert matplotlib.pyplot.show.call_count == 1, \
            "When not specifying -grace or -s, then matplotlib function to show the plot should be called"

    def test_when_called_with_grace_then_grace_and_not_saved_or_shown(self, inject_mocks):
        scripts.dispersion.main([get_phonon_file(), "-grace"])
        assert scripts.dispersion.output_grace.call_count == 1, \
            "When specifying the -grace function, the euphonic output_grace function should be called"

    def test_when_called_with_btol_then_btol_is_used(self, inject_mocks):
        btol = 13.0
        scripts.dispersion.main([get_phonon_file(), "-btol={}".format(btol)])
        assert scripts.dispersion.plot_dispersion.call_args[1]["btol"] == btol

    def test_when_called_with_mev_units_then_mev_is_used(self, inject_mocks):
        units = "meV"
        scripts.dispersion.main([get_phonon_file(), "-units={}".format(units)])
        assert mock_has_method_call(inject_mocks["phonon_mock"], "convert_e_units", units)


@pytest.mark.integration
class TestRegression:

    @pytest.fixture
    def inject_mocks(self, mocker):
        # Prevent calls to show so we can get the current figure using gcf()
        mocker.patch("matplotlib.pyplot.show")

        mocker.resetall()

    def test_plots_produce_expected_xydata(self, inject_mocks):
        scripts.dispersion.main([get_phonon_file()])

        lines = matplotlib.pyplot.gcf().axes[0].lines

        for filenum, file in iter_dispersion_data_files():
            file_contents = np.genfromtxt(file, delimiter=",")
            assert np.array_equal(file_contents, lines[filenum].get_xydata().T)
