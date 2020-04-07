import pytest
import os
from unittest.mock import MagicMock
from typing import Dict
# Required for mocking
import matplotlib.pyplot

from ..utils import get_data_path

import scripts.dispersion


class SharedCode:

    @staticmethod
    def get_bands_file():
        return os.path.join(get_data_path(), "NaH.bands")

    @staticmethod
    def get_phonon_file():
        return os.path.join(get_data_path(), "NaH.phonon")


class MockArgsRecorder:

    def __init__(self):
        self.args = None
        self.kwargs = None
        self.called_funcs = []

    def record_args(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def func_called(self, func):
        print("Mock call checked: {}".format(func))
        return func in self.called_funcs

    def reset(self):
        self.args = None
        self.kwargs = None
        self.called_funcs = []

    def __getattr__(self, item):
        print("Mock call: {}".format(item))
        self.called_funcs.append(item)
        return self.record_args


@pytest.mark.unit
class TestUnit:

    @pytest.fixture
    def inject_mocks(self, monkeypatch):
        phonon_mock = MockArgsRecorder()
        phonon_mock_args_recorder = MockArgsRecorder()

        def get_phonon_mock(*args, **kwargs):
            phonon_mock_args_recorder.from_castep(*args, **kwargs)
            return phonon_mock
        monkeypatch.setattr(scripts.dispersion.PhononData, "from_castep", get_phonon_mock)

        bands_mock = MockArgsRecorder()
        bands_mock_args_recorder = MockArgsRecorder()

        def get_bands_mock(*args, **kwargs):
            bands_mock_args_recorder.from_castep(*args, **kwargs)
            return bands_mock
        monkeypatch.setattr(scripts.dispersion.BandsData, "from_castep", get_bands_mock)

        output_grace_mock = MockArgsRecorder()

        def call_output_grace_mock(*args, **kwargs):
            output_grace_mock.output_grace(*args, **kwargs)
        monkeypatch.setattr(scripts.dispersion, "output_grace", call_output_grace_mock)

        plot_dispersion_args_recorder = MockArgsRecorder()
        plot_dispersion_mock = MagicMock()

        def get_plot_dispersion_mock(*args, **kwargs):
            plot_dispersion_args_recorder.plot_dispersion(*args, **kwargs)
            return plot_dispersion_mock
        monkeypatch.setattr(scripts.dispersion, "plot_dispersion", get_plot_dispersion_mock)

        matplotlib_mock = MockArgsRecorder()
        monkeypatch.setattr("matplotlib.pyplot", matplotlib_mock)

        mocks: Dict[str, MockArgsRecorder] = {
            "phonon_mock": phonon_mock, "phonon_mock_args_recorder": phonon_mock_args_recorder,
            "bands_mock": bands_mock, "bands_mock_args_recorder": bands_mock_args_recorder,
            "output_grace_mock": output_grace_mock,
            "plot_dispersion_args_recorder": plot_dispersion_args_recorder,
            "matplotlib_mock": matplotlib_mock
        }

        for mock in mocks.values():
            mock.reset()
        print("Mock calls reset")

        return mocks

    @pytest.mark.parametrize("data_file", [SharedCode.get_bands_file(), SharedCode.get_phonon_file()])
    def test_when_not_called_with_up_and_down_both_are_true(self, inject_mocks, data_file):
        plot_dispersion_args_recorder = inject_mocks["plot_dispersion_args_recorder"]
        scripts.dispersion.main([data_file])
        assert plot_dispersion_args_recorder.kwargs["up"]
        assert plot_dispersion_args_recorder.kwargs["down"]

    @pytest.mark.parametrize("data_file", [SharedCode.get_bands_file(), SharedCode.get_phonon_file()])
    def test_when_called_with_up_then_down_is_false(self, inject_mocks, data_file):
        plot_dispersion_args_recorder = inject_mocks["plot_dispersion_args_recorder"]
        scripts.dispersion.main([data_file, "-up"])
        assert plot_dispersion_args_recorder.kwargs["up"]
        assert not plot_dispersion_args_recorder.kwargs["down"]

    @pytest.mark.parametrize("data_file", [SharedCode.get_bands_file(), SharedCode.get_phonon_file()])
    def test_when_called_with_down_then_up_is_false(self, inject_mocks, data_file):
        plot_dispersion_args_recorder = inject_mocks["plot_dispersion_args_recorder"]
        scripts.dispersion.main([data_file, "-down"])
        assert not plot_dispersion_args_recorder.kwargs["up"]
        assert plot_dispersion_args_recorder.kwargs["down"]

    @pytest.mark.parametrize("data_mock_args_recorder_name, other_mock_args_recorder_name, data_file", [
        ("bands_mock_args_recorder", "phonon_mock_args_recorder", SharedCode.get_bands_file()),
        ("phonon_mock_args_recorder", "bands_mock_args_recorder", SharedCode.get_phonon_file())
    ])
    def test_when_called_with_data_file_then_correct_data_is_loaded(self, inject_mocks,
                                                                    data_mock_args_recorder_name,
                                                                    other_mock_args_recorder_name,
                                                                    data_file):
        scripts.dispersion.main([data_file])
        assert inject_mocks[data_mock_args_recorder_name].func_called("from_castep")
        assert not inject_mocks[other_mock_args_recorder_name].func_called("from_castep")

    @pytest.mark.parametrize("data_mock_name, data_file", [
        ("bands_mock", SharedCode.get_bands_file()),
        ("phonon_mock", SharedCode.get_phonon_file())
    ])
    def test_when_called_then_data_is_converted_to_units(self, inject_mocks, data_mock_name, data_file):
        data_mock = inject_mocks[data_mock_name]
        scripts.dispersion.main([data_file])
        assert data_mock.func_called("convert_e_units")

    @pytest.mark.parametrize("data_mock_name, data_file", [
        ("bands_mock", SharedCode.get_bands_file()),
        ("phonon_mock", SharedCode.get_phonon_file())
    ])
    def test_when_called_with_reorder_freqs_then_freqs_are_reordered(self, inject_mocks, data_mock_name, data_file):
        data_mock = inject_mocks[data_mock_name]
        scripts.dispersion.main([data_file, "-reorder"])
        assert data_mock.func_called("reorder_freqs")

    @pytest.mark.parametrize("data_file", [SharedCode.get_bands_file(), SharedCode.get_phonon_file()])
    def test_when_called_without_grace_but_save_then_saved(self, inject_mocks, data_file):
        matplotlib_mock = inject_mocks["matplotlib_mock"]
        scripts.dispersion.main([data_file, "-s FileName.plot"])
        assert matplotlib_mock.func_called("savefig")

    @pytest.mark.parametrize("data_file", [SharedCode.get_bands_file(), SharedCode.get_phonon_file()])
    def test_when_called_without_grace_or_show_then_saved(self, inject_mocks, data_file):
        matplotlib_mock = inject_mocks["matplotlib_mock"]
        scripts.dispersion.main([data_file])
        assert matplotlib_mock.func_called("show")

    @pytest.mark.parametrize("data_file", [SharedCode.get_bands_file(), SharedCode.get_phonon_file()])
    def test_when_called_with_grace_then_grace_and_not_saved_or_shown(self, inject_mocks, data_file):
        output_grace_mock = inject_mocks["output_grace_mock"]
        scripts.dispersion.main([data_file, "-grace"])
        assert output_grace_mock.func_called("output_grace")
