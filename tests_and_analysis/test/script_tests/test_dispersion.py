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


# Record calls and call args in order to assert certain calls with args have been made
class MockCallsRecorder:

    def __init__(self):
        self.calls = []

    def _record_args(self, *args, **kwargs):
        # Overwrite the last_args and last_kwargs with new args and kwargs
        self.calls[-1]["args"] = args
        self.calls[-1]["kwargs"] = kwargs

    @property
    def last_args(self):
        return self.calls[-1]["args"]

    @property
    def last_kwargs(self):
        return self.calls[-1]["kwargs"]

    def func_called(self, func):
        # True if func has been called since the last reset
        for call in self.calls:
            if call["item"] == func:
                return True
        else:
            return False

    def func_called_with(self, func, *args, **kwargs):
        # True if func has been called with the specified args and kwargs
        # since the last reset
        for call in self.calls:
            if call["item"] == func \
                    and all(arg == call["args"][index] for index, arg in enumerate(args)) \
                    and all(call["kwargs"][kwargs] == kwarg for kwarg in kwargs):
                print(call)
                return True
        else:
            return False

    def reset(self):
        # Reset calls and args
        self.calls = []

    def __getattr__(self, item):
        # Record that item has been called and record the arguments
        self.calls.append({"item": item})
        return self._record_args


@pytest.mark.unit
class TestUnit:

    @pytest.fixture
    def inject_mocks(self, monkeypatch):
        # Mock out calls to the phonon data structure
        phonon_mock = MockCallsRecorder()
        phonon_mock_args_recorder = MockCallsRecorder()

        def get_phonon_mock(*args, **kwargs):
            phonon_mock_args_recorder.from_castep(*args, **kwargs)
            return phonon_mock
        monkeypatch.setattr(scripts.dispersion.PhononData, "from_castep", get_phonon_mock)

        # Mock out calls to the bands data structure
        bands_mock = MockCallsRecorder()
        bands_mock_args_recorder = MockCallsRecorder()

        def get_bands_mock(*args, **kwargs):
            bands_mock_args_recorder.from_castep(*args, **kwargs)
            return bands_mock
        monkeypatch.setattr(scripts.dispersion.BandsData, "from_castep", get_bands_mock)

        # Mock out calls to the output_grace function
        output_grace_mock = MockCallsRecorder()

        def call_output_grace_mock(*args, **kwargs):
            output_grace_mock.output_grace(*args, **kwargs)
        monkeypatch.setattr(scripts.dispersion, "output_grace", call_output_grace_mock)

        # Mock out calls to the plot_dispersion function
        plot_dispersion_args_recorder = MockCallsRecorder()
        plot_dispersion_mock = MagicMock()

        def get_plot_dispersion_mock(*args, **kwargs):
            plot_dispersion_args_recorder.plot_dispersion(*args, **kwargs)
            return plot_dispersion_mock
        monkeypatch.setattr(scripts.dispersion, "plot_dispersion", get_plot_dispersion_mock)

        # Mock out calls to pyplot
        matplotlib_mock = MockCallsRecorder()
        monkeypatch.setattr("matplotlib.pyplot", matplotlib_mock)

        # Gather mocks together
        mocks: Dict[str, MockCallsRecorder] = {
            "phonon_mock": phonon_mock, "phonon_mock_args_recorder": phonon_mock_args_recorder,
            "bands_mock": bands_mock, "bands_mock_args_recorder": bands_mock_args_recorder,
            "output_grace_mock": output_grace_mock,
            "plot_dispersion_args_recorder": plot_dispersion_args_recorder,
            "matplotlib_mock": matplotlib_mock
        }

        # Reset mock calls to prevent any confusion in tests
        for mock in mocks.values():
            mock.reset()
        print("Mock calls reset")

        # Inject the mocks
        return mocks

    @pytest.mark.parametrize("data_file", [SharedCode.get_bands_file(), SharedCode.get_phonon_file()])
    def test_when_not_called_with_up_and_down_both_are_true(self, inject_mocks, data_file):
        plot_dispersion_args_recorder = inject_mocks["plot_dispersion_args_recorder"]
        scripts.dispersion.main([data_file])
        assert plot_dispersion_args_recorder.last_kwargs["up"], \
            "Neither -up or -down pass, so both should be set to true"
        assert plot_dispersion_args_recorder.last_kwargs["down"], \
            "Neither -up or -down pass, so both should be set to true"

    @pytest.mark.parametrize("data_file", [SharedCode.get_bands_file(), SharedCode.get_phonon_file()])
    def test_when_called_with_up_then_down_is_false(self, inject_mocks, data_file):
        plot_dispersion_args_recorder = inject_mocks["plot_dispersion_args_recorder"]
        scripts.dispersion.main([data_file, "-up"])
        assert plot_dispersion_args_recorder.last_kwargs["up"], "-up passed as param so up should be true"
        assert not plot_dispersion_args_recorder.last_kwargs["down"], "-up passed as param so down should be false"

    @pytest.mark.parametrize("data_file", [SharedCode.get_bands_file(), SharedCode.get_phonon_file()])
    def test_when_called_with_down_then_up_is_false(self, inject_mocks, data_file):
        plot_dispersion_args_recorder = inject_mocks["plot_dispersion_args_recorder"]
        scripts.dispersion.main([data_file, "-down"])
        assert not plot_dispersion_args_recorder.last_kwargs["up"], "-down passed as param, so up should be false"
        assert plot_dispersion_args_recorder.last_kwargs["down"], "-down passed as param, so down should be true"

    @pytest.mark.parametrize("data_mock_args_recorder_name, other_mock_args_recorder_name, data_file", [
        ("bands_mock_args_recorder", "phonon_mock_args_recorder", SharedCode.get_bands_file()),
        ("phonon_mock_args_recorder", "bands_mock_args_recorder", SharedCode.get_phonon_file())
    ])
    def test_when_called_with_data_file_then_correct_data_is_loaded(self, inject_mocks,
                                                                    data_mock_args_recorder_name,
                                                                    other_mock_args_recorder_name,
                                                                    data_file):
        scripts.dispersion.main([data_file])
        assert inject_mocks[data_mock_args_recorder_name].func_called("from_castep"), \
            "If {} is a .bands file should call into a BandsData object, " \
            "or if not a PhononData object".format(data_file)
        assert not inject_mocks[other_mock_args_recorder_name].func_called("from_castep"), \
            "If {} is a .bands file should call into a BandsData object, " \
            "or if not a PhononData object".format(data_file)

    @pytest.mark.parametrize("data_mock_name, data_file", [
        ("bands_mock", SharedCode.get_bands_file()),
        ("phonon_mock", SharedCode.get_phonon_file())
    ])
    def test_when_called_then_data_is_converted_to_units(self, inject_mocks, data_mock_name, data_file):
        data_mock = inject_mocks[data_mock_name]
        scripts.dispersion.main([data_file])
        assert data_mock.func_called("convert_e_units"), \
            "Should always convert to the correct units, even if no units are passed"

    @pytest.mark.parametrize("data_mock_name, data_file", [
        ("bands_mock", SharedCode.get_bands_file()),
        ("phonon_mock", SharedCode.get_phonon_file())
    ])
    def test_when_called_with_reorder_freqs_then_freqs_are_reordered(self, inject_mocks, data_mock_name, data_file):
        data_mock = inject_mocks[data_mock_name]
        scripts.dispersion.main([data_file, "-reorder"])
        assert data_mock.func_called("reorder_freqs"), \
            "If -reorder is specified, then the correct function should be called to reorder them"

    @pytest.mark.parametrize("data_mock_name, data_file", [
        ("bands_mock", SharedCode.get_bands_file()),
        ("phonon_mock", SharedCode.get_phonon_file())
    ])
    def test_when_called_without_reorder_freqs_then_freqs_are_not_reordered(self, inject_mocks, data_mock_name,
                                                                            data_file):
        data_mock = inject_mocks[data_mock_name]
        scripts.dispersion.main([data_file])
        assert not data_mock.func_called("reorder_freqs"), \
            "If -reorder is not specified, then the function to reorder them should not be called"

    @pytest.mark.parametrize("data_file", [SharedCode.get_bands_file(), SharedCode.get_phonon_file()])
    def test_when_called_without_grace_but_save_then_saved(self, inject_mocks, data_file):
        matplotlib_mock = inject_mocks["matplotlib_mock"]
        filename = "FileName.plot"
        scripts.dispersion.main([data_file, "-s=FileName.plot"])
        assert matplotlib_mock.func_called("savefig"), \
            "When specifying -s and a filename, should use matplotlib function to save figure"
        assert filename in matplotlib_mock.last_args

    @pytest.mark.parametrize("data_file", [SharedCode.get_bands_file(), SharedCode.get_phonon_file()])
    def test_when_called_without_grace_or_show_then_saved(self, inject_mocks, data_file):
        matplotlib_mock = inject_mocks["matplotlib_mock"]
        scripts.dispersion.main([data_file])
        assert matplotlib_mock.func_called("show"), \
            "When not specifying -grace or -s, then matplotlib function to show the plot should be called"

    @pytest.mark.parametrize("data_file", [SharedCode.get_bands_file(), SharedCode.get_phonon_file()])
    def test_when_called_with_grace_then_grace_and_not_saved_or_shown(self, inject_mocks, data_file):
        output_grace_mock = inject_mocks["output_grace_mock"]
        scripts.dispersion.main([data_file, "-grace"])
        assert output_grace_mock.func_called("output_grace"), \
            "When specifying the -grace function, the euphonic output_grace function should be called"

    @pytest.mark.parametrize("data_file", [SharedCode.get_bands_file(), SharedCode.get_phonon_file()])
    def test_when_called_with_btol_then_btol_is_used(self, inject_mocks, data_file):
        plot_dispersion_args_recorder = inject_mocks["plot_dispersion_args_recorder"]
        btol = 13.0
        scripts.dispersion.main([data_file, "-btol={}".format(btol)])
        assert plot_dispersion_args_recorder.last_kwargs["btol"] == btol

    @pytest.mark.parametrize("data_mock_name, data_file", [
        ("bands_mock", SharedCode.get_bands_file()),
        ("phonon_mock", SharedCode.get_phonon_file())
    ])
    def test_when_called_with_mev_units_then_mev_is_used(self, inject_mocks, data_mock_name, data_file):
        data_mock = inject_mocks[data_mock_name]
        units = "meV"
        scripts.dispersion.main([data_file, "-units={}".format(units)])
        assert data_mock.func_called_with("convert_e_units", units)
