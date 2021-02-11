import pytest
import os
import math
# Required for mocking
from random import random
import time
from unittest.mock import Mock
from euphonic import ForceConstants
from euphonic.cli.optimise_eta import calculate_optimum_eta
from tests_and_analysis.test.utils import get_castep_path


quartz_castep_bin = get_castep_path("quartz", "quartz.castep_bin")


class SharedCode:

    @staticmethod
    def get_lowest_time_per_qpt_and_index(etas_time_per_qpts):
        # Search etas_time_per_qpts for lowest time
        lowest_time_per_qpt = math.inf
        lowest_time_per_qpt_index = None
        for index, time_per_qpt in enumerate(etas_time_per_qpts):
            if time_per_qpt < lowest_time_per_qpt:
                lowest_time_per_qpt = time_per_qpt
                lowest_time_per_qpt_index = index
        return lowest_time_per_qpt, lowest_time_per_qpt_index

    @staticmethod
    def get_calculate_optimum_eta_kwargs():
        return [
            {},
            {"eta_min": 0.25, "eta_max": 1.35, "eta_step": 0.2, "n": 20},
            {"eta_min": 0.5, "eta_max": 1.75, "eta_step": 0.3, "n": 10}
        ]


class MockTime:

    def __init__(self):
        self.last_time = 0

    def get_time(self):
        self.last_time += random()
        return self.last_time


@pytest.mark.unit
class TestUnit:

    # Run the fixture for multiple permutations of arguments
    @pytest.fixture(params=SharedCode.get_calculate_optimum_eta_kwargs())
    def call_with_params_and_quartz_unit(self, request, monkeypatch):
        # Mock getting the data from a castep file
        force_constants_mock = Mock()
        monkeypatch.setattr(
            ForceConstants, "from_castep", lambda *args, **kwargs: Mock())

        # Simulate time
        time_manager: MockTime = MockTime()
        monkeypatch.setattr(
            time, "time", lambda *args, **kwargs: time_manager.get_time())

        kwargs = request.param
        return calculate_optimum_eta(quartz_castep_bin, **kwargs)

    def test_optimal_has_lowest_time_per_qpt(
            self, call_with_params_and_quartz_unit):
        # Unpack data
        optimal_eta = call_with_params_and_quartz_unit[0]
        optimal_time_per_qpt = call_with_params_and_quartz_unit[2]
        etas = call_with_params_and_quartz_unit[3]
        etas_time_per_qpts = call_with_params_and_quartz_unit[5]

        lowest_time_per_qpt, lowest_time_per_qpt_index = \
            SharedCode.get_lowest_time_per_qpt_and_index(etas_time_per_qpts)

        # Check optimal time per q-point is the lowest detected q-point
        assert optimal_time_per_qpt == lowest_time_per_qpt
        # Check eta from lowest detected q-point index matches our
        # optimal q-point
        assert etas[lowest_time_per_qpt_index] == optimal_eta


@pytest.mark.integration
class TestRegression:

    @pytest.fixture
    def call_with_defaults_and_quartz_integration(self):
        return calculate_optimum_eta(quartz_castep_bin)

    # Regression test
    def test_optimal_is_0_75(self, call_with_defaults_and_quartz_integration):
        optimal_eta = call_with_defaults_and_quartz_integration[0]
        assert optimal_eta == 0.75


@pytest.mark.integration
class TestIntegration:

    # Run the fixture for multiple permutations of arguments
    @pytest.fixture(params=SharedCode.get_calculate_optimum_eta_kwargs())
    def call_with_params_and_quartz_integration(self, request):
        kwargs = request.param
        return calculate_optimum_eta(quartz_castep_bin, **kwargs)

    def test_optimal_has_lowest_time_per_qpt(
            self, call_with_params_and_quartz_integration):
        # Unpack data
        optimal_eta = call_with_params_and_quartz_integration[0]
        optimal_time_per_qpt = call_with_params_and_quartz_integration[2]
        etas = call_with_params_and_quartz_integration[3]
        etas_time_per_qpts = call_with_params_and_quartz_integration[5]

        lowest_time_per_qpt, lowest_time_per_qpt_index = \
            SharedCode.get_lowest_time_per_qpt_and_index(etas_time_per_qpts)

        # Check optimal time per q-point is the lowest detected q-point
        assert optimal_time_per_qpt == lowest_time_per_qpt
        # Check eta from lowest detected q-point index matches our
        # optimal q-point
        assert etas[lowest_time_per_qpt_index] == optimal_eta
