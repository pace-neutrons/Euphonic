import pytest
import os
import math
# Required for mocking
from random import random
import time
from unittest.mock import Mock
from euphonic import ForceConstants
from euphonic.cli.optimise_dipole_parameter import calculate_optimum_dipole_parameter
from tests_and_analysis.test.utils import get_castep_path
import euphonic.cli.optimise_dipole_parameter


quartz_castep_bin = get_castep_path("quartz", "quartz.castep_bin")


class SharedCode:

    @staticmethod
    def get_lowest_time_per_qpt_and_index(dipole_parameters_time_per_qpts):
        # Search dipole_parameters_time_per_qpts for lowest time
        lowest_time_per_qpt = math.inf
        lowest_time_per_qpt_index = None
        for index, time_per_qpt in enumerate(dipole_parameters_time_per_qpts):
            if time_per_qpt < lowest_time_per_qpt:
                lowest_time_per_qpt = time_per_qpt
                lowest_time_per_qpt_index = index
        return lowest_time_per_qpt, lowest_time_per_qpt_index

    @staticmethod
    def get_calculate_optimum_dipole_parameter_kwargs():
        return [
            {},
            {"dipole_parameter_min": 0.25, "dipole_parameter_max": 1.35,
             "dipole_parameter_step": 0.2, "n": 20},
            {"dipole_parameter_min": 0.5, "dipole_parameter_max": 1.75,
             "dipole_parameter_step": 0.3, "n": 10}
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
    @pytest.fixture(
        params=SharedCode.get_calculate_optimum_dipole_parameter_kwargs())
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
        return calculate_optimum_dipole_parameter(quartz_castep_bin, **kwargs)

    def test_optimal_has_lowest_time_per_qpt(
            self, call_with_params_and_quartz_unit):
        # Unpack data
        optimal_dipole_parameter = call_with_params_and_quartz_unit[0]
        optimal_time_per_qpt = call_with_params_and_quartz_unit[2]
        dipole_parameters = call_with_params_and_quartz_unit[3]
        dipole_parameters_time_per_qpts = call_with_params_and_quartz_unit[5]

        lowest_time_per_qpt, lowest_time_per_qpt_index = \
            SharedCode.get_lowest_time_per_qpt_and_index(
                dipole_parameters_time_per_qpts)

        # Check optimal time per q-point is the lowest detected q-point
        assert optimal_time_per_qpt == lowest_time_per_qpt
        # Check dipole_parameter from lowest detected q-point index matches
        # our optimal q-point
        assert dipole_parameters[
            lowest_time_per_qpt_index] == optimal_dipole_parameter


@pytest.mark.integration
class TestRegression:

    def test_optimal_is_0_75(self):
        optimal_dipole_parameter = calculate_optimum_dipole_parameter(
            quartz_castep_bin)[0]
        assert optimal_dipole_parameter == 0.75

    def test_time_per_qpt_with_use_c_is_lower_than_disable_c(self):
        kwargs = {'n': 10, 'dipole_parameter_min': 0.5,
                  'dipole_parameter_max': 0.5}
        optimum_time_qpt_c = calculate_optimum_dipole_parameter(
            quartz_castep_bin, use_c=True, **kwargs)[2]
        optimum_time_qpt_noc = calculate_optimum_dipole_parameter(
            quartz_castep_bin, use_c=False, **kwargs)[2]
        assert optimum_time_qpt_c < optimum_time_qpt_noc

    def test_fc_with_no_born_emits_user_warning(self):
        with pytest.warns(UserWarning):
            euphonic.cli.optimise_dipole_parameter.main([get_castep_path(
                'LZO', 'La2Zr2O7.castep_bin')])



@pytest.mark.integration
class TestIntegration:

    # Run the fixture for multiple permutations of arguments
    @pytest.fixture(
        params=SharedCode.get_calculate_optimum_dipole_parameter_kwargs())
    def call_with_params_and_quartz_integration(self, request):
        kwargs = request.param
        return calculate_optimum_dipole_parameter(quartz_castep_bin, **kwargs)

    def test_optimal_has_lowest_time_per_qpt(
            self, call_with_params_and_quartz_integration):
        # Unpack data
        optimal_dipole_parameter = call_with_params_and_quartz_integration[0]
        optimal_time_per_qpt = call_with_params_and_quartz_integration[2]
        dipole_parameters = call_with_params_and_quartz_integration[3]
        dipole_parameters_time_per_qpts = \
            call_with_params_and_quartz_integration[5]

        lowest_time_per_qpt, lowest_time_per_qpt_index = \
            SharedCode.get_lowest_time_per_qpt_and_index(
                dipole_parameters_time_per_qpts)

        # Check optimal time per q-point is the lowest detected q-point
        assert optimal_time_per_qpt == lowest_time_per_qpt
        # Check dipole_parameter from lowest detected q-point index matches
        # our optimal q-point
        assert dipole_parameters[
            lowest_time_per_qpt_index] == optimal_dipole_parameter
