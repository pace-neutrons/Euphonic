import pytest
import os
import math
# Required for mocking
from random import random
import time
from unittest.mock import Mock
from euphonic import ForceConstants
from euphonic.cli.optimise_dipole_parameter import calculate_optimum_dipole_parameter
from tests_and_analysis.test.utils import get_castep_path, get_phonopy_path
import euphonic.cli.optimise_dipole_parameter


quartz_castep_bin = get_castep_path("quartz", "quartz.castep_bin")
lzo_castep_bin = get_castep_path("LZO", "La2Zr2O7.castep_bin")
nacl_default_yaml = get_phonopy_path("NaCl_default", "phonopy.yaml")
quick_calc_params = ['-n=10', '--min=0.5', '--max=0.5']


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


    @pytest.mark.parametrize('optimise_dipole_parameter_args', [
        [lzo_castep_bin, '--asr=reciprocal', *quick_calc_params],
        [quartz_castep_bin, '--n-threads=2', *quick_calc_params],
        [quartz_castep_bin, '--disable-c', *quick_calc_params]])
    def test_called_without_errors(self, optimise_dipole_parameter_args):
        euphonic.cli.optimise_dipole_parameter.main(
            optimise_dipole_parameter_args)

    def test_reading_nacl_default_reads_born(self):
        # BORN should be read by default so no warning should
        # be raised
        with pytest.warns(None) as record:
            euphonic.cli.optimise_dipole_parameter.main([
                nacl_default_yaml, *quick_calc_params])
        assert len(record) == 0

    def test_fc_with_no_born_emits_user_warning(self):
        with pytest.warns(UserWarning):
            euphonic.cli.optimise_dipole_parameter.main([
                lzo_castep_bin, *quick_calc_params])

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

    @pytest.mark.parametrize('kwargs', [
            {},
            {"dipole_parameter_min": 0.25, "dipole_parameter_max": 1.35,
             "dipole_parameter_step": 0.2, "n": 20},
            {"dipole_parameter_min": 0.5, "dipole_parameter_max": 1.75,
             "dipole_parameter_step": 0.3, "n": 10}
        ])
    def test_optimal_has_lowest_time_per_qpt(
            self, kwargs):
        params = calculate_optimum_dipole_parameter(
            quartz_castep_bin, **kwargs)
        # Unpack data
        optimal_dipole_parameter = params[0]
        optimal_time_per_qpt = params[2]
        dipole_parameters = params[3]
        dipole_parameters_time_per_qpts = params[5]

        lowest_time_per_qpt, lowest_time_per_qpt_index = \
            self.get_lowest_time_per_qpt_and_index(
                dipole_parameters_time_per_qpts)

        # Check optimal time per q-point is the lowest detected q-point
        assert optimal_time_per_qpt == lowest_time_per_qpt
        # Check dipole_parameter from lowest detected q-point index matches
        # our optimal q-point
        assert dipole_parameters[
            lowest_time_per_qpt_index] == optimal_dipole_parameter

