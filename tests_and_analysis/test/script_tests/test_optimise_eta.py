import os
import math
import pytest
from scripts.optimise_eta import calculate_optimum_eta
from tests_and_analysis.test.utils import get_data_path

quartz_castep_bin = os.path.join(get_data_path(), "interpolation", "quartz", "quartz")


@pytest.fixture
def call_with_defaults_and_quartz():
    return calculate_optimum_eta(quartz_castep_bin)


# Regression test
@pytest.mark.integration
def test_optimal_is_0_75(call_with_defaults_and_quartz):
    optimal_eta = call_with_defaults_and_quartz[0]
    assert optimal_eta == 0.75


# Create multiple fixtures to call with multiple selections of keyword args
@pytest.fixture(params=[
    {},
    {"eta_min": 0.25, "eta_max": 1.35, "eta_step": 0.2, "n": 20},
    {"eta_min": -0.25, "eta_max": 1.75, "eta_step": 0.2, "n": 20}
])
def call_with_params_and_quartz(request):
    kwargs = request.param
    return calculate_optimum_eta(quartz_castep_bin, **kwargs)


def get_lowest_time_per_qpt_and_index(etas_time_per_qpts):
    # Search etas_time_per_qpts for lowest time
    lowest_time_per_qpt = math.inf
    lowest_time_per_qpt_index = None
    for index, time_per_qpt in enumerate(etas_time_per_qpts):
        if time_per_qpt < lowest_time_per_qpt:
            lowest_time_per_qpt = time_per_qpt
            lowest_time_per_qpt_index = index
    return lowest_time_per_qpt, lowest_time_per_qpt_index


@pytest.mark.integration
def test_optimal_has_lowest_time_per_qpt(call_with_params_and_quartz):
    # Unpack data
    optimal_eta = call_with_params_and_quartz[0]
    optimal_time_per_qpt = call_with_params_and_quartz[2]
    etas = call_with_params_and_quartz[3]
    etas_time_per_qpts = call_with_params_and_quartz[5]

    lowest_time_per_qpt, lowest_time_per_qpt_index = \
        get_lowest_time_per_qpt_and_index(etas_time_per_qpts)

    # Check optimal time per q-point is the lowest detected q-point
    assert optimal_time_per_qpt == lowest_time_per_qpt
    # Check eta from lowest detected q-point index matches our optimal q-point
    assert etas[lowest_time_per_qpt_index] == optimal_eta
