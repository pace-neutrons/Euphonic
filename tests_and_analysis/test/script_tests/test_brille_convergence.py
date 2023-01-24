import os
import sys
import json
from unittest.mock import patch

import pytest
import numpy.testing as npt
import numpy as np

from euphonic import Spectrum2D
from tests_and_analysis.test.utils import get_data_path, get_castep_path, get_phonopy_path
from tests_and_analysis.test.script_tests.utils import (
    get_script_test_data_path, get_all_plot_line_data, get_all_figs, args_to_key)

pytestmark = pytest.mark.multiple_extras
# These tests require both Brille and Matplotlib, allow tests with
# multiple_extras marker to be collected and deselected if
# either is not installed
try:
    import matplotlib.pyplot
    import euphonic.cli.brille_convergence
except ModuleNotFoundError:
    pass

graphite_fc_file = get_castep_path('graphite', 'graphite.castep_bin')
nacl_prim_fc_file = get_phonopy_path('NaCl_prim', 'phonopy_nacl.yaml')
brille_conv_output_file = os.path.join(get_script_test_data_path(),
                                              'brille-convergence.json')

quick_calc_params = ['--npts=2', '--brille-npts=10']
brille_conv_params = [
    [graphite_fc_file, *quick_calc_params]]

class TestRegression:

    @pytest.fixture
    def inject_mocks(self, mocker):
        # Prevent calls to show so we can get the current figure using
        # gcf()
        mocker.patch('matplotlib.pyplot.show')
        mocker.resetall()

    def teardown_method(self):
        # Ensure figures are closed
        matplotlib.pyplot.close('all')

    def run_brille_conv_and_test_result(
            self, brille_conv_args):
        euphonic.cli.brille_convergence.main(brille_conv_args)
        figs = get_all_figs()
        all_plot_data = get_all_plot_line_data(figs)

        with open(brille_conv_output_file, 'r') as expected_data_file:
            expected_all_plot_data = json.load(expected_data_file)[
                args_to_key(brille_conv_args)]

        # Loop over expected figures
        for expected_plot_data, plot_data in zip(expected_all_plot_data, all_plot_data):
            for key, expected_val in expected_plot_data.items():
                if key not in ['xy_data', 'x_ticklabels']:
                    assert expected_val == plot_data[key]

#            if key = 'xy_data':
#                # xy_data is indexed (n_axes, n_series, xy_axes, n_points)
#                expected_val = np.array(expected_val)
#                val = np.array(all_plot_data[i][key])
#                if i <= 1: # First 2 plots are residual scatter plots
#                    # x data are Euphonic frequencies
#                    # they are predictable so we can test them
#                    npt.assert_allclose(
#                        expected_val[:, :, 0], val[:, :, 0],
#                        atol=sys.float_info.epsilon)
#                    # Note i = 1 is 3D ax, not sure how to test this yet
#                    # y data are residuals from low density brille grid
                    # hard to predict - just test they are below threshold
                    #assert np.all(val[:, :, 1] < val[:, :, 0])
                # The following plots are 1-D averaged or per-qpt
                # structure factor line plots
#                else:
#                    pass


    @pytest.mark.parametrize('brille_conv_args', brille_conv_params)
    def test_brille_conv_plots(
            self, inject_mocks, brille_conv_args):
        self.run_brille_conv_and_test_result(brille_conv_args)

    @pytest.mark.skip
    @pytest.mark.brille
    @pytest.mark.multiple_extras
    @pytest.mark.parametrize('brille_conv_args, expected_kwargs', [
        (['--use-brille', '--brille-npts', '25', '--disable-c'],
         {'grid_npts': 25, 'grid_type': 'trellis',
          'interpolation_kwargs': {'use_c': False}}),
        (['--use-brille', '--brille-grid-type', 'mesh', '--use-c',
          '--n-threads', '2'],
          {'grid_type': 'mesh', 'interpolation_kwargs': {'use_c': True,
                                                         'n_threads': 2}})
    ])
    def test_brille_interpolator_from_force_constants_kwargs_passed(
            self, inject_mocks, mocker, brille_conv_args, expected_kwargs):
        from euphonic.brille import BrilleInterpolator
        # Stop execution once from_fc has been called - we're only
        # checking here that the correct arguments have been passed
        # through
        class MockException(Exception):
            pass
        mock = mocker.patch.object(BrilleInterpolator, 'from_force_constants',
                                   side_effect=MockException())
        try:
            euphonic.cli.brille_convergence.main(
                [graphite_fc_file] + brille_conv_args + quick_calc_params)
        except MockException:
            pass
        default_interp_kwargs =  {'asr': None, 'dipole_parameter': 1.0,
                                  'n_threads': None, 'use_c': None}
        default_kwargs = {'grid_type': 'trellis', 'grid_npts': 5000,
                          'grid_density': None}
        expected_interp_kwargs = {
            **default_interp_kwargs,
            **expected_kwargs.pop('interpolation_kwargs', {})}
        expected_kwargs = {**default_kwargs, **expected_kwargs}
        expected_kwargs['interpolation_kwargs'] = expected_interp_kwargs
        assert mock.call_args[1] == expected_kwargs


@patch('matplotlib.pyplot.show')
@pytest.mark.skip(reason='Only run if you want to regenerate the test data')
def test_regenerate_brille_conv_data(_):
    # Read from existing file first to allow option of only replacing for
    # certain test cases or keys
    try:
        with open(brille_conv_output_file, 'r') as json_file:
            json_data = json.load(json_file)
    except FileNotFoundError:
        json_data = {}
    for brille_conv_param in (brille_conv_params):
        # Generate figures for us to retrieve with gcf
        euphonic.cli.brille_convergence.main(brille_conv_param)

        figs = get_all_figs()
        all_plot_data = get_all_plot_line_data(figs)
        # Optionally only write certain keys
        keys_to_replace = []
        if len(keys_to_replace) > 0:
            for i, plot_data in enumerate(all_plot_data):
                for key in keys_to_replace:
                    json_data[args_to_key(brille_conv_param)][i][key] = plot_data[key]
        else:
            json_data[args_to_key(brille_conv_param)] = all_plot_data

    with open(brille_conv_output_file, 'w+') as json_file:
        json.dump(json_data, json_file, indent=4)
