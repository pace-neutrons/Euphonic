from contextlib import suppress
import json
import sys
from unittest.mock import patch
import warnings

import numpy as np
import numpy.testing as npt
import pytest

from tests_and_analysis.test.script_tests.utils import (
    args_to_key,
    get_all_figs,
    get_all_plot_line_data,
    get_script_test_data_path,
)
from tests_and_analysis.test.utils import (
    get_castep_path,
    get_phonopy_path,
    ignore_openmp_warning,
)

pytestmark = [pytest.mark.multiple_extras, pytest.mark.brille,
              pytest.mark.matplotlib]
# These tests require both Brille and Matplotlib, allow tests with
# these markers to be collected and deselected if
# either is not installed

# Required for mocking
with suppress(ModuleNotFoundError):
    import matplotlib.pyplot  # noqa: ICN001

graphite_fc_file = get_castep_path('graphite', 'graphite.castep_bin')
nacl_prim_fc_file = get_phonopy_path('NaCl_prim', 'phonopy_nacl.yaml')
brille_conv_output_file = get_script_test_data_path('brille-convergence.json')
quartz_phonon_file = get_castep_path('quartz', 'quartz_split_qpts.phonon')

quick_calc_params = ['--npts=3', '--brille-npts=10']
brille_conv_params = [
    [graphite_fc_file, *quick_calc_params],
    [graphite_fc_file, *quick_calc_params, '--eb=0.5', '--shape=lorentz'],
    [graphite_fc_file, '--npts=3', '--brille-npts', '10', '15'],
    [nacl_prim_fc_file, *quick_calc_params, '-n=2', '--ebins=5', '--e-min=80',
     '--e-max=160', '-u=1/cm']]

@ignore_openmp_warning
class TestRegression:
    @pytest.fixture
    def inject_mocks(self, mocker):
        # Prevent calls to show so we can get the current figure
        # using gcf()
        mocker.patch('matplotlib.pyplot.show')
        mocker.resetall()

    def teardown_method(self):
        # Ensure figures are closed
        matplotlib.pyplot.close('all')

    def test_no_fc_error(self):
        import euphonic.cli.brille_convergence

        with pytest.raises(TypeError, match='Force constants are required'):
            euphonic.cli.brille_convergence.main([quartz_phonon_file])

    def run_brille_conv_and_test_result(self, brille_conv_args):
        import euphonic.cli.brille_convergence

        atol = sys.float_info.epsilon
        euphonic.cli.brille_convergence.main(brille_conv_args)
        figs = get_all_figs()
        all_plot_data = get_all_plot_line_data(figs)

        with open(brille_conv_output_file) as expected_data_file:
            expected_all_plot_data = json.load(expected_data_file)[
                args_to_key(brille_conv_args)]

        for expected_plot_data, plot_data in zip(expected_all_plot_data,
                                                 all_plot_data,
                                                 strict=True):
            for key, expected_val in expected_plot_data.items():
                if key == 'xy_data':
                    # Float values for small statistics hard to check
                    # Just test shape
                    for expected_item, plot_item in zip(
                            expected_val, plot_data[key], strict=True):
                        assert (np.array(expected_item).shape
                                == np.array(plot_item).shape)

                elif key != 'x_ticklabels':
                    # Check titles and ax labels
                    # Don't care about tick labels
                    assert expected_val == plot_data[key]

        # Check specific properties of the different figures
        # 0:  frequency residual vs frequency scatter plot
        # 1:  structure factor residual vs frequency vs sf 3D scatter
        # 2:  has 2 axes - 1D intensity average vs frequency line plot
        #                - intensity residual vs frequency line plot
        # 3+: has 2 axes - 1D intensity at q-point vs frequency line plot
        #                - intensity residual vs frequency line plot
        for i, fig in enumerate(figs):
            # Note: xy_data is indexed (n_axes, n_series, xy_axes, n_points)
            ex_xy_data = [np.array(axd)
                          for axd in expected_all_plot_data[i]['xy_data']]
            xy_data = [np.array(axd) for axd in all_plot_data[i]['xy_data']]

            # x data are Euphonic frequencies or bins - can be tested
            for ax in range(len(xy_data)):
                npt.assert_allclose(ex_xy_data[ax][:, 0],
                                    xy_data[ax][:, 0],
                                    atol=atol)
            if i == 0:
                # Only check frequency residuals plot
                # structure factors are unstable

                # y axis is residual, check is smaller than frequencies
                # themselves on the x axis
                # also ignore acoustic (first 3) frequencies
                assert np.all(
                    xy_data[0][:, -1, 3:] < 0.1*xy_data[0][:, -2, 3:])
            elif i > 1:
                # Check differences between Brille and Euphonic series
                # on first axis match residuals on 2nd axis
                for j in range(len(xy_data[1])):
                    npt.assert_allclose(
                        xy_data[0][j + 1, 1] - xy_data[0][0, 1],
                        xy_data[1][j, 1])

            markers = [ln.get_marker() for ax in fig.axes for ln in ax.lines]
            linestyles = [ln.get_ls() for ax in fig.axes for ln in ax.lines]
            if i < 2:
                assert all(mk == 'x' for mk in markers)  # scatter plots
                assert all(ls == 'None' for ls in linestyles)
            else:
                assert all(mk == 'None' for mk in markers)  # line plots
                assert all(ls == '-' for ls in linestyles)

    @pytest.mark.parametrize('brille_conv_args', brille_conv_params)
    def test_brille_conv_plots(self, inject_mocks, brille_conv_args):
        self.run_brille_conv_and_test_result(brille_conv_args)

    @pytest.mark.parametrize('brille_conv_args, expected_kwargs', [
        (['--brille-npts', '25', '--disable-c'],
         {'grid_npts': 25, 'grid_type': 'trellis',
          'interpolation_kwargs': {'use_c': False}}),
        (['--brille-grid-type', 'mesh', '--use-c',
          '--n-threads', '2', '--asr', 'realspace'],
          {'grid_type': 'mesh', 'interpolation_kwargs': {'asr': 'realspace',
                                                         'use_c': True,
                                                         'n_threads': 2}}),
    ])
    def test_brille_interpolator_from_force_constants_kwargs_passed(
            self,
            inject_mocks,
            mocker,
            brille_conv_args,
            expected_kwargs):
        from euphonic.brille import BrilleInterpolator
        import euphonic.cli.brille_convergence
        # Stop execution once from_fc has been called - we're only
        # checking here that the correct arguments have been passed
        # through
        class MockError(Exception):
            """Throw exception to stop working on Brille interpolator"""

        mock = mocker.patch.object(BrilleInterpolator, 'from_force_constants',
                                   side_effect=MockError())
        with suppress(MockError):
            euphonic.cli.brille_convergence.main(
                [graphite_fc_file, *brille_conv_args])

        default_interp_kwargs =  {'asr': None, 'dipole_parameter': 1.0,
                                  'n_threads': None, 'use_c': None}
        default_kwargs = {'grid_type': 'trellis', 'grid_npts': 5000}
        expected_interp_kwargs = {
            **default_interp_kwargs,
            **expected_kwargs.pop('interpolation_kwargs', {})}
        expected_kwargs = {**default_kwargs, **expected_kwargs}
        expected_kwargs['interpolation_kwargs'] = expected_interp_kwargs
        assert mock.call_args[1] == expected_kwargs


@patch('matplotlib.pyplot.show')
@pytest.mark.skip(reason='Only run if you want to regenerate the test data')
def test_regenerate_brille_conv_data(_):
    import euphonic.cli.brille_convergence

    # Read from existing file first to allow option of only replacing for
    # certain test cases or keys
    try:
        with open(brille_conv_output_file) as json_file:
            json_data = json.load(json_file)
    except FileNotFoundError:
        json_data = {}
    for brille_conv_param in (brille_conv_params):
        # Ensure figures from previous runs are closed
        matplotlib.pyplot.close('all')
        # Generate figures for us to retrieve with gcf
        euphonic.cli.brille_convergence.main(brille_conv_param)

        figs = get_all_figs()
        all_plot_data = get_all_plot_line_data(figs)
        # Optionally only write certain keys
        keys_to_replace = []
        if len(keys_to_replace) > 0:
            for i, plot_data in enumerate(all_plot_data):
                for key in keys_to_replace:
                    json_data[args_to_key(brille_conv_param)][i][key] \
                        = plot_data[key]
        else:
            json_data[args_to_key(brille_conv_param)] = all_plot_data

    with open(brille_conv_output_file, 'w+') as json_file:
        json.dump(json_data, json_file, indent=4)
