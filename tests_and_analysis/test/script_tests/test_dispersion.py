import sys
import os
import json
from unittest.mock import patch
from platform import platform

import pytest
import numpy.testing as npt
from packaging import version
from scipy import __version__ as scipy_ver

from tests_and_analysis.test.utils import (
    get_data_path, get_castep_path, get_phonopy_path)
from tests_and_analysis.test.script_tests.utils import (
    get_script_test_data_path, get_current_plot_line_data, args_to_key)

pytestmark = pytest.mark.matplotlib
# Allow tests with matplotlib marker to be collected and
# deselected if Matplotlib is not installed
try:
    import matplotlib.pyplot
    import euphonic.cli.dispersion
except ModuleNotFoundError:
    pass

cahgo2_fc_file = get_phonopy_path('CaHgO2', 'mp-7041-20180417.yaml')
lzo_fc_file = os.path.join(
    get_data_path(), 'force_constants', 'LZO_force_constants.json')
nacl_fc_file = get_phonopy_path('NaCl_cli_test', 'force_constants.hdf5')
nacl_phonon_file = os.path.join(
    get_phonopy_path('NaCl', 'band'), 'band.yaml')
nacl_phonon_hdf5_file = os.path.join(
    get_phonopy_path('NaCl', 'band'), 'band.hdf5')
quartz_phonon_file = os.path.join(
    get_data_path(), 'qpoint_phonon_modes', 'quartz',
    'quartz_bandstructure_qpoint_phonon_modes.json')
disp_output_file = os.path.join(get_script_test_data_path(), 'dispersion.json')
disp_params =  [
    [lzo_fc_file],
    [quartz_phonon_file],
    [quartz_phonon_file, '--btol=1000']]
disp_params_from_phonopy =  [
    [cahgo2_fc_file],
    [cahgo2_fc_file, '--energy-unit=hartree'],
    [cahgo2_fc_file, '--x-label=wavenumber', '--y-label=Energy (meV)',
     '--title=CaHgO2'],
    [cahgo2_fc_file, '-u=1/cm', '--e-min=200'],
    [cahgo2_fc_file, '--e-min=30', '--e-max=100'],
    [cahgo2_fc_file, '--length-unit=bohr', '--q-spacing=0.04'],
    [cahgo2_fc_file, '--q-spacing=0.02'],
    [cahgo2_fc_file, '--asr'],
    [cahgo2_fc_file, '--asr=realspace'],
    [nacl_fc_file],
    [nacl_phonon_file],
    [nacl_phonon_hdf5_file]]
disp_params_macos_segfault =  [[cahgo2_fc_file, '--reorder']]


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

    def run_dispersion_and_test_result(self, dispersion_args):
        euphonic.cli.dispersion.main(dispersion_args)

        line_data = get_current_plot_line_data()

        with open(disp_output_file, 'r') as f:
            expected_line_data = json.load(f)[args_to_key(dispersion_args)]
        # Increase tolerance if asr present - can give slightly
        # different results with different libs
        if any(['--asr' in arg for arg in dispersion_args]):
            atol = 5e-6
        else:
            atol = sys.float_info.epsilon
        for key, value in line_data.items():
            if key == 'xy_data':
                # numpy can only auto convert 2D lists - xy_data has
                # dimensions (n_lines, 2, n_points) so check in a loop
                for idx, line in enumerate(value):
                    npt.assert_allclose(
                        line, expected_line_data[key][idx],
                        atol=atol)
            else:
                assert value == expected_line_data[key]


    @pytest.mark.parametrize('dispersion_args', disp_params)
    def test_dispersion_plot_data(self, inject_mocks, dispersion_args):
        self.run_dispersion_and_test_result(dispersion_args)

    @pytest.mark.phonopy_reader
    @pytest.mark.parametrize('dispersion_args', disp_params_from_phonopy)
    def test_dispersion_plot_data_from_phonopy(
            self, inject_mocks, dispersion_args):
        self.run_dispersion_and_test_result(dispersion_args)

    @pytest.mark.phonopy_reader
    @pytest.mark.parametrize('dispersion_args', disp_params_macos_segfault)
    @pytest.mark.skipif(
        (any([s in platform() for s in ['Darwin', 'macOS']])
         and version.parse(scipy_ver) > version.parse('1.1.0')),
        reason=('Segfaults on some MacOS platforms with Scipy > 1.1.0, may '
                'be related to https://github.com/google/jax/issues/432'))
    def test_dispersion_plot_data_macos_segfault(
            self, inject_mocks, dispersion_args):
        self.run_dispersion_and_test_result(dispersion_args)

    @pytest.mark.parametrize('dispersion_args', [
        [quartz_phonon_file, '--save-to'],
        [quartz_phonon_file, '-s']])
    def test_plot_save_to_file(self, inject_mocks, tmpdir, dispersion_args):
        output_file = str(tmpdir.join('test.png'))
        euphonic.cli.dispersion.main(dispersion_args + [output_file])
        assert os.path.exists(output_file)

    @pytest.mark.parametrize('dispersion_args', [
        [os.path.join(get_data_path(), 'crystal', 'crystal_LZO.json')],
        [os.path.join(get_data_path(), 'force_constants', 'NaCl',
                      'FORCE_CONSTANTS')]])
    def test_invalid_file_raises_value_error(self, dispersion_args):
        with pytest.raises(ValueError):
            euphonic.cli.dispersion.main(dispersion_args)


@patch('matplotlib.pyplot.show')
@pytest.mark.skip(reason='Only run if you want to regenerate the test data')
def test_regenerate_disp_data(_):
    json_data = {}
    for disp_param in disp_params:
        # Generate current figure for us to retrieve with gcf
        euphonic.cli.dispersion.main(disp_param)
        # Retrieve with gcf and write to file
        json_data[args_to_key(disp_param)
                  ] = get_current_plot_line_data()
    with open(disp_output_file, 'w+') as json_file:
        json.dump(json_data, json_file, indent=4)
