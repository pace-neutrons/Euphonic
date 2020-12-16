import json
import os
import sys
from unittest.mock import patch

import pytest
import numpy as np
import numpy.testing as npt
# Required for mocking
import matplotlib.pyplot

from tests_and_analysis.test.utils import get_data_path
from tests_and_analysis.test.script_tests.utils import (
    get_script_test_data_path, get_current_plot_line_data,
    args_to_key)
import euphonic.cli.dos


nah_phonon_file = os.path.join(
    get_data_path(), 'qpoint_phonon_modes', 'NaH', 'NaH.phonon')
quartz_fc_file = os.path.join(
    get_data_path(), 'force_constants', 'quartz', 'quartz.castep_bin')
dos_output_file = os.path.join(get_script_test_data_path(), 'dos.json')
dos_params = [
    [nah_phonon_file],
    [nah_phonon_file, '--energy-broadening=2'],
    [nah_phonon_file, '--ebins=100'],
    [nah_phonon_file, '--energy-broadening=2.3', '--ebins=50'],
    [nah_phonon_file, '--energy-unit=1/cm', '--eb=20', '--e-max=750'],
    [nah_phonon_file, '--eb=5e-5', '--shape=lorentz', '--e-min=1.5e-3',
     '-u=hartree'],
    [nah_phonon_file, '--title=NaH', '--x-label=Energy (meV)',
     '--y-label=DOS'],
    [quartz_fc_file],
    [quartz_fc_file, '--grid', '3', '3', '3']]


@pytest.mark.integration
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

    @pytest.mark.parametrize('dos_args', dos_params)
    def test_plots_contain_expected_data(self, inject_mocks, dos_args):
        euphonic.cli.dos.main(dos_args)

        line_data = get_current_plot_line_data()

        with open(dos_output_file, 'r') as f:
            expected_line_data = json.load(f)[args_to_key(dos_args)]
        for key, value in line_data.items():
            if isinstance(value, list) and isinstance(value[0], float):
                npt.assert_allclose(
                    value, expected_line_data[key],
                    atol=5*sys.float_info.epsilon)
            else:
                assert value == expected_line_data[key]


    @pytest.mark.parametrize('dos_args', [
        [nah_phonon_file, '--save-to'],
        [nah_phonon_file, '-s']])
    def test_plot_save_to_file(self, inject_mocks, tmpdir, dos_args):
        output_file = str(tmpdir.join('test.png'))
        euphonic.cli.dos.main(dos_args + [output_file])
        assert os.path.exists(output_file)

    @pytest.mark.parametrize('dos_args', [
        [os.path.join(get_data_path(), 'structure_factor', 'quartz',
                      'quartz_structure_factor.json')]])
    def test_invalid_file_raises_value_error(self, dos_args):
        with pytest.raises(ValueError):
            euphonic.cli.dos.main(dos_args)


@patch('matplotlib.pyplot.show')
@pytest.mark.skip(reason='Only run if you want to regenerate the test data')
def test_regenerate_dos_data(_):
    json_data = {}
    for dos_param in dos_params:
        # Generate current figure for us to retrieve with gcf
        euphonic.cli.dos.main(dos_param)
        # Retrieve with gcf and write to file
        json_data[args_to_key(dos_param)
                  ] = get_current_plot_line_data()
    with open(dos_output_file, 'w+') as json_file:
        json.dump(json_data, json_file, indent=4)
