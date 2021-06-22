import os
import json
from unittest.mock import patch

import pytest
import numpy.testing as npt
# Required for mocking
import matplotlib.pyplot

from tests_and_analysis.test.utils import get_data_path, get_castep_path
from tests_and_analysis.test.script_tests.utils import (
    get_script_test_data_path, get_current_plot_image_data, args_to_key)
import euphonic.cli.intensity_map


quartz_phonon_file = os.path.join(
    get_data_path(), 'qpoint_phonon_modes', 'quartz',
    'quartz_bandstructure_qpoint_phonon_modes.json')
lzo_phonon_file = get_castep_path('LZO', 'La2Zr2O7.phonon')
graphite_fc_file = get_castep_path('graphite', 'graphite.castep_bin')
intensity_map_output_file = os.path.join(get_script_test_data_path(),
                                         'intensity-map.json')
intensity_map_params = [
    [graphite_fc_file],
    [graphite_fc_file, '--v-min=0', '--v-max=1e-10'],
    [graphite_fc_file, '--energy-unit=meV'],
    [graphite_fc_file, '--weights=coherent', '--cmap=bone'],
    [graphite_fc_file, '--weights=coherent', '--temperature=800'],
    [graphite_fc_file, '-w', 'dos', '--y-label=DOS', '--title=DOS TITLE'],
    [graphite_fc_file, '--e-min=50', '-u=cm^-1', '--x-label=wavenumber'],
    [graphite_fc_file, '--e-min=-100', '--e-max=1000', '--ebins=100',
     '--energy-unit=cm^-1'],
    [graphite_fc_file, '--energy-broadening=2e-3', '-u=eV'],
    [graphite_fc_file, '--q-spacing=0.05', '--length-unit=bohr',
     '--q-broadening=0.1'],
    [graphite_fc_file, '--qb=0.01', '--eb=1.5', '--shape=lorentz'],
    [graphite_fc_file, '--asr'],
    [graphite_fc_file, '--asr=realspace'],
    [quartz_phonon_file],
    [quartz_phonon_file, '--btol=1000'],
    [lzo_phonon_file]]


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

    @pytest.mark.parametrize('intensity_map_args', intensity_map_params)
    def test_plots_produce_expected_image(
            self, inject_mocks, intensity_map_args):
        euphonic.cli.intensity_map.main(intensity_map_args)

        image_data = get_current_plot_image_data()

        with open(intensity_map_output_file, 'r') as expected_data_file:
            expected_data_key = args_to_key(intensity_map_args)
            expected_image_data = json.load(
                expected_data_file)[expected_data_key]
        for key, value in image_data.items():
            if key == 'extent':
                # Lower bound of y-data (energy) varies by up to ~2e-6 on
                # different systems when --asr is used, compared to
                # the upper bound of 100s of meV this is effectively zero,
                # so increase tolerance to allow for this
                npt.assert_allclose(value, expected_image_data[key], atol=2e-6)
            elif isinstance(value, list) and isinstance(value[0], float):
                # Errors of 2-4 epsilon seem to be common when using
                # broadening, so slightly increase tolerance
                npt.assert_allclose(value, expected_image_data[key],
                                    atol=1e-14)
            else:
                assert value == expected_image_data[key]

    @pytest.mark.parametrize('intensity_map_args', [
        [quartz_phonon_file, '--save-to'],
        [quartz_phonon_file, '-s']])
    def test_plot_save_to_file(self, inject_mocks, tmpdir, intensity_map_args):
        output_file = str(tmpdir.join('test.png'))
        euphonic.cli.intensity_map.main(intensity_map_args + [output_file])
        assert os.path.exists(output_file)

    @pytest.mark.parametrize('intensity_map_args', [
        [os.path.join(get_data_path(), 'util', 'qgrid_444.txt')]])
    def test_invalid_file_raises_value_error(self, intensity_map_args):
        with pytest.raises(ValueError):
            euphonic.cli.intensity_map.main(intensity_map_args)

    @pytest.mark.parametrize('intensity_map_args', [
        [quartz_phonon_file, '-w=incoherent']])
    def test_invalid_weights_raises_causes_exit(self, intensity_map_args):
        # Argparse should call sys.exit on invalid choices
        with pytest.raises(SystemExit) as err:
            euphonic.cli.intensity_map.main(intensity_map_args)
        assert err.type == SystemExit
        assert err.value.code == 2


@patch('matplotlib.pyplot.show')
@pytest.mark.skip(reason='Only run if you want to regenerate the test data')
def test_regenerate_intensity_map_data(_):
    # Read from existing file first to allow option of only replacing for
    # certain test cases or keys
    try:
        with open(intensity_map_output_file, 'r') as json_file:
            json_data = json.load(json_file)
    except FileNotFoundError:
        json_data = {}

    for intensity_map_param in intensity_map_params:
        # Generate current figure for us to retrieve with gcf
        euphonic.cli.intensity_map.main(intensity_map_param)

        # Retrieve with gcf and write to file
        image_data = get_current_plot_image_data()
        # Optionally only write certain keys
        keys_to_replace = []
        if len(keys_to_replace) > 0:
            for key in keys_to_replace:
                json_data[args_to_key(intensity_map_param)][key] = image_data[key]
        else:
            json_data[args_to_key(intensity_map_param)] = image_data

    with open(intensity_map_output_file, 'w+') as json_file:
        json.dump(json_data, json_file, indent=4)
