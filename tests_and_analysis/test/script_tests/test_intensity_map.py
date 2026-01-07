from contextlib import suppress
import json
from unittest.mock import patch

import numpy.testing as npt
import pytest

from euphonic import Spectrum2D
from tests_and_analysis.test.script_tests.utils import (
    args_to_key,
    get_current_plot_image_data,
    get_script_test_data_path,
)
from tests_and_analysis.test.utils import get_castep_path, get_data_path

pytestmark = pytest.mark.matplotlib
# Allow tests with matplotlib marker to be collected and
# deselected if Matplotlib is not installed
with suppress(ModuleNotFoundError):
    import matplotlib.pyplot  # noqa: ICN001

    import euphonic.cli.intensity_map

quartz_json_file = get_data_path(
    'qpoint_phonon_modes', 'quartz',
    'quartz_bandstructure_qpoint_phonon_modes.json')
quartz_no_evec_json_file = get_data_path(
    'qpoint_frequencies', 'quartz',
    'quartz_bandstructure_qpoint_frequencies.json')
lzo_phonon_file = get_castep_path('LZO', 'La2Zr2O7.phonon')
graphite_fc_file = get_castep_path('graphite', 'graphite.castep_bin')
intensity_map_output_file = get_script_test_data_path('intensity-map.json')
intensity_map_params = [
    [graphite_fc_file],
    [graphite_fc_file, '--scale=-0.1'],
    [graphite_fc_file, '--vmin=0', '--vmax=1e-10'],
    [graphite_fc_file, '--energy-unit=meV'],
    [graphite_fc_file, '-w', 'dos', '--ylabel=DOS', '--title=DOS TITLE'],
    [graphite_fc_file, '--e-min=50', '-u=cm^-1', '--xlabel=wavenumber'],
    [graphite_fc_file, '--e-min=-100', '--e-max=1000', '--ebins=100',
     '--energy-unit=cm^-1'],
    [graphite_fc_file, '--energy-broadening=2e-3', '-u=eV'],
    [graphite_fc_file, '--asr'],
    [graphite_fc_file, '--asr=realspace'],
    [quartz_json_file],
    [quartz_json_file, '--btol=1000'],
    [lzo_phonon_file],
    [quartz_no_evec_json_file],
    [graphite_fc_file, '--weighting=coherent', '--cmap=bone'],
    [graphite_fc_file, '--weighting=coherent', '--temperature=800']]
broadening_warning_expected_params = [
    [graphite_fc_file, '--q-spacing=0.05', '--length-unit=bohr',
     '--q-broadening=0.1'],
    [graphite_fc_file, '--qb=0.01', '--eb=1.5', '--shape=lorentz'],
]


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

    def run_intensity_map_and_test_result(
            self, intensity_map_args):
        euphonic.cli.intensity_map.main(intensity_map_args)

        image_data = get_current_plot_image_data()

        with open(intensity_map_output_file) as expected_data_file:
            expected_image_data = json.load(expected_data_file)[
                args_to_key(intensity_map_args)]

        for key, expected_val in expected_image_data.items():
            if key == 'extent':
                # Lower bound of y-data (energy) varies by up to ~2e-6 on
                # different systems when --asr is used, compared to
                # the upper bound of 100s of meV this is effectively zero,
                # so increase tolerance to allow for this
                npt.assert_allclose(expected_val, image_data[key], atol=2e-6)
            elif key in ['data_1', 'data_2']:
                # Errors of 2-4 epsilon seem to be common when using
                # broadening, so slightly increase tolerance
                npt.assert_allclose(expected_val, image_data[key],
                                    atol=1e-14)
            else:
                assert expected_val == image_data[key]

    @pytest.mark.parametrize('intensity_map_args', intensity_map_params)
    def test_intensity_map_image_data(
            self, inject_mocks, intensity_map_args):
        self.run_intensity_map_and_test_result(intensity_map_args)

    @pytest.mark.parametrize('intensity_map_args',
                             broadening_warning_expected_params)
    def test_intensity_map_image_data_width_warning(
            self, inject_mocks, intensity_map_args):
        with pytest.warns(UserWarning, match='x_data bin widths are not equal'):
            self.run_intensity_map_and_test_result(intensity_map_args)

    @pytest.mark.parametrize('intensity_map_args', [
        [quartz_json_file, '--save-to'],
        [quartz_json_file, '-s']])
    def test_plot_save_to_file(self, inject_mocks, tmp_path, intensity_map_args):
        output_file = tmp_path / 'test.png'
        euphonic.cli.intensity_map.main([*intensity_map_args, str(output_file)])
        assert output_file.exists()

    @pytest.mark.parametrize('intensity_map_args', [
        [quartz_json_file, '--save-json']])
    def test_plot_save_to_json(self, inject_mocks, tmp_path, intensity_map_args):
        output_file = tmp_path / 'test.json'
        euphonic.cli.intensity_map.main([*intensity_map_args, str(output_file)])
        spec = Spectrum2D.from_json_file(output_file)
        assert isinstance(spec, Spectrum2D)

    @pytest.mark.parametrize('intensity_map_args', [
        [get_data_path('util', 'qgrid_444.txt')]])
    def test_invalid_file_raises_value_error(self, intensity_map_args):
        with pytest.raises(ValueError):
            euphonic.cli.intensity_map.main(intensity_map_args)

    @pytest.mark.parametrize('intensity_map_args', [
        [quartz_no_evec_json_file, '--weighting', 'coherent']])
    def test_qpoint_frequencies_incompatible_args_raises_type_error(
            self, intensity_map_args):
        with pytest.raises(TypeError, match='Eigenvectors are required'):
            euphonic.cli.intensity_map.main(intensity_map_args)

    @pytest.mark.parametrize('intensity_map_args', [
        [quartz_json_file, '--weighting', 'coherent',
         '--temperature', '300']])
    def test_qpoint_modes_debyewaller_raises_type_error(
            self, intensity_map_args):
        with pytest.raises(TypeError, match='Force constants data'):
            euphonic.cli.intensity_map.main(intensity_map_args)

    @pytest.mark.parametrize('intensity_map_args', [
        [quartz_json_file, '--qb=0.01']])
    def test_broaden_unequal_q_axis_emits_user_warning(
            self, inject_mocks, intensity_map_args):
        with pytest.warns(UserWarning):
            euphonic.cli.intensity_map.main(intensity_map_args)

    # Until --pdos is implemented for intensity-map, check we haven't
    # accidentally allowed it as an argument
    @pytest.mark.parametrize('intensity_map_args', [
        [quartz_json_file, '-w=dos', '--pdos=Si']])
    def test_pdos_raises_causes_exit(self, intensity_map_args):
        with pytest.raises(SystemExit) as err:
            euphonic.cli.intensity_map.main(intensity_map_args)
        assert err.value.code == 2

    @pytest.mark.parametrize('intensity_map_args', [
        [quartz_json_file, '-w=incoherent-dos']])
    def test_invalid_weighting_raises_causes_exit(self, intensity_map_args):
        # Argparse should call sys.exit on invalid choices
        with pytest.raises(SystemExit) as err:
            euphonic.cli.intensity_map.main(intensity_map_args)
        assert err.value.code == 2


@patch('matplotlib.pyplot.show')
@pytest.mark.skip(reason='Only run if you want to regenerate the test data')
def test_regenerate_intensity_map_data(_):
    # Read from existing file first to allow option of only replacing for
    # certain test cases or keys
    try:
        with open(intensity_map_output_file) as json_file:
            json_data = json.load(json_file)
    except FileNotFoundError:
        json_data = {}

    for intensity_map_param in intensity_map_params + broadening_warning_expected_params:
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
