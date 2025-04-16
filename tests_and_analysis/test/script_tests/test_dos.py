import json
import os
import sys
from unittest.mock import patch

import numpy.testing as npt
import pytest

from euphonic import Spectrum1D
from euphonic.cli.utils import _get_pdos_weighting
from tests_and_analysis.test.script_tests.utils import (
    args_to_key,
    get_plot_line_data,
    get_script_test_data_path,
)
from tests_and_analysis.test.utils import (
    get_castep_path,
    get_data_path,
    get_phonopy_path,
)

pytestmark = pytest.mark.matplotlib
# Allow tests with matplotlib marker to be collected and
# deselected if Matplotlib is not installed
try:
    import matplotlib.pyplot  # noqa: ICN001

    import euphonic.cli.dos
except ModuleNotFoundError:
    pass

nah_phonon_file = get_castep_path('NaH', 'NaH.phonon')
nacl_no_evec_yaml_file = get_phonopy_path('NaCl', 'mesh', 'mesh_no_evec.yaml')
quartz_fc_file = get_castep_path('quartz', 'quartz.castep_bin')
dos_output_file = get_script_test_data_path('dos.json')
dos_params = [
    [nah_phonon_file],
    [nah_phonon_file, '--energy-broadening=2'],
    [nah_phonon_file, '--ebins=100'],
    [nah_phonon_file, '--ebins=100', '--scale=1e2'],
    [nah_phonon_file, '--energy-broadening=2.3', '--ebins=50'],
    [nah_phonon_file, '--instrument-broadening', '1.', '0.1', '-0.001',
     '--ebins=100', '--shape', 'gauss'],
    [nah_phonon_file, '--energy-unit=1/cm', '--eb=20', '--e-max=750'],
    [nah_phonon_file, '--eb=5e-5', '--shape=lorentz', '--e-min=1.5e-3',
     '-u=hartree'],
    [nah_phonon_file, '--title=NaH', '--x-label=Energy (meV)',
     '--y-label=DOS'],
    [nah_phonon_file, '--weighting=coherent-plus-incoherent-dos', '--pdos'],
    [nah_phonon_file, '--weighting=coherent-dos', '--pdos', 'Na', 'H'],
    [nah_phonon_file, '--weighting=incoherent-dos', '--pdos', 'Na'],
    [quartz_fc_file, '--grid-spacing=0.1', '--length-unit=bohr'],
    [quartz_fc_file, '--grid', '5', '5', '4'],
    [quartz_fc_file, '--grid', '5', '5', '4', '--adaptive', '--pdos'],
    [quartz_fc_file, '--grid', '5', '5', '4', '--adaptive'],
    [quartz_fc_file, '--grid', '5', '5', '4', '--adaptive', '--eb', '2'],
    [quartz_fc_file, '--grid', '5', '5', '4', '--adaptive',
     '--adaptive-method=fast'],
    [quartz_fc_file, '--grid', '5', '5', '4', '--adaptive',
     '--adaptive-method=fast', '--adaptive-error=0.05']]
dos_params_from_phonopy = [[nacl_no_evec_yaml_file]]

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

    def run_dos_and_test_result(self, dos_args):
        euphonic.cli.dos.main(dos_args)

        line_data = get_plot_line_data()
        # Only use first axis xy_data to save space
        # and avoid regenerating data after refactoring
        line_data['xy_data'] = line_data['xy_data'][0]

        with open(dos_output_file) as f:
            expected_line_data = json.load(f)[args_to_key(dos_args)]
        for key, expected_val in expected_line_data.items():
            # We don't care about the details of tick labels for DOS
            if key == 'x_ticklabels':
                pass
            elif key == 'xy_data':
                npt.assert_allclose(
                    expected_val, line_data[key],
                    atol=5*sys.float_info.epsilon)
            else:
                assert expected_val == line_data[key]

    @pytest.mark.parametrize('dos_args', dos_params)
    def test_dos_plot_data(self, inject_mocks, dos_args):
        self.run_dos_and_test_result(dos_args)

    @pytest.mark.phonopy_reader
    @pytest.mark.multiple_extras
    @pytest.mark.parametrize('dos_args', dos_params_from_phonopy)
    def test_dos_plot_data_from_phonopy(self, inject_mocks, dos_args):
        self.run_dos_and_test_result(dos_args)

    @pytest.mark.parametrize('dos_args', [
        [nah_phonon_file, '--save-to'],
        [nah_phonon_file, '-s']])
    def test_plot_save_to_file(self, inject_mocks, tmpdir, dos_args):
        output_file = str(tmpdir.join('test.png'))
        euphonic.cli.dos.main(dos_args + [output_file])
        assert os.path.exists(output_file)

    @pytest.mark.parametrize('dos_args', [
        [nah_phonon_file, '--save-json']])
    def test_plot_save_to_json(self, inject_mocks, tmpdir, dos_args):
        output_file = str(tmpdir.join('test.json'))
        euphonic.cli.dos.main(dos_args + [output_file])
        spec = Spectrum1D.from_json_file(output_file)
        assert isinstance(spec, Spectrum1D)

    @pytest.mark.parametrize('dos_args', [
        [get_data_path('crystal', 'crystal_LZO.json')]])
    def test_invalid_file_raises_value_error(self, dos_args):
        with pytest.raises(ValueError):
            euphonic.cli.dos.main(dos_args)

    @pytest.mark.phonopy_reader
    @pytest.mark.multiple_extras
    @pytest.mark.parametrize('dos_args', [
        [nacl_no_evec_yaml_file, '--pdos'],
        [nacl_no_evec_yaml_file, '--weighting', 'coherent-dos']])
    def test_qpoint_frequencies_and_incompatible_args_raises_type_error(
            self, dos_args):
        with pytest.raises(TypeError):
            euphonic.cli.dos.main(dos_args)

    def test_qpoint_modes_and_adaptive_raises_type_error(self):
        with pytest.raises(TypeError):
            euphonic.cli.dos.main([nah_phonon_file, '--adaptive'])

    def test_get_pdos_weighting_without_dash_raises_valueerror(self):
        with pytest.raises(ValueError):
            _get_pdos_weighting('coherentdos')


@patch('matplotlib.pyplot.show')
@pytest.mark.skip(reason='Only run if you want to regenerate the test data')
def test_regenerate_dos_data(_):
    # Read from existing file first to allow option of only replacing for
    # certain test cases or keys
    try:
        with open(dos_output_file) as json_file:
            json_data = json.load(json_file)
    except FileNotFoundError:
        json_data = {}

    for dos_param in dos_params + dos_params_from_phonopy:
        # Generate current figure for us to retrieve with gcf
        euphonic.cli.dos.main(dos_param)

        # Retrieve with gcf and write to file
        line_data = get_plot_line_data()
        # Only use first axis xy_data to save space
        # and avoid regenerating data after refactoring
        line_data['xy_data'] = line_data['xy_data'][0]

        # Optionally only write certain keys
        keys_to_replace = []
        if len(keys_to_replace) > 0:
            for key in keys_to_replace:
                json_data[args_to_key(dos_param)][key] = line_data[key]
        else:
            json_data[args_to_key(dos_param)] = line_data

    with open(dos_output_file, 'w+') as json_file:
        json.dump(json_data, json_file, indent=4)
