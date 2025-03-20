import os
import json
from unittest.mock import patch

import pytest
import numpy.testing as npt

from euphonic import Spectrum2D
from tests_and_analysis.test.utils import get_data_path, get_castep_path, get_phonopy_path
from tests_and_analysis.test.script_tests.utils import (
    get_script_test_data_path, get_current_plot_image_data, args_to_key)

pytestmark = pytest.mark.matplotlib
# Allow tests with matplotlib marker to be collected and
# deselected if Matplotlib is not installed
try:
    import matplotlib.pyplot
    import euphonic.cli.powder_map
except ModuleNotFoundError:
    pass

graphite_fc_file = get_castep_path('graphite', 'graphite.castep_bin')
nacl_prim_fc_file = get_phonopy_path('NaCl_prim', 'phonopy_nacl.yaml')
powder_map_output_file = os.path.join(get_script_test_data_path(),
                                      'powder-map.json')

quick_calc_params = ['--npts=10', '--npts-min=10', '--q-spacing=1']
powder_map_params = [
    [graphite_fc_file, '-w', 'dos', '--y-label=DOS', '--title=DOS TITLE',
     *quick_calc_params],
    [graphite_fc_file, '-w', 'dos', '--y-label=DOS', '--scale=10',
     *quick_calc_params],
    [graphite_fc_file, '--e-min=50', '-u=cm^-1', '--x-label=wavenumber',
     *quick_calc_params, '-w=coherent-dos'],
    [graphite_fc_file, '--e-min=-100', '--e-max=1000', '--ebins=100',
     '--energy-unit=cm^-1', *quick_calc_params],
    [graphite_fc_file, '--energy-broadening=2e-3', '-u=eV',
     *quick_calc_params],
    [graphite_fc_file, '--q-spacing=0.05', '--length-unit=bohr',
     '--q-broadening=0.1', *quick_calc_params],
    [graphite_fc_file, '--qb=0.01', '--eb=1.5', '--shape=lorentz',
     *quick_calc_params],
    [graphite_fc_file, '--energy-broadening', '10', '1e-2', '-u=1/cm',
     *quick_calc_params],
    [graphite_fc_file, '--energy-broadening', '10', '1e-2', '-u=1/cm',
     '--q-broadening', '0.2', '-0.01', *quick_calc_params],
    [graphite_fc_file, '--asr', *quick_calc_params],
    [graphite_fc_file, '--asr=realspace', '--dipole-parameter=0.75',
     *quick_calc_params],
    [graphite_fc_file, '--e-i=15', '--ebins=50', *quick_calc_params],
    [graphite_fc_file, '--e-i=15', '--e-max=20', '--ebins=50',
     *quick_calc_params],
    [graphite_fc_file, '--e-f=15', '--ebins=50', '--q-max=16',
     '--angle-range', '20', '230', *quick_calc_params]]
powder_map_params_from_phonopy = [
    [nacl_prim_fc_file],
    [nacl_prim_fc_file, '--temperature=1000', *quick_calc_params],
    [nacl_prim_fc_file, '--v-min=0', '--v-max=1e-10', *quick_calc_params],
    [nacl_prim_fc_file, '--energy-unit=meV', *quick_calc_params],
    [nacl_prim_fc_file, '--weighting=coherent', '--c-map=bone',
     *quick_calc_params],
    [nacl_prim_fc_file, '-w=incoherent-dos', '--pdos=Na', '--no-widget',
     *quick_calc_params],
    [nacl_prim_fc_file, '-w=coherent-plus-incoherent-dos', '--pdos=Cl',
     *quick_calc_params],
    [nacl_prim_fc_file, '--temperature=1000', '--weighting=coherent',
     *quick_calc_params]]
powder_map_params_brille = [[graphite_fc_file, '--use-brille',
                             '--brille-npts', '10',
                             '--brille-grid-type', 'nest',
                             '-w', 'coherent',
                             *quick_calc_params]]

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

    def run_powder_map_and_test_result(
            self, powder_map_args, keys_to_omit=('x_ticklabels',)):
        euphonic.cli.powder_map.main(powder_map_args)

        image_data = get_current_plot_image_data()

        with open(powder_map_output_file, 'r') as expected_data_file:
            expected_image_data = json.load(expected_data_file)[
                args_to_key(powder_map_args)]
        for key, expected_val in expected_image_data.items():
            if key in keys_to_omit:
                # We don't care about the details of tick labels for powder map
                pass
            elif key == 'extent':
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

    @pytest.mark.parametrize('powder_map_args', powder_map_params)
    def test_powder_map_plot_image(
            self, inject_mocks, powder_map_args):
        self.run_powder_map_and_test_result(powder_map_args)

    @pytest.mark.phonopy_reader
    @pytest.mark.multiple_extras
    @pytest.mark.parametrize(
        'powder_map_args', powder_map_params_from_phonopy)
    def test_powder_map_plot_image_from_phonopy(
            self, inject_mocks, powder_map_args):
        self.run_powder_map_and_test_result(powder_map_args)

    @pytest.mark.brille
    @pytest.mark.multiple_extras
    @pytest.mark.parametrize(
        'powder_map_args', powder_map_params_brille)
    def test_powder_map_plot_image_with_brille(
            self, inject_mocks, powder_map_args):
        # Different numerical results on different platforms
        # unless a very dense, computationally expensive grid
        # is used. Just check that the program runs and a plot
        # is produced, by omitting check of 'data_1', 'data_2'
        self.run_powder_map_and_test_result(
            powder_map_args,
            keys_to_omit=['x_ticklabels', 'data_1', 'data_2'])

    @pytest.mark.brille
    @pytest.mark.multiple_extras
    @pytest.mark.parametrize('powder_map_args, expected_kwargs', [
        (['--use-brille', '--brille-npts', '25', '--disable-c'],
         {'grid_npts': 25, 'grid_type': 'trellis',
          'interpolation_kwargs': {'use_c': False}}),
        (['--use-brille', '--brille-grid-type', 'mesh', '--use-c',
          '--n-threads', '2'],
          {'grid_type': 'mesh', 'interpolation_kwargs': {'use_c': True,
                                                         'n_threads': 2}})
    ])
    def test_brille_interpolator_from_force_constants_kwargs_passed(
            self, inject_mocks, mocker, powder_map_args, expected_kwargs):
        from euphonic.brille import BrilleInterpolator
        # Stop execution once from_fc has been called - we're only
        # checking here that the correct arguments have been passed
        # through
        class MockException(Exception):
            pass
        mock = mocker.patch.object(BrilleInterpolator, 'from_force_constants',
                                   side_effect=MockException())
        try:
            euphonic.cli.powder_map.main(
                [graphite_fc_file] + powder_map_args + quick_calc_params)
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

    @pytest.mark.phonopy_reader
    @pytest.mark.multiple_extras
    @pytest.mark.parametrize('powder_map_args', [
        [nacl_prim_fc_file, '--save-to'],
        [nacl_prim_fc_file, '-s']])
    def test_plot_save_to_file(self, inject_mocks, tmpdir, powder_map_args):
        output_file = str(tmpdir.join('test.png'))
        euphonic.cli.powder_map.main(powder_map_args + [output_file]
                                     + quick_calc_params)
        assert os.path.exists(output_file)

    @pytest.mark.parametrize('powder_map_args', [
        [graphite_fc_file, '--save-json']])
    def test_plot_save_to_json(self, inject_mocks, tmpdir, powder_map_args):
        output_file = str(tmpdir.join('test.json'))
        euphonic.cli.powder_map.main(powder_map_args + [output_file]
                                     + quick_calc_params)
        spec = Spectrum2D.from_json_file(output_file)
        assert isinstance(spec, Spectrum2D)

    @pytest.mark.parametrize('powder_map_args', [
        [os.path.join(get_data_path(), 'util', 'qgrid_444.txt')]])
    def test_invalid_file_raises_value_error(self, powder_map_args):
        with pytest.raises(ValueError):
            euphonic.cli.powder_map.main(powder_map_args)

    @pytest.mark.phonopy_reader
    @pytest.mark.multiple_extras
    @pytest.mark.parametrize('powder_map_args', [
        [get_data_path(
            'qpoint_phonon_modes', 'quartz',
            'quartz_bandstructure_qpoint_phonon_modes.json')]])
    def test_qpoint_modes_raises_type_error(self, powder_map_args):
        with pytest.raises(TypeError):
            euphonic.cli.powder_map.main(powder_map_args)

    @pytest.mark.parametrize('powder_map_args', [
        [nacl_prim_fc_file, '-w=incoherent']])
    def test_invalid_weighting_raises_causes_exit(self, powder_map_args):
        # Argparse should call sys.exit on invalid choices
        with pytest.raises(SystemExit) as err:
            euphonic.cli.powder_map.main(powder_map_args)
        assert err.type == SystemExit
        assert err.value.code == 2

    @pytest.mark.brille
    @pytest.mark.phonopy_reader
    @pytest.mark.multiple_extras
    @pytest.mark.parametrize('powder_map_args', [
        [nacl_prim_fc_file, '--use-brille', '--brille-grid-type', 'grid']])
    def test_invalid_brille_grid_raises_causes_exit(self, powder_map_args):
        # Argparse should call sys.exit on invalid choices
        with pytest.raises(SystemExit) as err:
            euphonic.cli.powder_map.main(powder_map_args)
        assert err.type == SystemExit
        assert err.value.code == 2

    @pytest.mark.phonopy_reader
    @pytest.mark.multiple_extras
    @pytest.mark.parametrize('powder_map_args', [
        [nacl_prim_fc_file, '-w=coherent', '--pdos', 'Na']])
    def test_coherent_weighting_and_pdos_raises_value_error(
            self, powder_map_args):
        with pytest.raises(ValueError):
            euphonic.cli.powder_map.main(powder_map_args + quick_calc_params)

    @pytest.mark.phonopy_reader
    @pytest.mark.multiple_extras
    @pytest.mark.parametrize('powder_map_args', [
        [nacl_prim_fc_file, '--pdos']])
    def test_no_pdos_args_raises_causes_exit(self, powder_map_args):
        with pytest.raises(SystemExit) as err:
            euphonic.cli.powder_map.main(powder_map_args)
        assert err.type == SystemExit
        assert err.value.code == 2

    @pytest.mark.phonopy_reader
    @pytest.mark.multiple_extras
    @pytest.mark.parametrize('powder_map_args', [
        [nacl_prim_fc_file, '--pdos', 'Na', 'Cl']])
    def test_multiple_pdos_args_raises_causes_exit(self, powder_map_args):
        with pytest.raises(SystemExit) as err:
            euphonic.cli.powder_map.main(powder_map_args)
        assert err.type == SystemExit
        assert err.value.code == 2


@patch('matplotlib.pyplot.show')
@pytest.mark.skip(reason='Only run if you want to regenerate the test data')
def test_regenerate_powder_map_data(_):
    # Read from existing file first to allow option of only replacing for
    # certain test cases or keys
    try:
        with open(powder_map_output_file, 'r') as json_file:
            json_data = json.load(json_file)
    except FileNotFoundError:
        json_data = {}

    for powder_map_param in (powder_map_params
                             + powder_map_params_from_phonopy
                             + powder_map_params_brille):
        # Generate current figure for us to retrieve with gcf
        euphonic.cli.powder_map.main(powder_map_param)

        # Retrieve with gcf and write to file
        image_data = get_current_plot_image_data()
        # Optionally only write certain keys
        keys_to_replace = []
        if len(keys_to_replace) > 0:
            for key in keys_to_replace:
                json_data[args_to_key(powder_map_param)][key] = image_data[key]
        else:
            json_data[args_to_key(powder_map_param)] = image_data

    with open(powder_map_output_file, 'w+') as json_file:
        json.dump(json_data, json_file, indent=4)
