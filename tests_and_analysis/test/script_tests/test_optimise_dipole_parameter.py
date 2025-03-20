from contextlib import ExitStack

# Required for mocking
from typing import Any, Optional

import numpy as np
import pytest

import euphonic.cli.optimise_dipole_parameter
from euphonic import ForceConstants
from euphonic.cli.optimise_dipole_parameter import calculate_optimum_dipole_parameter
from tests_and_analysis.test.utils import get_castep_path, get_phonopy_path

quartz_castep_bin = get_castep_path("quartz", "quartz.castep_bin")
lzo_castep_bin = get_castep_path("LZO", "La2Zr2O7.castep_bin")
nacl_default_yaml = get_phonopy_path("NaCl_default", "phonopy.yaml")
quick_calc_params = ['-n=10', '--min=0.5', '--max=0.5']


class TestRegression:
    @staticmethod
    def call_cli(fc_file: str, args: list[Any], warning: Optional[str]) -> None:
        """Call optimise-dipole-parameter, checking for warning if appropriate"""
        with ExitStack() as stack:
            if warning is not None:
                stack.enter_context(pytest.warns(UserWarning, match=warning))

            return euphonic.cli.optimise_dipole_parameter.main([fc_file, *args])

    @pytest.mark.parametrize(
        'fc_file, opt_dipole_par_args, expected_n_qpts, '
        'expected_dipole_pars, expected_calc_qpt_ph_modes_kwargs, '
        'warning', [
            (quartz_castep_bin, ['-n=5'], 5, np.linspace(0.25, 1.5, 6), {}, None),
            (quartz_castep_bin,
             ['-n=10', '--n-threads=2', '--dipole-parameter-min=0.5',
              '--dipole-parameter-max=1.0'], 10, np.linspace(0.5, 1.0, 3),
             {'n_threads': 2}, None),
            (lzo_castep_bin,
             ['-n=15', '--asr=reciprocal', '--disable-c',
              '--dipole-parameter-min=0.1', '--dipole-parameter-max=0.4',
              '--dipole-parameter-step=0.1'], 15, np.linspace(0.1, 0.4, 4),
             {'asr': 'reciprocal', 'use_c': False},
             "Born charges not found for this material")
        ])
    def test_calc_qpt_phonon_modes_called_with_correct_args(
            self, mocker, fc_file, opt_dipole_par_args, expected_n_qpts,
            expected_dipole_pars, expected_calc_qpt_ph_modes_kwargs,
            warning):
        fc = ForceConstants.from_castep(fc_file)
        mock = mocker.patch.object(ForceConstants, 'calculate_qpoint_phonon_modes',
            wraps=fc.calculate_qpoint_phonon_modes)

        self.call_cli(fc_file, opt_dipole_par_args, warning=warning)

        default_kwargs = {'asr': None, 'n_threads': None, 'use_c': None}

        # Called twice for each dipole_parameter value - once to measure
        # initialisation time, and once to measure time per qpt
        # time/qpt call has extra argument 'reduce_qpts'
        expected_kwargs = {**default_kwargs,
                           **expected_calc_qpt_ph_modes_kwargs}
        expected_kwargs_per_qpt = expected_kwargs.copy()
        expected_kwargs_per_qpt['reduce_qpts'] = False
        assert mock.call_count == len(expected_dipole_pars)*2
        for i, dipole_par in enumerate(expected_dipole_pars):
            expected_kwargs['dipole_parameter'] = dipole_par
            expected_kwargs_per_qpt['dipole_parameter'] = dipole_par
            # Ensure initialisation call and time/qpt call are made
            # with correct kwargs
            assert mock.call_args_list[2*i][1] == expected_kwargs
            assert mock.call_args_list[2*i + 1][1] == expected_kwargs_per_qpt
            # Ensure time/qpt call is made with correct number of qpts
            assert len(mock.call_args_list[2*i + 1][0][0]) == expected_n_qpts

    # Ensure by correct number of default qpts are passed,
    # but don't actually run because this would be expensive
    def test_default_correct_n_qpts_passed(self, mocker):
        mock = mocker.patch(
            'euphonic.cli.optimise_dipole_parameter'
            '.calculate_optimum_dipole_parameter')
        euphonic.cli.optimise_dipole_parameter.main(
            [quartz_castep_bin])
        default_n_qpts = 500
        assert mock.call_args[1]['n'] == default_n_qpts

    @pytest.mark.phonopy_reader
    def test_reading_nacl_default_reads_born(self, recwarn):
        # BORN should be read by default so no warning should
        # be raised
        euphonic.cli.optimise_dipole_parameter.main([
           nacl_default_yaml, *quick_calc_params])
        assert len(recwarn) == 0

    def test_qpoint_modes_raises_type_error(self):
        with pytest.raises(TypeError):
            euphonic.cli.optimise_dipole_parameter.main([
                get_castep_path("quartz", "quartz-666-grid.phonon")])

    def test_optimal_has_lowest_time_per_qpt(self):
        (opt_param, _, opt_t, all_params,
         _, all_t) = calculate_optimum_dipole_parameter(quartz_castep_bin)
        assert opt_t == np.amin(all_t)
        assert opt_param == all_params[np.argmin(all_t)]
