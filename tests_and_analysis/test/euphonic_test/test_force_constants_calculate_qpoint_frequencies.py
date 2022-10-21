import json

import pytest
import numpy as np
import numpy.testing as npt

from euphonic import ForceConstants, ureg
from euphonic.util import mp_grid
from tests_and_analysis.test.utils import (
    get_castep_path, get_test_qpts, sum_at_degenerate_modes)
from tests_and_analysis.test.euphonic_test.test_qpoint_frequencies import (
    check_qpt_freqs, get_expected_qpt_freqs)
from tests_and_analysis.test.euphonic_test.test_force_constants import (
    get_fc, get_fc_path)


class TestForceConstantsCalculateQPointFrequencies:

    def get_lzo_fc():
        return ForceConstants.from_castep(
            get_castep_path('LZO', 'La2Zr2O7.castep_bin'))

    lzo_params = [
        (get_lzo_fc(), 'LZO',
         [get_test_qpts(), {}], 'LZO_no_asr_qpoint_frequencies.json'),
        (get_lzo_fc(), 'LZO',
         [get_test_qpts(), {'asr': 'realspace'}],
         'LZO_realspace_qpoint_frequencies.json')]

    def get_quartz_fc():
        return ForceConstants.from_castep(
            get_castep_path('quartz', 'quartz.castep_bin'))

    quartz_params = [
        (get_quartz_fc(), 'quartz',
         [get_test_qpts(), {'asr': 'reciprocal', 'splitting': False}],
         'quartz_reciprocal_qpoint_frequencies.json'),
        (get_quartz_fc(), 'quartz',
         [get_test_qpts(), {'asr': 'reciprocal', 'splitting': False,
                            'dipole_parameter': 0.75}],
         'quartz_reciprocal_qpoint_frequencies.json'),
        (get_quartz_fc(), 'quartz',
         [get_test_qpts('split'), {'asr': 'reciprocal', 'splitting': True,
                                   'insert_gamma': False}],
         'quartz_split_reciprocal_qpoint_frequencies.json'),
        (get_quartz_fc(), 'quartz',
         [get_test_qpts('split_insert_gamma'),
          {'asr': 'reciprocal', 'splitting': True, 'insert_gamma': True}],
         'quartz_split_reciprocal_qpoint_frequencies.json')]

    @pytest.mark.parametrize(
        'fc, material, all_args, expected_qpoint_frequencies_file',
        lzo_params + quartz_params)
    @pytest.mark.parametrize(
        'reduce_qpts, n_threads',
        [(False, 0), (True, 0), (True, 1), (True, 2)])
    def test_calculate_qpoint_frequencies(
            self, fc, material, all_args, expected_qpoint_frequencies_file,
            reduce_qpts, n_threads):
        func_kwargs = all_args[1]
        func_kwargs['reduce_qpts'] = reduce_qpts
        if n_threads == 0:
            func_kwargs['use_c'] = False
        else:
            func_kwargs['use_c'] = True
            func_kwargs['n_threads'] = n_threads
        qpt_freqs = fc.calculate_qpoint_frequencies(all_args[0], **func_kwargs)
        expected_qpt_freqs = get_expected_qpt_freqs(
            material, expected_qpoint_frequencies_file)
        # Only give gamma-acoustic modes special treatment if the acoustic
        # sum rule has been applied
        tol_kwargs = {}
        if 'asr' in func_kwargs.keys():
            tol_kwargs['acoustic_gamma_atol'] = 0.55
        # Use larger tolerances with reciprocal ASR - formalism works
        # only at gamma but is applied to all q, so problem is less
        # well conditioned leading to larger f.p errors on different systems
        if func_kwargs.get('asr') == 'reciprocal':
            tol_kwargs['frequencies_atol'] = 0.01
        check_qpt_freqs(qpt_freqs,
                        expected_qpt_freqs,
                        **tol_kwargs)

    @pytest.mark.parametrize(
        ('fc, material, all_args, expected_qpoint_frequencies_file, '
         'expected_modg_file'),
        [(get_quartz_fc(),
          'quartz',
          [mp_grid([5, 5, 4]),
           {'return_mode_gradients': True}],
          'quartz_554_full_qpoint_frequencies.json',
          'quartz_554_full_mode_gradients.json'),
         (get_lzo_fc(),
          'LZO',
          [mp_grid([2, 2, 2]),
           {'asr': 'reciprocal', 'return_mode_gradients': True}],
          'lzo_222_full_qpoint_frequencies.json',
          'lzo_222_full_mode_gradients.json')])
    @pytest.mark.parametrize(
        'n_threads',
        [0, 2])
    def test_calculate_qpt_freqs_with_mode_gradients(
            self, fc, material, all_args, expected_qpoint_frequencies_file,
            expected_modg_file, n_threads):
        func_kwargs = all_args[1]
        if n_threads == 0:
            func_kwargs['use_c'] = False
        else:
            func_kwargs['use_c'] = True
            func_kwargs['n_threads'] = n_threads
        qpt_freqs, modg = fc.calculate_qpoint_frequencies(
            all_args[0], **func_kwargs)
        with open(get_fc_path(expected_modg_file), 'r') as fp:
            modg_dict = json.load(fp)
        expected_modg = modg_dict['mode_gradients']*ureg(
                modg_dict['mode_gradients_unit'])
        expected_qpt_freqs = get_expected_qpt_freqs(
            material, expected_qpoint_frequencies_file)
        check_qpt_freqs(qpt_freqs,
                        expected_qpt_freqs)
        assert modg.units == expected_modg.units
        # Mode gradients are derived from eigenvectors - in the case of
        # degenerate modes they may not be in the same order
        summed_modg = sum_at_degenerate_modes(
            modg.magnitude,
            expected_qpt_freqs.frequencies.magnitude)
        summed_expected_modg = sum_at_degenerate_modes(
            expected_modg.magnitude,
            expected_qpt_freqs.frequencies.magnitude)
        npt.assert_allclose(summed_modg, summed_expected_modg,
                            atol=2e-5)

    weights = np.array([0.1, 0.05, 0.05, 0.2, 0.2, 0.15, 0.15, 0.2, 0.1])
    weights_output_split_gamma = np.array([
        0.1, 0.05, 0.025, 0.025, 0.2, 0.1, 0.1, 0.075, 0.075, 0.075, 0.075,
        0.2, 0.1])

    @pytest.mark.parametrize('fc, qpts, weights, expected_weights, kwargs', [
        (get_fc('quartz'), get_test_qpts(), weights, weights, {}),
        (get_fc('quartz'), get_test_qpts('split_insert_gamma'), weights,
         weights_output_split_gamma, {'insert_gamma': True})])
    def test_calculate_qpoint_frequencies_with_weights_sets_weights(
            self, fc, qpts, weights, expected_weights, kwargs):
        qpt_freqs_weighted = fc.calculate_qpoint_frequencies(
            qpts, weights=weights, **kwargs)
        npt.assert_allclose(qpt_freqs_weighted.weights, expected_weights)
