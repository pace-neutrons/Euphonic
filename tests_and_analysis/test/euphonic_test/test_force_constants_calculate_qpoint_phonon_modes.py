import os
import json
import pytest
import numpy as np
import numpy.testing as npt
from euphonic import ureg, ForceConstants
from tests_and_analysis.test.utils import get_data_path
from tests_and_analysis.test.euphonic_test.test_qpoint_phonon_modes import (
    ExpectedQpointPhononModes, check_qpt_ph_modes, get_qpt_ph_modes_dir)
from tests_and_analysis.test.euphonic_test.test_force_constants import (
    get_fc, get_fc_dir)


test_qpts = np.array([
    [0.00, 0.00, 0.00],
    [0.00, 0.00, 0.50],
    [-0.25, 0.50, 0.50],
    [-0.151515, 0.575758, 0.5],
    [0.30, 0.00, 0.00],
    [0.00, 0.40, 0.00],
    [0.60, 0.00, 0.20],
    [2.00, 2.00, 0.5],
    [1.75, 0.50, 2.50]])


test_split_qpts = np.array([
    [0.00, 0.00, 0.00],
    [0.00, 0.00, 0.50],
    [0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00],
    [-0.25, 0.50, 0.50],
    [0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00],
    [-0.151515, 0.575758, 0.5],
    [0.00, 0.00, 0.00]
])


test_split_qpts_insert_gamma = np.array([
    [0.00, 0.00, 0.00],
    [0.00, 0.00, 0.50],
    [0.00, 0.00, 0.00],
    [-0.25, 0.50, 0.50],
    [0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00],
    [-0.151515, 0.575758, 0.5],
    [0.00, 0.00, 0.00]
])


@pytest.mark.unit
class TestForceConstantsCalculateQPointPhononModes:

    @pytest.fixture(params=['json', 'castep'])
    def get_lzo_fc(self, request):
        if request.param == 'json':
            return get_fc('LZO')
        else:
            return ForceConstants.from_castep(
                os.path.join(get_fc_dir('LZO'), 'La2Zr2O7.castep_bin'))

    lzo_params = [
        (pytest.lazy_fixture('get_lzo_fc'), 'LZO',
         [test_qpts, {}], 'LZO_no_asr_qpoint_phonon_modes.json'),
        (pytest.lazy_fixture('get_lzo_fc'), 'LZO',
         [test_qpts, {'asr':'realspace'}],
         'LZO_realspace_qpoint_phonon_modes.json'),
        (pytest.lazy_fixture('get_lzo_fc'), 'LZO',
         [test_qpts, {'asr': 'reciprocal'}],
         'LZO_reciprocal_qpoint_phonon_modes.json')]

    @pytest.fixture(params=['json', 'castep'])
    def get_si2_fc(self, request):
        if request.param == 'json':
            return get_fc('Si2-sc-skew')
        else:
            return ForceConstants.from_castep(
                os.path.join(get_fc_dir('Si2-sc-skew'),
                             'Si2-sc-skew.castep_bin'))

    si2_params = [
        (pytest.lazy_fixture('get_si2_fc'), 'Si2-sc-skew',
         [test_qpts, {}], 'Si2_no_asr_qpoint_phonon_modes.json'),
        (pytest.lazy_fixture('get_si2_fc'), 'Si2-sc-skew',
         [test_qpts, {'asr':'realspace'}],
         'Si2_realspace_qpoint_phonon_modes.json'),
        (pytest.lazy_fixture('get_si2_fc'), 'Si2-sc-skew',
         [test_qpts, {'asr': 'reciprocal'}],
         'Si2_reciprocal_qpoint_phonon_modes.json')]


    @pytest.fixture(params=['json', 'castep'])
    def get_quartz_fc(self, request):
        if request.param == 'json':
            return get_fc('quartz')
        else:
            return ForceConstants.from_castep(
                os.path.join(get_fc_dir('quartz'), 'quartz.castep_bin'))

    quartz_params = [
        (pytest.lazy_fixture('get_quartz_fc'), 'quartz',
         [test_qpts, {'asr': 'reciprocal', 'splitting': False}],
         'quartz_reciprocal_qpoint_phonon_modes.json'),
        (pytest.lazy_fixture('get_quartz_fc'), 'quartz',
         [test_qpts, {'asr': 'reciprocal', 'splitting': False,
                      'eta_scale': 0.75}],
         'quartz_reciprocal_qpoint_phonon_modes.json'),
        (pytest.lazy_fixture('get_quartz_fc'), 'quartz',
         [test_split_qpts, {'asr': 'reciprocal', 'splitting': True,
                            'insert_gamma': False}],
         'quartz_split_reciprocal_qpoint_phonon_modes.json'),
        (pytest.lazy_fixture('get_quartz_fc'), 'quartz',
         [test_split_qpts_insert_gamma,
         {'asr': 'reciprocal', 'splitting': True, 'insert_gamma': True}],
         'quartz_split_reciprocal_qpoint_phonon_modes.json')]


    @pytest.fixture(params=['json', 'phonopy'])
    def get_nacl_fc(self, request):
        if request.param == 'json':
            return get_fc('NaCl')
        else:
            return ForceConstants.from_phonopy(
                path=get_fc_dir('NaCl'),
                summary_name='phonopy_nacl.yaml')

    nacl_params = [
        (pytest.lazy_fixture('get_nacl_fc'), 'NaCl',
         [test_qpts, {'asr': 'reciprocal'}],
         'NaCl_reciprocal_qpoint_phonon_modes.json')]


    @pytest.fixture(params=['phonopy'])
    def get_cahgo2_fc(self, request):
        if request.param == 'json':
            return get_fc('CaHgO2')
        else:
            return ForceConstants.from_phonopy(
                path=get_fc_dir('CaHgO2'),
                summary_name='mp-1818-20180417.yaml')

    cahgo2_params = [
        (pytest.lazy_fixture('get_cahgo2_fc'), 'CaHgO2',
         [test_qpts, {'asr': 'reciprocal'}],
         'CaHgO2_reciprocal_qpoint_phonon_modes.json')]


    @pytest.mark.parametrize(
        'fc, material, all_args, expected_qpoint_phonon_modes_file',
        lzo_params + quartz_params + nacl_params + si2_params + cahgo2_params)
    @pytest.mark.parametrize(
        'reduce_qpts, n_threads',
        [(False, 0), (True, 0), (True, 1), (True, 2)])
    def test_calculate_qpoint_phonon_modes(
            self, fc, material, all_args, expected_qpoint_phonon_modes_file,
            reduce_qpts, n_threads):
        func_kwargs = all_args[1]
        func_kwargs['reduce_qpts'] = reduce_qpts
        if n_threads == 0:
            func_kwargs['use_c'] = False
        else:
            func_kwargs['use_c'] = True
            func_kwargs['n_threads'] = n_threads
            func_kwargs['fall_back_on_python'] = False
        qpoint_phonon_modes = fc.calculate_qpoint_phonon_modes(
            all_args[0], **func_kwargs)
        expected_qpoint_phonon_modes = ExpectedQpointPhononModes(
            os.path.join(get_qpt_ph_modes_dir(material),
                         expected_qpoint_phonon_modes_file))
        # Only give gamma-acoustic modes special treatment if the acoustic
        # sum rule has been applied
        if not 'asr' in func_kwargs.keys():
            gamma_atol = None
        else:
            gamma_atol = 0.5
        check_qpt_ph_modes(qpoint_phonon_modes,
                           expected_qpoint_phonon_modes,
                           frequencies_atol=1e-4,
                           frequencies_rtol=2e-5,
                           acoustic_gamma_atol=gamma_atol)

    weights = np.array([0.1, 0.05, 0.05, 0.2, 0.2, 0.15, 0.15, 0.2, 0.1])
    weights_output_split_gamma = np.array([
        0.1, 0.05, 0.025, 0.025, 0.2, 0.1, 0.1, 0.075, 0.075, 0.075, 0.075,
        0.2, 0.1])

    @pytest.mark.parametrize('fc, qpts, weights, expected_weights, kwargs', [
        (get_fc('quartz'), test_qpts, weights, weights, {}),
        (get_fc('quartz'), test_split_qpts_insert_gamma, weights,
         weights_output_split_gamma, {'insert_gamma': True})])
    def test_calculate_qpoint_phonon_modes_with_weights_sets_weights(
            self, fc, qpts, weights, expected_weights, kwargs):
        qpt_ph_modes_weighted = fc.calculate_qpoint_phonon_modes(
            qpts, weights=weights, **kwargs)
        npt.assert_allclose(qpt_ph_modes_weighted.weights, expected_weights)

    @pytest.mark.parametrize('fc, qpts, weights, expected_weights, kwargs', [
        (get_fc('quartz'), test_qpts, weights, weights, {}),
        (get_fc('quartz'), test_split_qpts_insert_gamma, weights,
         weights_output_split_gamma, {'insert_gamma': True})])
    def test_calculate_qpoint_phonon_modes_with_weights_doesnt_change_result(
            self, fc, qpts, weights, expected_weights, kwargs):
        qpt_ph_modes_weighted = fc.calculate_qpoint_phonon_modes(
            qpts, weights=weights, **kwargs)
        qpt_ph_modes_unweighted = fc.calculate_qpoint_phonon_modes(
            qpts, **kwargs)
        qpt_ph_modes_unweighted.weights = expected_weights
        check_qpt_ph_modes(qpt_ph_modes_weighted, qpt_ph_modes_unweighted)