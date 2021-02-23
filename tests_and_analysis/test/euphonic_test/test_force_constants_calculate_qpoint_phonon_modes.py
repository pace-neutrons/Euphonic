import os
import json

import pytest
import numpy as np
import numpy.testing as npt

from euphonic import ureg, ForceConstants
from euphonic.force_constants import ImportCError
from tests_and_analysis.test.utils import (get_data_path, get_phonopy_path,
    get_castep_path, get_test_qpts)
from tests_and_analysis.test.euphonic_test.test_qpoint_phonon_modes import (
    ExpectedQpointPhononModes, check_qpt_ph_modes, get_qpt_ph_modes_dir)
from tests_and_analysis.test.euphonic_test.test_force_constants import (
    get_fc, get_fc_dir)


@pytest.mark.unit
class TestForceConstantsCalculateQPointPhononModes:

    def get_lzo_fc():
        return ForceConstants.from_castep(
            get_castep_path('LZO', 'La2Zr2O7.castep_bin'))

    lzo_params = [
        (get_lzo_fc(), 'LZO',
         [get_test_qpts(), {}], 'LZO_no_asr_qpoint_phonon_modes.json'),
        (get_lzo_fc(), 'LZO',
         [get_test_qpts(), {'asr':'realspace'}],
         'LZO_realspace_qpoint_phonon_modes.json'),
        (get_lzo_fc(), 'LZO',
         [get_test_qpts(), {'asr': 'reciprocal'}],
         'LZO_reciprocal_qpoint_phonon_modes.json')]


    def get_si2_fc():
        return ForceConstants.from_castep(
            get_castep_path('Si2-sc-skew', 'Si2-sc-skew.castep_bin'))

    si2_params = [
        (get_si2_fc(), 'Si2-sc-skew',
         [get_test_qpts(), {}], 'Si2-sc-skew_no_asr_qpoint_phonon_modes.json'),
        (get_si2_fc(), 'Si2-sc-skew',
         [get_test_qpts(), {'asr':'realspace'}],
         'Si2-sc-skew_realspace_qpoint_phonon_modes.json'),
        (get_si2_fc(), 'Si2-sc-skew',
         [get_test_qpts(), {'asr': 'reciprocal'}],
         'Si2-sc-skew_reciprocal_qpoint_phonon_modes.json')]


    def get_quartz_fc():
        return ForceConstants.from_castep(
            get_castep_path('quartz', 'quartz.castep_bin'))

    quartz_params = [
        (get_quartz_fc(), 'quartz',
         [get_test_qpts(), {'asr': 'reciprocal', 'splitting': False}],
         'quartz_reciprocal_qpoint_phonon_modes.json'),
        (get_quartz_fc(), 'quartz',
         [get_test_qpts(), {'asr': 'reciprocal', 'splitting': False,
                            'eta_scale': 0.75}],
         'quartz_reciprocal_qpoint_phonon_modes.json'),
        (get_quartz_fc(), 'quartz',
         [get_test_qpts('split'), {'asr': 'reciprocal', 'splitting': True,
                                   'insert_gamma': False}],
         'quartz_split_reciprocal_qpoint_phonon_modes.json'),
        (get_quartz_fc(), 'quartz',
         [get_test_qpts('split_insert_gamma'),
          {'asr': 'reciprocal', 'splitting': True, 'insert_gamma': True}],
         'quartz_split_reciprocal_qpoint_phonon_modes.json')]


    nacl_params = [(
        ForceConstants.from_phonopy(
            path=get_phonopy_path('NaCl', ''),
            summary_name='phonopy_nacl.yaml'),
        'NaCl',
        [get_test_qpts(), {'asr': 'reciprocal'}],
        'NaCl_reciprocal_qpoint_phonon_modes.json')]


    cahgo2_params = [(
        ForceConstants.from_phonopy(
            path=get_phonopy_path('CaHgO2', ''),
            summary_name='mp-7041-20180417.yaml'),
        'CaHgO2',
        [get_test_qpts(), {'asr': 'reciprocal'}],
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

    # ForceConstants stores some values (supercell image list, vectors
    # for the Ewald sum) so check repeated calculations give the same
    # result
    def test_repeated_calculate_qpoint_phonon_modes_doesnt_change_result(self):
        fc = get_fc('quartz')
        qpt_ph_modes1 = fc.calculate_qpoint_phonon_modes(
            get_test_qpts(), asr='realspace')
        qpt_ph_modes2 = fc.calculate_qpoint_phonon_modes(
            get_test_qpts(), asr='realspace')
        check_qpt_ph_modes(qpt_ph_modes1, qpt_ph_modes2)

    @pytest.mark.parametrize(
        'fc, material, qpt, kwargs, expected_qpt_ph_modes_file',
        [(get_fc('quartz'), 'quartz', np.array([[1., 1., 1.]]),
         {'splitting': True}, 'quartz_single_qpoint_phonon_modes.json')])
    def test_calculate_qpoint_phonon_modes_single_qpt(
            self, fc, material, qpt, kwargs, expected_qpt_ph_modes_file):
        qpoint_phonon_modes = fc.calculate_qpoint_phonon_modes(
            qpt, **kwargs)
        expected_qpoint_phonon_modes = ExpectedQpointPhononModes(
            os.path.join(get_qpt_ph_modes_dir(material),
                         expected_qpt_ph_modes_file))
        check_qpt_ph_modes(qpoint_phonon_modes,
                           expected_qpoint_phonon_modes,
                           frequencies_atol=1e-4,
                           frequencies_rtol=2e-5)

    weights = np.array([0.1, 0.05, 0.05, 0.2, 0.2, 0.15, 0.15, 0.2, 0.1])
    weights_output_split_gamma = np.array([
        0.1, 0.05, 0.025, 0.025, 0.2, 0.1, 0.1, 0.075, 0.075, 0.075, 0.075,
        0.2, 0.1])

    @pytest.mark.parametrize('fc, qpts, weights, expected_weights, kwargs', [
        (get_fc('quartz'), get_test_qpts(), weights, weights, {}),
        (get_fc('quartz'), get_test_qpts('split_insert_gamma'), weights,
         weights_output_split_gamma, {'insert_gamma': True})])
    def test_calculate_qpoint_phonon_modes_with_weights_sets_weights(
            self, fc, qpts, weights, expected_weights, kwargs):
        qpt_ph_modes_weighted = fc.calculate_qpoint_phonon_modes(
            qpts, weights=weights, **kwargs)
        npt.assert_allclose(qpt_ph_modes_weighted.weights, expected_weights)

    @pytest.mark.parametrize('fc, qpts, weights, expected_weights, kwargs', [
        (get_fc('quartz'), get_test_qpts(), weights, weights, {}),
        (get_fc('quartz'), get_test_qpts('split_insert_gamma'), weights,
         weights_output_split_gamma, {'insert_gamma': True})])
    def test_calculate_qpoint_phonon_modes_with_weights_doesnt_change_result(
            self, fc, qpts, weights, expected_weights, kwargs):
        qpt_ph_modes_weighted = fc.calculate_qpoint_phonon_modes(
            qpts, weights=weights, **kwargs)
        qpt_ph_modes_unweighted = fc.calculate_qpoint_phonon_modes(
            qpts, **kwargs)
        qpt_ph_modes_unweighted.weights = expected_weights
        check_qpt_ph_modes(qpt_ph_modes_weighted, qpt_ph_modes_unweighted)

    @pytest.mark.parametrize('asr', ['realspace', 'reciprocal'])
    def test_calc_qpt_ph_mds_asr_with_nonsense_fc_raises_warning(
            self, asr):
        fc = ForceConstants.from_json_file(
            os.path.join(get_fc_dir(), 'quartz_random_force_constants.json'))
        with pytest.warns(UserWarning):
            fc.calculate_qpoint_phonon_modes(get_test_qpts(), asr=asr)


@pytest.mark.unit
class TestForceConstantsCalculateQPointPhononModesWithoutCExtensionInstalled:
    def test_calculate_qpoint_phonon_modes_with_use_c_true_raises_error(
            self, mocker):
        # Mock import of euphonic._euphonic to raise ImportError
        import builtins
        real_import = builtins.__import__
        def mocked_import(name, *args, **kwargs):
            if name == 'euphonic._euphonic':
                raise ImportError
            return real_import(name, *args, **kwargs)
        mocker.patch('builtins.__import__', side_effect=mocked_import)

        fc = get_fc('quartz')
        with pytest.raises(ImportCError):
            fc.calculate_qpoint_phonon_modes(get_test_qpts(),
                                             use_c=True,
                                             fall_back_on_python=False)
