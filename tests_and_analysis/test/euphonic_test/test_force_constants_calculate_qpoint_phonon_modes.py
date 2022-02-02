import os
import json
from multiprocessing import cpu_count

import pytest
import numpy as np
import numpy.testing as npt

import euphonic
from euphonic import ureg, ForceConstants
from euphonic.util import mp_grid
from euphonic.force_constants import ImportCError
from tests_and_analysis.test.utils import (get_data_path, get_phonopy_path,
    get_castep_path, get_test_qpts, sum_at_degenerate_modes)
from tests_and_analysis.test.euphonic_test.test_qpoint_phonon_modes import (
    ExpectedQpointPhononModes, check_qpt_ph_modes, get_qpt_ph_modes_path)
from tests_and_analysis.test.euphonic_test.test_force_constants import (
    get_fc, get_fc_path)


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
         [get_test_qpts(), {'splitting': False}],
         'quartz_no_asr_qpoint_phonon_modes.json'),
        (get_quartz_fc(), 'quartz',
         [get_test_qpts(), {'asr': 'reciprocal', 'splitting': False}],
         'quartz_reciprocal_qpoint_phonon_modes.json'),
        (get_quartz_fc(), 'quartz',
         [get_test_qpts(), {'asr': 'reciprocal', 'splitting': False,
                            'dipole_parameter': 0.75}],
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


    @staticmethod
    def calculate_and_check_qpoint_phonon_modes(
            fc, material, args, expected_file, reduce_qpts, n_threads):
        func_kwargs = args[1]
        func_kwargs['reduce_qpts'] = reduce_qpts
        if n_threads == 0:
            func_kwargs['use_c'] = False
        else:
            func_kwargs['use_c'] = True
            func_kwargs['n_threads'] = n_threads
        qpoint_phonon_modes = fc.calculate_qpoint_phonon_modes(
            args[0], **func_kwargs)
        expected_qpoint_phonon_modes = ExpectedQpointPhononModes(
            get_qpt_ph_modes_path(material, expected_file))
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
        check_qpt_ph_modes(qpoint_phonon_modes,
                           expected_qpoint_phonon_modes,
                           **tol_kwargs)

    @pytest.mark.parametrize(
        'fc, material, all_args, expected_qpoint_phonon_modes_file',
        lzo_params + quartz_params + si2_params)
    @pytest.mark.parametrize(
        'reduce_qpts, n_threads',
        [(False, 0), (True, 0), (True, 1), (True, 2)])
    def test_calculate_qpoint_phonon_modes(
            self, fc, material, all_args, expected_qpoint_phonon_modes_file,
            reduce_qpts, n_threads):
        self.calculate_and_check_qpoint_phonon_modes(
            fc, material, all_args, expected_qpoint_phonon_modes_file,
            reduce_qpts, n_threads)

    @pytest.mark.phonopy_reader
    @pytest.mark.parametrize(
        'fc_kwargs, material, all_args, expected_qpoint_phonon_modes_file', [
        ({'path': get_phonopy_path('NaCl'),
          'summary_name': 'phonopy_nacl.yaml'},
         'NaCl',
         [get_test_qpts(), {'asr': 'reciprocal'}],
         'NaCl_reciprocal_qpoint_phonon_modes.json'),
        ({'path': get_phonopy_path('CaHgO2'),
          'summary_name': 'mp-7041-20180417.yaml'},
         'CaHgO2',
         [get_test_qpts(), {'asr': 'reciprocal'}],
         'CaHgO2_reciprocal_qpoint_phonon_modes.json')])
    @pytest.mark.parametrize(
        'reduce_qpts, n_threads',
        [(False, 0), (True, 0), (True, 1), (True, 2)])
    def test_calculate_qpoint_phonon_modes_from_phonopy(
            self, fc_kwargs, material, all_args,
            expected_qpoint_phonon_modes_file, reduce_qpts, n_threads):
        fc = ForceConstants.from_phonopy(**fc_kwargs)
        self.calculate_and_check_qpoint_phonon_modes(
            fc, material, all_args, expected_qpoint_phonon_modes_file,
            reduce_qpts, n_threads)

    @pytest.mark.parametrize(
        ('fc, material, all_args, expected_qpoint_phonon_modes_file, '
         'expected_modg_file'), [
        (get_quartz_fc(),
         'quartz',
         [mp_grid([5, 5, 4]),
          {'return_mode_gradients': True}],
         'quartz_554_full_qpoint_phonon_modes.json',
         'quartz_554_full_mode_gradients.json'),
        (get_lzo_fc(),
         'LZO',
         [mp_grid([2, 2, 2]),
          {'asr': 'reciprocal', 'return_mode_gradients': True}],
         'lzo_222_full_qpoint_phonon_modes.json',
         'lzo_222_full_mode_gradients.json')])
    @pytest.mark.parametrize(
        'n_threads',
        [0, 2])
    def test_calculate_qpoint_phonon_modes_with_mode_gradients(
            self, fc, material, all_args, expected_qpoint_phonon_modes_file,
            expected_modg_file, n_threads):
        func_kwargs = all_args[1]
        if n_threads == 0:
            func_kwargs['use_c'] = False
        else:
            func_kwargs['use_c'] = True
            func_kwargs['n_threads'] = n_threads
        qpoint_phonon_modes, modg = fc.calculate_qpoint_phonon_modes(
            all_args[0], **func_kwargs)
        with open(get_fc_path(expected_modg_file), 'r') as fp:
            modg_dict = json.load(fp)
        expected_modg = modg_dict['mode_gradients']*ureg(
                modg_dict['mode_gradients_unit'])
        expected_qpoint_phonon_modes = ExpectedQpointPhononModes(
            get_qpt_ph_modes_path(material, expected_qpoint_phonon_modes_file))
        check_qpt_ph_modes(qpoint_phonon_modes,
                           expected_qpoint_phonon_modes)
        assert modg.units == expected_modg.units
        # Mode gradients are derived from eigenvectors - in the case of
        # degenerate modes they may not be in the same order
        summed_modg = sum_at_degenerate_modes(
            modg.magnitude,
            expected_qpoint_phonon_modes.frequencies.magnitude)
        summed_expected_modg = sum_at_degenerate_modes(
            expected_modg.magnitude,
            expected_qpoint_phonon_modes.frequencies.magnitude)
        npt.assert_allclose(summed_modg, summed_expected_modg,
                            atol=2e-5)

    @pytest.mark.parametrize(
        ('fc, material, all_args, expected_qpoint_phonon_modes_file, '
         'expected_modw_file'), [
        (get_quartz_fc(),
         'quartz',
         [mp_grid([5, 5, 4]),
          {'return_mode_widths': True}],
         'quartz_554_full_qpoint_phonon_modes.json',
         'quartz_554_full_mode_widths.json'),
        (get_lzo_fc(),
         'LZO',
         [mp_grid([2, 2, 2]),
          {'asr': 'reciprocal', 'return_mode_widths': True}],
         'lzo_222_full_qpoint_phonon_modes.json',
         'lzo_222_full_mode_widths.json')])
    @pytest.mark.parametrize(
        'n_threads',
        [0, 2])
    def test_calculate_qpoint_phonon_modes_with_mode_widths(
            self, fc, material, all_args, expected_qpoint_phonon_modes_file,
            expected_modw_file, n_threads):
        func_kwargs = all_args[1]
        if n_threads == 0:
            func_kwargs['use_c'] = False
        else:
            func_kwargs['use_c'] = True
            func_kwargs['n_threads'] = n_threads
        qpoint_phonon_modes, modw = fc.calculate_qpoint_phonon_modes(
            all_args[0], **func_kwargs)
        with open(get_fc_path(expected_modw_file), 'r') as fp:
            modw_dict = json.load(fp)
        expected_modw = modw_dict['mode_widths']*ureg(
                modw_dict['mode_widths_unit'])
        expected_qpoint_phonon_modes = ExpectedQpointPhononModes(
            get_qpt_ph_modes_path(material, expected_qpoint_phonon_modes_file))
        check_qpt_ph_modes(qpoint_phonon_modes,
                           expected_qpoint_phonon_modes)
        assert modw.units == expected_modw.units
        # Mode widths are derived from eigenvectors - in the case of
        # degenerate modes they may not be in the same order
        summed_modw = sum_at_degenerate_modes(
            modw.magnitude,
            expected_qpoint_phonon_modes.frequencies.magnitude)
        summed_expected_modw = sum_at_degenerate_modes(
            expected_modw.magnitude,
            expected_qpoint_phonon_modes.frequencies.magnitude)
        npt.assert_allclose(summed_modw, summed_expected_modw,
                            rtol=1e-4)


    def test_calc_qpt_ph_modes_with_mode_widths_raises_deprecation_warning(self):
        fc = get_fc('quartz')
        with pytest.warns(DeprecationWarning):
            fc.calculate_qpoint_phonon_modes(get_test_qpts(),
                                             return_mode_widths=True)

    def test_calc_qpt_ph_modes_with_eta_scale_raises_deprecation_warning(self):
        fc = get_fc('quartz')
        with pytest.warns(DeprecationWarning):
            fc.calculate_qpoint_phonon_modes(get_test_qpts(),
                                             eta_scale=0.9)

    def test_calc_qpt_ph_modes_with_large_complex_mode_gradients_raises_warning(
            self, mocker):
        # Mock force constants to return nonsense gradients with high
        # imaginary terms
        class MockFC(ForceConstants):
            def _calculate_phonons_at_q(self, *args):
                freqs, evecs, grads = ForceConstants._calculate_phonons_at_q(
                    self, *args)
                mocked_grads = np.ones(grads.shape) + np.ones(grads.shape)*1j
                return freqs, evecs, mocked_grads
        mocker.patch('euphonic.ForceConstants', MockFC)
        fc = euphonic.ForceConstants.from_json_file(
            get_fc_path('quartz_force_constants.json'))
        with pytest.warns(UserWarning):
            fc.calculate_qpoint_phonon_modes(get_test_qpts(), use_c=False,
                                             return_mode_gradients=True)

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
            get_qpt_ph_modes_path(material, expected_qpt_ph_modes_file))
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
            get_fc_path('quartz_random_force_constants.json'))
        with pytest.warns(UserWarning):
            fc.calculate_qpoint_phonon_modes(get_test_qpts(), asr=asr)


class TestForceConstantsCalculateQPointPhononModesWithoutCExtensionInstalled:

    @pytest.fixture
    def mocked_cext_with_importerror(self, mocker):
        # Mock import of euphonic._euphonic to raise ImportError
        import builtins
        real_import = builtins.__import__
        def mocked_import(name, *args, **kwargs):
            if name == 'euphonic._euphonic':
                raise ImportError
            return real_import(name, *args, **kwargs)
        mocker.patch('builtins.__import__', side_effect=mocked_import)

    def test_with_use_c_true_raises_importcerror(
            self, mocked_cext_with_importerror):
        fc = get_fc('quartz')
        with pytest.raises(ImportCError):
            fc.calculate_qpoint_phonon_modes(get_test_qpts(),
                                             use_c=True)

    def test_with_use_c_default_warns(
            self, mocked_cext_with_importerror):
        fc = get_fc('quartz')
        with pytest.warns(None) as warn_record:
            fc.calculate_qpoint_phonon_modes(get_test_qpts())
        assert len(warn_record) == 1

    def test_with_use_c_false_doesnt_raise_error_or_warn(
            self, mocked_cext_with_importerror):
        fc = get_fc('quartz')
        with pytest.warns(None) as warn_record:
            fc.calculate_qpoint_phonon_modes(get_test_qpts(), use_c=False)
        assert len(warn_record) == 0


class TestForceConstantsCalculateQPointPhononModesWithCExtensionInstalled:

    @pytest.fixture
    def mocked_cext(self, mocker):
        return mocker.patch('euphonic._euphonic.calculate_phonons')

    def test_cext_called_with_use_c_true(self, mocked_cext):
        fc = get_fc('quartz')
        fc.calculate_qpoint_phonon_modes(get_test_qpts(), use_c=True)
        mocked_cext.assert_called()

    def test_cext_called_with_use_c_default(self, mocked_cext):
        fc = get_fc('quartz')
        fc.calculate_qpoint_phonon_modes(get_test_qpts())
        mocked_cext.assert_called()

    def test_cext_not_called_with_use_c_false(self, mocked_cext):
        fc = get_fc('quartz')
        fc.calculate_qpoint_phonon_modes(get_test_qpts(), use_c=False)
        mocked_cext.assert_not_called()

    # The following only tests that the C extension was called with the
    # correct n_threads, rather than testing that that number of threads
    # have actually been spawned, but I can't think of a way to test that
    def test_cext_called_with_n_threads_arg(self, mocked_cext):
        n_threads = 3
        fc = get_fc('quartz')
        fc.calculate_qpoint_phonon_modes(get_test_qpts(), n_threads=n_threads)
        assert mocked_cext.call_args[0][-1] == n_threads

    def test_cext_called_with_n_threads_default_and_env_var(self, mocked_cext):
        n_threads = 4
        os.environ['EUPHONIC_NUM_THREADS'] = str(n_threads)
        fc = get_fc('quartz')
        fc.calculate_qpoint_phonon_modes(get_test_qpts())
        assert mocked_cext.call_args[0][-1] == n_threads

    def test_cext_called_with_n_threads_default_and_no_env_var(self, mocked_cext):
        n_threads = cpu_count()
        try:
            os.environ.pop('EUPHONIC_NUM_THREADS')
        except KeyError:
            pass
        fc = get_fc('quartz')
        fc.calculate_qpoint_phonon_modes(get_test_qpts())
        assert mocked_cext.call_args[0][-1] == n_threads

