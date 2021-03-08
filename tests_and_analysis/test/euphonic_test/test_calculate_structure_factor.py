import json
import os

import pytest
import numpy.testing as npt

from euphonic import ureg, ForceConstants, QpointPhononModes
from tests_and_analysis.test.utils import (get_castep_path, get_phonopy_path,
    get_test_qpts)
from tests_and_analysis.test.euphonic_test.test_qpoint_phonon_modes import (
    get_qpt_ph_modes, get_qpt_ph_modes_dir)
from tests_and_analysis.test.euphonic_test.test_debye_waller import (
    get_dw)
from tests_and_analysis.test.euphonic_test.test_force_constants import (
    get_fc_dir)
from tests_and_analysis.test.euphonic_test.test_structure_factor import (
    check_structure_factor, get_sf_dir)


@pytest.mark.integration
class TestCalculateStructureFactorFromForceConstants:

    def get_multithreaded_kwargs(self, n_threads):
        kwargs = {}
        if n_threads == 0:
            kwargs['use_c'] = False
        else:
            kwargs['use_c'] = True
            kwargs['n_threads'] = n_threads
        kwargs['asr'] = 'reciprocal'
        return kwargs

    @pytest.fixture(params=[0, 1, 2])
    def get_quartz_qpt_ph_modes(self, request):
        fc = ForceConstants.from_castep(
            get_castep_path('quartz', 'quartz.castep_bin'))
        kwargs = self.get_multithreaded_kwargs(request.param)
        return fc.calculate_qpoint_phonon_modes(
            get_test_qpts('split'), **kwargs)

    @pytest.fixture(params=[0, 1, 2])
    def get_si2_qpt_ph_modes(self, request):
        fc = ForceConstants.from_castep(
            get_castep_path('Si2-sc-skew', 'Si2-sc-skew.castep_bin'))
        kwargs = self.get_multithreaded_kwargs(request.param)
        return fc.calculate_qpoint_phonon_modes(get_test_qpts(), **kwargs)

    @pytest.fixture(params=[0, 1, 2])
    def get_cahgo2_qpt_ph_modes(self, request):
        fc = ForceConstants.from_phonopy(
            path=get_phonopy_path('CaHgO2', ''),
            summary_name='mp-7041-20180417.yaml')
        kwargs = self.get_multithreaded_kwargs(request.param)
        return fc.calculate_qpoint_phonon_modes(get_test_qpts(), **kwargs)

    @pytest.mark.parametrize(
        'material, qpt_ph_modes, dw_file, expected_sf_file', [
            ('quartz', pytest.lazy_fixture('get_quartz_qpt_ph_modes'),
             'quartz_666_0K_debye_waller.json',
             'quartz_0K_fc_structure_factor.json'),
            ('quartz', pytest.lazy_fixture('get_quartz_qpt_ph_modes'),
             'quartz_666_300K_debye_waller.json',
             'quartz_300K_fc_structure_factor.json'),
            ('quartz', pytest.lazy_fixture('get_quartz_qpt_ph_modes'),
             None,
             'quartz_fc_structure_factor.json'),
            ('Si2-sc-skew', pytest.lazy_fixture('get_si2_qpt_ph_modes'),
             'Si2-sc-skew_666_300K_debye_waller.json',
             'Si2-sc-skew_300K_fc_structure_factor.json'),
            ('CaHgO2', pytest.lazy_fixture('get_cahgo2_qpt_ph_modes'),
             'CaHgO2_666_300K_debye_waller.json',
             'CaHgO2_300K_fc_structure_factor.json')])
    def test_calculate_structure_factor(self, material, qpt_ph_modes,
                                        dw_file, expected_sf_file):
        if dw_file is not None:
            sf = qpt_ph_modes.calculate_structure_factor(
                dw=get_dw(material, dw_file))
        else:
            sf = qpt_ph_modes.calculate_structure_factor()
        sf_file = os.path.join(get_sf_dir(material), expected_sf_file)
        expected_sf = sf.from_json_file(sf_file)
        check_structure_factor(sf, expected_sf, sf_atol=8e-14, sf_rtol=3e-4,
                               freq_atol=1e-4, freq_rtol=2e-5,
                               freq_gamma_atol=0.5, sf_gamma_atol=1e-7)


@pytest.mark.unit
class TestCalculateStructureFactorFromQpointPhononModes:

    def get_quartz_qpt_ph_modes():
        return QpointPhononModes.from_castep(
            get_castep_path('quartz', 'quartz_nosplit.phonon'))

    def get_si2_qpt_ph_modes():
        return QpointPhononModes.from_castep(
            get_castep_path('Si2-sc-skew', 'Si2-sc-skew.phonon'))

    def get_cahgo2_qpt_ph_modes():
        return QpointPhononModes.from_phonopy(
            path=get_phonopy_path('CaHgO2', ''),
            summary_name='mp-7041-20180417.yaml',
            phonon_name='qpoints.yaml')

    @pytest.mark.parametrize(
        'material, qpt_ph_modes, dw_file, expected_sf_file', [
            ('quartz', get_quartz_qpt_ph_modes(),
             'quartz_666_0K_debye_waller.json',
             'quartz_0K_structure_factor.json'),
            ('quartz', get_quartz_qpt_ph_modes(),
             'quartz_666_300K_debye_waller.json',
             'quartz_300K_structure_factor.json'),
            ('quartz', get_quartz_qpt_ph_modes(),
             None,
             'quartz_structure_factor.json'),
            ('Si2-sc-skew', get_si2_qpt_ph_modes(),
             'Si2-sc-skew_666_300K_debye_waller.json',
             'Si2-sc-skew_300K_structure_factor.json'),
            ('CaHgO2', get_cahgo2_qpt_ph_modes(),
             'CaHgO2_666_300K_debye_waller.json',
             'CaHgO2_300K_structure_factor.json')])
    def test_calculate_structure_factor(self, material, qpt_ph_modes,
                                        dw_file, expected_sf_file):
        if dw_file is not None:
            sf = qpt_ph_modes.calculate_structure_factor(
                dw=get_dw(material, dw_file))
        else:
            sf = qpt_ph_modes.calculate_structure_factor()
        sf_file = os.path.join(get_sf_dir(material), expected_sf_file)
        expected_sf = sf.from_json_file(sf_file)
        check_structure_factor(sf, expected_sf)

    @pytest.mark.parametrize(
        'qpt_ph_modes, dw_material, dw_file', [
            (get_quartz_qpt_ph_modes(),
             'Si2-sc-skew', 'Si2-sc-skew_666_300K_debye_waller.json')])
    def test_incompatible_debye_waller_raises_valueerror(
            self, qpt_ph_modes, dw_material, dw_file):
        with pytest.raises(ValueError):
            sf = qpt_ph_modes.calculate_structure_factor(
                dw=get_dw(dw_material, dw_file))


@pytest.mark.unit
class TestCalculateStructureFactorUsingReferenceData:
    @pytest.fixture
    def quartz_modes(self):
        return get_qpt_ph_modes('quartz')

    @pytest.fixture
    def fake_quartz_data(self):
        return {
            "description": "fake data for testing",
            "physical_property": {"coherent_scattering_length":
                                  {"__units__": "fm",
                                   "Si": {"__complex__": True,
                                          "real": 4.0, "imag": -0.70},
                                   "O": 5.803}}}

    @staticmethod
    def _dump_data(data, tmpdir, filename):
        filename = tmpdir.join(filename)
        with open(filename, 'wt') as fd:
            json.dump(data, fd)
        return str(filename)

    def test_structure_factor_with_named_ref(self, quartz_modes):
        fm = ureg['fm']
        sf_direct = quartz_modes.calculate_structure_factor(
            scattering_lengths={'Si': 4.1491*fm, 'O': 5.803*fm})
        sf_named = quartz_modes.calculate_structure_factor(
            scattering_lengths='Sears1992')

        aa2 = ureg['angstrom']**2
        npt.assert_allclose(sf_direct.structure_factors.to(aa2).magnitude,
                            sf_named.structure_factors.to(aa2).magnitude)

    def test_structure_factor_with_file_ref(self, quartz_modes,
                                            tmpdir, fake_quartz_data):
        fm = ureg['fm']

        filename = self._dump_data(fake_quartz_data, tmpdir, 'fake_data')

        sf_direct = quartz_modes.calculate_structure_factor(
            scattering_lengths={'Si': complex(4., -0.7)*fm, 'O': 5.803*fm})
        sf_from_file = quartz_modes.calculate_structure_factor(
            scattering_lengths=filename)

        aa2 = ureg['angstrom']**2
        npt.assert_allclose(sf_direct.structure_factors.to(aa2).magnitude,
                            sf_from_file.structure_factors.to(aa2).magnitude)
