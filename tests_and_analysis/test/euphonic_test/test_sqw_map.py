import unittest
import os
import numpy.testing as npt
import numpy as np
import unittest.mock as mock
import pytest
from euphonic import ureg, ForceConstants, QpointPhononModes
from ..utils import get_data_path


class TestSqwMapQpointPhononModesLZO(unittest.TestCase):

    def setUp(self):
        seedname = 'La2Zr2O7'
        data_path = get_data_path()
        self.sqw_path = os.path.join(data_path, 'sqw_map')
        self.sf_path = os.path.join(data_path, 'structure_factor', 'LZO')
        self.data = QpointPhononModes.from_castep(
            os.path.join(data_path, 'La2Zr2O7.phonon'))
        fm = ureg.fm
        self.scattering_lengths = {'La': 8.24*fm, 'Zr': 7.16*fm, 'O': 5.803*fm}
        self.ebins = np.arange(0, 100, 1.)*ureg('meV')

        # QpointPhononModes object for DW grid
        self.dw_data = QpointPhononModes.from_castep(
            os.path.join(self.sf_path, 'La2Zr2O7-grid.phonon'))

    def test_sqw_T5(self):
        sf = self.data.calculate_structure_factor(self.scattering_lengths)
        sqw_map = sf.calculate_sqw_map(self.ebins, temperature=5*ureg('K'))
        npt.assert_allclose(sqw_map.y_data.magnitude, self.ebins)
        expected_sqw_map = np.loadtxt(os.path.join(
            self.sqw_path, 'sqw_map_pdata_T5.txt'))
        npt.assert_allclose(sqw_map.z_data.to('bohr**2').magnitude, expected_sqw_map, rtol=1e-6)

    def test_sqw_T5_dw(self):
        dw5 = self.dw_data.calculate_debye_waller(5*ureg('K'))
        sf = self.data.calculate_structure_factor(self.scattering_lengths, dw=dw5)
        sqw_map = sf.calculate_sqw_map(self.ebins)

        npt.assert_allclose(sqw_map.y_data.magnitude, self.ebins)
        expected_sqw_map = np.loadtxt(os.path.join(
            self.sqw_path, 'sqw_map_pdata_T5_dw.txt'))
        npt.assert_allclose(sqw_map.z_data.to('bohr**2').magnitude, expected_sqw_map, rtol=1e-6)

    def test_sqw_T100(self):
        sf = self.data.calculate_structure_factor(self.scattering_lengths)
        sqw_map = sf.calculate_sqw_map(self.ebins, temperature=100*ureg('K'))

        npt.assert_allclose(sqw_map.y_data.magnitude, self.ebins)
        expected_sqw_map = np.loadtxt(os.path.join(
            self.sqw_path, 'sqw_map_pdata_T100.txt'))
        npt.assert_allclose(sqw_map.z_data.to('bohr**2').magnitude, expected_sqw_map, rtol=1e-6)

    def test_sqw_T100_dw(self):
        dw100 = self.dw_data.calculate_debye_waller(100*ureg('K'))
        sf = self.data.calculate_structure_factor(self.scattering_lengths, dw=dw100)
        sqw_map = sf.calculate_sqw_map(self.ebins)

        npt.assert_allclose(sqw_map.y_data.magnitude, self.ebins)
        expected_sqw_map = np.loadtxt(os.path.join(
            self.sqw_path, 'sqw_map_pdata_T100_dw.txt'))
        npt.assert_allclose(sqw_map.z_data.to('bohr**2').magnitude, expected_sqw_map, rtol=1e-6)


class TestSqwMapForceConstantsLZOSerial(unittest.TestCase):

    sf_path = os.path.join(get_data_path(), 'structure_factor', 'LZO')

    def setUp(self):
        data_path = get_data_path()
        self.interpolation_path = os.path.join(data_path, 'interpolation', 'LZO')
        self.sqw_path = os.path.join(data_path, 'sqw_map')
        pdata = QpointPhononModes.from_castep(
            os.path.join(data_path, 'La2Zr2O7.phonon'))
        fc = ForceConstants.from_castep(
            os.path.join(interpolation_path, 'La2Zr2O7.castep_bin'))
        self.data = fc.calculate_qpoint_phonon_modes(pdata.qpts, asr='realspace')
        fm = ureg.fm
        self.scattering_lengths = {'La': 8.24*fm, 'Zr': 7.16*fm, 'O': 5.803*fm}
        self.ebins = np.arange(0, 100, 1.)

    # Mock the calculate_structure_factor function and return a value from a
    # file because we're only testing sqw_map here
    @pytest.mark.skip(reason='Cant mock structure factor output yet')
    @mock.patch('euphonic.QpointPhononModes.calculate_structure_factor',
                return_value=np.loadtxt(os.path.join(
                    sf_path, 'sf_idata_T5.txt')))
    def test_sqw_T5(self, calculate_structure_factor_function):
        self.data.calculate_sqw_map(
            self.scattering_lengths, self.ebins, temperature=5*ureg('K'))

        npt.assert_allclose(self.data.sqw_ebins.magnitude, self.ebins)
        expected_sqw_map = np.loadtxt(os.path.join(
            self.sqw_path, 'sqw_map_idata_T5.txt'))
        # Mask out first few energy bins to avoid testing unstable Bragg peaks
        npt.assert_allclose(self.data.sqw_map[:, 5], expected_sqw_map[:, 5])

    @pytest.mark.skip(reason='Cant mock structure factor output yet')
    @mock.patch('euphonic.QpointPhononModes.calculate_structure_factor',
                return_value=np.loadtxt(os.path.join(
                    sf_path, 'sf_idata_T100.txt')))
    def test_sqw_T100(self, calculate_structure_factor_function):
        self.data.calculate_sqw_map(
            self.scattering_lengths, self.ebins, temperature=100*ureg('K'))

        npt.assert_allclose(self.data.sqw_ebins.magnitude, self.ebins)
        expected_sqw_map = np.loadtxt(os.path.join(
            self.sqw_path, 'sqw_map_idata_T100.txt'))
        npt.assert_allclose(self.data.sqw_map[:, 5], expected_sqw_map[:, 5])

    @pytest.mark.skip(reason='Cant mock structure factor output yet')
    @mock.patch('euphonic.QpointPhononModes.calculate_structure_factor',
                return_value=np.loadtxt(os.path.join(
                    sf_path, 'sf_idata_T100.txt')))
    def test_sqw_arguments_passed(self, calculate_structure_factor_function):
        temp = 100*ureg('K')
        dw = self.data.calculate_debye_waller(100*ureg('K'))
        self.data.calculate_sqw_map(
            self.scattering_lengths, self.ebins, temperature=temp,
            dw=dw)
        sf_args, sf_kwargs = calculate_structure_factor_function.call_args

        # Check calculate_structure_factor was called with the correct args
        # (make sure they were passed through from sqw_map)
        self.assertEqual(sf_args[0], self.scattering_lengths)
        self.assertEqual(sf_kwargs['temperature'], temp)
        self.assertEqual(sf_kwargs['calc_bose'], False)
        self.assertEqual(sf_kwargs['dw'], dw)

class TestSqwMapForceConstantsLZOSerialC(
    TestSqwMapForceConstantsLZOSerial):

    def setUp(self):
        self.seedname = 'La2Zr2O7'
        data_path = get_data_path()
        self.interpolation_path = os.path.join(data_path, 'interpolation', 'LZO')
        self.sqw_path = os.path.join(data_path, 'sqw_map')
        pdata = QpointPhononModes.from_castep(
            os.path.join(data_path, 'La2Zr2O7.phonon'))
        fc = ForceConstants.from_castep(
            os.path.join(interpolation_path, 'La2Zr2O7.castep_bin'))
        self.data = fc.calculate_qpoint_phonon_modes(
            pdata.qpts, asr='realspace', use_c=True,
            fall_back_on_python=False)
        fm = ureg.fm
        self.scattering_lengths = {'La': 8.24*fm, 'Zr': 7.16*fm, 'O': 5.803*fm}
        self.ebins = np.arange(0, 100, 1.)

class TestSqwMapForceConstantsLZOParallelC(
    TestSqwMapForceConstantsLZOSerial):

    def setUp(self):
        self.seedname = 'La2Zr2O7'
        data_path = get_data_path()
        self.interpolation_path = os.path.join(data_path, 'interpolation', 'LZO')
        self.sqw_path = os.path.join(data_path, 'sqw_map')
        pdata = QpointPhononModes.from_castep(
            os.path.join(data_path, 'La2Zr2O7.phonon'))
        fc = ForceConstants.from_castep(
            os.path.join(interpolation_path, 'La2Zr2O7.castep_bin'))
        self.data = fc.calculate_qpoint_phonon_modes(
            pdata.qpts, asr='realspace', use_c=True,
            n_threads=2, fall_back_on_python=False)
        fm = ureg.fm
        self.scattering_lengths = {'La': 8.24*fm, 'Zr': 7.16*fm, 'O': 5.803*fm}
        self.ebins = np.arange(0, 100, 1.)

