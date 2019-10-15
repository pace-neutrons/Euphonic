import unittest
import numpy.testing as npt
import numpy as np
# Before Python 3.3 mock is an external module
try:
    import unittest.mock as mock
except ImportError:
    import mock
from euphonic.data.phonon import PhononData
from euphonic.data.interpolation import InterpolationData


class TestSqwMapPhononDataLZO(unittest.TestCase):

    def setUp(self):
        seedname = 'La2Zr2O7'
        phonon_path = 'test/data/'
        self.sqw_path = 'test/data/sqw_map/'
        self.sf_path = 'test/data/structure_factor/LZO/'
        self.data = PhononData(seedname, path=phonon_path)
        self.scattering_lengths = {'La': 8.24, 'Zr': 7.16, 'O': 5.803}
        self.ebins = np.arange(0, 100, 1.)

    def test_sqw_T5(self):
        self.data.calculate_sqw_map(self.scattering_lengths, self.ebins, T=5)

        npt.assert_allclose(self.data.sqw_ebins.magnitude, self.ebins)
        expected_sqw_map = np.loadtxt(self.sqw_path + 'sqw_map_pdata_T5.txt')
        npt.assert_allclose(self.data.sqw_map, expected_sqw_map)

    def test_sqw_T5_dw(self):
        self.data.calculate_sqw_map(
            self.scattering_lengths, self.ebins, T=5, dw_arg='La2Zr2O7-grid',
            path=self.sf_path)

        npt.assert_allclose(self.data.sqw_ebins.magnitude, self.ebins)
        expected_sqw_map = np.loadtxt(
            self.sqw_path + 'sqw_map_pdata_T5_dw.txt')
        npt.assert_allclose(self.data.sqw_map, expected_sqw_map)

    def test_sqw_T100(self):
        self.data.calculate_sqw_map(self.scattering_lengths, self.ebins, T=100)

        npt.assert_allclose(self.data.sqw_ebins.magnitude, self.ebins)
        expected_sqw_map = np.loadtxt(self.sqw_path + 'sqw_map_pdata_T100.txt')
        npt.assert_allclose(self.data.sqw_map, expected_sqw_map)

    def test_sqw_T100_dw(self):
        self.data.calculate_sqw_map(
            self.scattering_lengths, self.ebins, T=100, dw_arg='La2Zr2O7-grid',
            path=self.sf_path)

        npt.assert_allclose(self.data.sqw_ebins.magnitude, self.ebins)
        expected_sqw_map = np.loadtxt(
            self.sqw_path + 'sqw_map_pdata_T100_dw.txt')
        npt.assert_allclose(self.data.sqw_map, expected_sqw_map)


class TestSqwMapInterpolationDataLZOSerial(unittest.TestCase):

    sf_path = 'test/data/structure_factor/LZO/'

    def setUp(self):
        self.seedname = 'La2Zr2O7'
        phonon_path = 'test/data/'
        self.interpolation_path = 'test/data/interpolation/LZO'
        self.sqw_path = 'test/data/sqw_map/'
        pdata = PhononData(self.seedname, path=phonon_path)
        self.data = InterpolationData(
            self.seedname, path=self.interpolation_path)
        self.data.calculate_fine_phonons(pdata.qpts, asr='realspace')
        self.scattering_lengths = {'La': 8.24, 'Zr': 7.16, 'O': 5.803}
        self.ebins = np.arange(0, 100, 1.)

    # Mock the calculate_structure_factor function and return a value from a
    # file because we're only testing sqw_map here
    @mock.patch('euphonic.data.phonon.PhononData.calculate_structure_factor',
                return_value=np.loadtxt(sf_path + 'sf_idata_T5.txt'))
    def test_sqw_T5(self, calculate_structure_factor_function):
        self.data.calculate_sqw_map(self.scattering_lengths, self.ebins, T=5)

        npt.assert_allclose(self.data.sqw_ebins.magnitude, self.ebins)
        expected_sqw_map = np.loadtxt(self.sqw_path + 'sqw_map_idata_T5.txt')
        # Mask out first few energy bins to avoid testing unstable Bragg peaks
        npt.assert_allclose(self.data.sqw_map[:, 5], expected_sqw_map[:, 5])

    @mock.patch('euphonic.data.phonon.PhononData.calculate_structure_factor',
                return_value=np.loadtxt(sf_path + 'sf_idata_T100.txt'))
    def test_sqw_T100(self, calculate_structure_factor_function):
        self.data.calculate_sqw_map(self.scattering_lengths, self.ebins, T=100)

        npt.assert_allclose(self.data.sqw_ebins.magnitude, self.ebins)
        expected_sqw_map = np.loadtxt(self.sqw_path + 'sqw_map_idata_T100.txt')
        npt.assert_allclose(self.data.sqw_map[:, 5], expected_sqw_map[:, 5])

    @mock.patch('euphonic.data.phonon.PhononData.calculate_structure_factor',
                return_value=np.loadtxt(sf_path + 'sf_idata_T100.txt'))
    def test_sqw_arguments_passed(self, calculate_structure_factor_function):
        self.data.calculate_sqw_map(
            self.scattering_lengths, self.ebins, T=100, dw_arg=[4, 4, 4],
            asr='reciprocal', model='CASTEP')
        sf_args, sf_kwargs = calculate_structure_factor_function.call_args

        # Check calculate_structure_factor was called with the correct args
        # (make sure they were passed through from sqw_map)
        self.assertEqual(sf_args[0], self.scattering_lengths)
        self.assertEqual(sf_kwargs['T'], 100)
        self.assertEqual(sf_kwargs['calc_bose'], False)
        self.assertEqual(sf_kwargs['dw_arg'], [4, 4, 4])
        self.assertEqual(sf_kwargs['asr'], 'reciprocal')
        self.assertEqual(sf_kwargs['model'], 'CASTEP')

class TestSqwMapInterpolationDataLZOParallel(
    TestSqwMapInterpolationDataLZOSerial):

    def setUp(self):
        self.seedname = 'La2Zr2O7'
        phonon_path = 'test/data/'
        self.interpolation_path = 'test/data/interpolation/LZO'
        self.sqw_path = 'test/data/sqw_map/'
        pdata = PhononData(self.seedname, path=phonon_path)
        self.data = InterpolationData(
            self.seedname, path=self.interpolation_path)
        self.data.calculate_fine_phonons(pdata.qpts, asr='realspace', nprocs=2)
        self.scattering_lengths = {'La': 8.24, 'Zr': 7.16, 'O': 5.803}
        self.ebins = np.arange(0, 100, 1.)
