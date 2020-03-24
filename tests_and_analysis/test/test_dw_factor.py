import os
import unittest
import numpy as np
import numpy.testing as npt
from euphonic.data.phonon import PhononData
from euphonic.data.interpolation import InterpolationData

class TestDWFactorLZO(unittest.TestCase):

    def setUp(self):
        seedname = 'La2Zr2O7-grid'
        path = os.path.join('data', 'structure_factor', 'LZO')
        self.data = PhononData.from_castep(seedname, path=path)
        self.dw_path = os.path.join('data','dw_factor', 'LZO')

    def test_dw_T5(self):
        dw = self.data._dw_coeff(5)
        expected_dw = np.reshape(
            np.loadtxt(os.path.join(self.dw_path, 'dw_T5.txt')),
            (self.data.n_ions, 3, 3))
        npt.assert_allclose(dw, expected_dw)

    def test_dw_T100(self):
        dw = self.data._dw_coeff(100)
        expected_dw = np.reshape(
            np.loadtxt(os.path.join(self.dw_path, 'dw_T100.txt')),
            (self.data.n_ions, 3, 3))
        npt.assert_allclose(dw, expected_dw, rtol=5e-7)

class TestDWFactorQuartz(unittest.TestCase):

    def setUp(self):
        self.seedname = 'quartz'
        self.path = os.path.join('data', 'interpolation', 'quartz')
        self.data = InterpolationData.from_castep(self.seedname, path=self.path)
        qpts = np.loadtxt(os.path.join('data', 'qgrid_444.txt'))
        self.data.calculate_fine_phonons(qpts, asr='reciprocal')
        self.dw_path = os.path.join('data', 'dw_factor', 'quartz')

    def test_dw_T5(self):
        dw = self.data._dw_coeff(5)
        expected_dw = np.reshape(
            np.loadtxt(os.path.join(self.dw_path, 'dw_T5.txt')),
            (self.data.n_ions, 3, 3))
        npt.assert_allclose(dw, expected_dw, atol=2e-14)

    def test_dw_T100(self):
        dw = self.data._dw_coeff(100)
        expected_dw = np.reshape(
            np.loadtxt(os.path.join(self.dw_path, 'dw_T100.txt')),
            (self.data.n_ions, 3, 3))
        npt.assert_allclose(dw, expected_dw, atol=5e-10)

    def test_empty_idata_raises_exception(self):
        empty_data = InterpolationData.from_castep(self.seedname, self.path)
        self.assertRaises(Exception, empty_data._dw_coeff)
