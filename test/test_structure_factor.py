import unittest
import numpy.testing as npt
import numpy as np
from casteppy.data.phonon import PhononData
from casteppy.calculate.scattering import structure_factor


class TestStructureFactorLZO(unittest.TestCase):

    def setUp(self):
        seedname = 'La2Zr2O7'
        phonon_path = 'test/data/'
        self.sf_path = 'test/data/scattering/'
        self.data = PhononData(seedname, phonon_path)
        self.scattering_lengths = {'La': 8.24, 'Zr': 7.16, 'O': 5.803}

    def test_sf_T5(self):
        sf = structure_factor(self.data, self.scattering_lengths, T=5)
        expected_sf = np.loadtxt(self.sf_path + 'sf_T5.txt')
        npt.assert_allclose(sf, expected_sf)

    def test_sf_T100(self):
        sf = structure_factor(self.data, self.scattering_lengths, T=100)
        expected_sf = np.loadtxt(self.sf_path + 'sf_T100.txt')
        npt.assert_allclose(sf, expected_sf)