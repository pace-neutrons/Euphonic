import unittest
import warnings
import numpy.testing as npt
import numpy as np
from simphony.data.phonon import PhononData
from simphony.data.interpolation import InterpolationData


class TestStructureFactorPhononDataLZO(unittest.TestCase):

    def setUp(self):
        seedname = 'La2Zr2O7'
        phonon_path = 'test/data/'
        self.sf_path = 'test/data/scattering/'
        self.data = PhononData(seedname, path=phonon_path)
        self.scattering_lengths = {'La': 8.24, 'Zr': 7.16, 'O': 5.803}

    def test_sf_T5(self):
        sf = self.data.structure_factor(self.scattering_lengths, T=5)
        expected_sf = np.loadtxt(self.sf_path + 'sf_pdata_T5.txt')
        npt.assert_allclose(sf, expected_sf)

    def test_sf_T5_dw(self):
        sf = self.data.structure_factor(
            self.scattering_lengths, T=5, dw_arg='La2Zr2O7-grid',
            path=self.sf_path)
        expected_sf = np.loadtxt(self.sf_path + 'sf_pdata_dw_T5.txt')
        npt.assert_allclose(sf, expected_sf)

    def test_sf_T100(self):
        sf = self.data.structure_factor(self.scattering_lengths, T=100)
        expected_sf = np.loadtxt(self.sf_path + 'sf_pdata_T100.txt')
        npt.assert_allclose(sf, expected_sf)

    def test_sf_T100_dw(self):
        sf = self.data.structure_factor(
            self.scattering_lengths, T=100, dw_arg='La2Zr2O7-grid',
            path=self.sf_path)
        expected_sf = np.loadtxt(self.sf_path + 'sf_pdata_dw_T100.txt')
        npt.assert_allclose(sf, expected_sf)


class TestStructureFactorInterpolationDataLZO(unittest.TestCase):

    def setUp(self):
        # Need to separately test SF calculation with interpolated phonon data
        # to test eigenvector calculations
        self.seedname = 'La2Zr2O7'
        phonon_path = 'test/data/'
        self.interpolation_path = 'test/data/interpolation/LZO'
        self.sf_path = 'test/data/scattering/'
        pdata = PhononData(self.seedname, path=phonon_path)
        self.data = InterpolationData(
            self.seedname, path=self.interpolation_path)
        self.data.calculate_fine_phonons(pdata.qpts, asr='realspace')
        self.scattering_lengths = {'La': 8.24, 'Zr': 7.16, 'O': 5.803}

        # Due to 1/w factor in structure factor, calculation can be unstable
        # at q=0. Create a mask to mask out gamma values and test these
        # separately with a higher tolerance
        mask = np.ones((self.data.freqs.shape))
        mask[-3, :] = 0  # Mask Gamma point nodes
        self.mask = mask

    def test_sf_T5(self):
        sf = self.data.structure_factor(self.scattering_lengths, T=5)
        expected_sf = np.loadtxt(self.sf_path + 'sf_idata_T5.txt')
        # Due to 1/w factor in structure factor, calculation can be unstable
        # at q=0. Test q=0 and non q=0 values separately with different
        # tolerances
        #npt.assert_allclose(np.delete(sf, -3, axis=0),
        #                    np.delete(expected_sf, -3, axis=0), atol=2e-13)
        # Test gamma point values
        # Also exclude acoustic modes as they can be very unstable due to 1/w
        #npt.assert_allclose(sf[-3, 3:], expected_sf[-3, 3:],
        #                    rtol=5e-3, atol=4e-10)
        sf_sum = np.zeros(sf.shape)
        expected_sf_sum = np.zeros(sf.shape)
        for q in range(self.data.n_qpts):
            TOL = 1e-8
            diff = np.append(TOL + 1, np.diff(self.data.freqs[q]))
            unique_index = np.where(diff > TOL)[0]
            x = np.zeros(self.data.n_branches, dtype=np.int32)
            x[unique_index] = 1
            unique_modes = np.cumsum(x) - 1
            sf_sum[q, :len(unique_index)] = np.bincount(unique_modes, sf[q])
            expected_sf_sum[q, :len(unique_index)] = np.bincount(unique_modes, expected_sf[q])
        npt.assert_allclose(np.delete(sf_sum, -3, axis=0),
                            np.delete(expected_sf_sum, -3, axis=0))
        npt.assert_allclose(sf_sum[-3, 3:], expected_sf_sum[-3, 3:],
                            rtol=5e-3, atol=1e-12)



    def test_sf_T5_dw_grid(self):
        sf = self.data.structure_factor(
            self.scattering_lengths, T=5, dw_arg=[4, 4, 4],
            path=self.interpolation_path, asr='realspace')*self.mask
        expected_sf = np.loadtxt(
            self.sf_path + 'sf_idata_dw_grid_T5.txt')*self.mask
        npt.assert_allclose(np.delete(sf, -3, axis=0),
                            np.delete(expected_sf, -3, axis=0), atol=2e-13)
        npt.assert_allclose(sf[-3], expected_sf[-3], rtol=5e-3, atol=1e-10)

    def test_sf_T5_dw_seedname(self):
        sf = self.data.structure_factor(
            self.scattering_lengths, T=5, dw_arg='La2Zr2O7-grid',
            path=self.sf_path)*self.mask
        expected_sf = np.loadtxt(
            self.sf_path + 'sf_idata_dw_seedname_T5.txt')*self.mask
        npt.assert_allclose(np.delete(sf, -3, axis=0),
                            np.delete(expected_sf, -3, axis=0), atol=2e-13)
        npt.assert_allclose(sf[-3], expected_sf[-3], rtol=5e-3, atol=1e-10)

    def test_sf_T100(self):
        sf = self.data.structure_factor(
            self.scattering_lengths, T=100)*self.mask
        expected_sf = np.loadtxt(self.sf_path + 'sf_idata_T100.txt')*self.mask
        npt.assert_allclose(np.delete(sf, -3, axis=0),
                            np.delete(expected_sf, -3, axis=0), atol=2e-13)
        npt.assert_allclose(sf[-3], expected_sf[-3], rtol=5e-3, atol=1e-10)

    def test_sf_T100_dw_grid(self):
        sf = self.data.structure_factor(
            self.scattering_lengths, T=100, dw_arg=[4, 4, 4],
            path=self.interpolation_path, asr='realspace')*self.mask
        expected_sf = np.loadtxt(
            self.sf_path + 'sf_idata_dw_grid_T100.txt')*self.mask
        npt.assert_allclose(np.delete(sf, -3, axis=0),
                            np.delete(expected_sf, -3, axis=0), atol=2e-13)
        npt.assert_allclose(sf[-3], expected_sf[-3], rtol=5e-3, atol=1e-10)

    def test_sf_T100_dw_seedname(self):
        sf = self.data.structure_factor(
            self.scattering_lengths, T=100, dw_arg='La2Zr2O7-grid',
            path=self.sf_path)*self.mask
        expected_sf = np.loadtxt(
            self.sf_path + 'sf_idata_dw_seedname_T100.txt')*self.mask
        npt.assert_allclose(np.delete(sf, -3, axis=0),
                            np.delete(expected_sf, -3, axis=0), atol=2e-13)
        npt.assert_allclose(sf[-3], expected_sf[-3], rtol=5e-3, atol=1e-10)

    def test_empty_interpolation_data_raises_warning(self):
        empty_data = InterpolationData(
            self.seedname, path=self.interpolation_path)
        # Test that trying to call structure factor on an empty object
        # raises a warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            empty_data.structure_factor(self.scattering_lengths)
            assert issubclass(w[-1].category, UserWarning)
