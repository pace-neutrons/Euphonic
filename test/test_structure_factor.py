import unittest
import warnings
import numpy.testing as npt
import numpy as np
from euphonic.data.phonon import PhononData
from euphonic.data.interpolation import InterpolationData


class TestStructureFactorPhononDataLZO(unittest.TestCase):

    def setUp(self):
        seedname = 'La2Zr2O7'
        phonon_path = 'test/data/'
        self.sf_path = 'test/data/structure_factor/LZO/'
        self.data = PhononData(seedname, path=phonon_path)
        self.scattering_lengths = {'La': 8.24, 'Zr': 7.16, 'O': 5.803}

    def test_sf_T5(self):
        sf = self.data.calculate_structure_factor(self.scattering_lengths, T=5)
        expected_sf = np.loadtxt(self.sf_path + 'sf_pdata_T5.txt')
        npt.assert_allclose(sf, expected_sf)

    def test_sf_T5_dw(self):
        sf = self.data.calculate_structure_factor(
            self.scattering_lengths, T=5, dw_arg='La2Zr2O7-grid',
            path=self.sf_path)
        expected_sf = np.loadtxt(self.sf_path + 'sf_pdata_dw_T5.txt')
        npt.assert_allclose(sf, expected_sf)

    def test_sf_T100(self):
        sf = self.data.calculate_structure_factor(
            self.scattering_lengths, T=100)
        expected_sf = np.loadtxt(self.sf_path + 'sf_pdata_T100.txt')
        npt.assert_allclose(sf, expected_sf)

    def test_sf_T100_dw(self):
        sf = self.data.calculate_structure_factor(
            self.scattering_lengths, T=100, dw_arg='La2Zr2O7-grid',
            path=self.sf_path)
        expected_sf = np.loadtxt(self.sf_path + 'sf_pdata_dw_T100.txt')
        npt.assert_allclose(sf, expected_sf)


class TestStructureFactorInterpolationDataLZO(unittest.TestCase):

    def setUp(self):
        # Need to separately test SF calculation with interpolated phonon data
        # to test eigenvector calculations
        self.seedname = 'La2Zr2O7'
        self.interpolation_path = 'test/data/interpolation/LZO'
        self.sf_path = 'test/data/structure_factor/LZO/'
        self.scattering_lengths = {'La': 8.24, 'Zr': 7.16, 'O': 5.803}
        qpts = np.loadtxt(self.sf_path + 'qpts.txt')

        self.data = InterpolationData(
            self.seedname, path=self.interpolation_path)
        self.data.calculate_fine_phonons(qpts, asr='realspace')

    def test_sf_T5(self):
        sf = self.data.calculate_structure_factor(self.scattering_lengths, T=5)
        expected_sf = np.loadtxt(self.sf_path + 'sf_idata_T5.txt')

        # Structure factors not necessarily equal due to degenerate modes
        # So sum structure factors over degenerate modes, these should be equal
        sf_sum = np.zeros(sf.shape)
        expected_sf_sum = np.zeros(sf.shape)
        for q in range(self.data.n_qpts):
            TOL = 1e-8
            diff = np.append(TOL + 1, np.diff(self.data.freqs[q].magnitude))
            unique_index = np.where(diff > TOL)[0]
            x = np.zeros(self.data.n_branches, dtype=np.int32)
            x[unique_index] = 1
            unique_modes = np.cumsum(x) - 1
            sf_sum[q, :len(unique_index)] = np.bincount(unique_modes, sf[q])
            expected_sf_sum[q, :len(unique_index)] = np.bincount(
                unique_modes, expected_sf[q])
        # Due to 1/w factor in structure factor, calculation can be unstable
        # at q=0. Test q=0 and non q=0 values separately with different
        # tolerances
        npt.assert_allclose(np.delete(sf_sum, -3, axis=0),
                            np.delete(expected_sf_sum, -3, axis=0))
        npt.assert_allclose(sf_sum[-3, 3:], expected_sf_sum[-3, 3:],
                            rtol=5e-3, atol=1e-12)

    def test_sf_T5_dw_grid(self):
        sf = self.data.calculate_structure_factor(
            self.scattering_lengths, T=5, dw_arg=[4, 4, 4],
            path=self.interpolation_path, asr='realspace')
        expected_sf = np.loadtxt(
            self.sf_path + 'sf_idata_dw_grid_T5.txt')
        sf_sum = np.zeros(sf.shape)
        expected_sf_sum = np.zeros(sf.shape)
        for q in range(self.data.n_qpts):
            TOL = 1e-8
            diff = np.append(TOL + 1, np.diff(self.data.freqs[q].magnitude))
            unique_index = np.where(diff > TOL)[0]
            x = np.zeros(self.data.n_branches, dtype=np.int32)
            x[unique_index] = 1
            unique_modes = np.cumsum(x) - 1
            sf_sum[q, :len(unique_index)] = np.bincount(unique_modes, sf[q])
            expected_sf_sum[q, :len(unique_index)] = np.bincount(
                unique_modes, expected_sf[q])
        npt.assert_allclose(np.delete(sf_sum, -3, axis=0),
                            np.delete(expected_sf_sum, -3, axis=0))
        npt.assert_allclose(sf_sum[-3, 3:], expected_sf_sum[-3, 3:],
                            rtol=5e-3, atol=1e-12)

    def test_sf_T5_dw_seedname(self):
        sf = self.data.calculate_structure_factor(
            self.scattering_lengths, T=5, dw_arg='La2Zr2O7-grid',
            path=self.sf_path)
        expected_sf = np.loadtxt(
            self.sf_path + 'sf_idata_dw_seedname_T5.txt')

        sf_sum = np.zeros(sf.shape)
        expected_sf_sum = np.zeros(sf.shape)
        for q in range(self.data.n_qpts):
            TOL = 1e-8
            diff = np.append(TOL + 1, np.diff(self.data.freqs[q].magnitude))
            unique_index = np.where(diff > TOL)[0]
            x = np.zeros(self.data.n_branches, dtype=np.int32)
            x[unique_index] = 1
            unique_modes = np.cumsum(x) - 1
            sf_sum[q, :len(unique_index)] = np.bincount(unique_modes, sf[q])
            expected_sf_sum[q, :len(unique_index)] = np.bincount(
                unique_modes, expected_sf[q])
        npt.assert_allclose(np.delete(sf_sum, -3, axis=0),
                            np.delete(expected_sf_sum, -3, axis=0))
        npt.assert_allclose(sf_sum[-3, 3:], expected_sf_sum[-3, 3:],
                            rtol=5e-3, atol=1e-12)

    def test_sf_T100(self):
        sf = self.data.calculate_structure_factor(
            self.scattering_lengths, T=100)
        expected_sf = np.loadtxt(self.sf_path + 'sf_idata_T100.txt')

        sf_sum = np.zeros(sf.shape)
        expected_sf_sum = np.zeros(sf.shape)
        for q in range(self.data.n_qpts):
            TOL = 1e-8
            diff = np.append(TOL + 1, np.diff(self.data.freqs[q].magnitude))
            unique_index = np.where(diff > TOL)[0]
            x = np.zeros(self.data.n_branches, dtype=np.int32)
            x[unique_index] = 1
            unique_modes = np.cumsum(x) - 1
            sf_sum[q, :len(unique_index)] = np.bincount(unique_modes, sf[q])
            expected_sf_sum[q, :len(unique_index)] = np.bincount(
                unique_modes, expected_sf[q])
        npt.assert_allclose(np.delete(sf_sum, -3, axis=0),
                            np.delete(expected_sf_sum, -3, axis=0))
        npt.assert_allclose(sf_sum[-3, 3:], expected_sf_sum[-3, 3:],
                            rtol=5e-3, atol=1e-12)

    def test_sf_T100_dw_grid(self):
        sf = self.data.calculate_structure_factor(
            self.scattering_lengths, T=100, dw_arg=[4, 4, 4],
            path=self.interpolation_path, asr='realspace')
        expected_sf = np.loadtxt(
            self.sf_path + 'sf_idata_dw_grid_T100.txt')

        sf_sum = np.zeros(sf.shape)
        expected_sf_sum = np.zeros(sf.shape)
        for q in range(self.data.n_qpts):
            TOL = 1e-8
            diff = np.append(TOL + 1, np.diff(self.data.freqs[q].magnitude))
            unique_index = np.where(diff > TOL)[0]
            x = np.zeros(self.data.n_branches, dtype=np.int32)
            x[unique_index] = 1
            unique_modes = np.cumsum(x) - 1
            sf_sum[q, :len(unique_index)] = np.bincount(unique_modes, sf[q])
            expected_sf_sum[q, :len(unique_index)] = np.bincount(
                unique_modes, expected_sf[q])
        npt.assert_allclose(np.delete(sf_sum, -3, axis=0),
                            np.delete(expected_sf_sum, -3, axis=0))
        npt.assert_allclose(sf_sum[-3, 3:], expected_sf_sum[-3, 3:],
                            rtol=5e-3, atol=1e-12)

    def test_sf_T100_dw_seedname(self):
        sf = self.data.calculate_structure_factor(
            self.scattering_lengths, T=100, dw_arg='La2Zr2O7-grid',
            path=self.sf_path)
        expected_sf = np.loadtxt(
            self.sf_path + 'sf_idata_dw_seedname_T100.txt')

        sf_sum = np.zeros(sf.shape)
        expected_sf_sum = np.zeros(sf.shape)
        for q in range(self.data.n_qpts):
            TOL = 1e-8
            diff = np.append(TOL + 1, np.diff(self.data.freqs[q].magnitude))
            unique_index = np.where(diff > TOL)[0]
            x = np.zeros(self.data.n_branches, dtype=np.int32)
            x[unique_index] = 1
            unique_modes = np.cumsum(x) - 1
            sf_sum[q, :len(unique_index)] = np.bincount(unique_modes, sf[q])
            expected_sf_sum[q, :len(unique_index)] = np.bincount(
                unique_modes, expected_sf[q])
        npt.assert_allclose(np.delete(sf_sum, -3, axis=0),
                            np.delete(expected_sf_sum, -3, axis=0))
        npt.assert_allclose(sf_sum[-3, 3:], expected_sf_sum[-3, 3:],
                            rtol=5e-3, atol=1e-12)

    def test_empty_interpolation_data_raises_warning(self):
        empty_data = InterpolationData(
            self.seedname, path=self.interpolation_path)
        # Test that trying to call structure factor on an empty object
        # raises a warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            empty_data.calculate_structure_factor(self.scattering_lengths)
            assert issubclass(w[-1].category, UserWarning)


class TestStructureFactorInterpolationDataQuartz(unittest.TestCase):

    def setUp(self):
        # Need to separately test SF calculation with interpolated phonon data
        # to test eigenvector calculations
        self.seedname = 'quartz'
        self.interpolation_path = 'test/data/interpolation/quartz'
        self.sf_path = 'test/data/structure_factor/quartz/'
        self.scattering_lengths = {'Si': 4.1491, 'O': 5.803}
        qpts = np.loadtxt(self.sf_path + 'qpts.txt')

        self.data = InterpolationData(
            self.seedname, path=self.interpolation_path)
        self.data.calculate_fine_phonons(qpts, asr='reciprocal')

    def test_sf_T0(self):
        sf = self.data.calculate_structure_factor(self.scattering_lengths, T=0)
        expected_sf = np.loadtxt(self.sf_path + 'sf_idata_T0.txt')

        # Structure factors not necessarily equal due to degenerate modes
        # So sum structure factors over degenerate modes, these should be equal
        sf_sum = np.zeros(sf.shape)
        expected_sf_sum = np.zeros(sf.shape)
        for q in range(self.data.n_qpts):
            TOL = 1e-8
            diff = np.append(TOL + 1, np.diff(self.data.freqs[q].magnitude))
            unique_index = np.where(diff > TOL)[0]
            x = np.zeros(self.data.n_branches, dtype=np.int32)
            x[unique_index] = 1
            unique_modes = np.cumsum(x) - 1
            sf_sum[q, :len(unique_index)] = np.bincount(unique_modes, sf[q])
            expected_sf_sum[q, :len(unique_index)] = np.bincount(
                unique_modes, expected_sf[q])
        # Due to 1/w factor in structure factor, calculation can be unstable
        # at q=0 for acoustic modes. Don't test these
        gamma_qpts = [16]  # Index of gamma pts in qpts array
        npt.assert_allclose(np.delete(sf_sum, gamma_qpts, axis=0),
                            np.delete(expected_sf_sum, gamma_qpts, axis=0),
                            rtol=7e-4, atol=3e-16)
        npt.assert_allclose(sf_sum[gamma_qpts, 3:],
                            expected_sf_sum[gamma_qpts, 3:],
                            rtol=8e-6, atol=3e-14)

    def test_sf_T0_dw_grid(self):
        sf = self.data.calculate_structure_factor(
            self.scattering_lengths, T=0, dw_arg=[4, 4, 4],
            path=self.interpolation_path, asr='reciprocal')
        expected_sf = np.loadtxt(
            self.sf_path + 'sf_idata_dw_grid_T0.txt')
        sf_sum = np.zeros(sf.shape)
        expected_sf_sum = np.zeros(sf.shape)
        for q in range(self.data.n_qpts):
            TOL = 1e-8
            diff = np.append(TOL + 1, np.diff(self.data.freqs[q].magnitude))
            unique_index = np.where(diff > TOL)[0]
            x = np.zeros(self.data.n_branches, dtype=np.int32)
            x[unique_index] = 1
            unique_modes = np.cumsum(x) - 1
            sf_sum[q, :len(unique_index)] = np.bincount(unique_modes, sf[q])
            expected_sf_sum[q, :len(unique_index)] = np.bincount(
                unique_modes, expected_sf[q])
        gamma_qpts = [16]
        npt.assert_allclose(np.delete(sf_sum, gamma_qpts, axis=0),
                            np.delete(expected_sf_sum, gamma_qpts, axis=0),
                            rtol=2e-3, atol=3e-16)
        npt.assert_allclose(sf_sum[gamma_qpts, 3:],
                            expected_sf_sum[gamma_qpts, 3:],
                            rtol=8e-6, atol=3e-14)

    def test_sf_T100(self):
        sf = self.data.calculate_structure_factor(
            self.scattering_lengths, T=100)
        expected_sf = np.loadtxt(self.sf_path + 'sf_idata_T100.txt')

        sf_sum = np.zeros(sf.shape)
        expected_sf_sum = np.zeros(sf.shape)
        for q in range(self.data.n_qpts):
            TOL = 1e-8
            diff = np.append(TOL + 1, np.diff(self.data.freqs[q].magnitude))
            unique_index = np.where(diff > TOL)[0]
            x = np.zeros(self.data.n_branches, dtype=np.int32)
            x[unique_index] = 1
            unique_modes = np.cumsum(x) - 1
            sf_sum[q, :len(unique_index)] = np.bincount(unique_modes, sf[q])
            expected_sf_sum[q, :len(unique_index)] = np.bincount(
                unique_modes, expected_sf[q])
        gamma_qpts = [16]
        npt.assert_allclose(np.delete(sf_sum, gamma_qpts, axis=0),
                            np.delete(expected_sf_sum, gamma_qpts, axis=0),
                            rtol=7e-4, atol=3e-16)
        npt.assert_allclose(sf_sum[gamma_qpts, 3:],
                            expected_sf_sum[gamma_qpts, 3:],
                            rtol=8e-6, atol=3e-14)

    def test_sf_T100_dw_grid(self):
        sf = self.data.calculate_structure_factor(
            self.scattering_lengths, T=100, dw_arg=[4, 4, 4],
            path=self.interpolation_path, asr='reciprocal')
        expected_sf = np.loadtxt(
            self.sf_path + 'sf_idata_dw_grid_T100.txt')

        sf_sum = np.zeros(sf.shape)
        expected_sf_sum = np.zeros(sf.shape)
        for q in range(self.data.n_qpts):
            TOL = 1e-8
            diff = np.append(TOL + 1, np.diff(self.data.freqs[q].magnitude))
            unique_index = np.where(diff > TOL)[0]
            x = np.zeros(self.data.n_branches, dtype=np.int32)
            x[unique_index] = 1
            unique_modes = np.cumsum(x) - 1
            sf_sum[q, :len(unique_index)] = np.bincount(unique_modes, sf[q])
            expected_sf_sum[q, :len(unique_index)] = np.bincount(
                unique_modes, expected_sf[q])
        gamma_qpts = [16]
        npt.assert_allclose(np.delete(sf_sum, gamma_qpts, axis=0),
                            np.delete(expected_sf_sum, gamma_qpts, axis=0),
                            rtol=7e-4, atol=3e-16)
        npt.assert_allclose(sf_sum[gamma_qpts, 3:],
                            expected_sf_sum[gamma_qpts, 3:],
                            rtol=8e-6, atol=3e-14)
