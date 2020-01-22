# Remove current directory from path, so when running tests it uses installed
# euphonic, not from the ./euphonic foler (otherwise import euphonic._euphonic
# as euphonic_c won't work so C tests can't run
import sys
try:
    sys.path.remove('')
except ValueError:
    pass

import os
import unittest
import numpy.testing as npt
import numpy as np
from euphonic.data.phonon import PhononData
from euphonic.data.interpolation import InterpolationData


class TestStructureFactorPhononDataLZO(unittest.TestCase):

    def setUp(self):
        seedname = 'La2Zr2O7'
        phonon_path = 'data'
        self.sf_path = os.path.join('data', 'structure_factor', 'LZO')
        self.data = PhononData.from_castep(seedname, path=phonon_path)
        self.scattering_lengths = {'La': 8.24, 'Zr': 7.16, 'O': 5.803}
        self.dw_data = PhononData.from_castep(
            'La2Zr2O7-grid', path=self.sf_path)

    def test_sf_T5(self):
        sf = self.data.calculate_structure_factor(self.scattering_lengths, T=5)
        expected_sf = np.loadtxt(os.path.join(self.sf_path, 'sf_pdata_T5.txt'))
        npt.assert_allclose(sf, expected_sf, rtol=2e-7)

    def test_sf_T5_dw(self):
        sf = self.data.calculate_structure_factor(
            self.scattering_lengths, T=5, dw_data=self.dw_data)
        expected_sf = np.loadtxt(os.path.join(
            self.sf_path, 'sf_pdata_dw_T5.txt'))
        npt.assert_allclose(sf, expected_sf, rtol=2e-7, atol=1e-18)

    def test_sf_T100(self):
        sf = self.data.calculate_structure_factor(
            self.scattering_lengths, T=100)
        expected_sf = np.loadtxt(os.path.join(
            self.sf_path, 'sf_pdata_T100.txt'))
        npt.assert_allclose(sf, expected_sf, rtol=2e-7)

    def test_sf_T100_dw(self):
        sf = self.data.calculate_structure_factor(
            self.scattering_lengths, T=100, dw_data=self.dw_data)
        expected_sf = np.loadtxt(os.path.join(
            self.sf_path, 'sf_pdata_dw_T100.txt'))
        npt.assert_allclose(sf, expected_sf, rtol=2e-6)


class TestStructureFactorInterpolationDataLZOSerial(unittest.TestCase):

    def setUp(self):
        # Need to separately test SF calculation with interpolated phonon data
        # to test eigenvector calculations
        self.seedname = 'La2Zr2O7'
        self.interpolation_path = os.path.join('data', 'interpolation', 'LZO')
        self.sf_path = os.path.join('data', 'structure_factor' , 'LZO')
        self.scattering_lengths = {'La': 8.24, 'Zr': 7.16, 'O': 5.803}
        qpts = np.loadtxt(os.path.join(self.sf_path, 'qpts.txt'))

        self.data = InterpolationData.from_castep(
            self.seedname, path=self.interpolation_path)
        self.data.calculate_fine_phonons(qpts, asr='realspace')

        # InterpolationData object for DW grid
        self.dw_idata = InterpolationData.from_castep(
            self.seedname, path=self.interpolation_path)
        self.dw_idata.calculate_fine_phonons(
            np.loadtxt(os.path.join('data', 'qgrid_444.txt')), asr='realspace')

        # PhononData object for DW grid
        self.dw_pdata = PhononData.from_castep(
            'La2Zr2O7-grid', path=self.sf_path)

    def test_sf_T5(self):
        sf = self.data.calculate_structure_factor(self.scattering_lengths, T=5)
        expected_sf = np.loadtxt(os.path.join(self.sf_path, 'sf_idata_T5.txt'))

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

    def test_sf_T5_dw_idata(self):
        sf = self.data.calculate_structure_factor(
            self.scattering_lengths, T=5, dw_data=self.dw_idata)
        expected_sf = np.loadtxt(os.path.join(
            self.sf_path, 'sf_idata_dw_grid_T5.txt'))
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

    def test_sf_T5_dw_pdata(self):
        sf = self.data.calculate_structure_factor(
            self.scattering_lengths, T=5, dw_data=self.dw_pdata)
        expected_sf = np.loadtxt(os.path.join(
            self.sf_path, 'sf_idata_dw_seedname_T5.txt'))

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
                            np.delete(expected_sf_sum, -3, axis=0), rtol=1e-6)
        npt.assert_allclose(sf_sum[-3, 3:], expected_sf_sum[-3, 3:],
                            rtol=5e-3, atol=1e-12)

    def test_sf_T100(self):
        sf = self.data.calculate_structure_factor(
            self.scattering_lengths, T=100)
        expected_sf = np.loadtxt(os.path.join(
            self.sf_path, 'sf_idata_T100.txt'))

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

    def test_sf_T100_dw_idata(self):
        sf = self.data.calculate_structure_factor(
            self.scattering_lengths, T=100, dw_data=self.dw_idata)
        expected_sf = np.loadtxt(os.path.join(
            self.sf_path, 'sf_idata_dw_grid_T100.txt'))

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
                            np.delete(expected_sf_sum, -3, axis=0), rtol=1e-6)
        npt.assert_allclose(sf_sum[-3, 3:], expected_sf_sum[-3, 3:],
                            rtol=5e-3, atol=1e-12)

    def test_sf_T100_dw_pdata(self):
        sf = self.data.calculate_structure_factor(
            self.scattering_lengths, T=100, dw_data=self.dw_pdata)
        expected_sf = np.loadtxt(os.path.join(
            self.sf_path, 'sf_idata_dw_seedname_T100.txt'))

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
                            np.delete(expected_sf_sum, -3, axis=0), rtol=1e-6)
        npt.assert_allclose(sf_sum[-3, 3:], expected_sf_sum[-3, 3:],
                            rtol=5e-3, atol=1e-12)

    def test_empty_interpolation_data_raises_exception(self):
        empty_data = InterpolationData.from_castep(
            self.seedname, path=self.interpolation_path)
        # Test that trying to call structure factor on an empty object
        # raises an Exception
        self.assertRaises(Exception, empty_data.calculate_structure_factor)

    def test_incompatible_dw_arg_raises_exception(self):
        quartz_data = InterpolationData.from_castep(
            'quartz', path=self.interpolation_path + '/../quartz')
        # Test that trying to call structure factor with a dw_arg for another
        # material raises an Exception
        self.assertRaises(Exception, self.data.calculate_structure_factor,
            self.scattering_lengths, dw_data=quartz_data)

class TestStructureFactorInterpolationDataLZOSerialC(
    TestStructureFactorInterpolationDataLZOSerial):

    def setUp(self):
        # Need to separately test SF calculation with interpolated phonon data
        # to test eigenvector calculations
        self.seedname = 'La2Zr2O7'
        self.interpolation_path = os.path.join('data', 'interpolation', 'LZO')
        self.sf_path = os.path.join('data', 'structure_factor', 'LZO')
        self.scattering_lengths = {'La': 8.24, 'Zr': 7.16, 'O': 5.803}
        qpts = np.loadtxt(os.path.join(self.sf_path, 'qpts.txt'))

        self.data = InterpolationData.from_castep(
            self.seedname, path=self.interpolation_path)
        self.data.calculate_fine_phonons(qpts, asr='realspace', use_c=True)

        # InterpolationData object for DW grid
        self.dw_idata = InterpolationData.from_castep(
            self.seedname, path=self.interpolation_path)
        self.dw_idata.calculate_fine_phonons(
            np.loadtxt(os.path.join('data', 'qgrid_444.txt')), asr='realspace',
            use_c=True)

        # PhononData object for DW grid
        self.dw_pdata = PhononData.from_castep(
            'La2Zr2O7-grid', path=self.sf_path)


class TestStructureFactorInterpolationDataLZOParallelC(
    TestStructureFactorInterpolationDataLZOSerial):

    def setUp(self):
        # Need to separately test SF calculation with interpolated phonon data
        # to test eigenvector calculations
        self.seedname = 'La2Zr2O7'
        self.interpolation_path = os.path.join('data', 'interpolation', 'LZO')
        self.sf_path = os.path.join('data', 'structure_factor', 'LZO')
        self.scattering_lengths = {'La': 8.24, 'Zr': 7.16, 'O': 5.803}
        qpts = np.loadtxt(os.path.join(self.sf_path, 'qpts.txt'))

        self.data = InterpolationData.from_castep(
            self.seedname, path=self.interpolation_path)
        self.data.calculate_fine_phonons(qpts, asr='realspace', use_c=True,
                                         n_threads=2)

        # InterpolationData object for DW grid
        self.dw_idata = InterpolationData.from_castep(
            self.seedname, path=self.interpolation_path)
        self.dw_idata.calculate_fine_phonons(
            np.loadtxt(os.path.join('data', 'qgrid_444.txt')), asr='realspace',
            use_c=True)

        # PhononData object for DW grid
        self.dw_pdata = PhononData.from_castep(
            'La2Zr2O7-grid', path=self.sf_path)

class TestStructureFactorInterpolationDataQuartzSerial(unittest.TestCase):

    def setUp(self):
        # Need to separately test SF calculation with interpolated phonon data
        # to test eigenvector calculations
        self.seedname = 'quartz'
        self.interpolation_path = os.path.join('data', 'interpolation', 'quartz')
        self.sf_path = os.path.join('data', 'structure_factor', 'quartz')
        self.scattering_lengths = {'Si': 4.1491, 'O': 5.803}
        qpts = np.loadtxt(os.path.join(self.sf_path, 'qpts.txt'))

        self.data = InterpolationData.from_castep(
            self.seedname, path=self.interpolation_path)
        self.data.calculate_fine_phonons(qpts, asr='reciprocal')

        # InterpolationData object for DW grid
        self.dw_data = InterpolationData.from_castep(
            self.seedname, path=self.interpolation_path)
        self.dw_data.calculate_fine_phonons(
            np.loadtxt(os.path.join('data', 'qgrid_444.txt')), asr='reciprocal')

    def test_sf_T0(self):
        sf = self.data.calculate_structure_factor(self.scattering_lengths, T=0)
        expected_sf = np.loadtxt(os.path.join(self.sf_path, 'sf_idata_T0.txt'))

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
            self.scattering_lengths, T=0, dw_data=self.dw_data)
        expected_sf = np.loadtxt(os.path.join(
            self.sf_path, 'sf_idata_dw_grid_T0.txt'))
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
        expected_sf = np.loadtxt(os.path.join(self.sf_path, 'sf_idata_T100.txt'))

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
            self.scattering_lengths, T=100, dw_data=self.dw_data)
        expected_sf = np.loadtxt(os.path.join(
            self.sf_path, 'sf_idata_dw_grid_T100.txt'))

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

class TestStructureFactorInterpolationDataQuartzSerialC(
    TestStructureFactorInterpolationDataQuartzSerial):

    def setUp(self):
        # Need to separately test SF calculation with interpolated phonon data
        # to test eigenvector calculations
        self.seedname = 'quartz'
        self.interpolation_path = os.path.join('data', 'interpolation', 'quartz')
        self.sf_path = os.path.join('data', 'structure_factor', 'quartz')
        self.scattering_lengths = {'Si': 4.1491, 'O': 5.803}
        qpts = np.loadtxt(os.path.join(self.sf_path, 'qpts.txt'))

        self.data = InterpolationData.from_castep(
            self.seedname, path=self.interpolation_path)
        self.data.calculate_fine_phonons(qpts, asr='reciprocal', use_c=True)

        # InterpolationData object for DW grid
        self.dw_data = InterpolationData.from_castep(
            self.seedname, path=self.interpolation_path)
        self.dw_data.calculate_fine_phonons(
            np.loadtxt(os.path.join('data', 'qgrid_444.txt')), asr='reciprocal')

class TestStructureFactorInterpolationDataQuartzParallelC(
    TestStructureFactorInterpolationDataQuartzSerial):

    def setUp(self):
        # Need to separately test SF calculation with interpolated phonon data
        # to test eigenvector calculations
        self.seedname = 'quartz'
        self.interpolation_path = os.path.join('data', 'interpolation', 'quartz')
        self.sf_path = os.path.join('data', 'structure_factor', 'quartz')
        self.scattering_lengths = {'Si': 4.1491, 'O': 5.803}
        qpts = np.loadtxt(os.path.join(self.sf_path, 'qpts.txt'))

        self.data = InterpolationData.from_castep(
            self.seedname, path=self.interpolation_path)
        self.data.calculate_fine_phonons(qpts, asr='reciprocal', use_c=True,
                                         n_threads=2)

        # InterpolationData object for DW grid
        self.dw_data = InterpolationData.from_castep(
            self.seedname, path=self.interpolation_path)
        self.dw_data.calculate_fine_phonons(
            np.loadtxt(os.path.join('data', 'qgrid_444.txt')), asr='reciprocal')
