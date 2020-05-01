import os
import unittest
import numpy.testing as npt
import numpy as np
from euphonic.data.phonon import PhononData
from euphonic.data.interpolation import InterpolationData
from ..utils import get_data_path


class TestStructureFactorPhononDataNaCl(unittest.TestCase):

    def setUp(self):
        data_path = os.path.join(
            get_data_path(), 'phonopy_data', 'NaCl')
        self.sf_path = os.path.join(data_path, 'structure_factor')
        self.data = PhononData.from_phonopy(
            path=os.path.join(data_path, 'qpoints'), phonon_name='qpoints.yaml',
            summary_name='phonopy.yaml')
        self.scattering_lengths = {'Na': 3.63, 'Cl': 9.577}
        self.dw_data = PhononData.from_phonopy(
            phonon_name='mesh.yaml', path=os.path.join(data_path, 'mesh'))

    def test_sf_T5(self):
        sf = self.data.calculate_structure_factor(self.scattering_lengths, T=5)
        expected_sf = np.loadtxt(os.path.join(self.sf_path, 'sf_pdata_T5.txt'))
        npt.assert_allclose(sf, expected_sf, atol=1e-24)

    def test_sf_T5_dw(self):
        sf = self.data.calculate_structure_factor(
            self.scattering_lengths, T=5, dw_data=self.dw_data)
        expected_sf = np.loadtxt(os.path.join(
            self.sf_path, 'sf_pdata_dw_T5.txt'))
        npt.assert_allclose(sf, expected_sf, atol=1e-24)

    def test_sf_T100(self):
        sf = self.data.calculate_structure_factor(
            self.scattering_lengths, T=100)
        expected_sf = np.loadtxt(os.path.join(
            self.sf_path, 'sf_pdata_T100.txt'))
        npt.assert_allclose(sf, expected_sf, atol=1e-24)

    def test_sf_T100_dw(self):
        sf = self.data.calculate_structure_factor(
            self.scattering_lengths, T=100, dw_data=self.dw_data)
        expected_sf = np.loadtxt(os.path.join(
            self.sf_path, 'sf_pdata_dw_T100.txt'))
        npt.assert_allclose(sf, expected_sf, atol=1e-24)


class TestStructureFactorInterpolationDataNaClSerial(unittest.TestCase):

    def setUp(self):
        # Need to separately test SF calculation with interpolated phonon data
        # to test eigenvector calculations
        data_path = os.path.join(
            get_data_path(), 'phonopy_data', 'NaCl')
        self.interpolation_path = os.path.join(data_path, 'interpolation')
        self.sf_path = os.path.join(data_path, 'structure_factor')
        self.scattering_lengths = {'Na': 3.63, 'Cl': 9.577}
        qpts = np.array([[0., 0., 0.],
                         [0., 0., 0.5],
                         [-0.25, 0.5, 0.5],
                         [-0.151515, 0.575758, 0.5],
                         [1., 1., 1.]])
        self.gamma_idx = 4

        self.data = InterpolationData.from_phonopy(
            path=self.interpolation_path, summary_name='phonopy.yaml')
        self.data.calculate_fine_phonons(qpts, asr='reciprocal')

        # InterpolationData object for DW grid
        self.dw_idata = InterpolationData.from_phonopy(
            path=self.interpolation_path, summary_name='phonopy.yaml')
        self.dw_idata.calculate_fine_phonons(
            np.loadtxt(os.path.join(get_data_path(), 'qgrid_444.txt')),
            asr='reciprocal')

        # PhononData object for DW grid
        self.dw_pdata = PhononData.from_phonopy(
            phonon_name='mesh.yaml', path=os.path.join(data_path, 'mesh'))

    def test_sf_T5(self):
        sf = self.data.calculate_structure_factor(self.scattering_lengths, T=5)
        expected_sf = np.loadtxt(os.path.join(self.sf_path, 'sf_idata_T5.txt'))

        # Structure factors not necessarily equal due to degenerate modes
        # So sum structure factors over degenerate modes, these should be equal
        sf_sum = np.zeros(sf.shape)
        expected_sf_sum = np.zeros(sf.shape)
        for q in range(self.data.n_qpts):
            TOL = 5e-4
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
        gi = self.gamma_idx
        npt.assert_allclose(np.delete(sf_sum, gi, axis=0),
                            np.delete(expected_sf_sum, gi, axis=0),
                            atol=1e-20)
        npt.assert_allclose(sf_sum[gi, 3:], expected_sf_sum[gi, 3:],
                            atol=1e-20)

    def test_sf_T5_dw_idata(self):
        sf = self.data.calculate_structure_factor(
            self.scattering_lengths, T=5, dw_data=self.dw_idata)
        expected_sf = np.loadtxt(os.path.join(
            self.sf_path, 'sf_idata_dwidata_T5.txt'))
        sf_sum = np.zeros(sf.shape)
        expected_sf_sum = np.zeros(sf.shape)
        for q in range(self.data.n_qpts):
            TOL = 5e-4
            diff = np.append(TOL + 1, np.diff(self.data.freqs[q].magnitude))
            unique_index = np.where(diff > TOL)[0]
            x = np.zeros(self.data.n_branches, dtype=np.int32)
            x[unique_index] = 1
            unique_modes = np.cumsum(x) - 1
            sf_sum[q, :len(unique_index)] = np.bincount(unique_modes, sf[q])
            expected_sf_sum[q, :len(unique_index)] = np.bincount(
                unique_modes, expected_sf[q])
        gi = self.gamma_idx
        npt.assert_allclose(np.delete(sf_sum, gi, axis=0),
                            np.delete(expected_sf_sum, gi, axis=0),
                            atol=1e-20)
        npt.assert_allclose(sf_sum[gi, 3:], expected_sf_sum[gi, 3:],
                            atol=1e-20)

    def test_sf_T5_dw_pdata(self):
        sf = self.data.calculate_structure_factor(
            self.scattering_lengths, T=5, dw_data=self.dw_pdata)
        expected_sf = np.loadtxt(os.path.join(
            self.sf_path, 'sf_idata_dwpdata_T5.txt'))

        sf_sum = np.zeros(sf.shape)
        expected_sf_sum = np.zeros(sf.shape)
        for q in range(self.data.n_qpts):
            TOL = 5e-4
            diff = np.append(TOL + 1, np.diff(self.data.freqs[q].magnitude))
            unique_index = np.where(diff > TOL)[0]
            x = np.zeros(self.data.n_branches, dtype=np.int32)
            x[unique_index] = 1
            unique_modes = np.cumsum(x) - 1
            sf_sum[q, :len(unique_index)] = np.bincount(unique_modes, sf[q])
            expected_sf_sum[q, :len(unique_index)] = np.bincount(
                unique_modes, expected_sf[q])
        gi = self.gamma_idx
        npt.assert_allclose(np.delete(sf_sum, gi, axis=0),
                            np.delete(expected_sf_sum, gi, axis=0),
                            atol=1e-20)
        npt.assert_allclose(sf_sum[gi, 3:], expected_sf_sum[gi, 3:],
                            atol=1e-20)

    def test_sf_T100(self):
        sf = self.data.calculate_structure_factor(
            self.scattering_lengths, T=100)
        expected_sf = np.loadtxt(os.path.join(
            self.sf_path, 'sf_idata_T100.txt'))

        sf_sum = np.zeros(sf.shape)
        expected_sf_sum = np.zeros(sf.shape)
        for q in range(self.data.n_qpts):
            TOL = 5e-4
            diff = np.append(TOL + 1, np.diff(self.data.freqs[q].magnitude))
            unique_index = np.where(diff > TOL)[0]
            x = np.zeros(self.data.n_branches, dtype=np.int32)
            x[unique_index] = 1
            unique_modes = np.cumsum(x) - 1
            sf_sum[q, :len(unique_index)] = np.bincount(unique_modes, sf[q])
            expected_sf_sum[q, :len(unique_index)] = np.bincount(
                unique_modes, expected_sf[q])
        gi = self.gamma_idx
        npt.assert_allclose(np.delete(sf_sum, gi, axis=0),
                            np.delete(expected_sf_sum, gi, axis=0),
                            atol=1e-20)
        npt.assert_allclose(sf_sum[gi, 3:], expected_sf_sum[gi, 3:],
                            atol=1e-20)

    def test_sf_T100_dw_idata(self):
        sf = self.data.calculate_structure_factor(
            self.scattering_lengths, T=100, dw_data=self.dw_idata)
        expected_sf = np.loadtxt(os.path.join(
            self.sf_path, 'sf_idata_dwidata_T100.txt'))

        sf_sum = np.zeros(sf.shape)
        expected_sf_sum = np.zeros(sf.shape)
        for q in range(self.data.n_qpts):
            TOL = 5e-4
            diff = np.append(TOL + 1, np.diff(self.data.freqs[q].magnitude))
            unique_index = np.where(diff > TOL)[0]
            x = np.zeros(self.data.n_branches, dtype=np.int32)
            x[unique_index] = 1
            unique_modes = np.cumsum(x) - 1
            sf_sum[q, :len(unique_index)] = np.bincount(unique_modes, sf[q])
            expected_sf_sum[q, :len(unique_index)] = np.bincount(
                unique_modes, expected_sf[q])
        gi = self.gamma_idx
        npt.assert_allclose(np.delete(sf_sum, gi, axis=0),
                            np.delete(expected_sf_sum, gi, axis=0),
                            atol=1e-20)
        npt.assert_allclose(sf_sum[gi, 3:], expected_sf_sum[gi, 3:],
                            atol=1e-20)

    def test_sf_T100_dw_pdata(self):
        sf = self.data.calculate_structure_factor(
            self.scattering_lengths, T=100, dw_data=self.dw_pdata)
        expected_sf = np.loadtxt(os.path.join(
            self.sf_path, 'sf_idata_dwpdata_T100.txt'))

        sf_sum = np.zeros(sf.shape)
        expected_sf_sum = np.zeros(sf.shape)
        for q in range(self.data.n_qpts):
            TOL = 5e-4
            diff = np.append(TOL + 1, np.diff(self.data.freqs[q].magnitude))
            unique_index = np.where(diff > TOL)[0]
            x = np.zeros(self.data.n_branches, dtype=np.int32)
            x[unique_index] = 1
            unique_modes = np.cumsum(x) - 1
            sf_sum[q, :len(unique_index)] = np.bincount(unique_modes, sf[q])
            expected_sf_sum[q, :len(unique_index)] = np.bincount(
                unique_modes, expected_sf[q])
        gi = self.gamma_idx
        npt.assert_allclose(np.delete(sf_sum, gi, axis=0),
                            np.delete(expected_sf_sum, gi, axis=0),
                            atol=1e-20)
        npt.assert_allclose(sf_sum[gi, 3:], expected_sf_sum[gi, 3:],
                            atol=1e-20)


class TestStructureFactorInterpolationDataNaClSerialC(TestStructureFactorInterpolationDataNaClSerial):

    def setUp(self):
        # Need to separately test SF calculation with interpolated phonon data
        # to test eigenvector calculations
        data_path = os.path.join(
            get_data_path(), 'phonopy_data', 'NaCl')
        self.interpolation_path = os.path.join(data_path, 'interpolation')
        self.sf_path = os.path.join(data_path, 'structure_factor')
        self.scattering_lengths = {'Na': 3.63, 'Cl': 9.577}
        qpts = np.array([[0., 0., 0.],
                         [0., 0., 0.5],
                         [-0.25, 0.5, 0.5],
                         [-0.151515, 0.575758, 0.5],
                         [1., 1., 1.]])
        self.gamma_idx = 4

        self.data = InterpolationData.from_phonopy(
            path=self.interpolation_path, summary_name='phonopy.yaml')
        self.data.calculate_fine_phonons(qpts, asr='reciprocal', use_c=True,
                                         fall_back_on_python=False)

        # InterpolationData object for DW grid
        self.dw_idata = InterpolationData.from_phonopy(
            path=self.interpolation_path, summary_name='phonopy.yaml')
        self.dw_idata.calculate_fine_phonons(
            np.loadtxt(os.path.join(get_data_path(), 'qgrid_444.txt')),
            asr='reciprocal', use_c=True, fall_back_on_python=False)

        # PhononData object for DW grid
        self.dw_pdata = PhononData.from_phonopy(
            phonon_name='mesh.yaml', path=os.path.join(data_path, 'mesh'))


class TestStructureFactorInterpolationDataNaClParallelC(TestStructureFactorInterpolationDataNaClSerial):

    def setUp(self):
        # Need to separately test SF calculation with interpolated phonon data
        # to test eigenvector calculations
        data_path = os.path.join(
            get_data_path(), 'phonopy_data', 'NaCl')
        self.interpolation_path = os.path.join(data_path, 'interpolation')
        self.sf_path = os.path.join(data_path, 'structure_factor')
        self.scattering_lengths = {'Na': 3.63, 'Cl': 9.577}
        qpts = np.array([[0., 0., 0.],
                         [0., 0., 0.5],
                         [-0.25, 0.5, 0.5],
                         [-0.151515, 0.575758, 0.5],
                         [1., 1., 1.]])
        self.gamma_idx = 4

        self.data = InterpolationData.from_phonopy(
            path=self.interpolation_path, summary_name='phonopy.yaml')
        self.data.calculate_fine_phonons(qpts, asr='reciprocal', use_c=True,
                                         fall_back_on_python=False, n_threads=2)

        # InterpolationData object for DW grid
        self.dw_idata = InterpolationData.from_phonopy(
            path=self.interpolation_path, summary_name='phonopy.yaml')
        self.dw_idata.calculate_fine_phonons(
            np.loadtxt(os.path.join(get_data_path(), 'qgrid_444.txt')),
            asr='reciprocal', use_c=True, fall_back_on_python=False,
            n_threads=2)

        # PhononData object for DW grid
        self.dw_pdata = PhononData.from_phonopy(
            phonon_name='mesh.yaml', path=os.path.join(data_path, 'mesh'))