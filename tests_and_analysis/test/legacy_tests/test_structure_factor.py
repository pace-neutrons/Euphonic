import os
import unittest
import numpy.testing as npt
import numpy as np
from euphonic import ureg, ForceConstants, QpointPhononModes
from euphonic.util import _bose_factor
from tests_and_analysis.test.utils import get_data_path


class TestStructureFactorQpointPhononModesLZO(unittest.TestCase):

    def setUp(self):
        seedname = 'La2Zr2O7'
        data_path = get_data_path()
        self.sf_path = os.path.join(data_path, 'structure_factor', 'LZO')
        self.data = QpointPhononModes.from_castep(
            os.path.join(data_path, 'La2Zr2O7.phonon'))
        fm = ureg.fm
        self.scattering_lengths = {'La': 8.24*fm, 'Zr': 7.16*fm, 'O': 5.803*fm}
        self.dw_data = QpointPhononModes.from_castep(
            os.path.join(self.sf_path, 'La2Zr2O7-grid.phonon'))

    def test_sf_T5(self):
        sf_obj = self.data.calculate_structure_factor(
            self.scattering_lengths)
        bose = _bose_factor(sf_obj._frequencies, 5)
        sf = sf_obj.structure_factors.to('bohr**2').magnitude*bose
        expected_sf = np.loadtxt(os.path.join(self.sf_path, 'sf_pdata_T5.txt'))
        npt.assert_allclose(sf, expected_sf, rtol=2e-7)

    def test_sf_T5_dw(self):
        dw5 = self.dw_data.calculate_debye_waller(5*ureg('K'))
        sf_obj = self.data.calculate_structure_factor(
            self.scattering_lengths, dw=dw5)
        bose = _bose_factor(sf_obj._frequencies, 5)
        sf = sf_obj.structure_factors.to('bohr**2').magnitude*bose
        expected_sf = np.loadtxt(os.path.join(
            self.sf_path, 'sf_pdata_dw_T5.txt'))
        npt.assert_allclose(sf, expected_sf, rtol=2e-7, atol=1e-18)

    def test_sf_T100(self):
        sf_obj = self.data.calculate_structure_factor(
            self.scattering_lengths)
        bose = _bose_factor(sf_obj._frequencies, 100)
        sf = sf_obj.structure_factors.to('bohr**2').magnitude*bose
        expected_sf = np.loadtxt(os.path.join(
            self.sf_path, 'sf_pdata_T100.txt'))
        npt.assert_allclose(sf, expected_sf, rtol=2e-7)

    def test_sf_T100_dw(self):
        dw100 = self.dw_data.calculate_debye_waller(100*ureg('K'))
        sf_obj = self.data.calculate_structure_factor(
            self.scattering_lengths, dw=dw100)
        bose = _bose_factor(sf_obj._frequencies, 100)
        sf = sf_obj.structure_factors.to('bohr**2').magnitude*bose
        expected_sf = np.loadtxt(os.path.join(
            self.sf_path, 'sf_pdata_dw_T100.txt'))
        npt.assert_allclose(sf, expected_sf, rtol=2e-6)


class TestStructureFactorForceConstantsLZOSerial(unittest.TestCase):

    def setUp(self):
        # Need to separately test SF calculation with interpolated phonon data
        # to test eigenvector calculations
        data_path = get_data_path()
        self.interpolation_path = os.path.join(
            data_path, 'force_constants', 'LZO')
        self.sf_path = os.path.join(data_path, 'structure_factor' , 'LZO')
        fm = ureg.fm
        self.scattering_lengths = {'La': 8.24*fm, 'Zr': 7.16*fm, 'O': 5.803*fm}
        qpts = np.loadtxt(os.path.join(self.sf_path, 'qpts.txt'))

        fc_filename = os.path.join(
            self.interpolation_path, 'La2Zr2O7.castep_bin')
        fc = ForceConstants.from_castep(fc_filename)
        self.idata = fc.calculate_qpoint_phonon_modes(qpts, asr='realspace')

        # ForceConstants object for DW grid
        dw_fc = ForceConstants.from_castep(fc_filename)
        self.dw_idata = dw_fc.calculate_qpoint_phonon_modes(
            np.loadtxt(os.path.join(data_path, 'qgrid_444.txt')),
            asr='realspace')

        # QpointPhononModes object for DW grid
        self.dw_pdata = QpointPhononModes.from_castep(
            os.path.join(self.sf_path, 'La2Zr2O7-grid.phonon'))

        # Tolerance for assuming degenerate modes
        self.TOL = 1e-4


    def test_sf_T5(self):
        sf_obj = self.idata.calculate_structure_factor(
            self.scattering_lengths)
        bose = _bose_factor(sf_obj._frequencies, 5)
        sf = sf_obj.structure_factors.to('bohr**2').magnitude*bose
        expected_sf = np.loadtxt(os.path.join(self.sf_path, 'sf_idata_T5.txt'))

        # Structure factors not necessarily equal due to degenerate modes
        # So sum structure factors over degenerate modes, these should be equal
        sf_sum = np.zeros(sf.shape)
        expected_sf_sum = np.zeros(sf.shape)
        for q in range(self.idata.n_qpts):
            diff = np.append(self.TOL + 1,
                             np.diff(self.idata.frequencies[q].magnitude))
            unique_index = np.where(diff > self.TOL)[0]
            x = np.zeros(3*self.idata.crystal.n_atoms, dtype=np.int32)
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
        dw5 = self.dw_idata.calculate_debye_waller(5*ureg('K'))
        sf_obj = self.idata.calculate_structure_factor(
            self.scattering_lengths, dw=dw5)
        bose = _bose_factor(sf_obj._frequencies, 5)
        sf = sf_obj.structure_factors.to('bohr**2').magnitude*bose
        expected_sf = np.loadtxt(os.path.join(
            self.sf_path, 'sf_idata_dw_grid_T5.txt'))
        sf_sum = np.zeros(sf.shape)
        expected_sf_sum = np.zeros(sf.shape)
        for q in range(self.idata.n_qpts):
            diff = np.append(self.TOL + 1,
                             np.diff(self.idata.frequencies[q].magnitude))
            unique_index = np.where(diff > self.TOL)[0]
            x = np.zeros(3*self.idata.crystal.n_atoms, dtype=np.int32)
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
        dw5 = self.dw_pdata.calculate_debye_waller(5*ureg('K'))
        sf_obj = self.idata.calculate_structure_factor(
            self.scattering_lengths, dw=dw5)
        bose = _bose_factor(sf_obj._frequencies, 5)
        sf = sf_obj.structure_factors.to('bohr**2').magnitude*bose
        expected_sf = np.loadtxt(os.path.join(
            self.sf_path, 'sf_idata_dw_seedname_T5.txt'))

        sf_sum = np.zeros(sf.shape)
        expected_sf_sum = np.zeros(sf.shape)
        for q in range(self.idata.n_qpts):
            diff = np.append(self.TOL + 1,
                             np.diff(self.idata.frequencies[q].magnitude))
            unique_index = np.where(diff > self.TOL)[0]
            x = np.zeros(3*self.idata.crystal.n_atoms, dtype=np.int32)
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
        sf_obj = self.idata.calculate_structure_factor(
            self.scattering_lengths)
        bose = _bose_factor(sf_obj._frequencies, 100)
        sf = sf_obj.structure_factors.to('bohr**2').magnitude*bose
        expected_sf = np.loadtxt(os.path.join(
            self.sf_path, 'sf_idata_T100.txt'))

        sf_sum = np.zeros(sf.shape)
        expected_sf_sum = np.zeros(sf.shape)
        for q in range(self.idata.n_qpts):
            diff = np.append(self.TOL + 1,
                             np.diff(self.idata.frequencies[q].magnitude))
            unique_index = np.where(diff > self.TOL)[0]
            x = np.zeros(3*self.idata.crystal.n_atoms, dtype=np.int32)
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
        dw100 = self.dw_idata.calculate_debye_waller(100*ureg('K'))
        sf_obj = self.idata.calculate_structure_factor(
            self.scattering_lengths, dw=dw100)
        bose = _bose_factor(sf_obj._frequencies, 100)
        sf = sf_obj.structure_factors.to('bohr**2').magnitude*bose
        expected_sf = np.loadtxt(os.path.join(
            self.sf_path, 'sf_idata_dw_grid_T100.txt'))

        sf_sum = np.zeros(sf.shape)
        expected_sf_sum = np.zeros(sf.shape)
        for q in range(self.idata.n_qpts):
            diff = np.append(self.TOL + 1,
                             np.diff(self.idata.frequencies[q].magnitude))
            unique_index = np.where(diff > self.TOL)[0]
            x = np.zeros(3*self.idata.crystal.n_atoms, dtype=np.int32)
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
        dw100 = self.dw_pdata.calculate_debye_waller(100*ureg('K'))
        sf_obj = self.idata.calculate_structure_factor(
            self.scattering_lengths, dw=dw100)
        bose = _bose_factor(sf_obj._frequencies, 100)
        sf = sf_obj.structure_factors.to('bohr**2').magnitude*bose
        expected_sf = np.loadtxt(os.path.join(
            self.sf_path, 'sf_idata_dw_seedname_T100.txt'))

        sf_sum = np.zeros(sf.shape)
        expected_sf_sum = np.zeros(sf.shape)
        for q in range(self.idata.n_qpts):
            diff = np.append(self.TOL + 1,
                             np.diff(self.idata.frequencies[q].magnitude))
            unique_index = np.where(diff > self.TOL)[0]
            x = np.zeros(3*self.idata.crystal.n_atoms, dtype=np.int32)
            x[unique_index] = 1
            unique_modes = np.cumsum(x) - 1
            sf_sum[q, :len(unique_index)] = np.bincount(unique_modes, sf[q])
            expected_sf_sum[q, :len(unique_index)] = np.bincount(
                unique_modes, expected_sf[q])
        npt.assert_allclose(np.delete(sf_sum, -3, axis=0),
                            np.delete(expected_sf_sum, -3, axis=0), rtol=1e-6)
        npt.assert_allclose(sf_sum[-3, 3:], expected_sf_sum[-3, 3:],
                            rtol=5e-3, atol=1e-12)

    def test_incompatible_dw_arg_raises_exception(self):
        quartz_data = ForceConstants.from_castep(os.path.join(
            self.interpolation_path, '../quartz', 'quartz.castep_bin'))
        # Test that trying to call structure factor with a dw_arg for another
        # material raises an Exception
        self.assertRaises(Exception, self.idata.calculate_structure_factor,
            self.scattering_lengths, dw_data=quartz_data)


class TestStructureFactorForceConstantsLZOSerialC(TestStructureFactorForceConstantsLZOSerial):

    def setUp(self):
        # Need to separately test SF calculation with interpolated phonon data
        # to test eigenvector calculations
        self.seedname = 'La2Zr2O7'
        data_path = get_data_path()
        self.interpolation_path = os.path.join(
            data_path, 'force_constants', 'LZO')
        self.sf_path = os.path.join(data_path, 'structure_factor', 'LZO')
        fm = ureg.fm
        self.scattering_lengths = {'La': 8.24*fm, 'Zr': 7.16*fm, 'O': 5.803*fm}
        qpts = np.loadtxt(os.path.join(self.sf_path, 'qpts.txt'))

        fc_filename = os.path.join(
            self.interpolation_path, 'La2Zr2O7.castep_bin')
        fc = ForceConstants.from_castep(fc_filename)
        self.idata = fc.calculate_qpoint_phonon_modes(
            qpts, asr='realspace', use_c=True, fall_back_on_python=False)

        # ForceConstants object for DW grid
        dw_fc = ForceConstants.from_castep(fc_filename)
        self.dw_idata = dw_fc.calculate_qpoint_phonon_modes(
            np.loadtxt(os.path.join(data_path, 'qgrid_444.txt')),
            asr='realspace', use_c=True, fall_back_on_python=False)

        # QpointPhononModes object for DW grid
        self.dw_pdata = QpointPhononModes.from_castep(
            os.path.join(self.sf_path, 'La2Zr2O7-grid.phonon'))

        # Tolerance for assuming degenerate modes
        self.TOL = 1e-4


class TestStructureFactorForceConstantsLZOParallelC(TestStructureFactorForceConstantsLZOSerial):

    def setUp(self):
        # Need to separately test SF calculation with interpolated phonon data
        # to test eigenvector calculations
        self.seedname = 'La2Zr2O7'
        data_path = get_data_path()
        self.interpolation_path = os.path.join(
            data_path, 'force_constants', 'LZO')
        self.sf_path = os.path.join(data_path, 'structure_factor', 'LZO')
        fm = ureg.fm
        self.scattering_lengths = {'La': 8.24*fm, 'Zr': 7.16*fm, 'O': 5.803*fm}
        qpts = np.loadtxt(os.path.join(self.sf_path, 'qpts.txt'))

        fc_filename = os.path.join(
            self.interpolation_path, 'La2Zr2O7.castep_bin')
        fc = ForceConstants.from_castep(fc_filename)
        self.idata = fc.calculate_qpoint_phonon_modes(
            qpts, asr='realspace', use_c=True, n_threads=2,
            fall_back_on_python=False)

        # ForceConstants object for DW grid
        dw_fc = ForceConstants.from_castep(fc_filename)
        self.dw_idata = dw_fc.calculate_qpoint_phonon_modes(
            np.loadtxt(os.path.join(data_path, 'qgrid_444.txt')),
            asr='realspace', use_c=True, fall_back_on_python=False)

        # QpointPhononModes object for DW grid
        self.dw_pdata = QpointPhononModes.from_castep(
            os.path.join(self.sf_path, 'La2Zr2O7-grid.phonon'))

        # Tolerance for assuming degenerate modes
        self.TOL = 1e-4


class TestStructureFactorForceConstantsQuartzSerial(unittest.TestCase):

    def setUp(self):
        # Need to separately test SF calculation with interpolated phonon data
        # to test eigenvector calculations
        self.seedname = 'quartz'
        data_path = get_data_path()
        self.interpolation_path = os.path.join(
            data_path, 'force_constants', 'quartz')
        self.sf_path = os.path.join(data_path, 'structure_factor', 'quartz')
        fm = ureg.fm
        self.scattering_lengths = {'Si': 4.1491*fm, 'O': 5.803*fm}
        qpts = np.loadtxt(os.path.join(self.sf_path, 'qpts.txt'))

        fc_filename = os.path.join(
            self.interpolation_path, 'quartz.castep_bin')
        fc = ForceConstants.from_castep(fc_filename)
        self.idata = fc.calculate_qpoint_phonon_modes(qpts, asr='reciprocal')

        # ForceConstants object for DW grid
        dw_fc = ForceConstants.from_castep(fc_filename)
        self.dw_idata = dw_fc.calculate_qpoint_phonon_modes(
            np.loadtxt(os.path.join(data_path, 'qgrid_444.txt')), asr='reciprocal')

        # Tolerance for assuming degenerate modes - needs to be higher for
        # Quartz due to symmetry
        self.TOL = 0.05

    def test_sf_T0(self):
        sf_obj = self.idata.calculate_structure_factor(
            self.scattering_lengths)
        bose = _bose_factor(sf_obj._frequencies, 0)
        sf = sf_obj.structure_factors.to('bohr**2').magnitude*bose
        expected_sf = np.loadtxt(os.path.join(self.sf_path, 'sf_idata_T0.txt'))

        # Structure factors not necessarily equal due to degenerate modes
        # So sum structure factors over degenerate modes, these should be equal
        sf_sum = np.zeros(sf.shape)
        expected_sf_sum = np.zeros(sf.shape)
        for q in range(self.idata.n_qpts):
            diff = np.append(self.TOL + 1,
                             np.diff(self.idata.frequencies[q].magnitude))
            unique_index = np.where(diff > self.TOL)[0]
            x = np.zeros(3*self.idata.crystal.n_atoms, dtype=np.int32)
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
                            rtol=7e-4, atol=2e-13)
        npt.assert_allclose(sf_sum[gamma_qpts, 3:],
                            expected_sf_sum[gamma_qpts, 3:],
                            rtol=4e-3, atol=1.1e-11)

    def test_sf_T0_dw_grid(self):
        dw0 = self.dw_idata.calculate_debye_waller(0*ureg('K'))
        sf_obj = self.idata.calculate_structure_factor(
            self.scattering_lengths, dw=dw0)
        bose = _bose_factor(sf_obj._frequencies, 0)
        sf = sf_obj.structure_factors.to('bohr**2').magnitude*bose
        expected_sf = np.loadtxt(os.path.join(
            self.sf_path, 'sf_idata_dw_grid_T0.txt'))
        sf_sum = np.zeros(sf.shape)
        expected_sf_sum = np.zeros(sf.shape)
        for q in range(self.idata.n_qpts):
            diff = np.append(self.TOL + 1,
                             np.diff(self.idata.frequencies[q].magnitude))
            unique_index = np.where(diff > self.TOL)[0]
            x = np.zeros(3*self.idata.crystal.n_atoms, dtype=np.int32)
            x[unique_index] = 1
            unique_modes = np.cumsum(x) - 1
            sf_sum[q, :len(unique_index)] = np.bincount(unique_modes, sf[q])
            expected_sf_sum[q, :len(unique_index)] = np.bincount(
                unique_modes, expected_sf[q])
        gamma_qpts = [16]
        npt.assert_allclose(np.delete(sf_sum, gamma_qpts, axis=0),
                            np.delete(expected_sf_sum, gamma_qpts, axis=0),
                            rtol=2e-3, atol=2e-13)
        npt.assert_allclose(sf_sum[gamma_qpts, 3:],
                            expected_sf_sum[gamma_qpts, 3:],
                            rtol=4e-3, atol=1.1e-11)

    def test_sf_T100(self):
        sf_obj = self.idata.calculate_structure_factor(
            self.scattering_lengths)
        bose = _bose_factor(sf_obj._frequencies, 100)
        sf = sf_obj.structure_factors.to('bohr**2').magnitude*bose
        expected_sf = np.loadtxt(
            os.path.join(self.sf_path, 'sf_idata_T100.txt'))

        sf_sum = np.zeros(sf.shape)
        expected_sf_sum = np.zeros(sf.shape)
        for q in range(self.idata.n_qpts):
            diff = np.append(self.TOL + 1,
                             np.diff(self.idata.frequencies[q].magnitude))
            unique_index = np.where(diff > self.TOL)[0]
            x = np.zeros(3*self.idata.crystal.n_atoms, dtype=np.int32)
            x[unique_index] = 1
            unique_modes = np.cumsum(x) - 1
            sf_sum[q, :len(unique_index)] = np.bincount(unique_modes, sf[q])
            expected_sf_sum[q, :len(unique_index)] = np.bincount(
                unique_modes, expected_sf[q])
        gamma_qpts = [16]
        npt.assert_allclose(np.delete(sf_sum, gamma_qpts, axis=0),
                            np.delete(expected_sf_sum, gamma_qpts, axis=0),
                            rtol=7e-4, atol=2e-13)
        npt.assert_allclose(sf_sum[gamma_qpts, 3:],
                            expected_sf_sum[gamma_qpts, 3:],
                            rtol=4e-3, atol=1.1e-11)

    def test_sf_T100_dw_grid(self):
        dw100 = self.dw_idata.calculate_debye_waller(100*ureg('K'))
        sf_obj = self.idata.calculate_structure_factor(
            self.scattering_lengths, dw=dw100)
        bose = _bose_factor(sf_obj._frequencies, 100)
        sf = sf_obj.structure_factors.to('bohr**2').magnitude*bose
        expected_sf = np.loadtxt(os.path.join(
            self.sf_path, 'sf_idata_dw_grid_T100.txt'))

        sf_sum = np.zeros(sf.shape)
        expected_sf_sum = np.zeros(sf.shape)
        for q in range(self.idata.n_qpts):
            diff = np.append(self.TOL + 1,
                             np.diff(self.idata.frequencies[q].magnitude))
            unique_index = np.where(diff > self.TOL)[0]
            x = np.zeros(3*self.idata.crystal.n_atoms, dtype=np.int32)
            x[unique_index] = 1
            unique_modes = np.cumsum(x) - 1
            sf_sum[q, :len(unique_index)] = np.bincount(unique_modes, sf[q])
            expected_sf_sum[q, :len(unique_index)] = np.bincount(
                unique_modes, expected_sf[q])
        gamma_qpts = [16]
        npt.assert_allclose(np.delete(sf_sum, gamma_qpts, axis=0),
                            np.delete(expected_sf_sum, gamma_qpts, axis=0),
                            rtol=7e-4, atol=2e-13)
        npt.assert_allclose(sf_sum[gamma_qpts, 3:],
                            expected_sf_sum[gamma_qpts, 3:],
                            rtol=4e-3, atol=1.1e-11)


class TestStructureFactorForceConstantsQuartzSerialC(TestStructureFactorForceConstantsQuartzSerial):

    def setUp(self):
        # Need to separately test SF calculation with interpolated phonon data
        # to test eigenvector calculations
        self.seedname = 'quartz'
        data_path = get_data_path()
        self.interpolation_path = os.path.join(
            data_path, 'force_constants', 'quartz')
        self.sf_path = os.path.join(data_path, 'structure_factor', 'quartz')
        fm = ureg.fm
        self.scattering_lengths = {'Si': 4.1491*fm, 'O': 5.803*fm}
        qpts = np.loadtxt(os.path.join(self.sf_path, 'qpts.txt'))

        fc_filename = os.path.join(
            self.interpolation_path, 'quartz.castep_bin')
        fc = ForceConstants.from_castep(fc_filename)
        self.idata = fc.calculate_qpoint_phonon_modes(
            qpts, asr='reciprocal', use_c=True, fall_back_on_python=False)

        # ForceConstants object for DW grid
        dw_fc = ForceConstants.from_castep(fc_filename)
        self.dw_idata = dw_fc.calculate_qpoint_phonon_modes(
            np.loadtxt(os.path.join(data_path, 'qgrid_444.txt')),
            asr='reciprocal')

        # Tolerance for assuming degenerate modes - needs to be higher for
        # Quartz due to symmetry
        self.TOL = 0.05


class TestStructureFactorForceConstantsQuartzParallelC(TestStructureFactorForceConstantsQuartzSerial):

    def setUp(self):
        # Need to separately test SF calculation with interpolated phonon data
        # to test eigenvector calculations
        self.seedname = 'quartz'
        data_path = get_data_path()
        self.interpolation_path = os.path.join(
            data_path, 'force_constants', 'quartz')
        self.sf_path = os.path.join(data_path, 'structure_factor', 'quartz')
        fm = ureg.fm
        self.scattering_lengths = {'Si': 4.1491*fm, 'O': 5.803*fm}
        qpts = np.loadtxt(os.path.join(self.sf_path, 'qpts.txt'))

        fc_filename = os.path.join(
            self.interpolation_path, 'quartz.castep_bin')
        fc = ForceConstants.from_castep(fc_filename)
        self.idata = fc.calculate_qpoint_phonon_modes(
            qpts, asr='reciprocal', use_c=True, n_threads=2,
            fall_back_on_python=False)

        # ForceConstants object for DW grid
        dw_fc = ForceConstants.from_castep(fc_filename)
        self.dw_idata = dw_fc.calculate_qpoint_phonon_modes(
            np.loadtxt(os.path.join(data_path, 'qgrid_444.txt')),
            asr='reciprocal')

        # Tolerance for assuming degenerate modes - needs to be higher for
        # Quartz due to symmetry
        self.TOL = 0.05