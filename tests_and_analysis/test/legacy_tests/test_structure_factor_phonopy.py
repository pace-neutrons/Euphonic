import os
import unittest
import numpy.testing as npt
import numpy as np
from euphonic import ureg, ForceConstants, QpointPhononModes
from euphonic.util import _bose_factor
from tests_and_analysis.test.utils import get_data_path


class TestStructureFactorQpointPhononModesNaCl(unittest.TestCase):

    def setUp(self):
        data_path = os.path.join(
            get_data_path(), 'phonopy_data', 'NaCl')
        self.sf_path = os.path.join(data_path, 'structure_factor')
        self.data = QpointPhononModes.from_phonopy(
            path=os.path.join(data_path, 'qpoints'), phonon_name='qpoints.yaml',
            summary_name='phonopy.yaml')
        fm = ureg.fm
        self.scattering_lengths = {'Na': 3.63*fm, 'Cl': 9.577*fm}
        self.dw_data = QpointPhononModes.from_phonopy(
            phonon_name='mesh.yaml', path=os.path.join(data_path, 'mesh'))

    def test_sf_T5(self):
        sf_obj = self.data.calculate_structure_factor(
            self.scattering_lengths)
        bose = _bose_factor(sf_obj._frequencies, 5)
        sf = sf_obj.structure_factors.to('bohr**2').magnitude*bose
        expected_sf = np.loadtxt(os.path.join(self.sf_path, 'sf_pdata_T5.txt'))
        npt.assert_allclose(sf, expected_sf, atol=1e-24)

    def test_sf_T5_dw(self):
        dw5 = self.dw_data.calculate_debye_waller(5*ureg('K'))
        sf_obj = self.data.calculate_structure_factor(
            self.scattering_lengths, dw=dw5)
        bose = _bose_factor(sf_obj._frequencies, 5)
        sf = sf_obj.structure_factors.to('bohr**2').magnitude*bose
        expected_sf = np.loadtxt(os.path.join(
            self.sf_path, 'sf_pdata_dw_T5.txt'))
        npt.assert_allclose(sf, expected_sf, atol=1e-24)

    def test_sf_T100(self):
        sf_obj = self.data.calculate_structure_factor(
            self.scattering_lengths)
        bose = _bose_factor(sf_obj._frequencies, 100)
        sf = sf_obj.structure_factors.to('bohr**2').magnitude*bose
        expected_sf = np.loadtxt(os.path.join(
            self.sf_path, 'sf_pdata_T100.txt'))
        npt.assert_allclose(sf, expected_sf, atol=1e-24)

    def test_sf_T100_dw(self):
        dw100 = self.dw_data.calculate_debye_waller(100*ureg('K'))
        sf_obj = self.data.calculate_structure_factor(
            self.scattering_lengths, dw=dw100)
        bose = _bose_factor(sf_obj._frequencies, 100)
        sf = sf_obj.structure_factors.to('bohr**2').magnitude*bose
        expected_sf = np.loadtxt(os.path.join(
            self.sf_path, 'sf_pdata_dw_T100.txt'))
        npt.assert_allclose(sf, expected_sf, atol=1e-24)


class TestStructureFactorForceConstantsNaClSerial(unittest.TestCase):

    def setUp(self):
        # Need to separately test SF calculation with interpolated phonon data
        # to test eigenvector calculations
        data_path = os.path.join(
            get_data_path(), 'phonopy_data', 'NaCl')
        self.interpolation_path = os.path.join(data_path, 'force_constants')
        self.sf_path = os.path.join(data_path, 'structure_factor')
        fm = ureg.fm
        self.scattering_lengths = {'Na': 3.63*fm, 'Cl': 9.577*fm}
        qpts = np.array([[0., 0., 0.],
                         [0., 0., 0.5],
                         [-0.25, 0.5, 0.5],
                         [-0.151515, 0.575758, 0.5]])

        fc = ForceConstants.from_phonopy(
            path=self.interpolation_path, summary_name='phonopy.yaml')
        self.idata = fc.calculate_qpoint_phonon_modes(qpts, asr='reciprocal')

        # ForceConstants object for DW grid
        dw_fc = ForceConstants.from_phonopy(
            path=self.interpolation_path, summary_name='phonopy.yaml')
        self.dw_idata = dw_fc.calculate_qpoint_phonon_modes(
            np.loadtxt(os.path.join(get_data_path(), 'qgrid_444.txt')),
            asr='reciprocal')

        # QpointPhononModes object for DW grid
        self.dw_pdata = QpointPhononModes.from_phonopy(
            phonon_name='mesh.yaml', path=os.path.join(data_path, 'mesh'))

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
            TOL = 5e-4
            diff = np.append(TOL + 1, np.diff(self.idata.frequencies[q].magnitude))
            unique_index = np.where(diff > TOL)[0]
            x = np.zeros(3*self.idata.crystal.n_atoms, dtype=np.int32)
            x[unique_index] = 1
            unique_modes = np.cumsum(x) - 1
            sf_sum[q, :len(unique_index)] = np.bincount(unique_modes, sf[q])
            expected_sf_sum[q, :len(unique_index)] = np.bincount(
                unique_modes, expected_sf[q])
        # Due to 1/w factor in structure factor, calculation can be unstable
        # at q=0. Test q=0 and non q=0 values separately with different
        # tolerances
        npt.assert_allclose(sf_sum[:4], expected_sf_sum[:4], atol=3e-18)

    def test_sf_T5_dw_idata(self):
        dw5 = self.dw_idata.calculate_debye_waller(5*ureg('K'))
        sf_obj = self.idata.calculate_structure_factor(
            self.scattering_lengths, dw=dw5)
        bose = _bose_factor(sf_obj._frequencies, 5)
        sf = sf_obj.structure_factors.to('bohr**2').magnitude*bose
        expected_sf = np.loadtxt(os.path.join(
            self.sf_path, 'sf_idata_dwidata_T5.txt'))
        sf_sum = np.zeros(sf.shape)
        expected_sf_sum = np.zeros(sf.shape)
        for q in range(self.idata.n_qpts):
            TOL = 5e-4
            diff = np.append(TOL + 1, np.diff(self.idata.frequencies[q].magnitude))
            unique_index = np.where(diff > TOL)[0]
            x = np.zeros(3*self.idata.crystal.n_atoms, dtype=np.int32)
            x[unique_index] = 1
            unique_modes = np.cumsum(x) - 1
            sf_sum[q, :len(unique_index)] = np.bincount(unique_modes, sf[q])
            expected_sf_sum[q, :len(unique_index)] = np.bincount(
                unique_modes, expected_sf[q])
        npt.assert_allclose(sf_sum[:4], expected_sf_sum[:4], atol=3e-18)

    def test_sf_T5_dw_pdata(self):
        dw5 = self.dw_pdata.calculate_debye_waller(5*ureg('K'))
        sf_obj = self.idata.calculate_structure_factor(
            self.scattering_lengths, dw=dw5)
        bose = _bose_factor(sf_obj._frequencies, 5)
        sf = sf_obj.structure_factors.to('bohr**2').magnitude*bose
        expected_sf = np.loadtxt(os.path.join(
            self.sf_path, 'sf_idata_dwpdata_T5.txt'))

        sf_sum = np.zeros(sf.shape)
        expected_sf_sum = np.zeros(sf.shape)
        for q in range(self.idata.n_qpts):
            TOL = 5e-4
            diff = np.append(TOL + 1, np.diff(self.idata.frequencies[q].magnitude))
            unique_index = np.where(diff > TOL)[0]
            x = np.zeros(3*self.idata.crystal.n_atoms, dtype=np.int32)
            x[unique_index] = 1
            unique_modes = np.cumsum(x) - 1
            sf_sum[q, :len(unique_index)] = np.bincount(unique_modes, sf[q])
            expected_sf_sum[q, :len(unique_index)] = np.bincount(
                unique_modes, expected_sf[q])
        npt.assert_allclose(sf_sum[:4], expected_sf_sum[:4], atol=3e-18)

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
            TOL = 5e-4
            diff = np.append(TOL + 1, np.diff(self.idata.frequencies[q].magnitude))
            unique_index = np.where(diff > TOL)[0]
            x = np.zeros(3*self.idata.crystal.n_atoms, dtype=np.int32)
            x[unique_index] = 1
            unique_modes = np.cumsum(x) - 1
            sf_sum[q, :len(unique_index)] = np.bincount(unique_modes, sf[q])
            expected_sf_sum[q, :len(unique_index)] = np.bincount(
                unique_modes, expected_sf[q])
        npt.assert_allclose(sf_sum[:4], expected_sf_sum[:4], atol=4e-18)

    def test_sf_T100_dw_idata(self):
        dw100 = self.dw_idata.calculate_debye_waller(100*ureg('K'))
        sf_obj = self.idata.calculate_structure_factor(
            self.scattering_lengths, dw=dw100)
        bose = _bose_factor(sf_obj._frequencies, 100)
        sf = sf_obj.structure_factors.to('bohr**2').magnitude*bose
        expected_sf = np.loadtxt(os.path.join(
            self.sf_path, 'sf_idata_dwidata_T100.txt'))

        sf_sum = np.zeros(sf.shape)
        expected_sf_sum = np.zeros(sf.shape)
        for q in range(self.idata.n_qpts):
            TOL = 5e-4
            diff = np.append(TOL + 1, np.diff(self.idata.frequencies[q].magnitude))
            unique_index = np.where(diff > TOL)[0]
            x = np.zeros(3*self.idata.crystal.n_atoms, dtype=np.int32)
            x[unique_index] = 1
            unique_modes = np.cumsum(x) - 1
            sf_sum[q, :len(unique_index)] = np.bincount(unique_modes, sf[q])
            expected_sf_sum[q, :len(unique_index)] = np.bincount(
                unique_modes, expected_sf[q])
        npt.assert_allclose(sf_sum[:4], expected_sf_sum[:4], atol=4e-18)

    def test_sf_T100_dw_pdata(self):
        dw100 = self.dw_pdata.calculate_debye_waller(100*ureg('K'))
        sf_obj = self.idata.calculate_structure_factor(
            self.scattering_lengths, dw=dw100)
        bose = _bose_factor(sf_obj._frequencies, 100)
        sf = sf_obj.structure_factors.to('bohr**2').magnitude*bose
        expected_sf = np.loadtxt(os.path.join(
            self.sf_path, 'sf_idata_dwpdata_T100.txt'))

        sf_sum = np.zeros(sf.shape)
        expected_sf_sum = np.zeros(sf.shape)
        for q in range(self.idata.n_qpts):
            TOL = 5e-4
            diff = np.append(TOL + 1, np.diff(self.idata.frequencies[q].magnitude))
            unique_index = np.where(diff > TOL)[0]
            x = np.zeros(3*self.idata.crystal.n_atoms, dtype=np.int32)
            x[unique_index] = 1
            unique_modes = np.cumsum(x) - 1
            sf_sum[q, :len(unique_index)] = np.bincount(unique_modes, sf[q])
            expected_sf_sum[q, :len(unique_index)] = np.bincount(
                unique_modes, expected_sf[q])
        npt.assert_allclose(sf_sum[:4], expected_sf_sum[:4], atol=4e-18)


class TestStructureFactorForceConstantsNaClSerialC(TestStructureFactorForceConstantsNaClSerial):

    def setUp(self):
        # Need to separately test SF calculation with interpolated phonon data
        # to test eigenvector calculations
        data_path = os.path.join(
            get_data_path(), 'phonopy_data', 'NaCl')
        self.interpolation_path = os.path.join(data_path, 'force_constants')
        self.sf_path = os.path.join(data_path, 'structure_factor')
        fm = ureg.fm
        self.scattering_lengths = {'Na': 3.63*fm, 'Cl': 9.577*fm}
        qpts = np.array([[0., 0., 0.],
                         [0., 0., 0.5],
                         [-0.25, 0.5, 0.5],
                         [-0.151515, 0.575758, 0.5]])

        fc = ForceConstants.from_phonopy(
            path=self.interpolation_path, summary_name='phonopy.yaml')
        self.idata = fc.calculate_qpoint_phonon_modes(
            qpts, asr='reciprocal', use_c=True, fall_back_on_python=False)

        # ForceConstants object for DW grid
        dw_fc = ForceConstants.from_phonopy(
            path=self.interpolation_path, summary_name='phonopy.yaml')
        self.dw_idata = dw_fc.calculate_qpoint_phonon_modes(
            np.loadtxt(os.path.join(get_data_path(), 'qgrid_444.txt')),
            asr='reciprocal', use_c=True, fall_back_on_python=False)

        # QpointPhononModes object for DW grid
        self.dw_pdata = QpointPhononModes.from_phonopy(
            phonon_name='mesh.yaml', path=os.path.join(data_path, 'mesh'))


class TestStructureFactorForceConstantsNaClParallelC(TestStructureFactorForceConstantsNaClSerial):

    def setUp(self):
        # Need to separately test SF calculation with interpolated phonon data
        # to test eigenvector calculations
        data_path = os.path.join(
            get_data_path(), 'phonopy_data', 'NaCl')
        self.interpolation_path = os.path.join(data_path, 'force_constants')
        self.sf_path = os.path.join(data_path, 'structure_factor')
        fm = ureg.fm
        self.scattering_lengths = {'Na': 3.63*fm, 'Cl': 9.577*fm}
        qpts = np.array([[0., 0., 0.],
                         [0., 0., 0.5],
                         [-0.25, 0.5, 0.5],
                         [-0.151515, 0.575758, 0.5]])

        fc = ForceConstants.from_phonopy(
            path=self.interpolation_path, summary_name='phonopy.yaml')
        self.idata = fc.calculate_qpoint_phonon_modes(
            qpts, asr='reciprocal', use_c=True, fall_back_on_python=False,
            n_threads=2)

        # ForceConstants object for DW grid
        dw_fc = ForceConstants.from_phonopy(
            path=self.interpolation_path, summary_name='phonopy.yaml')
        self.dw_idata = dw_fc.calculate_qpoint_phonon_modes(
            np.loadtxt(os.path.join(get_data_path(), 'qgrid_444.txt')),
            asr='reciprocal', use_c=True, fall_back_on_python=False,
            n_threads=2)

        # QpointPhononModes object for DW grid
        self.dw_pdata = QpointPhononModes.from_phonopy(
            phonon_name='mesh.yaml', path=os.path.join(data_path, 'mesh'))