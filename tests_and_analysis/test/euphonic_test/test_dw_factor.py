import os
import unittest
import numpy as np
import numpy.testing as npt
from euphonic import ureg, ForceConstants, QpointPhononModes
from ..utils import get_data_path


class TestDWFactorLZO(unittest.TestCase):

    def setUp(self):
        seedname = 'La2Zr2O7-grid'
        data_path = get_data_path()
        path = os.path.join(data_path, 'structure_factor', 'LZO')
        self.data = QpointPhononModes.from_castep(seedname, path=path)
        self.dw_path = os.path.join(data_path, 'dw_factor', 'LZO')

    def test_dw_T5(self):
        dw = self.data.calculate_debye_waller(5*ureg('K'))
        expected_dw = np.reshape(
            np.loadtxt(os.path.join(self.dw_path, 'dw_T5.txt')),
            (self.data.crystal.n_atoms, 3, 3))
        # Note: dividing expected_dw by 2 because convention for Debye-Waller
        # has changed and a division of 2 is now included in the Debye-Waller
        # calculation that was previously contained in the structure factor
        # calculation. e^((W_ab*Q_a*Q_b)/2) has become e^(W_ab*Q_a*Q_b)
        # absorbing 1/2 into W_ab (the Debye-Waller factor)
        npt.assert_allclose(dw.debye_waller.to('bohr**2').magnitude,
                            expected_dw/2)

    def test_dw_T100(self):
        dw = self.data.calculate_debye_waller(100*ureg('K'))
        expected_dw = np.reshape(
            np.loadtxt(os.path.join(self.dw_path, 'dw_T100.txt')),
            (self.data.crystal.n_atoms, 3, 3))
        npt.assert_allclose(dw.debye_waller.to('bohr**2').magnitude,
                            expected_dw/2, rtol=5e-7)


class TestDWFactorQuartz(unittest.TestCase):

    def setUp(self):
        self.seedname = 'quartz'
        data_path = get_data_path()
        self.path = os.path.join(data_path, 'interpolation', 'quartz')
        fc = ForceConstants.from_castep(self.seedname, path=self.path)
        qpts = np.loadtxt(os.path.join(data_path, 'qgrid_444.txt'))
        self.data = fc.calculate_qpoint_phonon_modes(qpts, asr='reciprocal')
        self.dw_path = os.path.join(data_path, 'dw_factor', 'quartz')

    def test_dw_T5(self):
        dw = self.data.calculate_debye_waller(5*ureg('K'))
        expected_dw = np.reshape(
            np.loadtxt(os.path.join(self.dw_path, 'dw_T5.txt')),
            (self.data.crystal.n_atoms, 3, 3))
        npt.assert_allclose(dw.debye_waller.to('bohr**2').magnitude,
                            expected_dw/2, atol=2e-8)

    def test_dw_T100(self):
        dw = self.data.calculate_debye_waller(100*ureg('K'))
        expected_dw = np.reshape(
            np.loadtxt(os.path.join(self.dw_path, 'dw_T100.txt')),
            (self.data.crystal.n_atoms, 3, 3))
        npt.assert_allclose(dw.debye_waller.to('bohr**2').magnitude,
                            expected_dw/2, atol=6e-8)