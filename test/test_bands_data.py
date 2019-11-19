import unittest
import numpy.testing as npt
import numpy as np
from euphonic.data.bands import BandsData
from euphonic import ureg


class TestBandsDataNaH(unittest.TestCase):

    def setUp(self):
        # Test creation of BandsData object (which reads NaH.bands file in
        # test/data dir). BandsData object will also read extra data (ion_r
        # and ion_type) from the NaH.castep file

        # Create trivial object so attributes can be assigned to it
        expctd_data = type('', (), {})()  # Expected data

        expctd_data.cell_vec = np.array(
            [[0.000000, 4.534397, 4.534397],
             [4.534397, 0.000000, 4.534397],
             [4.534397, 4.534397, 0.000000]])*ureg('bohr')
        expctd_data.ion_r = np.array([[0.500000, 0.500000, 0.500000],
                                      [0.000000, 0.000000, 0.000000]])
        expctd_data.ion_type = np.array(['H', 'Na'])
        expctd_data.qpts = np.array([[-0.45833333, -0.37500000, -0.45833333],
                                     [-0.45833333, -0.37500000, -0.20833333]])
        expctd_data.weights = np.array([0.00347222, 0.00694444])
        expctd_data.freqs = np.array(
            [[-1.83230180, -0.83321119, -0.83021854,
              -0.83016941, -0.04792334],
             [-1.83229571, -0.83248269, -0.83078961,
              -0.83036048, -0.05738470]])*ureg('hartree')
        expctd_data.freq_down = np.array([])*ureg('hartree')
        expctd_data.fermi = np.array([-0.009615])*ureg('hartree')
        self.expctd_data = expctd_data

        seedname = 'NaH'
        path = 'data'
        data = BandsData.from_castep(seedname, path=path)
        self.data = data

    def test_cell_vec_read_nah_bands(self):
        npt.assert_allclose(self.data.cell_vec.to('bohr').magnitude,
                            self.expctd_data.cell_vec.to('bohr').magnitude)

    def test_ion_r_read_nah_bands(self):
        npt.assert_array_equal(self.data.ion_r,
                               self.expctd_data.ion_r)

    def test_ion_type_read_nah_bands(self):
        npt.assert_array_equal(self.data.ion_type,
                               self.expctd_data.ion_type)

    def test_qpts_read_nah_bands(self):
        npt.assert_array_equal(self.data.qpts,
                               self.expctd_data.qpts)

    def test_weights_read_nah_bands(self):
        npt.assert_array_equal(self.data.weights,
                               self.expctd_data.weights)

    def test_freqs_read_nah_bands(self):
        npt.assert_allclose(self.data.freqs.to('hartree').magnitude,
                            self.expctd_data.freqs.to('hartree').magnitude)

    def test_freq_down_read_nah_bands(self):
        npt.assert_allclose(self.data.freq_down.to('hartree').magnitude,
                            self.expctd_data.freq_down.to('hartree').magnitude)

    def test_fermi_read_nah_bands(self):
        npt.assert_allclose(self.data.fermi.to('hartree').magnitude,
                            self.expctd_data.fermi.to('hartree').magnitude)


class TestBandsDataFe(unittest.TestCase):

    def setUp(self):
        # Test creation of BandsData object (which reads Fe.bands file in
        # test/data dir). There is no Fe.castep file in test/data so the ion_r
        # and ion_pos attributes shouldn't exist

        # Create trivial function object so attributes can be assigned to it
        expctd_data = type('', (), {})()  # Expected data

        expctd_data.cell_vec = np.array(
            [[-2.708355, 2.708355, 2.708355],
             [2.708355, -2.708355, 2.708355],
             [2.708355, 2.708355, -2.708355]])*ureg('bohr')

        expctd_data.qpts = np.array([[-0.37500000, -0.45833333, 0.29166667],
                                     [-0.37500000, -0.37500000, 0.29166667]])
        expctd_data.weights = np.array([0.01388889, 0.01388889])
        expctd_data.freqs = np.array(
            [[0.02278248, 0.02644693, 0.12383402,
              0.15398152, 0.17125020, 0.43252010],
             [0.02760952, 0.02644911, 0.12442671,
              0.14597457, 0.16728951, 0.35463529]])*ureg('hartree')
        expctd_data.freq_down = np.array(
            [[0.08112495, 0.08345039, 0.19185076,
              0.22763689, 0.24912308, 0.46511567],
             [0.08778721, 0.08033338, 0.19288937,
              0.21817779, 0.24476910, 0.39214129]])*ureg('hartree')
        expctd_data.fermi = [0.173319, 0.173319]*ureg('hartree')
        self.expctd_data = expctd_data

        seedname = 'Fe'
        path = 'data'
        data = BandsData.from_castep(seedname, path=path)
        self.data = data

    def test_cell_vec_read_fe_bands(self):
        npt.assert_array_equal(self.data.cell_vec.to('bohr').magnitude,
                               self.expctd_data.cell_vec.to('bohr').magnitude)

    def test_ion_r_read_fe_bands(self):
        self.assertFalse(hasattr(self.data, 'ion_r'))

    def test_ion_type_read_fe_bands(self):
        self.assertFalse(hasattr(self.data, 'ion_type'))

    def test_qpts_read_fe_bands(self):
        npt.assert_array_equal(self.data.qpts,
                               self.expctd_data.qpts)

    def test_weights_read_fe_bands(self):
        npt.assert_array_equal(self.data.weights,
                               self.expctd_data.weights)

    def test_freqs_read_fe_bands(self):
        npt.assert_allclose(self.data.freqs.to('hartree').magnitude,
                            self.expctd_data.freqs.to('hartree').magnitude)

    def test_freq_down_read_fe_bands(self):
        npt.assert_allclose(self.data.freq_down.to('hartree').magnitude,
                            self.expctd_data.freq_down.to('hartree').magnitude)

    def test_fermi_read_fe_bands(self):
        npt.assert_allclose(self.data.fermi.to('hartree').magnitude,
                            self.expctd_data.fermi.to('hartree').magnitude)
