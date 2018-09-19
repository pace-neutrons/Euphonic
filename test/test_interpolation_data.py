import unittest
import os
import numpy.testing as npt
import numpy as np
from casteppy import ureg
from casteppy.data.interpolation import InterpolationData


class TestInputRead(unittest.TestCase):

    def setUp(self):
        # Create trivial function object so attributes can be assigned to it
        expctd_data = lambda:0
        expctd_data.n_ions = 22
        expctd_data.n_branches = 66
        expctd_data.cell_vec = np.array(
            [[7.58391282e+00, 1.84127921e-32, 0.00000000e+00],
             [3.79195641e+00, 3.79195641e+00, 5.36263618e+00],
             [3.79195641e+00, -3.79195641e+00, 5.36263618e+00]])
        expctd_data.ion_r = np.array([[0.125, 0.125, 0.125],
                                      [0.875, 0.875, 0.875],
                                      [0.41861943, 0.41861943, 0.83138057],
                                      [0.83138057, 0.83138057, 0.41861943],
                                      [0.41861943, 0.83138057, 0.41861943],
                                      [0.83138057, 0.41861943, 0.83138057],
                                      [0.83138057, 0.41861943, 0.41861943],
                                      [0.41861943, 0.83138057, 0.83138057],
                                      [0.16861943, 0.58138057, 0.16861943],
                                      [0.58138057, 0.16861943, 0.58138057],
                                      [0.16861943, 0.16861943, 0.58138057],
                                      [0.58138057, 0.58138057, 0.16861943],
                                      [0.16861943, 0.58138057, 0.58138057],
                                      [0.58138057, 0.16861943, 0.16861943],
                                      [0.5, 0.5, 0.5],
                                      [0.0, 0.5, 0.5],
                                      [0.5, 0.0, 0.5],
                                      [0.5, 0.5, 0.0],
                                      [0.0, 0.0, 0.0],
                                      [0.5, 0.0, 0.0],
                                      [0.0, 0.5, 0.0],
                                      [0.0, 0.0, 0.5]])
        expctd_data.ion_type = np.array([x.encode('ascii') for x in
                                        ['O', 'O', 'O', 'O', 'O', 'O', 'O',
                                         'O', 'O', 'O', 'O', 'O', 'O', 'O',
                                         'Zr', 'Zr', 'Zr', 'Zr',
                                         'La','La', 'La', 'La']])
        expctd_data.ion_mass = np.array(
        [15.99939999, 15.99939999, 15.99939999, 15.99939999, 15.99939999,
         15.99939999, 15.99939999, 15.99939999, 15.99939999, 15.99939999,
         15.99939999, 15.99939999, 15.99939999, 15.99939999, 91.22399997,
         91.22399997, 91.22399997, 91.22399997, 138.90549995,
         138.90549995, 138.90549995, 138.90549995])
        expctd_data.sc_matrix = np.array([[ 1, 1, -1],
                                          [ 1, -1, 1],
                                          [-1, 1, 1]])
        expctd_data.n_cells_in_sc = 4
        expctd_data.fc_mat_cell0_i0_j0 = np.array(
            [[1.26492555e-01, -2.31204635e-31, -1.16997352e-13],
             [-2.31204635e-31, 1.26492555e-01,  1.05181454e-30],
             [-1.16997352e-13, 3.15544362e-30,  1.26492555e-01]])
        expctd_data.fc_mat_cell3_i5_j10 = np.array(
            [[-8.32394989e-04, -2.22156212e-03, 7.33617499e-05],
             [-2.03285211e-03, -6.29315975e-04, 1.16282999e-03],
             [ 3.55359333e-04,  1.21568713e-03, 5.22410338e-05]])
        expctd_data.cell_origins = np.array([[0, 0, 0],
                                             [1, 0, 0],
                                             [0, 1, 0],
                                             [0, 0, 1]])
        self.expctd_data = expctd_data

        self.seedname = 'La2Zr2O7'
        self.path = os.path.join('test', 'data', 'interpolation')
        data = InterpolationData(self.seedname, self.path)
        self.data = data

    def test_n_ions_read(self):
        self.assertEqual(self.data.n_ions, self.expctd_data.n_ions)

    def test_n_branches_read(self):
        self.assertEqual(self.data.n_branches, self.expctd_data.n_branches)

    def test_cell_vec_read(self):
        npt.assert_allclose(self.data.cell_vec, self.expctd_data.cell_vec)

    def test_ion_r_read(self):
        npt.assert_allclose(self.data.ion_r, self.expctd_data.ion_r)

    def test_ion_type_read(self):
        npt.assert_array_equal(self.data.ion_type, self.expctd_data.ion_type)

    def test_ion_mass_read(self):
        npt.assert_allclose(self.data.ion_mass, self.expctd_data.ion_mass)

    def test_sc_matrix_read(self):
        npt.assert_allclose(self.data.sc_matrix, self.expctd_data.sc_matrix)

    def test_n_cells_in_sc_read(self):
        self.assertEqual(self.data.n_cells_in_sc,
                     self.expctd_data.n_cells_in_sc)

    def test_fc_mat_cell0_i0_j0_read(self):
        npt.assert_allclose(self.data.force_constants[0, 0, :, :],
                            self.expctd_data.fc_mat_cell0_i0_j0)

    def test_fc_mat_cell3_i5_j10_read(self):
        # 2nd fc index = cell*n_ions + j = 3*22 + 10 = 76
        npt.assert_allclose(self.data.force_constants[5, 76, :, :],
                            self.expctd_data.fc_mat_cell3_i5_j10)

    def test_fc_mat_read(self):
        expected_fc_mat = np.load(os.path.join(self.path, 'fc_mat_no_asr.npy'))
        npt.assert_allclose(self.data.force_constants, expected_fc_mat)

class TestInterpolatePhonons(unittest.TestCase):

    def setUp(self):
        seedname = 'La2Zr2O7'
        self.path = os.path.join('test', 'data', 'interpolation')
        data = InterpolationData(seedname, self.path)
        self.data = data

    def test_calculate_supercell_image_r_lim_1(self):
        expected_image_r = np.array([[-1, -1, -1],
                                     [-1, -1,  0],
                                     [-1, -1,  1],
                                     [-1,  0, -1],
                                     [-1,  0,  0],
                                     [-1,  0,  1],
                                     [-1,  1, -1],
                                     [-1,  1,  0],
                                     [-1,  1,  1],
                                     [ 0, -1, -1],
                                     [ 0, -1,  0],
                                     [ 0, -1,  1],
                                     [ 0,  0, -1],
                                     [ 0,  0,  0],
                                     [ 0,  0,  1],
                                     [ 0,  1, -1],
                                     [ 0,  1,  0],
                                     [ 0,  1,  1],
                                     [ 1, -1, -1],
                                     [ 1, -1,  0],
                                     [ 1, -1,  1],
                                     [ 1,  0, -1],
                                     [ 1,  0,  0],
                                     [ 1,  0,  1],
                                     [ 1,  1, -1],
                                     [ 1,  1,  0],
                                     [ 1,  1,  1]])
        lim = 1
        image_r = self.data._calculate_supercell_image_r(lim)
        npt.assert_equal(image_r, expected_image_r)

    def test_calculate_supercell_image_r_lim_2(self):
        expected_image_r = np.loadtxt(os.path.join(self.path, 'sc_image_r.txt'))
        lim = 2
        image_r = self.data._calculate_supercell_image_r(lim)
        npt.assert_equal(image_r, expected_image_r)

    def test_calculate_phases_qpt(self):
        lim = 2
        qpt = [-1, 9.35, 3.35]
        sc_image_r = np.loadtxt(os.path.join(self.path, 'sc_image_r.txt'))

        phase_data = np.loadtxt(os.path.join(self.path, 'phases.txt'))
        nc = phase_data[:, 0].astype(int)
        i = phase_data[:, 1].astype(int)
        expected_phases = np.zeros((4, (2*lim + 1)**3 + 1), dtype=np.complex128)
        expected_phases[nc, i] = (phase_data[:, 2].astype(float)
                                  + phase_data[:, 3].astype(float)*1j)

        phases = self.data._calculate_phases(lim, qpt, sc_image_r)
        npt.assert_allclose(phases, expected_phases)

    def test_calculate_phases_gamma_pt(self):
        lim = 2
        qpt = [0.0, 0.0, 0.0]
        sc_image_r = np.loadtxt(os.path.join(self.path, 'sc_image_r.txt'))

        expected_phases = np.zeros((4, (2*lim + 1)**3 + 1),
                                   dtype=np.complex128)
        # Last row of phases should always be zero, so that when summing
        # supercell image phases for the cumulant method, if there isn't an
        # image for that ion a phase of zero can be used
        expected_phases[:, :-1] = 1.0 + 0.0*1j

        phases = self.data._calculate_phases(lim, qpt, sc_image_r)
        npt.assert_equal(phases, expected_phases)

    def test_calculate_supercell_images_n_sc_images(self):
        # Supercell image calculation limit - 2 supercells in each direction
        lim = 2
        image_data = np.loadtxt(os.path.join(self.path, 'n_sc_images.txt'))
        i = image_data[:, 0].astype(int)
        j = image_data[:, 1].astype(int)
        n = image_data[:, 2].astype(int)
        expctd_n_images = np.zeros((22, 88)) # size = n_ions X n_ions*n_cells_in_sc
        expctd_n_images[i, j] = n
        self.data._calculate_supercell_images(lim)
        npt.assert_equal(self.data.n_sc_images, expctd_n_images)


    def test_calculate_supercell_images_sc_image_i(self):
        # Supercell image calculation limit - 2 supercells in each direction
        lim = 2
        image_data = np.loadtxt(os.path.join(self.path, 'sc_image_i.txt'))
        i = image_data[:, 0].astype(int)
        j = image_data[:, 1].astype(int)
        n = image_data[:, 2].astype(int)
        sc_i = image_data[:, 3].astype(int)
        # size = n_ions X n_ions*n_cells_in_sc X max supercell images
        expctd_sc_image_i = np.full((22, 88, (2*lim + 1)**3), -1)
        expctd_sc_image_i[i, j, n] = sc_i
        self.data._calculate_supercell_images(lim)
        npt.assert_equal(self.data.sc_image_i, expctd_sc_image_i)

    def test_calculate_fine_phonons_no_asr(self):
        qpts = np.array([[-1.00, 9.35, 3.35],
                         [-1.00, 9.30, 3.30]])
        expctd_freqs = np.array([[0.0002964463, 0.0002964463, 0.0003208033,
                                  0.0003501442, 0.0003501442, 0.0003903142,
                                  0.0004971960, 0.0004971960, 0.0005372831,
                                  0.0005438606, 0.0005861142, 0.0005861142,
                                  0.0007103592, 0.0007103592, 0.0007331462,
                                  0.0007786131, 0.0007874301, 0.0007929212,
                                  0.0008125973, 0.0008354708, 0.0008354708,
                                  0.0009078655, 0.0009078655, 0.0010160279,
                                  0.0010264260, 0.0010554380, 0.0011528116,
                                  0.0012094889, 0.0012304152, 0.0012410381,
                                  0.0012410381, 0.0012564502, 0.0013664021,
                                  0.0013664021, 0.0014355540, 0.0014355540,
                                  0.0014576131, 0.0015442769, 0.0015442769,
                                  0.0015449045, 0.0015505596, 0.0015937747,
                                  0.0017167580, 0.0017828415, 0.0017828415,
                                  0.0018048098, 0.0018598041, 0.0018598041,
                                  0.0018726136, 0.0019193752, 0.0020786720,
                                  0.0020786720, 0.0022934815, 0.0024275690,
                                  0.0024275690, 0.0024850294, 0.0025000806,
                                  0.0025179334, 0.0025179334, 0.0025401035,
                                  0.0025550180, 0.0025550180, 0.0028191045,
                                  0.0033473131, 0.0033680460, 0.0033680460],
                                 [0.0002824983, 0.0002824983, 0.0003072456,
                                  0.0003230949, 0.0003230949, 0.0003983840,
                                  0.0005007107, 0.0005007107, 0.0005054376,
                                  0.0005222212, 0.0005887989, 0.0005887989,
                                  0.0006746343, 0.0006861545, 0.0006861545,
                                  0.0007556994, 0.0007616131, 0.0007690864,
                                  0.0007737532, 0.0008536269, 0.0008536269,
                                  0.0009203968, 0.0009203968, 0.0010084528,
                                  0.0010433982, 0.0010561472, 0.0011814957,
                                  0.0012049367, 0.0012309021, 0.0012309021,
                                  0.0012705846, 0.0012715744, 0.0013684938,
                                  0.0013684938, 0.0014212562, 0.0014212562,
                                  0.0014500614, 0.0015262255, 0.0015262255,
                                  0.0015416714, 0.0015525612, 0.0016222745,
                                  0.0016979556, 0.0018021499, 0.0018021499,
                                  0.0018150461, 0.0018516882, 0.0018516882,
                                  0.0018554149, 0.0019187766, 0.0020983195,
                                  0.0020983195, 0.0022159595, 0.0024316898,
                                  0.0024316898, 0.0024829004, 0.0025025788,
                                  0.0025093281, 0.0025093281, 0.0025424669,
                                  0.0025499149, 0.0025499149, 0.0026935713,
                                  0.0033651415, 0.0033651415, 0.0033698188]])*ureg.hartree
        self.data.calculate_fine_phonons(qpts, asr=False)
        npt.assert_allclose(self.data.freqs, expctd_freqs, rtol=1e-3)

    def test_calculate_fine_phonons_asr(self):
        qpts = np.array([[-1.00, 9.35, 3.35],
                         [-1.00, 9.30, 3.30]])
        expctd_freqs = np.array([[0.0002964463, 0.0002964463, 0.0003208033,
                                  0.0003501442, 0.0003501442, 0.0003903142,
                                  0.0004971960, 0.0004971960, 0.0005372831,
                                  0.0005438606, 0.0005861142, 0.0005861142,
                                  0.0007103592, 0.0007103592, 0.0007331462,
                                  0.0007786131, 0.0007874301, 0.0007929212,
                                  0.0008125973, 0.0008354708, 0.0008354708,
                                  0.0009078655, 0.0009078655, 0.0010160279,
                                  0.0010264260, 0.0010554380, 0.0011528116,
                                  0.0012094889, 0.0012304152, 0.0012410381,
                                  0.0012410381, 0.0012564502, 0.0013664021,
                                  0.0013664021, 0.0014355540, 0.0014355540,
                                  0.0014576131, 0.0015442769, 0.0015442769,
                                  0.0015449045, 0.0015505596, 0.0015937747,
                                  0.0017167580, 0.0017828415, 0.0017828415,
                                  0.0018048098, 0.0018598041, 0.0018598041,
                                  0.0018726136, 0.0019193752, 0.0020786720,
                                  0.0020786720, 0.0022934815, 0.0024275690,
                                  0.0024275690, 0.0024850294, 0.0025000806,
                                  0.0025179334, 0.0025179334, 0.0025401035,
                                  0.0025550180, 0.0025550180, 0.0028191045,
                                  0.0033473131, 0.0033680460, 0.0033680460],
                                 [0.0002824983, 0.0002824983, 0.0003072456,
                                  0.0003230949, 0.0003230949, 0.0003983840,
                                  0.0005007107, 0.0005007107, 0.0005054376,
                                  0.0005222212, 0.0005887989, 0.0005887989,
                                  0.0006746343, 0.0006861545, 0.0006861545,
                                  0.0007556994, 0.0007616131, 0.0007690864,
                                  0.0007737532, 0.0008536269, 0.0008536269,
                                  0.0009203968, 0.0009203968, 0.0010084528,
                                  0.0010433982, 0.0010561472, 0.0011814957,
                                  0.0012049367, 0.0012309021, 0.0012309021,
                                  0.0012705846, 0.0012715744, 0.0013684938,
                                  0.0013684938, 0.0014212562, 0.0014212562,
                                  0.0014500614, 0.0015262255, 0.0015262255,
                                  0.0015416714, 0.0015525612, 0.0016222745,
                                  0.0016979556, 0.0018021499, 0.0018021499,
                                  0.0018150461, 0.0018516882, 0.0018516882,
                                  0.0018554149, 0.0019187766, 0.0020983195,
                                  0.0020983195, 0.0022159595, 0.0024316898,
                                  0.0024316898, 0.0024829004, 0.0025025788,
                                  0.0025093281, 0.0025093281, 0.0025424669,
                                  0.0025499149, 0.0025499149, 0.0026935713,
                                  0.0033651415, 0.0033651415, 0.0033698188]])*ureg.hartree
        self.data.calculate_fine_phonons(qpts, asr=True)
        npt.assert_allclose(self.data.freqs, expctd_freqs, rtol=1e-4)

    def test_enforce_acoustic_sum_rule(self):
        expected_fc_mat = np.load(os.path.join(self.path, 'fc_mat_asr.npy'))
        fc_mat = self.data._enforce_acoustic_sum_rule().to(ureg.hartree/ureg.bohr**2)
        npt.assert_allclose(fc_mat, expected_fc_mat, atol=1e-18)