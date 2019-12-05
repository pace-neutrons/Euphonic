import unittest
import os
import numpy.testing as npt
import numpy as np
from euphonic import ureg
from euphonic.data.interpolation import InterpolationData


class TestInputReadLZO(unittest.TestCase):

    def setUp(self):
        # Create trivial function object so attributes can be assigned to it
        expctd_data = type('', (), {})()
        expctd_data.n_ions = 22
        expctd_data.n_branches = 66
        expctd_data.cell_vec = (np.array(
            [[7.58391282e+00, 1.84127921e-32, 0.00000000e+00],
             [3.79195641e+00, 3.79195641e+00, 5.36263618e+00],
             [3.79195641e+00, -3.79195641e+00, 5.36263618e+00]])
            *ureg('angstrom'))
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
        expctd_data.ion_type = np.array(['O', 'O', 'O', 'O', 'O', 'O', 'O',
                                         'O', 'O', 'O', 'O', 'O', 'O', 'O',
                                         'Zr', 'Zr', 'Zr', 'Zr',
                                         'La', 'La', 'La', 'La'])
        expctd_data.ion_mass = np.array(
            [15.99939999, 15.99939999, 15.99939999, 15.99939999, 15.99939999,
             15.99939999, 15.99939999, 15.99939999, 15.99939999, 15.99939999,
             15.99939999, 15.99939999, 15.99939999, 15.99939999, 91.22399997,
             91.22399997, 91.22399997, 91.22399997, 138.90549995,
             138.90549995, 138.90549995, 138.90549995])*ureg('amu')
        expctd_data.sc_matrix = np.array([[1, 1, -1],
                                          [1, -1, 1],
                                          [-1, 1, 1]])
        expctd_data.n_cells_in_sc = 4
        expctd_data.fc_mat_cell0_i0_j0 = (np.array(
            [[1.26492555e-01, -2.31204635e-31, -1.16997352e-13],
             [-2.31204635e-31, 1.26492555e-01, 3.15544362e-30],
             [-1.16997352e-13, 1.05181454e-30, 1.26492555e-01]])
            *ureg('hartree/bohr**2'))
        expctd_data.fc_mat_cell3_i10_j5 = (np.array(
            [[-8.32394989e-04, -2.03285211e-03, 3.55359333e-04],
             [-2.22156212e-03, -6.29315975e-04, 1.21568713e-03],
             [7.33617499e-05, 1.16282999e-03, 5.22410338e-05]])
            *ureg('hartree/bohr**2'))
        expctd_data.cell_origins = np.array([[0, 0, 0],
                                             [1, 0, 0],
                                             [0, 1, 0],
                                             [0, 0, 1]])
        self.expctd_data = expctd_data

        self.seedname = 'La2Zr2O7'
        self.path = os.path.join('data', 'interpolation', 'LZO')
        data = InterpolationData.from_castep(self.seedname, path=self.path)
        self.data = data

    def test_n_ions_read(self):
        self.assertEqual(self.data.n_ions, self.expctd_data.n_ions)

    def test_n_branches_read(self):
        self.assertEqual(self.data.n_branches, self.expctd_data.n_branches)

    def test_cell_vec_read(self):
        npt.assert_allclose(self.data.cell_vec.to('bohr').magnitude,
                            self.expctd_data.cell_vec.to('bohr').magnitude)

    def test_ion_r_read(self):
        npt.assert_allclose(self.data.ion_r, self.expctd_data.ion_r)

    def test_ion_type_read(self):
        npt.assert_array_equal(self.data.ion_type, self.expctd_data.ion_type)

    def test_ion_mass_read(self):
        npt.assert_allclose(self.data.ion_mass.to('amu').magnitude,
                            self.expctd_data.ion_mass.to('amu').magnitude)

    def test_sc_matrix_read(self):
        npt.assert_allclose(self.data.sc_matrix, self.expctd_data.sc_matrix)

    def test_n_cells_in_sc_read(self):
        self.assertEqual(self.data.n_cells_in_sc,
                         self.expctd_data.n_cells_in_sc)

    def test_fc_mat_cell0_i0_j0_read(self):
        npt.assert_allclose(self.data.force_constants[0, 0:3, 0:3].magnitude,
                            self.expctd_data.fc_mat_cell0_i0_j0.magnitude)

    def test_fc_mat_cell3_i10_j5_read(self):
        npt.assert_allclose(
            self.data.force_constants[3, 30:33, 15:18].magnitude,
            self.expctd_data.fc_mat_cell3_i10_j5.magnitude)

    def test_fc_mat_read(self):
        expected_fc_mat = np.load(os.path.join(
            self.path, 'lzo_fc_mat_no_asr.npy'))*ureg('hartree/bohr**2')
        npt.assert_allclose(self.data.force_constants.magnitude,
                            expected_fc_mat.magnitude)


class TestInterpolatePhononsLZO(unittest.TestCase):

    def setUp(self):
        seedname = 'La2Zr2O7'
        self.path = os.path.join('data', 'interpolation', 'LZO')
        data = InterpolationData.from_castep(seedname, path=self.path)
        self.data = data

        self.qpts = np.array([[-1.00, 9.35, 3.35],
                              [-1.00, 9.00, 3.00]])
        self.expctd_freqs_asr = (np.array([
            [0.0002964449, 0.0002964449, 0.0003208033,
             0.0003501419, 0.0003501419, 0.0003903141,
             0.0004972138, 0.0004972138, 0.0005372809,
             0.0005438643, 0.0005861166, 0.0005861166,
             0.0007103804, 0.0007103804, 0.0007331639,
             0.0007786131, 0.0007874376, 0.0007929211,
             0.0008126016, 0.0008354861, 0.0008354861,
             0.0009078731, 0.0009078731, 0.0010160378,
             0.0010264374, 0.0010554444, 0.0011528145,
             0.0012094888, 0.0012304278, 0.0012410548,
             0.0012410548, 0.0012564500, 0.0013664070,
             0.0013664070, 0.0014355566, 0.0014355566,
             0.0014576129, 0.0015442745, 0.0015442745,
             0.0015449039, 0.0015505652, 0.0015937746,
             0.0017167608, 0.0017828465, 0.0017828465,
             0.0018048096, 0.0018598080, 0.0018598080,
             0.0018726170, 0.0019193824, 0.0020786777,
             0.0020786777, 0.0022934801, 0.0024275754,
             0.0024275754, 0.0024850292, 0.0025000804,
             0.0025179345, 0.0025179345, 0.0025401087,
             0.0025550191, 0.0025550191, 0.0028191070,
             0.0033473173, 0.0033680501, 0.0033680501],
            [-1.2527213902e-09, -1.2524650945e-09, -1.2522509615e-09,
             2.5186476888e-04, 2.5186476888e-04, 2.5186476888e-04,
             4.2115533128e-04, 4.2115533128e-04, 4.5919137201e-04,
             4.5919137201e-04, 4.5919137201e-04, 6.0460911494e-04,
             6.0460911494e-04, 6.0460911494e-04, 6.1121916807e-04,
             6.1121916807e-04, 6.1121916807e-04, 6.8315329115e-04,
             6.8315329115e-04, 8.9085325717e-04, 8.9085325717e-04,
             8.9085325717e-04, 1.0237996415e-03, 1.0237996415e-03,
             1.0237996415e-03, 1.1666034640e-03, 1.1744920636e-03,
             1.1744920636e-03, 1.1744920636e-03, 1.2802064794e-03,
             1.2802064794e-03, 1.2802064794e-03, 1.3122749877e-03,
             1.4159439055e-03, 1.4159439055e-03, 1.4159439055e-03,
             1.4813308344e-03, 1.4813308344e-03, 1.5084818310e-03,
             1.5084818310e-03, 1.5084818310e-03, 1.5957863393e-03,
             1.7192994802e-03, 1.7192994802e-03, 1.8119521571e-03,
             1.8119521571e-03, 1.8119521571e-03, 1.8609709896e-03,
             1.8609709896e-03, 1.8609709896e-03, 2.1913629570e-03,
             2.2033452584e-03, 2.2033452584e-03, 2.2033452584e-03,
             2.4420900293e-03, 2.4420900293e-03, 2.4420900293e-03,
             2.4754830417e-03, 2.4754830417e-03, 2.4754830417e-03,
             2.5106852083e-03, 2.5106852083e-03, 2.5106852083e-03,
             3.3517193438e-03, 3.3517193438e-03, 3.3517193438e-03]])
            *ureg('hartree'))
        self.expctd_freqs_no_asr = (np.array([
            [0.0002964623, 0.0002964623, 0.0003208033, 0.000350174,
             0.000350174, 0.0003903141, 0.0004972179, 0.0004972179,
             0.0005372886, 0.0005438642, 0.0005861163, 0.0005861163,
             0.0007103807, 0.0007103807, 0.0007331935, 0.0007786131,
             0.0007874315, 0.0007929211, 0.0008126019, 0.0008354958,
             0.0008354958, 0.000907874, 0.000907874, 0.0010160402,
             0.0010264376, 0.0010554468, 0.0011528125, 0.0012094888,
             0.0012304238, 0.0012410492, 0.0012410492, 0.00125645,
             0.0013664066, 0.0013664066, 0.0014355603, 0.0014355603,
             0.0014576129, 0.0015442837, 0.0015442837, 0.0015449061,
             0.0015505634, 0.0015937746, 0.0017167601, 0.0017828448,
             0.0017828448, 0.001804811, 0.0018598084, 0.0018598084,
             0.0018726182, 0.0019193824, 0.0020786769, 0.0020786769,
             0.002293487, 0.0024275755, 0.0024275755, 0.0024850292,
             0.0025000804, 0.002517934, 0.002517934, 0.0025401063,
             0.0025550198, 0.0025550198, 0.0028191102, 0.0033473155,
             0.0033680501, 0.0033680501],
            [1.2522582708e-05, 1.2522582708e-05, 1.2522582708e-05,
             2.5186476888e-04, 2.5186476888e-04, 2.5186476888e-04,
             4.2115533128e-04, 4.2115533128e-04, 4.5920462007e-04,
             4.5920462007e-04, 4.5920462007e-04, 6.0462274991e-04,
             6.0462274991e-04, 6.0462274991e-04, 6.1121916807e-04,
             6.1121916807e-04, 6.1121916807e-04, 6.8315329115e-04,
             6.8315329115e-04, 8.9089400855e-04, 8.9089400855e-04,
             8.9089400855e-04, 1.0238000223e-03, 1.0238000223e-03,
             1.0238000223e-03, 1.1666034640e-03, 1.1744920636e-03,
             1.1744920636e-03, 1.1744920636e-03, 1.2802064794e-03,
             1.2802064794e-03, 1.2802064794e-03, 1.3122749877e-03,
             1.4159439055e-03, 1.4159439055e-03, 1.4159439055e-03,
             1.4813308344e-03, 1.4813308344e-03, 1.5085078032e-03,
             1.5085078032e-03, 1.5085078032e-03, 1.5957863393e-03,
             1.7192994802e-03, 1.7192994802e-03, 1.8119544413e-03,
             1.8119544413e-03, 1.8119544413e-03, 1.8609709896e-03,
             1.8609709896e-03, 1.8609709896e-03, 2.1913629570e-03,
             2.2033465408e-03, 2.2033465408e-03, 2.2033465408e-03,
             2.4420900293e-03, 2.4420900293e-03, 2.4420900293e-03,
             2.4754830417e-03, 2.4754830417e-03, 2.4754830417e-03,
             2.5106852083e-03, 2.5106852083e-03, 2.5106852083e-03,
             3.3517193438e-03, 3.3517193438e-03, 3.3517193438e-03]])
            *ureg('hartree'))
        self.unique_sc_i = np.loadtxt(os.path.join(
            self.path, 'lzo_unique_sc_i.txt'), dtype=np.int32)
        self.unique_cell_i = np.loadtxt(os.path.join(
            self.path, 'lzo_unique_cell_i.txt'), dtype=np.int32)
        self.unique_sc_offsets = [[] for i in range(3)]
        with open(os.path.join(self.path, 'lzo_unique_sc_offsets.txt')) as f:
            for i in range(3):
                self.unique_sc_offsets[i] = [int(x)
                                             for x in f.readline().split()]
        self.unique_cell_origins = [[] for i in range(3)]
        with open(os.path.join(self.path, 'lzo_unique_cell_origins.txt')) as f:
            for i in range(3):
                self.unique_cell_origins[i] = [int(x)
                                               for x in f.readline().split()]

    def test_get_all_origins_lim_1(self):
        expected_image_r = np.array([[-1, -1, -1],
                                     [-1, -1, 0],
                                     [-1, -1, 1],
                                     [-1, 0, -1],
                                     [-1, 0, 0],
                                     [-1, 0, 1],
                                     [-1, 1, -1],
                                     [-1, 1, 0],
                                     [-1, 1, 1],
                                     [0, -1, -1],
                                     [0, -1, 0],
                                     [0, -1, 1],
                                     [0, 0, -1],
                                     [0, 0, 0],
                                     [0, 0, 1],
                                     [0, 1, -1],
                                     [0, 1, 0],
                                     [0, 1, 1],
                                     [1, -1, -1],
                                     [1, -1, 0],
                                     [1, -1, 1],
                                     [1, 0, -1],
                                     [1, 0, 0],
                                     [1, 0, 1],
                                     [1, 1, -1],
                                     [1, 1, 0],
                                     [1, 1, 1]])
        lim = 1
        image_r = self.data._get_all_origins(np.repeat(lim, 3) + 1,
                                             min_xyz=-np.repeat(lim, 3))
        npt.assert_equal(image_r, expected_image_r)

    def test_get_all_origins_lim_2(self):
        expected_image_r = np.loadtxt(
            os.path.join(self.path, 'lzo_sc_image_r.txt'))
        lim = 2
        image_r = self.data._get_all_origins(np.repeat(lim, 3) + 1,
                                             min_xyz=-np.repeat(lim, 3))
        npt.assert_equal(image_r, expected_image_r)

    def test_calculate_phases_qpt(self):
        qpt = np.array([-1, 9.35, 3.35])

        sc_phases, cell_phases = self.data._calculate_phases(
            qpt, self.unique_sc_offsets, self.unique_sc_i,
            self.unique_cell_origins, self.unique_cell_i)

        expected_sc_phases = np.loadtxt(os.path.join(
            self.path, 'lzo_sc_phases.txt'), dtype=np.complex128)
        expected_cell_phases = np.loadtxt(os.path.join(
            self.path, 'lzo_cell_phases.txt'), dtype=np.complex128)

        npt.assert_allclose(sc_phases, expected_sc_phases)
        npt.assert_allclose(cell_phases, expected_cell_phases)

    def test_calculate_phases_gamma_pt(self):
        qpt = np.array([0.0, 0.0, 0.0])

        sc_phases, cell_phases = self.data._calculate_phases(
            qpt, self.unique_sc_offsets, self.unique_sc_i,
            self.unique_cell_origins, self.unique_cell_i)

        expected_sc_phases = np.full(
            len(self.unique_sc_i), 1.0 + 0.0*1j, dtype=np.complex128)
        expected_cell_phases = np.full(
            len(self.unique_cell_i), 1.0 + 0.0*1j, dtype=np.complex128)

        npt.assert_equal(sc_phases, expected_sc_phases)
        npt.assert_equal(cell_phases, expected_cell_phases)

    def test_calculate_supercell_images_n_sc_images(self):
        # Supercell image calculation limit - 2 supercells in each direction
        lim = 2
        image_data = np.loadtxt(os.path.join(self.path, 'lzo_n_sc_images.txt'))
        i = image_data[:, 0].astype(int)
        j = image_data[:, 1].astype(int)
        n = image_data[:, 2].astype(int)
        expctd_n_images = np.zeros((22, 88))
        expctd_n_images[i, j] = n
        # After refactoring where the shape of n_sc_images was changed from
        # (n_ions, n_cells_in_sc*n_ions) to (n_cells_in_sc, n_ions, n_ions),
        # expctd_n_images must be reshaped to ensure tests still pass
        expctd_n_images = np.transpose(np.reshape(
            expctd_n_images, (22, 4, 22)), axes=[1, 0, 2])
        self.data._calculate_supercell_images(lim)
        npt.assert_equal(self.data._n_sc_images, expctd_n_images)

    def test_calculate_supercell_images_sc_image_i(self):
        # Supercell image calculation limit - 2 supercells in each direction
        lim = 2
        image_data = np.loadtxt(os.path.join(self.path, 'lzo_sc_image_i.txt'))
        i = image_data[:, 0].astype(int)
        j = image_data[:, 1].astype(int)
        n = image_data[:, 2].astype(int)
        sc_i = image_data[:, 3].astype(int)
        max_n = np.max(n) + 1
        # size = n_ions X n_ions*n_cells_in_sc X max supercell images
        expctd_sc_image_i = np.full((22, 88, max_n), -1)
        expctd_sc_image_i[i, j, n] = sc_i
        # After refactoring where the shape of sc_image_i was changed from
        # (n_ions, n_cells_in_sc*n_ions, (2*lim + 1)**3) to
        # (n_cells_in_sc, n_ions, n_ions, (2*lim + 1)**3),
        # expctd_image_i must be reshaped to ensure tests still pass
        expctd_sc_image_i = np.transpose(
            np.reshape(expctd_sc_image_i, (22, 4, 22, max_n)),
            axes=[1, 0, 2, 3])
        self.data._calculate_supercell_images(lim)
        npt.assert_equal(self.data._sc_image_i, expctd_sc_image_i)

    def test_calculate_fine_phonons_no_asr(self):
        self.data.calculate_fine_phonons(self.qpts)
        npt.assert_allclose(self.data.freqs.to('hartree').magnitude,
                            self.expctd_freqs_no_asr.to('hartree').magnitude,
                            atol=1e-10)

    def test_calculate_fine_phonons_no_asr_c(self):
        self.data.calculate_fine_phonons(self.qpts, use_c=True)
        npt.assert_allclose(self.data.freqs.to('hartree').magnitude,
                            self.expctd_freqs_no_asr.to('hartree').magnitude,
                            atol=1e-10)

    def test_calculate_fine_phonons_no_asr_c_2threads(self):
        self.data.calculate_fine_phonons(self.qpts, use_c=True, n_threads=2)
        npt.assert_allclose(self.data.freqs.to('hartree').magnitude,
                            self.expctd_freqs_no_asr.to('hartree').magnitude,
                            atol=1e-10)

    def test_calculate_fine_phonons_realspace_asr(self):
        self.data.calculate_fine_phonons(self.qpts, asr='realspace')
        npt.assert_allclose(self.data.freqs.to('hartree').magnitude,
                            self.expctd_freqs_asr.to('hartree').magnitude,
                            atol=1e-10)

    def test_calculate_fine_phonons_realspace_asr_c(self):
        self.data.calculate_fine_phonons(self.qpts, asr='realspace', use_c=True)
        npt.assert_allclose(self.data.freqs.to('hartree').magnitude,
                            self.expctd_freqs_asr.to('hartree').magnitude,
                            atol=1e-10)

    def test_calculate_fine_phonons_realspace_asr_c_2threads(self):
        self.data.calculate_fine_phonons(self.qpts, asr='realspace',
                                         use_c=True, n_threads=2)
        npt.assert_allclose(self.data.freqs.to('hartree').magnitude,
                            self.expctd_freqs_asr.to('hartree').magnitude,
                            atol=1e-10)

    def test_enforce_realspace_acoustic_sum_rule(self):
        expected_fc_mat = np.load(
            os.path.join(self.path, 'lzo_fc_mat_asr.npy'))
        fc_mat = self.data._enforce_realspace_asr()
        npt.assert_allclose(fc_mat, expected_fc_mat, atol=1e-18)


class TestInputReadGraphite(unittest.TestCase):

    def setUp(self):
        # Create trivial function object so attributes can be assigned to it
        expctd_data = type('', (), {})()
        expctd_data.n_ions = 4
        expctd_data.n_branches = 12
        expctd_data.cell_vec = np.array([
            [1.23158700E+00, -2.13317126E+00, 0.00000000E+00],
            [1.23158700E+00, 2.13317126E+00, 0.00000000E+00],
            [0.00000000E+00, 0.00000000E+00, 6.71000000E+00]])*ureg('angstrom')
        expctd_data.ion_r = np.array([[0.000, 0.000, 0.250],
                                      [0.000, 0.000, 0.750],
                                      [0.33333333, 0.66666667, 0.250],
                                      [0.66666667, 0.33333333, 0.750]])
        expctd_data.ion_type = np.array(['C', 'C', 'C', 'C'])
        expctd_data.ion_mass = np.array([12.0107000, 12.0107000,
                                         12.0107000, 12.0107000])*ureg('amu')
        expctd_data.sc_matrix = np.array([[7, 0, 0],
                                          [0, 7, 0],
                                          [0, 0, 2]])
        expctd_data.n_cells_in_sc = 98
        expctd_data.fc_mat_cell0_i0_j0 = (np.array([
            [6.35111387e-01, 2.76471554e-18, 0.00000000e+00],
            [2.05998413e-18, 6.35111387e-01, 0.00000000e+00],
            [0.00000000e+00, 0.00000000e+00, 1.52513691e-01]])
            *ureg('hartree/bohr**2'))
        expctd_data.fc_mat_cell10_i2_j3 = (np.array([
            [-8.16784177e-05, -1.31832252e-05, -1.11904290e-05],
            [-1.31832252e-05, 5.42461675e-05, 3.73913780e-06],
            [-1.11904290e-05, 3.73913780e-06, -2.99013850e-05]])
            *ureg('hartree/bohr**2'))
        expctd_data.cell_origins = np.array(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [5, 0, 0],
             [6, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0], [3, 1, 0], [4, 1, 0],
             [5, 1, 0], [6, 1, 0], [0, 2, 0], [1, 2, 0], [2, 2, 0], [3, 2, 0],
             [4, 2, 0], [5, 2, 0], [6, 2, 0], [0, 3, 0], [1, 3, 0], [2, 3, 0],
             [3, 3, 0], [4, 3, 0], [5, 3, 0], [6, 3, 0], [0, 4, 0], [1, 4, 0],
             [2, 4, 0], [3, 4, 0], [4, 4, 0], [5, 4, 0], [6, 4, 0], [0, 5, 0],
             [1, 5, 0], [2, 5, 0], [3, 5, 0], [4, 5, 0], [5, 5, 0], [6, 5, 0],
             [0, 6, 0], [1, 6, 0], [2, 6, 0], [3, 6, 0], [4, 6, 0], [5, 6, 0],
             [6, 6, 0], [0, 0, 1], [1, 0, 1], [2, 0, 1], [3, 0, 1], [4, 0, 1],
             [5, 0, 1], [6, 0, 1], [0, 1, 1], [1, 1, 1], [2, 1, 1], [3, 1, 1],
             [4, 1, 1], [5, 1, 1], [6, 1, 1], [0, 2, 1], [1, 2, 1], [2, 2, 1],
             [3, 2, 1], [4, 2, 1], [5, 2, 1], [6, 2, 1], [0, 3, 1], [1, 3, 1],
             [2, 3, 1], [3, 3, 1], [4, 3, 1], [5, 3, 1], [6, 3, 1], [0, 4, 1],
             [1, 4, 1], [2, 4, 1], [3, 4, 1], [4, 4, 1], [5, 4, 1], [6, 4, 1],
             [0, 5, 1], [1, 5, 1], [2, 5, 1], [3, 5, 1], [4, 5, 1], [5, 5, 1],
             [6, 5, 1], [0, 6, 1], [1, 6, 1], [2, 6, 1], [3, 6, 1], [4, 6, 1],
             [5, 6, 1], [6, 6, 1]])
        self.expctd_data = expctd_data

        self.seedname = 'graphite'
        self.path = os.path.join('data', 'interpolation', 'graphite')
        data = InterpolationData.from_castep(self.seedname, path=self.path)
        self.data = data

    def test_n_ions_read(self):
        self.assertEqual(self.data.n_ions, self.expctd_data.n_ions)

    def test_n_branches_read(self):
        self.assertEqual(self.data.n_branches, self.expctd_data.n_branches)

    def test_cell_vec_read(self):
        npt.assert_allclose(self.data.cell_vec.to('bohr').magnitude,
                            self.expctd_data.cell_vec.to('bohr').magnitude)

    def test_ion_r_read(self):
        npt.assert_allclose(self.data.ion_r, self.expctd_data.ion_r)

    def test_ion_type_read(self):
        npt.assert_array_equal(self.data.ion_type, self.expctd_data.ion_type)

    def test_ion_mass_read(self):
        npt.assert_allclose(self.data.ion_mass.magnitude,
                            self.expctd_data.ion_mass.magnitude)

    def test_sc_matrix_read(self):
        npt.assert_allclose(self.data.sc_matrix, self.expctd_data.sc_matrix)

    def test_n_cells_in_sc_read(self):
        self.assertEqual(self.data.n_cells_in_sc,
                         self.expctd_data.n_cells_in_sc)

    def test_cell_origins_read(self):
        npt.assert_allclose(self.data.cell_origins,
                            self.expctd_data.cell_origins)

    def test_fc_mat_cell0_i0_j0_read(self):
        npt.assert_allclose(self.data.force_constants[0, 0:3, 0:3].magnitude,
                            self.expctd_data.fc_mat_cell0_i0_j0.magnitude)

    def test_fc_mat_cell10_i2_j3read(self):
        npt.assert_allclose(self.data.force_constants[10, 6:9, 9:12].magnitude,
                            self.expctd_data.fc_mat_cell10_i2_j3.magnitude)

    def test_fc_mat_read(self):
        expected_fc_mat = np.load(os.path.join(
            self.path, 'graphite_fc_mat_no_asr.npy'))*ureg('hartree/bohr**2')
        npt.assert_allclose(self.data.force_constants.magnitude,
                            expected_fc_mat.magnitude)


class TestInterpolatePhononsGraphite(unittest.TestCase):

    def setUp(self):
        seedname = 'graphite'
        self.path = os.path.join('data', 'interpolation', 'graphite')
        self.n_ions = 4
        self.n_cells_in_sc = 98
        self.qpts = np.array([[0.00, 0.00, 0.00],
                              [0.001949, 0.001949, 0.00],
                              [0.50, 0.00, 0.00],
                              [0.25, 0.00, 0.00],
                              [0.00, 0.00, 0.50]])
        self.expctd_freqs_asr = (np.array([
            [-1.4954416353e-09, -1.4920078435e-09, -1.4801148666e-09,
             1.8361944616e-04, 1.8361944616e-04, 4.3310694904e-04,
             4.0094292783e-03, 4.0298656188e-03, 7.0857269505e-03,
             7.0857269505e-03, 7.1044827434e-03, 7.1044827434e-03],
            [5.1653999633e-06, 3.2069627188e-05, 5.1670788680e-05,
             1.8632833975e-04, 1.9060943285e-04, 4.3314392040e-04,
             4.0093920383e-03, 4.0298240624e-03, 7.0856315683e-03,
             7.0859645028e-03, 7.1043776174e-03, 7.1046906727e-03],
            [2.1681099868e-03, 2.2111725605e-03, 2.8193685159e-03,
             2.8197548439e-03, 2.9049709255e-03, 2.9141514127e-03,
             6.0277000188e-03, 6.0316368591e-03, 6.0977873315e-03,
             6.1104625371e-03, 6.3230408313e-03, 6.3262959952e-03],
            [7.6871083387e-04, 8.8171407452e-04, 2.1005257455e-03,
             2.1043480758e-03, 3.5723508265e-03, 3.5925792804e-03,
             3.7679565502e-03, 3.7804526590e-03, 6.6594311463e-03,
             6.6651464071e-03, 7.2007012704e-03, 7.2132353835e-03],
            [1.4065426444e-04, 1.4065426444e-04, 1.4065426444e-04,
             1.4065426444e-04, 3.2305904460e-04, 3.2305904460e-04,
             4.0222682768e-03, 4.0222682768e-03, 7.1591503492e-03,
             7.1591503492e-03, 7.1591503492e-03, 7.1591503492e-03]])
            *ureg('hartree'))
        self.expctd_freqs_no_asr = (np.array([
            [-1.4947813885e-05, -1.4947813885e-05, 1.7871973126e-05,
             1.8361944616e-04, 1.8361944616e-04, 4.3310694904e-04,
             4.0094292783e-03, 4.0298656188e-03, 7.0857269505e-03,
             7.0857269505e-03, 7.1044827434e-03, 7.1044827434e-03],
            [1.8594596548e-05, 2.8377223801e-05, 4.9463914519e-05,
             1.8632830014e-04, 1.9060940094e-04, 4.3314392073e-04,
             4.0093920384e-03, 4.0298240625e-03, 7.0856315682e-03,
             7.0859645027e-03, 7.1043776174e-03, 7.1046906727e-03],
            [2.1681107372e-03, 2.2111728203e-03, 2.8193681109e-03,
             2.8197547098e-03, 2.9049711018e-03, 2.9141516000e-03,
             6.0276999661e-03, 6.0316367945e-03, 6.0977872708e-03,
             6.1104623404e-03, 6.3230407709e-03, 6.3262959354e-03],
            [7.6869781440e-04, 8.8171407111e-04, 2.1005260482e-03,
             2.1043510112e-03, 3.5723508344e-03, 3.5925812023e-03,
             3.7679565852e-03, 3.7804526859e-03, 6.6594311404e-03,
             6.6651464171e-03, 7.2007012512e-03, 7.2132353897e-03],
            [1.4065426444e-04, 1.4065426444e-04, 1.4065426444e-04,
             1.4065426444e-04, 3.2305904460e-04, 3.2305904460e-04,
             4.0222682768e-03, 4.0222682768e-03, 7.1591503492e-03,
             7.1591503492e-03, 7.1591503492e-03, 7.1591503492e-03]])
            *ureg('hartree'))
        data = InterpolationData.from_castep(seedname, path=self.path)
        self.data = data

        self.unique_sc_i = np.loadtxt(os.path.join(
            self.path, 'graphite_unique_sc_i.txt'), dtype=np.int32)
        self.unique_cell_i = np.loadtxt(os.path.join(
            self.path, 'graphite_unique_cell_i.txt'), dtype=np.int32)
        self.unique_sc_offsets = [[] for i in range(3)]
        with open(os.path.join(
                self.path, 'graphite_unique_sc_offsets.txt')) as f:
            for i in range(3):
                self.unique_sc_offsets[i] = [int(x)
                                             for x in f.readline().split()]
        self.unique_cell_origins = [[] for i in range(3)]
        with open(os.path.join(
                self.path, 'graphite_unique_cell_origins.txt')) as f:
            for i in range(3):
                self.unique_cell_origins[i] = [int(x)
                                               for x in f.readline().split()]

    def test_calculate_phases_qpt(self):
        qpt = np.array([0.001949, 0.001949, 0.0])

        sc_phases, cell_phases = self.data._calculate_phases(
            qpt, self.unique_sc_offsets, self.unique_sc_i,
            self.unique_cell_origins, self.unique_cell_i)

        expected_sc_phases = np.loadtxt(os.path.join(
            self.path, 'graphite_sc_phases.txt'), dtype=np.complex128)
        expected_cell_phases = np.loadtxt(os.path.join(
            self.path, 'graphite_cell_phases.txt'), dtype=np.complex128)

        npt.assert_allclose(sc_phases, expected_sc_phases)
        npt.assert_allclose(cell_phases, expected_cell_phases)

    def test_calculate_phases_gamma_pt(self):
        qpt = np.array([0.0, 0.0, 0.0])

        sc_phases, cell_phases = self.data._calculate_phases(
            qpt, self.unique_sc_offsets, self.unique_sc_i,
            self.unique_cell_origins, self.unique_cell_i)

        expected_sc_phases = np.full(
            len(self.unique_sc_i), 1.0 + 0.0*1j, dtype=np.complex128)
        expected_cell_phases = np.full(
            len(self.unique_cell_i), 1.0 + 0.0*1j, dtype=np.complex128)

        npt.assert_equal(sc_phases, expected_sc_phases)
        npt.assert_equal(cell_phases, expected_cell_phases)

    def test_calculate_supercell_images_n_sc_images(self):
        # Supercell image calculation limit - 2 supercells in each direction
        lim = 2
        image_data = np.loadtxt(
            os.path.join(self.path, 'graphite_n_sc_images.txt'))
        i = image_data[:, 0].astype(int)
        j = image_data[:, 1].astype(int)
        n = image_data[:, 2].astype(int)
        expctd_n_images = np.zeros(
            (self.n_ions, self.n_ions*self.n_cells_in_sc))
        expctd_n_images[i, j] = n
        # After refactoring where the shape of n_sc_images was changed from
        # (n_ions, n_cells_in_sc*n_ions) to (n_cells_in_sc, n_ions, n_ions),
        # expctc_n_images must be reshaped to ensure tests still pass
        expctd_n_images = np.transpose(
            np.reshape(expctd_n_images,
                       (self.n_ions, self.n_cells_in_sc, self.n_ions)),
            axes=[1, 0, 2])
        self.data._calculate_supercell_images(lim)
        npt.assert_equal(self.data._n_sc_images, expctd_n_images)

    def test_calculate_supercell_images_sc_image_i(self):
        # Supercell image calculation limit - 2 supercells in each direction
        lim = 2
        image_data = np.loadtxt(
            os.path.join(self.path, 'graphite_sc_image_i.txt'))
        i = image_data[:, 0].astype(int)
        j = image_data[:, 1].astype(int)
        n = image_data[:, 2].astype(int)
        sc_i = image_data[:, 3].astype(int)
        max_n = np.max(n) + 1
        # size = n_ions X n_ions*n_cells_in_sc X max supercell images
        expctd_sc_image_i = np.full(
            (self.n_ions, self.n_ions*self.n_cells_in_sc, max_n), -1)
        expctd_sc_image_i[i, j, n] = sc_i
        # After refactoring where the shape of sc_image_i was changed from
        # (n_ions, n_cells_in_sc*n_ions, (2*lim + 1)**3) to
        # (n_cells_in_sc, n_ions, n_ions, (2*lim + 1)**3),
        # expctc_image_i must be reshaped to ensure tests still pass
        expctd_sc_image_i = np.transpose(
            np.reshape(expctd_sc_image_i, (self.n_ions, self.n_cells_in_sc,
                                           self.n_ions, max_n)),
            axes=[1, 0, 2, 3])
        self.data._calculate_supercell_images(lim)
        npt.assert_equal(self.data._sc_image_i, expctd_sc_image_i)

    def test_calculate_fine_phonons_no_asr(self):
        self.data.calculate_fine_phonons(self.qpts)
        npt.assert_allclose(self.data.freqs.to('hartree').magnitude,
                            self.expctd_freqs_no_asr.to('hartree').magnitude,
                            atol=1e-10)

    def test_calculate_fine_phonons_no_asr_c(self):
        self.data.calculate_fine_phonons(self.qpts, use_c=True)
        npt.assert_allclose(self.data.freqs.to('hartree').magnitude,
                            self.expctd_freqs_no_asr.to('hartree').magnitude,
                            atol=1e-10)

    def test_calculate_fine_phonons_no_asr_c_2threads(self):
        self.data.calculate_fine_phonons(self.qpts, use_c=True, n_threads=2)
        npt.assert_allclose(self.data.freqs.to('hartree').magnitude,
                            self.expctd_freqs_no_asr.to('hartree').magnitude,
                            atol=1e-10)

    def test_calculate_fine_phonons_realspace_asr(self):
        self.data.calculate_fine_phonons(self.qpts, asr='realspace')
        npt.assert_allclose(self.data.freqs.to('hartree').magnitude,
                            self.expctd_freqs_asr.to('hartree').magnitude,
                            atol=1e-10)

    def test_calculate_fine_phonons_realspace_asr_c(self):
        self.data.calculate_fine_phonons(self.qpts, asr='realspace',
                                         use_c=True)
        npt.assert_allclose(self.data.freqs.to('hartree').magnitude,
                            self.expctd_freqs_asr.to('hartree').magnitude,
                            atol=1e-10)

    def test_calculate_fine_phonons_realspace_asr_c_2threads(self):
        self.data.calculate_fine_phonons(self.qpts, asr='realspace',
                                         use_c=True, n_threads=2)
        npt.assert_allclose(self.data.freqs.to('hartree').magnitude,
                            self.expctd_freqs_asr.to('hartree').magnitude,
                            atol=1e-10)

    def test_enforce_realspace_acoustic_sum_rule(self):
        expected_fc_mat = np.load(os.path.join(self.path,
                                               'graphite_fc_mat_asr.npy'))
        fc_mat = self.data._enforce_realspace_asr()
        npt.assert_allclose(fc_mat, expected_fc_mat, atol=1e-18)


class TestInputReadQuartz(unittest.TestCase):

    def setUp(self):
        # Create trivial function object so attributes can be assigned to it
        expctd_data = type('', (), {})()
        expctd_data.n_ions = 9
        expctd_data.n_branches = 27
        expctd_data.cell_vec = np.array([
            [2.42617588, -4.20225989, 0.00000000],
            [2.42617588, 4.20225989, 0.00000000],
            [0.00000000, 0.00000000, 5.35030451]])*ureg('angstrom')
        expctd_data.ion_r = np.array([
            [0.41170811, 0.27568186, 0.27940760],
            [0.72431814, 0.13602626, 0.94607427],
            [0.86397374, 0.58829189, 0.61274094],
            [0.13602626, 0.72431814, 0.05392573],
            [0.58829189, 0.86397374, 0.38725906],
            [0.27568186, 0.41170811, 0.72059240],
            [0.46496011, 0.00000000, 0.16666667],
            [0.00000000, 0.46496011, 0.83333333],
            [0.53503989, 0.53503989, 0.50000000]])
        expctd_data.ion_type = np.array(
            ['O', 'O', 'O', 'O', 'O', 'O', 'Si', 'Si', 'Si'])
        expctd_data.ion_mass = np.array(
            [15.9994000, 15.9994000, 15.9994000,
             15.9994000, 15.9994000, 15.9994000,
             28.0855000, 28.0855000, 28.0855000])*ureg('amu')
        expctd_data.sc_matrix = np.array([[5, 0, 0],
                                          [0, 5, 0],
                                          [0, 0, 4]])
        expctd_data.n_cells_in_sc = 100
        expctd_data.fc_mat_cell0_i0_j0 = np.array([
            [0.22324308, 0.29855096, 0.31240272],
            [0.29855096, 0.43968813, 0.34437270],
            [0.31240272, 0.34437270, 0.33448280]])*ureg('hartree/bohr**2')
        expctd_data.fc_mat_cell4_i2_j7 = (np.array([
            [6.37874988e-05, -2.73116279e-04, -1.73624413e-04],
            [1.99404810e-08, 4.40511248e-04, 2.33322837e-04],
            [-1.85830030e-05, -9.05395392e-05, 2.17526152e-05]])
            *ureg('hartree/bohr**2'))
        expctd_data.cell_origins = np.array(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0],
             [0, 1, 0], [1, 1, 0], [2, 1, 0], [3, 1, 0], [4, 1, 0],
             [0, 2, 0], [1, 2, 0], [2, 2, 0], [3, 2, 0], [4, 2, 0],
             [0, 3, 0], [1, 3, 0], [2, 3, 0], [3, 3, 0], [4, 3, 0],
             [0, 4, 0], [1, 4, 0], [2, 4, 0], [3, 4, 0], [4, 4, 0],
             [0, 0, 1], [1, 0, 1], [2, 0, 1], [3, 0, 1], [4, 0, 1],
             [0, 1, 1], [1, 1, 1], [2, 1, 1], [3, 1, 1], [4, 1, 1],
             [0, 2, 1], [1, 2, 1], [2, 2, 1], [3, 2, 1], [4, 2, 1],
             [0, 3, 1], [1, 3, 1], [2, 3, 1], [3, 3, 1], [4, 3, 1],
             [0, 4, 1], [1, 4, 1], [2, 4, 1], [3, 4, 1], [4, 4, 1],
             [0, 0, 2], [1, 0, 2], [2, 0, 2], [3, 0, 2], [4, 0, 2],
             [0, 1, 2], [1, 1, 2], [2, 1, 2], [3, 1, 2], [4, 1, 2],
             [0, 2, 2], [1, 2, 2], [2, 2, 2], [3, 2, 2], [4, 2, 2],
             [0, 3, 2], [1, 3, 2], [2, 3, 2], [3, 3, 2], [4, 3, 2],
             [0, 4, 2], [1, 4, 2], [2, 4, 2], [3, 4, 2], [4, 4, 2],
             [0, 0, 3], [1, 0, 3], [2, 0, 3], [3, 0, 3], [4, 0, 3],
             [0, 1, 3], [1, 1, 3], [2, 1, 3], [3, 1, 3], [4, 1, 3],
             [0, 2, 3], [1, 2, 3], [2, 2, 3], [3, 2, 3], [4, 2, 3],
             [0, 3, 3], [1, 3, 3], [2, 3, 3], [3, 3, 3], [4, 3, 3],
             [0, 4, 3], [1, 4, 3], [2, 4, 3], [3, 4, 3], [4, 4, 3]])
        expctd_data.born = np.array([
            [[-1.43415314, -0.56480184, -0.51390346],
             [-0.51017266, -1.90914488, -0.58504551],
             [-0.44604816, -0.62037227, -1.73309837]],
            [[-2.25587456, 0.03575158, 0.76361601],
             [0.09038076, -1.08742347, -0.15253070],
             [0.76028223, -0.07610290, -1.73309837]],
            [[-1.32495121, 0.44710649, -0.24971254],
             [0.50174463, -2.01837869, 0.73757621],
             [-0.31423365, 0.69647517, -1.73309837]],
            [[-2.25587456, -0.03575158, -0.76361601],
             [-0.09038076, -1.08742347, -0.15253070],
             [-0.76028223, -0.07610290, -1.73309837]],
            [[-1.32495121, -0.44710649, 0.24971254],
             [-0.50174463, -2.01837869, 0.73757621],
             [0.31423365, 0.69647517, -1.73309837]],
            [[-1.43415314, 0.56480184, 0.51390346],
             [0.51017266, -1.90914488, -0.58504551],
             [0.44604816, -0.62037227, -1.73309837]],
            [[3.50466855, 0.27826283, 0.23601960],
             [0.27826283, 3.18335830, 0.13626485],
             [-0.27183190, -0.15694222, 3.46752205]],
            [[3.50466855, -0.27826283, -0.23601960],
             [-0.27826283, 3.18335830, 0.13626485],
             [0.27183190, -0.15694222, 3.46752205]],
            [[3.02266843, 0.00000000, 0.00000000],
             [0.00000000, 3.66532367, -0.27253140],
             [0.00000000, 0.31388444, 3.46752205]]])*ureg('e')
        expctd_data.dielectric = np.array([
            [2.49104301, 0.00000000, 0.00000000],
            [0.00000000, 2.49104301, 0.00000000],
            [0.00000000, 0.00000000, 2.52289805]])
        self.expctd_data = expctd_data

        self.seedname = 'quartz'
        self.path = os.path.join('data', 'interpolation', 'quartz')
        data = InterpolationData.from_castep(self.seedname, path=self.path)
        self.data = data

    def test_n_ions_read(self):
        self.assertEqual(self.data.n_ions, self.expctd_data.n_ions)

    def test_n_branches_read(self):
        self.assertEqual(self.data.n_branches, self.expctd_data.n_branches)

    def test_cell_vec_read(self):
        npt.assert_allclose(self.data.cell_vec.to('bohr').magnitude,
                            self.expctd_data.cell_vec.to('bohr').magnitude)

    def test_ion_r_read(self):
        npt.assert_allclose(self.data.ion_r, self.expctd_data.ion_r)

    def test_ion_type_read(self):
        npt.assert_array_equal(self.data.ion_type, self.expctd_data.ion_type)

    def test_ion_mass_read(self):
        npt.assert_allclose(self.data.ion_mass.magnitude,
                            self.expctd_data.ion_mass.magnitude)

    def test_sc_matrix_read(self):
        npt.assert_allclose(self.data.sc_matrix, self.expctd_data.sc_matrix)

    def test_n_cells_in_sc_read(self):
        self.assertEqual(self.data.n_cells_in_sc,
                         self.expctd_data.n_cells_in_sc)

    def test_cell_origins_read(self):
        npt.assert_allclose(self.data.cell_origins,
                            self.expctd_data.cell_origins)

    def test_born_read(self):
        npt.assert_allclose(self.data.born.magnitude,
                            self.expctd_data.born.magnitude)

    def test_dielctric_read(self):
        npt.assert_allclose(self.data.dielectric,
                            self.expctd_data.dielectric)

    def test_fc_mat_cell0_i0_j0_read(self):
        npt.assert_allclose(self.data.force_constants[0, 0:3, 0:3].magnitude,
                            self.expctd_data.fc_mat_cell0_i0_j0.magnitude)

    def test_fc_mat_cell4_i2_j7read(self):
        npt.assert_allclose(self.data.force_constants[4, 6:9, 21:24].magnitude,
                            self.expctd_data.fc_mat_cell4_i2_j7.magnitude)

    def test_fc_mat_read(self):
        expected_fc_mat = np.load(os.path.join(
            self.path, 'quartz_fc_mat_no_asr.npy'))*ureg('hartree/bohr**2')
        npt.assert_allclose(self.data.force_constants.magnitude,
                            expected_fc_mat.magnitude)


class TestInterpolatePhononsQuartz(unittest.TestCase):

    def setUp(self):
        seedname = 'quartz'
        self.path = os.path.join('data', 'interpolation', 'quartz')
        self.qpts = np.array([[0.00, 0.00, 0.00],
                              [0.00, 0.00, 0.50],
                              [-0.25, 0.50, 0.50],
                              [-0.151515, 0.575758, 0.5]])
        data = InterpolationData.from_castep(seedname, path=self.path)
        self.data = data
        self.expctd_freqs_no_asr = np.array([
            [-0.00009745, -0.00005474, -0.00005474, 0.00058293, 0.00058293,
             0.00101620, 0.00117558, 0.00117559, 0.00155134, 0.00157713,
             0.00172393, 0.00172393, 0.00200563, 0.00200563, 0.00209131,
             0.00222477, 0.00316467, 0.00316467, 0.00352001, 0.00363297,
             0.00363297, 0.00487607, 0.00487607, 0.00492351, 0.00495076,
             0.00525430, 0.00525431],
            [0.00021293, 0.00056589, 0.00056590, 0.00075771, 0.00083485,
             0.00083485, 0.00092258, 0.00119698, 0.00119698, 0.00175516,
             0.00182881, 0.00191025, 0.00191027, 0.00205705, 0.00205705,
             0.00259491, 0.00259491, 0.00355700, 0.00358489, 0.00362570,
             0.00362570, 0.00484908, 0.00492808, 0.00492808, 0.00505040,
             0.00553676, 0.00553677],
            [0.00029664, 0.00044691, 0.00045286, 0.00083218, 0.00085090,
             0.00107911, 0.00122936, 0.00131583, 0.00142863, 0.00154578,
             0.00166267, 0.00179816, 0.00189441, 0.00191534, 0.00205378,
             0.00246936, 0.00278196, 0.00315079, 0.00370003, 0.00376130,
             0.00380875, 0.00488787, 0.00490533, 0.00490608, 0.00526644,
             0.00531051, 0.00540936],
            [0.00028396, 0.00036024, 0.00049599, 0.00082340, 0.00084375,
             0.00109203, 0.00130479, 0.00134347, 0.00142754, 0.00155335,
             0.00159243, 0.00179022, 0.00188140, 0.00193608, 0.00205639,
             0.00243849, 0.00280079, 0.00315569, 0.00370591, 0.00370973,
             0.00385188, 0.00489424, 0.00489891, 0.00490631, 0.00527567,
             0.00533033, 0.00537392]])*ureg('hartree')
        self.expctd_freqs_asr = np.array([
            [0.00000000, 0.00000000, 0.00000000, 0.00058293, 0.00058293,
             0.00101620, 0.00117558, 0.00117559, 0.00155134, 0.00157718,
             0.00172394, 0.00172394, 0.00200565, 0.00200565, 0.00209131,
             0.00222481, 0.00316467, 0.00316467, 0.00352002, 0.00363298,
             0.00363298, 0.00487608, 0.00487608, 0.00492353, 0.00495076,
             0.00525430, 0.00525431],
            [0.00021598, 0.00056611, 0.00056686, 0.00075777, 0.00083532,
             0.00083554, 0.00092328, 0.00119699, 0.00119903, 0.00175523,
             0.00182882, 0.00191025, 0.00191045, 0.00205709, 0.00205738,
             0.00259496, 0.00259505, 0.00355706, 0.00358489, 0.00362570,
             0.00362572, 0.00484913, 0.00492809, 0.00492815, 0.00505041,
             0.00553678, 0.00553679],
            [0.00029928, 0.00044784, 0.00045479, 0.00083284, 0.00085117,
             0.00107950, 0.00122947, 0.00131602, 0.00142866, 0.00154594,
             0.00166280, 0.00179836, 0.00189458, 0.00191540, 0.00205403,
             0.00246941, 0.00278199, 0.00315087, 0.00370009, 0.00376137,
             0.00380880, 0.00488793, 0.00490540, 0.00490617, 0.00526645,
             0.00531057, 0.00540940],
            [0.00028671, 0.00036112, 0.00049916, 0.00082386, 0.00084410,
             0.00109225, 0.00130495, 0.00134356, 0.00142776, 0.00155337,
             0.00159266, 0.00179031, 0.00188151, 0.00193612, 0.00205664,
             0.00243853, 0.00280084, 0.00315578, 0.00370595, 0.00370979,
             0.00385195, 0.00489429, 0.00489894, 0.00490641, 0.00527571,
             0.00533034, 0.00537398]])*ureg('hartree')
        self.split_qpts = np.array([[0.00, 0.00, 0.00],
                         [0.00, 0.00, 0.50],
                         [0.00, 0.00, 0.00],
                         [-0.25, 0.50, 0.50],
                         [-0.151515, 0.575758, 0.5]])
        self.expctd_freqs_asr_splitting = np.array([
            [0.00000000, 0.00000000, 0.00000134, 0.00058293, 0.00058293,
             0.00101620, 0.00117558, 0.00117559, 0.00155134, 0.00168918,
             0.00172394, 0.00172394, 0.00200565, 0.00200565, 0.00209131,
             0.00245470, 0.00316467, 0.00316467, 0.00360268, 0.00363298,
             0.00363298, 0.00487608, 0.00487608, 0.00495076, 0.00525430,
             0.00525431, 0.00565964],
            [0.00021598, 0.00056611, 0.00056686, 0.00075777, 0.00083532,
             0.00083554, 0.00092328, 0.00119699, 0.00119903, 0.00175523,
             0.00182882, 0.00191025, 0.00191045, 0.00205709, 0.00205738,
             0.00259496, 0.00259505, 0.00355706, 0.00358489, 0.00362570,
             0.00362572, 0.00484913, 0.00492809, 0.00492815, 0.00505041,
             0.00553678, 0.00553679],
            [0.00000000, 0.00000000, 0.00000134, 0.00058293, 0.00058293,
             0.00101620, 0.00117558, 0.00117559, 0.00155134, 0.00168918,
             0.00172394, 0.00172394, 0.00200565, 0.00200565, 0.00209131,
             0.00245470, 0.00316467, 0.00316467, 0.00360268, 0.00363298,
             0.00363298, 0.00487608, 0.00487608, 0.00495076, 0.00525430,
             0.00525431, 0.00565964],
            [0.00029928, 0.00044784, 0.00045479, 0.00083284, 0.00085117,
             0.00107950, 0.00122947, 0.00131602, 0.00142866, 0.00154594,
             0.00166280, 0.00179836, 0.00189458, 0.00191540, 0.00205403,
             0.00246941, 0.00278199, 0.00315087, 0.00370009, 0.00376137,
             0.00380880, 0.00488793, 0.00490540, 0.00490617, 0.00526645,
             0.00531057, 0.00540940],
            [0.00028671, 0.00036112, 0.00049916, 0.00082386, 0.00084410,
             0.00109225, 0.00130495, 0.00134356, 0.00142776, 0.00155337,
             0.00159266, 0.00179031, 0.00188151, 0.00193612, 0.00205664,
             0.00243853, 0.00280084, 0.00315578, 0.00370595, 0.00370979,
             0.00385195, 0.00489429, 0.00489894, 0.00490641, 0.00527571,
             0.00533034, 0.00537398]])*ureg('hartree')
        self.expctd_split_freqs = np.array([
            [0.00000000, 0.00000000, 0.00000093, 0.00058293, 0.00058305,
             0.00101620, 0.00117558, 0.00117967, 0.00155134, 0.00161358,
             0.00172394, 0.00176372, 0.00200565, 0.00209027, 0.00209131,
             0.00238641, 0.00316467, 0.00317281, 0.00354729, 0.00363298,
             0.00367378, 0.00487608, 0.00490124, 0.00495076, 0.00524661,
             0.00525431, 0.00563557]])*ureg('hartree')
        self.expctd_split_i = np.array([2])

    def test_calculate_fine_phonons_dipole_no_asr(self):
        self.data.calculate_fine_phonons(
            self.qpts, dipole=True, splitting=False)
        npt.assert_allclose(
            self.data.freqs.to('hartree').magnitude,
            self.expctd_freqs_no_asr.to('hartree').magnitude,
            atol=1e-8)

    def test_calculate_fine_phonons_dipole_no_asr_c(self):
        self.data.calculate_fine_phonons(
            self.qpts, dipole=True, splitting=False, use_c=True)
        npt.assert_allclose(
            self.data.freqs.to('hartree').magnitude,
            self.expctd_freqs_no_asr.to('hartree').magnitude,
            atol=1e-8)

    def test_calculate_fine_phonons_dipole_no_asr_c_2threads(self):
        self.data.calculate_fine_phonons(
            self.qpts, dipole=True, splitting=False, use_c=True, n_threads=2)
        npt.assert_allclose(
            self.data.freqs.to('hartree').magnitude,
            self.expctd_freqs_no_asr.to('hartree').magnitude,
            atol=1e-8)

    def test_calculate_fine_phonons_dipole_recip_asr(self):
        self.data.calculate_fine_phonons(
            self.qpts, asr='reciprocal', dipole=True, splitting=False)
        npt.assert_allclose(
            self.data.freqs.to('hartree').magnitude,
            self.expctd_freqs_asr.to('hartree').magnitude,
            atol=1e-8)

    def test_calculate_fine_phonons_dipole_recip_asr_c(self):
        self.data.calculate_fine_phonons(
            self.qpts, asr='reciprocal', dipole=True, splitting=False,
            use_c=True)
        npt.assert_allclose(
            self.data.freqs.to('hartree').magnitude,
            self.expctd_freqs_asr.to('hartree').magnitude,
            atol=1e-8)

    def test_calculate_fine_phonons_dipole_recip_asr_c_2threads(self):
        self.data.calculate_fine_phonons(
            self.qpts, asr='reciprocal', dipole=True, splitting=False,
            use_c=True, n_threads=2)
        npt.assert_allclose(
            self.data.freqs.to('hartree').magnitude,
            self.expctd_freqs_asr.to('hartree').magnitude,
            atol=1e-8)

    def test_calculate_fine_phonons_dipole_recip_asr_split(self):
        self.data.calculate_fine_phonons(
            self.split_qpts, asr='reciprocal', dipole=True, splitting=True)
        npt.assert_array_equal(self.data.split_i, self.expctd_split_i)
        npt.assert_allclose(
            self.data.freqs.to('hartree').magnitude,
            self.expctd_freqs_asr_splitting.to('hartree').magnitude,
            atol=1e-8)
        npt.assert_allclose(
            self.data.split_freqs.to('hartree').magnitude,
            self.expctd_split_freqs.to('hartree').magnitude,
            atol=1e-8)

    def test_calculate_fine_phonons_dipole_recip_asr_split_c(self):
        self.data.calculate_fine_phonons(
            self.split_qpts, asr='reciprocal', dipole=True, splitting=True,
            use_c=True)
        npt.assert_array_equal(self.data.split_i, self.expctd_split_i)
        npt.assert_allclose(
            self.data.freqs.to('hartree').magnitude,
            self.expctd_freqs_asr_splitting.to('hartree').magnitude,
            atol=1e-8)
        npt.assert_allclose(
            self.data.split_freqs.to('hartree').magnitude,
            self.expctd_split_freqs.to('hartree').magnitude,
            atol=1e-8)

    def test_calculate_fine_phonons_dipole_recip_asr_split_c_2threads(self):
        self.data.calculate_fine_phonons(
            self.split_qpts, asr='reciprocal', dipole=True, splitting=True,
            use_c=True, n_threads=2)
        npt.assert_array_equal(self.data.split_i, self.expctd_split_i)
        npt.assert_allclose(
            self.data.freqs.to('hartree').magnitude,
            self.expctd_freqs_asr_splitting.to('hartree').magnitude,
            atol=1e-8)
        npt.assert_allclose(
            self.data.split_freqs.to('hartree').magnitude,
            self.expctd_split_freqs.to('hartree').magnitude,
            atol=1e-8)
