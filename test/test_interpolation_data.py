import unittest
import numpy.testing as npt
import numpy as np
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
        self.path = 'test/data'
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
        npt.assert_allclose(self.data.force_constants[0, 0:3, 0:3],
                            self.expctd_data.fc_mat_cell0_i0_j0)

    def test_fc_mat_cell3_i5_j10_read(self):
        npt.assert_allclose(self.data.force_constants[3, 15:18, 30:33],
                            self.expctd_data.fc_mat_cell3_i5_j10)

class TestInterpolatePhonons(unittest.TestCase):

    def setUp(self):
        seedname = 'La2Zr2O7'
        path = 'test/data'
        data = InterpolationData(seedname, path)
        self.data = data

    def test_calculate_supercell_image_r(self):
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

    def test_calculate_supercell_images_n_sc_images(self):
        # Supercell image calculation limit - 2 supercells in each direction
        lim = 2
        image_data = np.loadtxt('test/data/supercell_images/n_sc_images.txt')
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
        image_data = np.loadtxt('test/data/supercell_images/sc_image_i.txt')
        i = image_data[:, 0].astype(int)
        j = image_data[:, 1].astype(int)
        n = image_data[:, 2].astype(int)
        sc_i = image_data[:, 3].astype(int)
        expctd_sc_image_i = np.zeros((22, 88, (2*lim + 1)**3)) # size = n_ions X n_ions*n_cells_in_sc X max supercell images
        expctd_sc_image_i[i, j, n] = sc_i
        self.data._calculate_supercell_images(lim)
        npt.assert_equal(self.data.sc_image_i, expctd_sc_image_i)