import unittest
import os
import numpy.testing as npt
import numpy as np
from euphonic import ureg
from euphonic.data.interpolation import InterpolationData


class TestInputReadNaCL(unittest.TestCase):

    def setUp(self):
        # Create trivial function object so attributes can be assigned to it
        expctd_data = type('', (), {})()
        expctd_data.n_ions = 8
        expctd_data.n_branches = 24

        #TODO cell vec
        expctd_data.cell_vec = np.array([
            [2.42617588, -4.20225989, 0.00000000],
            [2.42617588, 4.20225989, 0.00000000],
            [0.00000000, 0.00000000, 5.35030451]])*ureg('angstrom')

        #TODO ion r
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
            ['Na', 'Na', 'Na', 'Na', 'Cl', 'Cl', 'Cl', 'Cl'])

        #TODO ion mass
        expctd_data.ion_mass = np.array(
            [15.9994000, 15.9994000, 15.9994000,
             15.9994000, 15.9994000, 15.9994000,
             28.0855000, 28.0855000, 28.0855000])*ureg('amu')

        #TODO sc matrix
        expctd_data.sc_matrix = np.array([[5, 0, 0],
                                          [0, 5, 0],
                                          [0, 0, 4]])
        #TODO n cells in sc
        expctd_data.n_cells_in_sc = 100

        #TODO fc mat cell0 i0 j0
        expctd_data.fc_mat_cell0_i0_j0 = np.array([
            [0.22324308, 0.29855096, 0.31240272],
            [0.29855096, 0.43968813, 0.34437270],
            [0.31240272, 0.34437270, 0.33448280]])*ureg('hartree/bohr**2')

        #TODO fc mat cell 10 i2 j3
        expctd_data.fc_mat_cell10_i2_j3 = (np.array([
            [6.37874988e-05, 1.99404810e-08, -1.85830030e-05],
            [-2.73116279e-04, 4.40511248e-04, -9.05395392e-05],
            [-1.73624413e-04, 2.33322837e-04, 2.17526152e-05]])
            *ureg('hartree/bohr**2'))

        #TODO cell origins
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

        #TODO born effective charge
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

        #TODO dielectric constant
        expctd_data.dielectric = np.array([
            [2.49104301, 0.00000000, 0.00000000],
            [0.00000000, 2.49104301, 0.00000000],
            [0.00000000, 0.00000000, 2.52289805]])

        self.expctd_data = expctd_data

        self.seedname = 'NaCl'
        self.path = os.path.join('test', 'phonopy_data')
        data = InterpolationData(self.seedname, path=self.path, model='phonopy')
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
        npt.assert_allclose(self.data.force_constants[4, 21:24, 6:9].magnitude,
                            self.expctd_data.fc_mat_cell10_i2_j3.magnitude)

    def test_fc_mat_read(self):
        expected_fc_mat = np.load(os.path.join(
            self.path, 'nacl_fc_mat.npy'))*ureg('hartree/bohr**2')
        # After refactoring where the shape of the force constants matrix was
        # changed from (3*n_cells_in_sc*n_ions, 3*n_ions) to
        # (n_cells_in_sc, 3*n_ions, 3*n_ions), expected_fc_mat must be
        # reshaped to ensure tests still pass

        #TODO reshape
        expected_fc_mat = np.reshape(
            expected_fc_mat,
            (self.expctd_data.n_cells_in_sc,
             3*self.expctd_data.n_ions,
             3*self.expctd_data.n_ions))
        npt.assert_allclose(self.data.force_constants.magnitude,
                            expected_fc_mat.magnitude)


class TestInterpolatePhononsQuartz(unittest.TestCase):

    def setUp(self):
        seedname = 'quartz'
        self.path = os.path.join('test', 'phonopy_data')
        self.qpts = np.array([[0.00, 0.00, 0.00],
                              [0.00, 0.00, 0.50],
                              [-0.25, 0.50, 0.50],
                              [-0.151515, 0.575758, 0.5]])
        data = InterpolationData(seedname, path=self.path, model='phonopy')
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

    def test_calculate_fine_phonons_dipole_no_asr_2procs(self):
        self.data.calculate_fine_phonons(
            self.qpts, dipole=True, splitting=False, nprocs=2)
        npt.assert_allclose(
            self.data.freqs.to('hartree').magnitude,
            self.expctd_freqs_no_asr.to('hartree').magnitude,
            atol=1e-8)

    def test_calculate_fine_phonons_dipole_no_asr_2procs_qchunk(self):
        self.data.calculate_fine_phonons(
            self.qpts, dipole=True, splitting=False, nprocs=2, _qchunk=2)
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

    def test_calculate_fine_phonons_dipole_recip_asr_2procs(self):
        self.data.calculate_fine_phonons(
            self.qpts, asr='reciprocal', dipole=True, splitting=False,
            nprocs=2)
        npt.assert_allclose(
            self.data.freqs.to('hartree').magnitude,
            self.expctd_freqs_asr.to('hartree').magnitude,
            atol=1e-8)

    def test_calculate_fine_phonons_dipole_recip_asr_2procs_qchunk(self):
        self.data.calculate_fine_phonons(
            self.qpts, asr='reciprocal', dipole=True, splitting=False,
            nprocs=2, _qchunk=2)
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

    def test_calculate_fine_phonons_dipole_recip_asr_split_2procs(self):
        self.data.calculate_fine_phonons(
            self.split_qpts, asr='reciprocal', dipole=True, splitting=True,
            nprocs=2)
        npt.assert_array_equal(self.data.split_i, self.expctd_split_i)
        npt.assert_allclose(
            self.data.freqs.to('hartree').magnitude,
            self.expctd_freqs_asr_splitting.to('hartree').magnitude,
            atol=1e-8)
        npt.assert_allclose(
            self.data.split_freqs.to('hartree').magnitude,
            self.expctd_split_freqs.to('hartree').magnitude,
            atol=1e-8)

    def test_calculate_fine_phonons_dipole_recip_asr_split_2procs_qchunk(self):
        self.data.calculate_fine_phonons(
            self.split_qpts, asr='reciprocal', dipole=True, splitting=True,
            nprocs=2, _qchunk=2)
        npt.assert_array_equal(self.data.split_i, self.expctd_split_i)
        npt.assert_allclose(
            self.data.freqs.to('hartree').magnitude,
            self.expctd_freqs_asr_splitting.to('hartree').magnitude,
            atol=1e-8)
        npt.assert_allclose(
            self.data.split_freqs.to('hartree').magnitude,
            self.expctd_split_freqs.to('hartree').magnitude,
            atol=1e-8)
