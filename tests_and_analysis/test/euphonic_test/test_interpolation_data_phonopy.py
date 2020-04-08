import unittest
import os
import numpy.testing as npt
import numpy as np
from euphonic import ureg
from euphonic.data.interpolation import InterpolationData
from ..utils import get_data_path

class TestReadInterpolationNaCl(unittest.TestCase):

    def setUp(self):

        self.path = os.path.join(
            get_data_path(), 'phonopy_data', 'NaCl', 'interpolation')

        # Create trivial function object so attributes can be assigned to it
        expctd_data = type('', (), {})()
        expctd_data.n_ions = 8
        expctd_data.n_branches = 24
        expctd_data.n_qpts = 53
        expctd_data.cell_vec = np.array(
           [[5.69030148, 0.        , 0.        ],
            [0.        , 5.69030148, 0.        ],
            [0.        , 0.        , 5.69030148]])*ureg('angstrom')
        expctd_data.ion_r = np.array(
            [[0. , 0. , 0. ],
             [0. , 0.5, 0.5],
             [0.5, 0. , 0.5],
             [0.5, 0.5, 0. ],
             [0.5, 0.5, 0.5],
             [0.5, 0. , 0. ],
             [0. , 0.5, 0. ],
             [0. , 0. , 0.5]])
        expctd_data.ion_type = np.array(
            ['Na', 'Na', 'Na', 'Na', 'Cl', 'Cl', 'Cl', 'Cl'])
        expctd_data.ion_mass = np.array(
                [22.989769, 22.989769, 22.989769, 22.989769,
                 35.453, 35.453, 35.453   , 35.453   ])*ureg('amu')
        expctd_data.sc_matrix = np.array(
            [[2, 0, 0],
             [0, 2, 0],
             [0, 0, 2]])
        expctd_data.n_cells_in_sc = 8
        expctd_data.cell_origins = np.array(
            [[0., 0., 0.],
             [1., 0., 0.],
             [0., 1., 0.],
             [1., 1., 0.],
             [0., 0., 1.],
             [1., 0., 1.],
             [0., 1., 1.],
             [1., 1., 1.]])
        expctd_data.born = np.array(
           [[[ 1.086875,  0.      ,  0.      ],
             [ 0.      ,  1.086875,  0.      ],
             [ 0.      ,  0.      ,  1.086875]],

            [[ 1.086875,  0.      ,  0.      ],
             [ 0.      ,  1.086875,  0.      ],
             [ 0.      ,  0.      ,  1.086875]],

            [[ 1.086875,  0.      ,  0.      ],
             [ 0.      ,  1.086875,  0.      ],
             [ 0.      ,  0.      ,  1.086875]],

            [[ 1.086875,  0.      ,  0.      ],
             [ 0.      ,  1.086875,  0.      ],
             [ 0.      ,  0.      ,  1.086875]],

            [[-1.086875,  0.      ,  0.      ],
             [ 0.      , -1.086875,  0.      ],
             [ 0.      ,  0.      , -1.086875]],

            [[-1.086875,  0.      ,  0.      ],
             [ 0.      , -1.086875,  0.      ],
             [ 0.      ,  0.      , -1.086875]],

            [[-1.086875,  0.      ,  0.      ],
             [ 0.      , -1.086875,  0.      ],
             [ 0.      ,  0.      , -1.086875]],

            [[-1.086875,  0.      ,  0.      ],
             [ 0.      , -1.086875,  0.      ],
             [ 0.      ,  0.      , -1.086875]]])*ureg('e')
        expctd_data.dielectric = np.array(
            [[2.43533967, 0.        , 0.        ],
             [0.        , 2.43533967, 0.        ],
             [0.        , 0.        , 2.43533967]])
        expctd_data.force_constants = np.load(
            self.path +'/force_constants_hdf5.npy')*ureg('hartree/bohr**2')
        self.expctd_data = expctd_data
        self.summary_nofc = 'phonopy_nofc.yaml'
        self.data = InterpolationData.from_phonopy(path=self.path)

        # Maximum difference in force constants read from plain text and hdf5
        # files
        self.fc_tol = 7e-18

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
                            self.expctd_data.ion_mass.magnitude, atol=1e-10)

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

    def test_fc_mat_read(self):
        npt.assert_allclose(self.data.force_constants.magnitude,
                            self.expctd_data.force_constants.magnitude,
                            atol = self.fc_tol)

    def test_fc_mat_read_fullfc(self):
        data_fullfc = InterpolationData.from_phonopy(
            path=self.path, summary_name=self.summary_nofc,
            fc_name='full_force_constants.hdf5')
        npt.assert_allclose(data_fullfc.force_constants.magnitude,
                            self.expctd_data.force_constants.magnitude)

    def test_fc_mat_read_FC(self):
        data_FC = InterpolationData.from_phonopy(
            path=self.path, summary_name=self.summary_nofc,
            fc_name='FORCE_CONSTANTS')
        npt.assert_allclose(data_FC.force_constants.magnitude,
                            self.expctd_data.force_constants.magnitude,
                            atol=self.fc_tol)

    def test_fc_mat_read_FULLFC(self):
        data_FULLFC = InterpolationData.from_phonopy(
            path=self.path, summary_name=self.summary_nofc,
            fc_name='FULL_FORCE_CONSTANTS')
        npt.assert_allclose(data_FULLFC.force_constants.magnitude,
                            self.expctd_data.force_constants.magnitude,
                            atol=self.fc_tol)

    def test_born_read_BORN(self):
        data_born = InterpolationData.from_phonopy(
            path=self.path, summary_name='phonopy_nofc_noborn.yaml',
            born_name='BORN')
        npt.assert_allclose(data_born.born.magnitude,
                            self.expctd_data.born.magnitude)


class TestInterpolatePhononsNaCl(unittest.TestCase):

    def setUp(self):
        self.path = os.path.join(
            get_data_path(), 'phonopy_data', 'NaCl', 'interpolation')

        self.qpts = np.array(
                        [[0.00, 0.00, 0.00],
                        [0.00, 0.00, 0.50],
                        [-0.25, 0.50, 0.50],
                        [-0.151515, 0.575758, 0.5]])


        expctd_data = type('', (), {})()
        expctd_data.split_qpts = np.empty([])
        expctd_data.split_freqs = np.empty([])*ureg('hartree')
        expctd_data.split_i = np.empty([])
        expctd_data.freqs_asr = np.empty([])
        expctd_data.freqs_asr_splitting = np.empty([])

        expctd_data.freqs_dipole_no_asr = np.array([
            [1.68554227,  1.68554227,  1.68554227, 13.03834226, 13.03834226,
             13.03834226, 13.70911614, 13.70911614, 13.70911614, 13.70911614,
             13.70911614, 13.70911614, 14.77619334, 14.77619334, 14.77619334,
             14.77619334, 14.77619334, 14.77619334, 15.29016584, 15.29016584,
             15.29016584, 26.8034935 , 26.8034935 , 26.8034935],
            [10.43378439, 10.43378439, 10.43378439, 10.43378439, 11.22623974,
             11.22623974, 13.41709858, 13.41709858, 13.41709858, 13.41709858,
             14.52964205, 14.52964205, 14.52964205, 14.52964205, 15.35508239,
             15.35508239, 16.77483373, 16.77483373, 21.52053536, 21.52053536,
             21.52053536, 21.52053536, 31.90031516, 31.90031516],
            [ 9.70009213,  9.70009213,  9.70009213,  9.70009213, 12.23105123,
             12.23105123, 12.23105123, 12.23105123, 12.89476859, 12.89476859,
             12.89476859, 12.89476859, 13.13693891, 13.13693891, 13.13693891,
             13.13693891, 22.06651219, 22.06651219, 22.06651219, 22.06651219,
             29.1978153 , 29.1978153 , 29.1978153 , 29.1978153 ],
            [ 8.70234882,  8.70234882, 10.00872202, 10.00872202, 12.85563966,
             12.85563966, 12.91640758, 12.91640758, 13.51524747, 13.51524747,
             13.69887901, 13.69887901, 13.76367323, 13.76367323, 14.54189731,
             14.54189731, 19.40187956, 19.40187956, 21.05182267, 21.05182267,
             28.28975768, 28.28975768, 29.77808371, 29.77808371]])*ureg('meV')
        expctd_data.freqs_dipole_realsp_asr = np.array([
            [-1.60758061, -1.60718395, -1.60708352, 13.25894361, 13.25895397,
             13.25895862, 13.7067571 , 13.7067572 , 13.70675726, 13.70675728,
             13.70675731, 13.70675743, 14.77838148, 14.77838151, 14.77838152,
             14.77838152, 14.77838152, 14.77838155, 15.28987378, 15.2898738,
              15.2898738 , 26.80366009, 26.8036601 , 26.8036601 ],
            [10.42911169, 10.42911169, 10.42918151, 10.42918158, 11.25702209,
             11.25705428, 13.42071458, 13.42071459, 13.42073099, 13.42073099,
             14.5376376 , 14.5376376 , 14.5376376 , 14.5376376 , 15.35508239,
             15.35508239, 16.77483373, 16.77483373, 21.51513497, 21.51513497,
             21.51513497, 21.51513497, 31.88946119, 31.88946565],
            [ 9.69770891,  9.69770893,  9.69770898,  9.69770898, 12.23182976,
             12.23182984, 12.23182984, 12.23182986, 12.89656098, 12.89656098,
             12.89656098, 12.89656098, 13.13865655, 13.13865656, 13.13865656,
             13.13865658, 22.06696896, 22.066969  , 22.066969  , 22.06696901,
             29.19637097, 29.19637097, 29.19637098, 29.19637098],
            [ 8.70045642,  8.70045874, 10.00553973, 10.00554104, 12.85834764,
             12.85834786, 12.91733905, 12.91734389, 13.51723245, 13.51723438,
             13.70221457, 13.70221553, 13.76553799, 13.76553943, 14.54322351,
             14.54322506, 19.40371223, 19.40371496, 21.0515387 , 21.05154028,
             28.28756337, 28.28756345, 29.77527494, 29.77527501]])*ureg('meV')

        expctd_data.freqs_dipole_recip_asr = np.array([
            [-0.00001183, -0.00000838,  0.00000036, 13.0330645, 13.0330645,
             13.0330645 , 13.70911614, 13.70911614, 13.70911614, 13.70911614,
             13.70911614, 13.70911614, 14.77619334, 14.77619334, 14.77619334,
             14.77619334, 14.77619334, 14.77619334, 15.29016584, 15.29016584,
             15.29016584, 26.8034935 , 26.8034935 , 26.8034935 ],
            [10.29421474, 10.29421474, 10.43378439, 10.43378439, 11.09817998,
             11.22623974, 13.41390631, 13.41390631, 13.41709858, 13.41709858,
             14.52964205, 14.52964205, 14.52964205, 14.52964205, 15.35508239,
             15.35508239, 16.77483373, 16.77483373, 21.52053536, 21.52053536,
             21.52053536, 21.52053536, 31.89843751, 31.90031516],
            [ 9.63825967,  9.63825967,  9.70009213,  9.70009213, 12.17415977,
             12.20855918, 12.20855918, 12.23105123, 12.88310822, 12.88310822,
             12.89476859, 12.89476859, 13.12198284, 13.13553123, 13.13553123,
             13.13693891, 22.04350987, 22.05424945, 22.05424945, 22.06651219,
             29.19174209, 29.19174209, 29.19468408, 29.1978153 ],
            [ 8.65321026,  8.65334846,  9.98387815,  9.98421216, 12.84875298,
             12.85137816, 12.86815343, 12.90260046, 13.51181836, 13.51450714,
             13.69142224, 13.69831554, 13.76153287, 13.76224884, 14.51838141,
             14.5334654 , 19.37183479, 19.39110497, 21.03444816, 21.04652258,
             28.28732511, 28.28804436, 29.77604856, 29.77629323]])*ureg('meV')

        self.expctd_data = expctd_data
        self.data = InterpolationData.from_phonopy(path=self.path)

    def test_calculate_fine_phonons_dipole_no_asr(self):
        self.data.calculate_fine_phonons(
            self.qpts, dipole=True, splitting=False, asr=None)
        npt.assert_allclose(
            self.data.freqs.to('hartree').magnitude,
            self.expctd_data.freqs_dipole_no_asr.to('hartree').magnitude,
            atol=1e-8)

    def test_calculate_fine_phonons_dipole_no_asr_c(self):
        self.data.calculate_fine_phonons(
            self.qpts, dipole=True, splitting=False, asr=None, use_c=True,
            fall_back_on_python=False)
        npt.assert_allclose(
            self.data.freqs.to('hartree').magnitude,
            self.expctd_data.freqs_dipole_no_asr.to('hartree').magnitude,
            atol=1e-8)

    def test_calculate_fine_phonons_dipole_no_asr_c_2threads(self):
        self.data.calculate_fine_phonons(
            self.qpts, dipole=True, splitting=False, use_c=True,
            fall_back_on_python=False, n_threads=2)
        npt.assert_allclose(
            self.data.freqs.to('hartree').magnitude,
            self.expctd_data.freqs_dipole_no_asr.to('hartree').magnitude,
            atol=1e-8)

    def test_calculate_fine_phonons_dipole_recip_asr(self):
        self.data.calculate_fine_phonons(
            self.qpts, dipole=True, splitting=False, asr='reciprocal')
        npt.assert_allclose(
            self.data.freqs.to('hartree').magnitude,
            self.expctd_data.freqs_dipole_recip_asr.to('hartree').magnitude,
            atol=1e-8)

    def test_calculate_fine_phonons_dipole_recip_asr_c(self):
        self.data.calculate_fine_phonons(
            self.qpts, dipole=True, splitting=False, asr='reciprocal',
            use_c=True, fall_back_on_python=False)
        npt.assert_allclose(
            self.data.freqs.to('hartree').magnitude,
            self.expctd_data.freqs_dipole_recip_asr.to('hartree').magnitude,
            atol=1e-8)

    def test_calculate_fine_phonons_dipole_recip_asr_c_2threads(self):
        self.data.calculate_fine_phonons(
            self.qpts, dipole=True, splitting=False, asr='reciprocal',
            use_c=True, fall_back_on_python=False, n_threads=2)
        npt.assert_allclose(
            self.data.freqs.to('hartree').magnitude,
            self.expctd_data.freqs_dipole_recip_asr.to('hartree').magnitude,
            atol=1e-8)

    def test_calculate_fine_phonons_dipole_realsp_asr(self):
        self.data.calculate_fine_phonons(
            self.qpts, dipole=True, splitting=False, asr='realspace')
        npt.assert_allclose(
            self.data.freqs.to('hartree').magnitude,
            self.expctd_data.freqs_dipole_realsp_asr.to('hartree').magnitude,
            atol=1e-8)

    def test_calculate_fine_phonons_dipole_realsp_asr_c(self):
        self.data.calculate_fine_phonons(
            self.qpts, dipole=True, splitting=False, asr='realspace',
            use_c=True, fall_back_on_python=False)
        npt.assert_allclose(
            self.data.freqs.to('hartree').magnitude,
            self.expctd_data.freqs_dipole_realsp_asr.to('hartree').magnitude,
            atol=1e-8)

    def test_calculate_fine_phonons_dipole_realsp_asr_c_2threads(self):
        self.data.calculate_fine_phonons(
            self.qpts, dipole=True, splitting=False, asr='realspace',
            use_c=True, fall_back_on_python=False, n_threads=2)
        npt.assert_allclose(
            self.data.freqs.to('hartree').magnitude,
            self.expctd_data.freqs_dipole_realsp_asr.to('hartree').magnitude,
            atol=1e-8)

