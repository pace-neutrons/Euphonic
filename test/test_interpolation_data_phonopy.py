import unittest
import os
import numpy.testing as npt
import numpy as np
from euphonic import ureg
from euphonic.data.interpolation import InterpolationData

class TestReadInterpolationNaCl(unittest.TestCase):

    def setUp(self):

        self.path = os.path.join('phonopy_data', 'interpolation')

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

        expctd_data.fc_mat = np.load(self.path +'/force_constants.npy')*ureg('hartree/bohr**2')

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

        self.expctd_data = expctd_data

        self.data = InterpolationData.from_phonopy(path=self.path, fc_name='force_constants.hdf5')
        self.data_fullfc = InterpolationData.from_phonopy(path=self.path, fc_name='full_force_constants.hdf5')
        self.data_FC = InterpolationData.from_phonopy(path=self.path, fc_name='FORCE_CONSTANTS')
        self.data_FULLFC = InterpolationData.from_phonopy(path=self.path, fc_name='FULL_FORCE_CONSTANTS')

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

    def test_fc_mat_read(self):
        npt.assert_allclose(self.data.force_constants.magnitude,
                            self.expctd_data.fc_mat)

    def test_fc_mat_read_fullfc(self):
        npt.assert_allclose(self.data_fullfc.force_constants.magnitude,
                            self.expctd_data.fc_mat)

    def test_fc_mat_read_FC(self):
        npt.assert_allclose(self.data_FC.force_constants.magnitude,
                            self.expctd_data.fc_mat)

    def test_fc_mat_read_FULLFC(self):
        npt.assert_allclose(self.data_FULLFC.force_constants.magnitude,
                            self.expctd_data.fc_mat)

class TestInterpolatePhononsNaCl(unittest.TestCase):

    def setUp(self):
        self.path = os.path.join('phonopy_data', 'interpolation')
        self.qpts = np.array(
                        [[0.00, 0.00, 0.00],
                        [0.00, 0.00, 0.50],
                        [-0.25, 0.50, 0.50],
                        [-0.151515, 0.575758, 0.5]])

        self.data = InterpolationData.from_phonopy(path=self.path, fc_name='force_constants.hdf5')

        expctd_data = type('', (), {})()
        expctd_data.split_qpts = np.empty([])
        expctd_data.split_freqs = np.empty([])*ureg('hartree')
        expctd_data.split_i = np.empty([])
        expctd_data.freqs_asr = np.empty([])
        expctd_data.freqs_asr_splitting = np.empty([])

        self.expctd_data.freqs_phonopy = np.load(self.path + '/phonopy_freqs.npy')

        expctd_data.freqs_dipole_no_asr = np.array(
            [[-1.71566899, -1.71566899, -1.71566899, 13.25831594, 13.25831594, 13.25831594,
              13.70675726, 13.70675726, 13.70675726, 13.70675726, 13.70675726, 13.70675726,
              14.77838152, 14.77838152, 14.77838152, 14.77838152, 14.77838152, 14.77838152,
              15.28987379, 15.28987379, 15.28987379, 26.80366010, 26.80366010,  26.8036601],
             [10.42911169, 10.42911169, 10.42911169, 10.42911169, 11.25702209, 11.25702209,
              13.42073099, 13.42073099, 13.42073099, 13.42073099, 14.53763760,  14.5376376,
              14.53763760, 14.53763760, 15.35508239, 15.35508239, 16.77483373, 16.77483373,
              21.51513497, 21.51513497, 21.51513497, 21.51513497, 31.88946565, 31.88946565],
             [ 9.69770898,  9.69770898,  9.69770898,  9.69770898, 12.23182985, 12.23182985,
              12.23182985, 12.23182985, 12.89656098, 12.89656098, 12.89656098, 12.89656098,
              13.13865656, 13.13865656, 13.13865656, 13.13865656, 22.06696901, 22.06696901,
              22.06696901, 22.06696901, 29.19637097, 29.19637097, 29.19637097, 29.19637097],
             [ 8.70045408,  8.70045408, 10.00554218, 10.00554218, 12.85834757, 12.85834757,
              12.91733821, 12.91733821, 13.51723425, 13.51723425, 13.70221505, 13.70221505,
              13.76553973, 13.76553973, 14.54322540, 14.5432254 , 19.40371191, 19.40371191,
              21.05154040, 21.05154040, 28.28756347, 28.28756347, 29.77527506, 29.77527506]])

        expctd_data.freqs_dipole_recip_asr = np.array(
            [[ 1.59447308,  1.59494716,  1.59508558, 13.03780753, 13.03780816,
              13.03780846, 13.70911598, 13.7091161 , 13.70911612, 13.70911614,
              13.70911619, 13.70911629, 14.77619333, 14.77619334, 14.77619334,
              14.77619335, 14.77619335, 14.77619336, 15.29016583, 15.29016583,
              15.29016585, 26.8034935 , 26.8034935 , 26.80349351],
             [10.43378439, 10.43378439, 10.43378587, 10.43378588, 11.22622985,
              11.22623974, 13.41709858, 13.41709858, 13.41712816, 13.41712822,
              14.52964205, 14.52964205, 14.52964205, 14.52964205, 15.35508239,
              15.35508239, 16.77483373, 16.77483373, 21.52053536, 21.52053536,
              21.52053536, 21.52053536, 31.90031516, 31.90032425],
             [ 9.70009213,  9.70009213,  9.70009218,  9.70009221, 12.23105123,
              12.23105124, 12.23105124, 12.23105131, 12.89476858, 12.89476858,
              12.89476859, 12.89476859, 13.13693889, 13.13693891, 13.13693891,
              13.13693891, 22.06651219, 22.0665122 , 22.0665122 , 22.06651223,
              29.1978153 , 29.1978153 , 29.1978153 , 29.19781531],
             [ 8.7023481 ,  8.70234881, 10.00872211, 10.00872245, 12.85563854,
              12.85563861, 12.91640757, 12.91640822, 13.51524781, 13.51525061,
              13.69887651, 13.69887892, 13.76367403, 13.76367505, 14.54189738,
              14.54189775, 19.40187929, 19.40187962, 21.05182276, 21.05182276,
              28.2897569 , 28.28975728, 29.77808445, 29.77808461]])

        expctd_data.freqs_dipole_realsp_asr = array(
            [[-1.18299439e-05, -8.35832858e-06,  2.17181083e-07,
              1.32635069e+01,  1.32635069e+01,  1.32635069e+01,
              1.37067573e+01,  1.37067573e+01,  1.37067573e+01,
              1.37067573e+01,  1.37067573e+01,  1.37067573e+01,
              1.47783815e+01,  1.47783815e+01,  1.47783815e+01,
              1.47783815e+01,  1.47783815e+01,  1.47783815e+01,
              1.52898738e+01,  1.52898738e+01,  1.52898738e+01,
              2.68036601e+01,  2.68036601e+01,  2.68036601e+01],
             [1.04291117e+01,  1.04291117e+01,  1.05713171e+01,
              1.05713171e+01,  1.12570221e+01,  1.13876368e+01,
              1.34207310e+01,  1.34207310e+01,  1.34242631e+01,
              1.34242631e+01,  1.45376376e+01,  1.45376376e+01,
              1.45376376e+01,  1.45376376e+01,  1.53550824e+01,
              1.53550824e+01,  1.67748337e+01,  1.67748337e+01,
              2.15151350e+01,  2.15151350e+01,  2.15151350e+01,
              2.15151350e+01,  3.18894656e+01,  3.18914015e+01],
             [9.69770898e+00,  9.69770898e+00,  9.75921890e+00,
              9.75921890e+00,  1.22318299e+01,  1.22549786e+01,
              1.22549786e+01,  1.22879539e+01,  1.28965610e+01,
              1.28965610e+01,  1.29096071e+01,  1.29096071e+01,
              1.31386566e+01,  1.31403902e+01,  1.31403902e+01,
              1.31559394e+01,  2.20669690e+01,  2.20797946e+01,
              2.20797946e+01,  2.20909691e+01,  2.91963710e+01,
              2.91996383e+01,  2.92026917e+01,  2.92026917e+01],
             [8.74784638e+00,  8.74800027e+00,  1.00319805e+01,
              1.00323200e+01,  1.28604427e+01,  1.28616809e+01,
              1.29334422e+01,  1.29702357e+01,  1.35180196e+01,
              1.35213418e+01,  1.37028261e+01,  1.37095319e+01,
              1.37671557e+01,  1.37679582e+01,  1.45522966e+01,
              1.45680862e+01,  1.94149899e+01,  1.94351445e+01,
              2.10571731e+01,  2.10698302e+01,  2.82893430e+01,
              2.82900943e+01,  2.97771378e+01,  2.97773925e+01]])

        self.expctd_data = expctd_data

    # NO ASR
    def test_calculate_fine_phonons_dipole_no_asr(self):
        self.data.calculate_fine_phonons(self.qpts, dipole=True, splitting=False, asr=None)
        npt.assert_allclose(
            self.data.freqs.to('hartree').magnitude,
            self.expctd_data.freqs_dipole_no_asr.to('hartree').magnitude,
            atol=1e-8)

    def test_calculate_fine_phonons_dipole_no_asr_c(self):
        self.data.calculate_fine_phonons(
            self.qpts, dipole=True, splitting=False, use_c=True)
        npt.assert_allclose(
            self.data.freqs.to('hartree').magnitude,
            self.expctd_data.freqs_dipole_no_asr.to('hartree').magnitude,
            atol=1e-8)

    def test_calculate_fine_phonons_dipole_no_asr_c_2threads(self):
        self.data.calculate_fine_phonons(
            self.qpts, dipole=True, splitting=False, use_c=True, n_threads=2)
        npt.assert_allclose(
            self.data.freqs.to('hartree').magnitude,
            self.expctd_data.freqs_dipole_no_asr.to('hartree').magnitude,
            atol=1e-8)

    # RECIP ASR
    def test_calculate_fine_phonons_dipole_recip_asr(self):
        self.data.calculate_fine_phonons(self.qpts, dipole=True, splitting=False, asr='reciprocal')
        npt.assert_allclose(
            self.data.freqs.to('hartree').magnitude,
            self.expctd_data.freqs_dipole_recip_asr.to('hartree').magnitude,
            atol=1e-8)

    def test_calculate_fine_phonons_dipole_recip_asr_c(self):
        self.data.calculate_fine_phonons(
            self.qpts, dipole=True, splitting=False, asr='reciprocal', use_c=True)
        npt.assert_allclose(
            self.data.freqs.to('hartree').magnitude,
            self.expctd_data.freqs_dipole_recip_asr.to('hartree').magnitude,
            atol=1e-8)

    def test_calculate_fine_phonons_dipole_recip_asr_c_2threads(self):
        self.data.calculate_fine_phonons(
            self.qpts, dipole=True, splitting=False, asr='reciprocal', use_c=True, n_threads=2)
        npt.assert_allclose(
            self.data.freqs.to('hartree').magnitude,
            self.expctd_data.freqs_dipole_recip_asr.to('hartree').magnitude,
            atol=1e-8)

    # REALSP ASR
    def test_calculate_fine_phonons_dipole_realsp_asr(self):
        self.data.calculate_fine_phonons(self.qpts, dipole=True, splitting=False, asr='realspace')
        npt.assert_allclose(
            self.data.freqs.to('hartree').magnitude,
            self.expctd_data.freqs_dipole_realsp_asr.to('hartree').magnitude,
            atol=1e-8)

    def test_calculate_fine_phonons_dipole_realsp_asr_c(self):
        self.data.calculate_fine_phonons(
            self.qpts, dipole=True, splitting=False, asr='realspace', use_c=True)
        npt.assert_allclose(
            self.data.freqs.to('hartree').magnitude,
            self.expctd_data.freqs_dipole_realsp_asr.to('hartree').magnitude,
            atol=1e-8)

    def test_calculate_fine_phonons_dipole_realsp_asr_c_2threads(self):
        self.data.calculate_fine_phonons(
            self.qpts, dipole=True, splitting=False, asr='realspace', use_c=True, n_threads=2)
        npt.assert_allclose(
            self.data.freqs.to('hartree').magnitude,
            self.expctd_data.freqs_dipole_realsp_asr.to('hartree').magnitude,
            atol=1e-8)

