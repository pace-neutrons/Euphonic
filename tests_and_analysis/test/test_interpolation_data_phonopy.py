import unittest
import os
import numpy.testing as npt
import numpy as np
from euphonic import ureg
from euphonic.data.interpolation import InterpolationData
from .utils import get_data_path

class TestReadInterpolationNaCl(unittest.TestCase):

    def setUp(self):

        self.path = os.path.join(get_data_path(), 'phonopy_data', 'NaCl', 'interpolation')

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


        expctd_data.force_constants = np.load(self.path +'/force_constants_hdf5.npy')*ureg('hartree/bohr**2')
        expctd_data.FORCE_CONSTANTS = np.load(self.path +'/force_constants.npy')*ureg('hartree/bohr**2')
        expctd_data.force_constants_yaml = np.load(self.path +'/force_constants_yaml.npy')*ureg('hartree/bohr**2')

        self.expctd_data = expctd_data

        self.data = InterpolationData.from_phonopy(path=self.path, fc_name='force_constants.hdf5')
        self.data_fullfc = InterpolationData.from_phonopy(path=self.path, fc_name='full_force_constants.hdf5')
        self.data_FC = InterpolationData.from_phonopy(path=self.path, fc_name='FORCE_CONSTANTS')
        self.data_FULLFC = InterpolationData.from_phonopy(path=self.path, fc_name='FULL_FORCE_CONSTANTS')
        self.data_fc_yaml = InterpolationData.from_phonopy(path=self.path, fc_format='yaml')
        self.data_fullfc_yaml = InterpolationData.from_phonopy(path=self.path, fc_format='yaml')

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
                            self.expctd_data.force_constants.magnitude)

    def test_fc_mat_read_fullfc(self):
        npt.assert_allclose(self.data_fullfc.force_constants.magnitude,
                            self.expctd_data.force_constants.magnitude)

    def test_fc_mat_read_FC(self):
        npt.assert_allclose(self.data_FC.force_constants.magnitude,
                            self.expctd_data.FORCE_CONSTANTS.magnitude)

    def test_fc_mat_read_FULLFC(self):
        npt.assert_allclose(self.data_FULLFC.force_constants.magnitude,
                            self.expctd_data.FORCE_CONSTANTS.magnitude)

    def test_fc_mat_read_fc_yaml(self):
        npt.assert_allclose(self.data_fc_yaml.force_constants.magnitude,
                            self.expctd_data.force_constants_yaml.magnitude)

    def test_fc_mat_read_fullfc_yaml(self):
        npt.assert_allclose(self.data_fullfc_yaml.force_constants.magnitude,
                            self.expctd_data.force_constants_yaml.magnitude)

        #self.data_fc_yaml = InterpolationData.from_phonopy(path=self.path, fc_format='yaml')
        #self.data_fullfc_yaml = InterpolationData.from_phonopy(path=self.path, fc_format='yaml')

class TestInterpolatePhononsNaCl(unittest.TestCase):

    def setUp(self):
        self.path = os.path.join(get_data_path(), 'phonopy_data', 'NaCl', 'interpolation')

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

        #NOTE test data calculated using hdf5 force constants
        expctd_data.freqs_phonopy = np.load(self.path + '/phonopy_freqs.npy')

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
              21.05154040, 21.05154040, 28.28756347, 28.28756347, 29.77527506, 29.77527506]])*ureg('meV')

        expctd_data.freqs_dipole_no_asr_c = np.array(
            [[  4.91744528,   4.91744528,   5.0918359 ,  12.93567212,
               12.93567212,  13.04517998,  13.05951676,  13.05951676,
               13.70911614,  14.00742491,  14.00742491,  14.77025694,
               14.77427058,  14.77427058,  14.77619334,  15.08006873,
               16.45304203,  16.45304203,  17.47674537,  17.47674537,
               17.47828672,  24.16958064,  24.1850628 ,  24.1850628 ],
             [-10.55515033,   7.6534125 ,   7.6534125 ,   7.99272109,
                7.99272109,  12.40868507,  12.40868507,  12.59989859,
               12.59989859,  13.36306087,  13.36306087,  14.31425794,
               14.31425794,  15.55266527,  15.55266527,  18.87780409,
               19.02652992,  19.02652992,  19.21587259,  19.21587259,
               23.43435431,  23.43435431,  26.28021246,  27.68423472],
             [ -6.79732314,  -6.52877662,   6.41967027,   7.97031544,
                8.53567189,   8.82418346,  11.38989656,  11.8524091 ,
               12.32396435,  12.83669301,  15.3897251 ,  17.57349784,
               17.71658408,  18.0134567 ,  18.52874789,  19.56591164,
               19.78336551,  20.41971358,  21.58845595,  22.63077994,
               22.98649865,  23.13670884,  24.61399916,  25.67683865],
             [ -8.75593967,  -8.03573449,  -2.81183047,   8.03953539,
                8.8093186 ,  10.0875571 ,  11.3349882 ,  12.57003131,
               13.14371587,  13.87520877,  14.9163484 ,  16.51999077,
               16.84054393,  17.34832194,  18.68782615,  18.91296193,
               19.34649579,  19.68041486,  21.16272369,  23.18949955,
               23.65981082,  24.03324903,  24.91073619,  26.56189863]])*ureg('meV')


        expctd_data.freqs_dipole_realsp_asr = np.array(
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
              28.2897569 , 28.28975728, 29.77808445, 29.77808461]])*ureg('meV')

        expctd_data.freqs_dipole_realsp_asr_c = np.array(
                [[  4.31275121,   4.31292014,   4.50500578,  12.95493019,
                   12.95493036,  13.25898064,  13.25899272,  13.26175015,
                   13.70675726,  14.00539587,  14.00539591,  14.77254171,
                   14.77648376,  14.77648376,  14.77838152,  15.07969465,
                   16.45078897,  16.45078963,  17.4686174 ,  17.46861989,
                   17.47000653,  24.16661102,  24.1823798 ,  24.18238062],
                 [-10.84095745,   7.27066626,   7.27068639,   7.9970153 ,
                    7.9970153 ,  12.41122062,  12.41122062,  12.8124337 ,
                   12.81243881,  13.36050414,  13.36050414,  14.31205953,
                   14.31205953,  15.55238069,  15.5523807 ,  18.87756772,
                   19.02832534,  19.02832535,  19.21408588,  19.21408588,
                   23.43454317,  23.43454317,  26.28038225,  27.78883558],
                 [ -7.02661042,  -6.7238804 ,   6.41344141,   7.9471861 ,
                    8.42863837,   8.83637886,  11.3873733 ,  11.84556824,
                   12.36458118,  12.83331296,  15.36454509,  17.58441525,
                   17.7183955 ,  18.04970674,  18.53092998,  19.58461383,
                   19.78553476,  20.42722718,  21.58652758,  22.64343171,
                   23.03580804,  23.15829366,  24.62331258,  25.68145159],
                 [ -8.90012076,  -8.20743774,  -2.81704786,   8.03404214,
                    8.81224076,  10.08311917,  11.34709915,  12.51745507,
                   13.15412181,  13.88429756,  14.91934187,  16.52876812,
                   16.85417494,  17.34856458,  18.69816404,  18.91790263,
                   19.35253901,  19.68743804,  21.1673609 ,  23.204019  ,
                   23.68041171,  24.03380563,  24.94102111,  26.56639684]])*ureg('meV')


        expctd_data.freqs_dipole_recip_asr = np.array(
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
              2.82900943e+01,  2.97771378e+01,  2.97773925e+01]])*ureg('meV')

        expctd_data.freqs_dipole_recip_asr_c = np.array(
              [[ 5.18269475,   5.18269475,   5.35084142,  12.93611118,
                 12.93611118,  13.05096752,  13.06459072,  13.06459072,
                 13.70911614,  14.00745762,  14.00745762,  14.77026413,
                 14.77427667,  14.77427667,  14.77619334,  15.08007973,
                 16.45414109,  16.45414109,  17.48105283,  17.48105283,
                 17.48248897,  24.17119402,  24.18651784,  24.18651784],
               [-10.41404752,   7.83062414,   7.83062414,   7.99272109,
                  7.99272109,  12.40868507,  12.40868507,  12.6132742 ,
                 12.6132742 ,  13.36306087,  13.36306087,  14.31425794,
                 14.31425794,  15.55266527,  15.55266527,  18.87780409,
                 19.02652992,  19.02652992,  19.21587259,  19.21587259,
                 23.43435431,  23.43435431,  26.28021246,  27.68644493],
               [ -6.59838406,  -6.31695364,   6.41967027,   7.97464002,
                  8.62491004,   8.83426519,  11.39497151,  11.85570632,
                 12.32445075,  12.83669301,  15.4105114 ,  17.5739    ,
                 17.71669668,  18.02143058,  18.52874789,  19.56671462,
                 19.78336551,  20.41993201,  21.59133369,  22.63471566,
                 22.99063095,  23.14083254,  24.61747425,  25.67748368],
               [ -8.60681942,  -7.86710124,  -2.80086785,   8.04934482,
                  8.815705  ,  10.09178607,  11.34126089,  12.63653753,
                 13.15072978,  13.88140829,  14.92371734,  16.52388202,
                 16.84623673,  17.35016988,  18.68820987,  18.91429496,
                 19.34774406,  19.68087525,  21.1662373 ,  23.19196158,
                 23.66286852,  24.03516123,  24.91293967,  26.56256556]])*ureg('meV')



        self.expctd_data = expctd_data

        self.data = InterpolationData.from_phonopy(path=self.path,
                            fc_name='force_constants.hdf5', read_born=True)

    # NO ASR
    def test_calculate_fine_phonons_dipole_no_asr(self):
        self.data.calculate_fine_phonons(self.qpts, dipole=True, splitting=False, asr=None)
        npt.assert_allclose(
            self.data.freqs.to('hartree').magnitude,
            self.expctd_data.freqs_dipole_no_asr.to('hartree').magnitude,
            atol=1e-8)

    def test_calculate_fine_phonons_dipole_no_asr_c(self):
        self.data.calculate_fine_phonons(self.qpts, dipole=True, splitting=False, use_c=True)
        npt.assert_allclose(
            self.data.freqs.to('hartree').magnitude,
            self.expctd_data.freqs_dipole_no_asr_c.to('hartree').magnitude,
            atol=1e-8)

    def test_calculate_fine_phonons_dipole_no_asr_c_2threads(self):
        self.data.calculate_fine_phonons(
            self.qpts, dipole=True, splitting=False, use_c=True, n_threads=2)
        npt.assert_allclose(
            self.data.freqs.to('hartree').magnitude,
            self.expctd_data.freqs_dipole_no_asr_c.to('hartree').magnitude,
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
            self.expctd_data.freqs_dipole_recip_asr_c.to('hartree').magnitude,
            atol=1e-8)

    def test_calculate_fine_phonons_dipole_recip_asr_c_2threads(self):
        self.data.calculate_fine_phonons(
            self.qpts, dipole=True, splitting=False, asr='reciprocal', use_c=True, n_threads=2)
        npt.assert_allclose(
            self.data.freqs.to('hartree').magnitude,
            self.expctd_data.freqs_dipole_recip_asr_c.to('hartree').magnitude,
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
            self.expctd_data.freqs_dipole_realsp_asr_c.to('hartree').magnitude,
            atol=1e-8)

    def test_calculate_fine_phonons_dipole_realsp_asr_c_2threads(self):
        self.data.calculate_fine_phonons(
            self.qpts, dipole=True, splitting=False, asr='realspace', use_c=True, n_threads=2)
        npt.assert_allclose(
            self.data.freqs.to('hartree').magnitude,
            self.expctd_data.freqs_dipole_realsp_asr_c.to('hartree').magnitude,
            atol=1e-8)

