import os
import unittest
import numpy.testing as npt
import numpy as np
from euphonic.data.phonon import PhononData
from euphonic import ureg
from .utils import get_data_path

class TestReadNaClPhononQPoints(unittest.TestCase):

    def setUp(self):
        self.path = os.path.join(get_data_path(), 'phonopy_data', 'qpoints')

        # Create trivial function object so attributes can be assigned to it
        expctd_data = type('', (), {})()
        expctd_data.n_ions = 8
        expctd_data.n_branches = 24
        expctd_data.n_qpts = 4

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

        expctd_data.ion_type = np.array(['Na', 'Na', 'Na', 'Na', 'Cl', 'Cl', 'Cl', 'Cl'])

        expctd_data.ion_mass = np.array(
                [22.989769, 22.989769, 22.989769, 22.989769,
                    35.453000, 35.453000, 35.453000, 35.453000])*ureg('amu')

        expctd_data.qpts = np.array(
                [[ 0.      ,  0.      ,  0.      ],
                 [ 0.      ,  0.      ,  0.5     ],
                 [-0.25    ,  0.5     ,  0.5     ],
                 [-0.151515,  0.575758,  0.5     ]])

        expctd_data.weights = np.array([0.25, 0.25, 0.25, 0.25])

        expctd_data.freqs = np.array(
                [[-0.15305672, -0.15305672, -0.15305672,  9.98275858,  9.98275858,
                   9.98275858,  9.98275858,  9.98275858,  9.98275858, 16.81664714,
                  16.81664714, 16.81664714, 19.05903109, 19.05903109, 19.05903109,
                  20.12732011, 20.12732011, 20.12732011, 20.12732011, 20.12732011,
                  20.12732011, 21.73566019, 21.73566019, 21.73566019],
                 [ 7.17689487,  7.17689487,  7.1768949 ,  7.1768949 , 14.16528667,
                  14.16528667, 14.1652867 , 14.1652867 , 15.51176817, 15.51176848,
                  16.24673037, 16.24673037, 18.02355311, 18.02355311, 19.57717104,
                  19.57717104, 19.57717104, 19.57717104, 20.923021  , 20.923021  ,
                  20.923021  , 20.923021  , 24.72369461, 24.72369461],
                 [12.22838658, 12.22838658, 12.22838672, 12.22838675, 13.479923  ,
                  13.47992306, 13.47992306, 13.47992315, 15.57333968, 15.57333982,
                  15.57333982, 15.57333987, 17.84863311, 17.84863323, 17.84863323,
                  17.84863341, 19.67521227, 19.67521256, 19.67521256, 19.67521287,
                  25.34815896, 25.34815901, 25.34815901, 25.34815904],
                 [10.99948579, 10.99948609, 12.44158214, 12.44158215, 12.80530205,
                  12.80530249, 13.86189376, 13.86189382, 15.64634206, 15.64634209,
                  15.83247592, 15.83247604, 18.05815279, 18.05815302, 18.92102697,
                  18.92102704, 19.11364648, 19.11364655, 19.58231829, 19.58231858,
                  23.97564755, 23.9756476 , 25.51022627, 25.51022678]])*ureg('meV')

        expctd_data.eigenvecs = np.load(self.path + '/phonopy_eigenvecs.npy')

        expctd_data.freq_down = np.array([])

        self.expctd_data = expctd_data

        self.data = PhononData.from_phonopy(path=self.path,
                            phonon_name='qpoints.yaml', summary_name='phonopy.yaml')

        self.data_hdf5 = PhononData.from_phonopy(path=self.path,
                            phonon_name='qpoints.hdf5', summary_name='phonopy.yaml')

    def test_n_qpts_read_nacl_phonon(self):
        npt.assert_equal(self.data.n_qpts,
                            self.expctd_data.n_qpts)

    def test_n_ions_read_nacl_phonon(self):
        npt.assert_equal(self.data.n_ions,
                            self.expctd_data.n_ions)

    def test_n_branches_read_nacl_phonon(self):
        npt.assert_equal(self.data.n_branches,
                            self.expctd_data.n_branches)

    def test_cell_vec_read_nacl_phonon(self):
        npt.assert_allclose(self.data.cell_vec.to('bohr').magnitude,
                            self.expctd_data.cell_vec.to('bohr').magnitude)

    def test_ion_r_read_nacl_phonon(self):
        npt.assert_array_equal(self.data.ion_r,
                               self.expctd_data.ion_r)

    def test_ion_type_read_nacl_phonon(self):
        npt.assert_array_equal(self.data.ion_type,
                               self.expctd_data.ion_type)

    def test_ion_mass_read_nacl_phonon(self):
        npt.assert_allclose(self.data.ion_mass.to('amu').magnitude,
                               self.expctd_data.ion_mass.to('amu').magnitude)

    def test_qpts_read_nacl_phonon(self):
        npt.assert_array_equal(self.data.qpts,
                               self.expctd_data.qpts)

    def test_weights_read_nacl_phonon(self):
        npt.assert_array_equal(self.data.weights,
                               self.expctd_data.weights)

    def test_freqs_read_nacl_phonon(self):
        npt.assert_allclose(
            self.data.freqs.to('hartree', 'spectroscopy').magnitude,
            self.expctd_data.freqs.to('hartree', 'spectroscopy').magnitude)

    def test_eigenvecs_read_nacl_phonon(self):
        npt.assert_array_equal(self.data.eigenvecs,
                               self.expctd_data.eigenvecs)
