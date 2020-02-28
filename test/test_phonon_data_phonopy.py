import os
import unittest
import numpy.testing as npt
import numpy as np
from euphonic.data.phonon import PhononData
from euphonic import ureg

#TODO phonon
#read phonopy.yaml
#read band.yaml, cell_vec, recip_vec, n_qpts, n_ions, ion_r, ion_mass, ion_type, qpts, freqs, eigenvecs,
#               split_i, split_freqs, split_eigenvecs
#read mesh.yaml
#read qpoints.yaml
#read band.hdf5
#read mesh.hdf5
#read qpoints.hdf5

#TODO load all output types? mesh, qpoints, band
class TestReadNaClPhononQPoints(unittest.TestCase):

    def setUp(self):

        root = '/home/connorjp/Pace/Euphonic' #TODO remove
        #self.path = os.path.join(root, 'test', 'phonopy_data', 'qpoints') #TODO Testing from elsewhere
        self.path = os.path.join('phonopy_data', 'qpoints')

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

        expctd_data.ion_type = np.array(['Na', 'Na', 'Na', 'Na', 'Cl', 'Cl', 'Cl', 'Cl'])

        expctd_data.ion_mass = np.array(
                [22.989769, 22.989769, 22.989769, 22.989769,
                    35.453000, 35.453000, 35.453000, 35.453000])*ureg('amu')

        expctd_data.qpts = np.load(self.path + '/qpts.npy')

        expctd_data.weights = np.load(self.path + '/weights.npy')

        expctd_data.freqs = np.load(self.path + '/freqs.npy')*ureg('meV')

        expctd_data.freq_down = np.array([])

        expctd_data.eigenvecs = np.load(self.path + '/eigenvecs.npy')

        self.expctd_data = expctd_data

        #self.path = os.path.join('test', 'phonopy_data', 'qpoints') #TODO load all output types

        data = PhononData.from_phonopy(path=self.path, phonon_name='qpoints.yaml')
        self.data = data

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
        npt.assert_array_equal(self.data.ion_mass.to('amu').magnitude,
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

test = TestReadNaClPhononQPoints()
test.setUp()
