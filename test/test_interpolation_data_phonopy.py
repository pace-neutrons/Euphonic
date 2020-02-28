import unittest
import os
import numpy.testing as npt
import numpy as np
from euphonic import ureg
from euphonic.data.interpolation import InterpolationData

class TestReadNaCl(unittest.TestCase):

    def setUp(self):

        root = '/home/connorjp/Pace/Euphonic' #TODO remove
        self.path = os.path.join(root, 'test', 'phonopy_data', 'interpolation') #TODO Testing from elsewhere
        #self.path = os.path.join('phonopy_data', 'interpolation')

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

            [[-1.086875,  0.      ,  0.      ],
             [ 0.      , -1.086875,  0.      ],
             [ 0.      ,  0.      , -1.086875]]])*ureg('e')

        expctd_data.dielectric = np.array(
            [[2.43533967, 0.        , 0.        ],
             [0.        , 2.43533967, 0.        ],
             [0.        , 0.        , 2.43533967]])

        self.expctd_data = expctd_data

        data = InterpolationData.from_phonopy(path=self.path)
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

    def test_fc_mat_read(self):
        npt.assert_allclose(self.data.force_constants.magnitude,
                            self.expctd_data.fc_mat)


class TestInterpolatePhononsNaCl(unittest.TestCase):

    def setUp(self):
        self.path = os.path.join('phonopy_data', 'interpolation')
        self.qpts = np.array([[0.00, 0.00, 0.00],
                              [0.00, 0.00, 0.50],
                              [-0.25, 0.50, 0.50],
                              [-0.151515, 0.575758, 0.5]])

        data = InterpolationData.from_phonopy(path=self.path, fc_name='FORCE_CONSTANTS', fc_format='phonopy')
        self.data = data
        self.expctd_data.freqs = np.array()*ureg('hartree')
        self.expctd_data.split_qpts = np.empty([])
        self.expctd_data.split_freqs = np.empty([])*ureg('hartree')
        self.expctd_data.split_i = np.empty([])

    def test_calculate_fine_phonons_dipole_no_asr(self):
        self.data.calculate_fine_phonons(
            self.qpts, dipole=True, splitting=False)
        npt.assert_allclose(
            self.data.freqs.to('hartree').magnitude,
            self.expctd_data.freqs_no_asr.to('hartree').magnitude,
            atol=1e-8)

    def test_calculate_fine_phonons_dipole_no_asr_2procs(self):
        self.data.calculate_fine_phonons(
            self.qpts, dipole=True, splitting=False, nprocs=2)
        npt.assert_allclose(
            self.data.freqs.to('hartree').magnitude,
            self.expctd_data.freqs_no_asr.to('hartree').magnitude,
            atol=1e-8)

    def test_calculate_fine_phonons_dipole_no_asr_2procs_qchunk(self):
        self.data.calculate_fine_phonons(
            self.qpts, dipole=True, splitting=False, nprocs=2, _qchunk=2)
        npt.assert_allclose(
            self.data.freqs.to('hartree').magnitude,
            self.expctd_data.freqs_no_asr.to('hartree').magnitude,
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
