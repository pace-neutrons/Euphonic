import unittest
import numpy.testing as npt
import numpy as np
from euphonic.data.phonon import PhononData
from euphonic import ureg


class TestReadInputFileNaHPhonon(unittest.TestCase):

    def setUp(self):
        # Create trivial function object so attributes can be assigned to it
        expctd_data = type('', (), {})()
        expctd_data.cell_vec = np.array(
            [[0.000000, 2.399500, 2.399500],
             [2.399500, 0.000000, 2.399500],
             [2.399500, 2.399500, 0.000000]])*ureg('angstrom')
        expctd_data.ion_r = np.array([[0.500000, 0.500000, 0.500000],
                                      [0.000000, 0.000000, 0.000000]])
        expctd_data.ion_type = np.array(['H', 'Na'])
        expctd_data.ion_mass = np.array([1.007940, 22.989770])*ureg('amu')
        expctd_data.qpts = np.array([[-0.250000, -0.250000, -0.250000],
                                     [-0.250000, -0.500000, -0.500000]])
        expctd_data.weights = np.array([0.125, 0.375])
        expctd_data.freqs = np.array(
            [[91.847109, 91.847109, 166.053018,
              564.508299, 564.508299, 884.068976],
             [132.031513, 154.825631, 206.213940,
              642.513551, 690.303338, 832.120011]])*ureg('1/cm')
        expctd_data.freq_down = np.array([])*ureg('1/cm')
        expctd_data.eigenvecs = np.array(
            [[[[-0.061613336996 - 0.060761142686*1j,
                -0.005526816216 - 0.006379010526*1j,
                0.067140153211 + 0.067140153211*1j],
               [0.666530886823 - 0.004641603630*1j,
                0.064846864124 + 0.004641603630*1j,
                -0.731377750947 + 0.000000000000*1j]],
              [[-0.043088481348 - 0.041294487960*1j,
                0.074981829953 + 0.073187836565*1j,
                -0.031893348605 - 0.031893348605*1j],
               [0.459604449490 - 0.009771253020*1j,
                -0.807028225834 + 0.009771253020*1j,
                0.347423776344 + 0.000000000000*1j]],
              [[-0.062303354995 - 0.062303354995*1j,
                -0.062303354995 - 0.062303354995*1j,
                -0.062303354995 - 0.062303354995*1j],
               [0.570587344099 - 0.000000000000*1j,
                0.570587344099 - 0.000000000000*1j,
                0.570587344099 + 0.000000000000*1j]],
              [[0.286272749085 + 0.286272749085*1j,
                0.286272749085 + 0.286272749085*1j,
                -0.572545498170 - 0.572545498170*1j],
               [0.052559422840 - 0.000000000000*1j,
                0.052559422840 + 0.000000000000*1j,
                -0.105118845679 + 0.000000000000*1j]],
              [[-0.459591797004 + 0.529611084985*1j,
                0.459591797004 - 0.529611084985*1j,
                0.000000000000 - 0.000000000000*1j],
               [0.006427739587 + 0.090808385909*1j,
                -0.006427739587 - 0.090808385909*1j,
                0.000000000000 + 0.000000000000*1j]],
              [[-0.403466180272 - 0.403466180272*1j,
                -0.403466180272 - 0.403466180272*1j,
                -0.403466180272 - 0.403466180272*1j],
               [-0.088110249616 - 0.000000000000*1j,
                -0.088110249616 - 0.000000000000*1j,
                -0.088110249616 + 0.000000000000*1j]]],
             [[[0.000000000000 + 0.000000000000*1j,
                0.031866260273 - 0.031866260273*1j,
                -0.031866260273 + 0.031866260273*1j],
               [-0.000000000000 - 0.000000000000*1j,
                -0.705669244698 + 0.000000000000*1j,
                0.705669244698 + 0.000000000000*1j]],
              [[-0.001780156891 + 0.001780156891*1j,
                -0.012680513033 + 0.012680513033*1j,
                -0.012680513033 + 0.012680513033*1j],
               [-0.582237273385 + 0.000000000000*1j,
                0.574608665929 - 0.000000000000*1j,
                0.574608665929 + 0.000000000000*1j]],
              [[-0.021184502078 + 0.021184502078*1j,
                -0.011544287510 + 0.011544287510*1j,
                -0.011544287510 + 0.011544287510*1j],
               [0.812686635458 - 0.000000000000*1j,
                0.411162853378 + 0.000000000000*1j,
                0.411162853378 + 0.000000000000*1j]],
              [[0.000000000000 + 0.000000000000*1j,
                -0.498983508201 + 0.498983508201*1j,
                0.498983508201 - 0.498983508201*1j],
               [0.000000000000 + 0.000000000000*1j,
                -0.045065697460 - 0.000000000000*1j,
                0.045065697460 + 0.000000000000*1j]],
              [[0.400389305548 - 0.400389305548*1j,
                -0.412005183792 + 0.412005183792*1j,
                -0.412005183792 + 0.412005183792*1j],
               [0.009657696420 - 0.000000000000*1j,
                -0.012050954709 + 0.000000000000*1j,
                -0.012050954709 + 0.000000000000*1j]],
              [[-0.582440084400 + 0.582440084400*1j,
                -0.282767859813 + 0.282767859813*1j,
                -0.282767859813 + 0.282767859813*1j],
               [-0.021140457173 + 0.000000000000*1j,
                -0.024995270201 - 0.000000000000*1j,
                -0.024995270201 + 0.000000000000*1j]]]])
        self.expctd_data = expctd_data

        self.seedname = 'NaH'
        self.path = 'data'
        data = PhononData.from_castep(self.seedname, path=self.path)
        self.data = data

    def test_cell_vec_read_nah_phonon(self):
        npt.assert_allclose(self.data.cell_vec.to('bohr').magnitude,
                            self.expctd_data.cell_vec.to('bohr').magnitude)

    def test_ion_r_read_nah_phonon(self):
        npt.assert_array_equal(self.data.ion_r,
                               self.expctd_data.ion_r)

    def test_ion_type_read_nah_phonon(self):
        npt.assert_array_equal(self.data.ion_type,
                               self.expctd_data.ion_type)

    def test_ion_mass_read_nah_phonon(self):
        npt.assert_allclose(self.data.ion_mass.to('amu').magnitude,
                            self.expctd_data.ion_mass.to('amu').magnitude)

    def test_qpts_read_nah_phonon(self):
        npt.assert_array_equal(self.data.qpts,
                               self.expctd_data.qpts)

    def test_weights_read_nah_phonon(self):
        npt.assert_array_equal(self.data.weights,
                               self.expctd_data.weights)

    def test_freqs_read_nah_phonon(self):
        npt.assert_allclose(
            self.data.freqs.to('hartree', 'spectroscopy').magnitude,
            self.expctd_data.freqs.to('hartree', 'spectroscopy').magnitude)

    def test_eigenvecs_read_nah_phonon(self):
        npt.assert_array_equal(self.data.eigenvecs,
                               self.expctd_data.eigenvecs)
