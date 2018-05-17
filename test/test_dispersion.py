import unittest
import math
import seekpath
import numpy as np
import numpy.testing as npt
from io import StringIO
from pint import UnitRegistry
from casteppy import dispersion as disp


class TestSetUpUnitRegistry(unittest.TestCase):

    def test_returns_unit_registry(self):
        self.assertIsInstance(disp.set_up_unit_registry(),
                              type(UnitRegistry()))

    def test_has_rydberg_units(self):
        ureg = disp.set_up_unit_registry()
        test_ev = 1 * ureg.Ry
        test_ev.ito(ureg.eV)
        self.assertEqual(test_ev.magnitude, 13.605693009)


class TestReadInputFileNaHBands(unittest.TestCase):

    def setUp(self):
        # Create trivial function object so attributes can be assigned to it
        NaH_bands = lambda:0
        # Need to use actual files here rather than simulating their content
        # with StringIO, in order to test the way the read_input_file function
        # searches for missing data (e.g. ion_pos) in other files
        NaH_bands_file = 'test/NaH.bands'
        units = 'hartree'
        up = False
        down = False

        with open(NaH_bands_file, 'r') as f:
            (NaH_bands.cell_vec, NaH_bands.ion_pos, NaH_bands.ion_type,
                NaH_bands.kpts, NaH_bands.weights, NaH_bands.freqs,
                NaH_bands.freq_down, NaH_bands.eigenvecs,
                NaH_bands.fermi) = disp.read_input_file(
                    f, units, up, down)

        NaH_bands.expected_cell_vec = [[0.000000, 4.534397, 4.534397],
                                       [4.534397, 0.000000, 4.534397],
                                       [4.534397, 4.534397, 0.000000]]
        NaH_bands.expected_ion_pos = [[0.500000, 0.500000, 0.500000],
                                      [0.000000, 0.000000, 0.000000]]
        NaH_bands.expected_ion_type = ['H', 'Na']
        NaH_bands.expected_kpts = [[-0.45833333, -0.37500000, -0.45833333],
                                   [-0.45833333, -0.37500000, -0.20833333]]
        NaH_bands.expected_weights = [0.00347222, 0.00694444]
        NaH_bands.expected_freqs = [[-1.83230180, -0.83321119, -0.83021854,
                                     -0.83016941, -0.04792334],
                                    [-1.83229571, -0.83248269, -0.83078961,
                                     -0.83036048, -0.05738470]]
        NaH_bands.expected_freq_down = []
        NaH_bands.expected_fermi = [-0.009615]
        self.NaH_bands = NaH_bands

    def test_cell_vec_read_nah_bands(self):
        npt.assert_array_equal(self.NaH_bands.cell_vec,
                               self.NaH_bands.expected_cell_vec)

    def test_ion_pos_read_nah_bands(self):
        npt.assert_array_equal(self.NaH_bands.ion_pos,
                               self.NaH_bands.expected_ion_pos)

    def test_ion_type_read_nah_bands(self):
        npt.assert_array_equal(self.NaH_bands.ion_type,
                               self.NaH_bands.expected_ion_type)

    def test_kpts_read_nah_bands(self):
        npt.assert_array_equal(self.NaH_bands.kpts,
                               self.NaH_bands.expected_kpts)

    def test_weights_read_nah_bands(self):
        npt.assert_array_equal(self.NaH_bands.weights,
                               self.NaH_bands.expected_weights)

    def test_freqs_read_nah_bands(self):
        npt.assert_array_equal(self.NaH_bands.freqs,
                               self.NaH_bands.expected_freqs)

    def test_freq_down_read_nah_bands(self):
        npt.assert_array_equal(self.NaH_bands.freq_down,
                               self.NaH_bands.expected_freq_down)

    def test_eigenvecs_read_nah_bands(self):
        npt.assert_array_equal(self.NaH_bands.eigenvecs, [])

    def test_fermi_read_nah_bands(self):
        npt.assert_array_equal(self.NaH_bands.fermi,
                               self.NaH_bands.expected_fermi)


class TestReadInputFileNaHPhonon(unittest.TestCase):

    def setUp(self):
        # Create trivial function object so attributes can be assigned to it
        NaH_phonon = lambda:0
        # Need to use actual files here rather than simulating their content
        # with StringIO, in order to test the way the read_input_file function
        # searches for missing data (e.g. ion_pos) in other files
        NaH_phonon_file = 'test/NaH.phonon'
        units = '1/cm'
        up = False
        down = False
        read_eigenvecs = True

        with open(NaH_phonon_file, 'r') as f:
            (NaH_phonon.cell_vec, NaH_phonon.ion_pos, NaH_phonon.ion_type,
                NaH_phonon.kpts, NaH_phonon.weights, NaH_phonon.freqs,
                NaH_phonon.freq_down, NaH_phonon.eigenvecs,
                NaH_phonon.fermi) = disp.read_input_file(
                    f, units, up, down, read_eigenvecs)

        NaH_phonon.expected_cell_vec = [[0.000000, 2.399500, 2.399500],
                                        [2.399500, 0.000000, 2.399500],
                                        [2.399500, 2.399500, 0.000000]]
        NaH_phonon.expected_ion_pos = [[0.500000, 0.500000, 0.500000],
                                       [0.000000, 0.000000, 0.000000]]
        NaH_phonon.expected_ion_type = ['H', 'Na']
        NaH_phonon.expected_kpts = [[-0.250000, -0.250000, -0.250000],
                                    [-0.250000, -0.500000, -0.500000]]
        NaH_phonon.expected_weights = [0.125, 0.375]
        NaH_phonon.expected_freqs = [[91.847109, 91.847109, 166.053018,
                                      564.508299, 564.508299, 884.068976],
                                     [132.031513, 154.825631, 206.213940,
                                      642.513551, 690.303338, 832.120011]]
        NaH_phonon.expected_freq_down = []
        NaH_phonon.expected_eigenvecs = [[[-0.061613336996 - 0.060761142686*1j,
                                           -0.005526816216 - 0.006379010526*1j,
                                            0.067140153211 + 0.067140153211*1j],
                                          [ 0.666530886823 - 0.004641603630*1j,
                                            0.064846864124 + 0.004641603630*1j,
                                           -0.731377750947 + 0.000000000000*1j],
                                          [-0.043088481348 - 0.041294487960*1j,
                                            0.074981829953 + 0.073187836565*1j,
                                           -0.031893348605 - 0.031893348605*1j],
                                          [ 0.459604449490 - 0.009771253020*1j,
                                           -0.807028225834 + 0.009771253020*1j,
                                            0.347423776344 + 0.000000000000*1j],
                                          [-0.062303354995 - 0.062303354995*1j,
                                           -0.062303354995 - 0.062303354995*1j,
                                           -0.062303354995 - 0.062303354995*1j],
                                          [ 0.570587344099 - 0.000000000000*1j,
                                            0.570587344099 - 0.000000000000*1j,
                                            0.570587344099 + 0.000000000000*1j],
                                          [ 0.286272749085 + 0.286272749085*1j,
                                            0.286272749085 + 0.286272749085*1j,
                                           -0.572545498170 - 0.572545498170*1j],
                                          [ 0.052559422840 - 0.000000000000*1j,
                                            0.052559422840 + 0.000000000000*1j,
                                           -0.105118845679 + 0.000000000000*1j],
                                          [-0.459591797004 + 0.529611084985*1j,
                                            0.459591797004 - 0.529611084985*1j,
                                            0.000000000000 - 0.000000000000*1j],
                                          [ 0.006427739587 + 0.090808385909*1j,
                                           -0.006427739587 - 0.090808385909*1j,
                                            0.000000000000 + 0.000000000000*1j],
                                          [-0.403466180272 - 0.403466180272*1j,
                                           -0.403466180272 - 0.403466180272*1j,
                                           -0.403466180272 - 0.403466180272*1j],
                                          [-0.088110249616 - 0.000000000000*1j,
                                           -0.088110249616 - 0.000000000000*1j,
                                           -0.088110249616 + 0.000000000000*1j]],
                                         [[ 0.000000000000 + 0.000000000000*1j,
                                            0.031866260273 - 0.031866260273*1j,
                                           -0.031866260273 + 0.031866260273*1j],
                                          [-0.000000000000 - 0.000000000000*1j,
                                           -0.705669244698 + 0.000000000000*1j,
                                            0.705669244698 + 0.000000000000*1j],
                                          [-0.001780156891 + 0.001780156891*1j,
                                           -0.012680513033 + 0.012680513033*1j,
                                           -0.012680513033 + 0.012680513033*1j],
                                          [-0.582237273385 + 0.000000000000*1j,
                                            0.574608665929 - 0.000000000000*1j,
                                            0.574608665929 + 0.000000000000*1j],
                                          [-0.021184502078 + 0.021184502078*1j,
                                           -0.011544287510 + 0.011544287510*1j,
                                           -0.011544287510 + 0.011544287510*1j],
                                          [ 0.812686635458 - 0.000000000000*1j,
                                            0.411162853378 + 0.000000000000*1j,
                                            0.411162853378 + 0.000000000000*1j],
                                          [ 0.000000000000 + 0.000000000000*1j,
                                           -0.498983508201 + 0.498983508201*1j,
                                            0.498983508201 - 0.498983508201*1j],
                                          [ 0.000000000000 + 0.000000000000*1j,
                                           -0.045065697460 - 0.000000000000*1j,
                                            0.045065697460 + 0.000000000000*1j],
                                          [ 0.400389305548 - 0.400389305548*1j,
                                           -0.412005183792 + 0.412005183792*1j,
                                           -0.412005183792 + 0.412005183792*1j],
                                          [ 0.009657696420 - 0.000000000000*1j,
                                           -0.012050954709 + 0.000000000000*1j,
                                           -0.012050954709 + 0.000000000000*1j],
                                          [-0.582440084400 + 0.582440084400*1j,
                                           -0.282767859813 + 0.282767859813*1j,
                                           -0.282767859813 + 0.282767859813*1j],
                                          [-0.021140457173 + 0.000000000000*1j,
                                           -0.024995270201 - 0.000000000000*1j,
                                           -0.024995270201 + 0.000000000000*1j]]]

        NaH_phonon.expected_fermi = []
        self.NaH_phonon = NaH_phonon

    def test_cell_vec_read_nah_phonon(self):
        npt.assert_array_equal(self.NaH_phonon.cell_vec,
                               self.NaH_phonon.expected_cell_vec)

    def test_ion_pos_read_nah_phonon(self):
        npt.assert_array_equal(self.NaH_phonon.ion_pos,
                               self.NaH_phonon.expected_ion_pos)

    def test_ion_type_read_nah_phonon(self):
        npt.assert_array_equal(self.NaH_phonon.ion_type,
                               self.NaH_phonon.expected_ion_type)

    def test_kpts_read_nah_phonon(self):
        npt.assert_array_equal(self.NaH_phonon.kpts,
                               self.NaH_phonon.expected_kpts)

    def test_weights_read_nah_phonon(self):
        npt.assert_array_equal(self.NaH_phonon.weights,
                               self.NaH_phonon.expected_weights)

    def test_freqs_read_nah_phonon(self):
        npt.assert_array_equal(self.NaH_phonon.freqs,
                               self.NaH_phonon.expected_freqs)

    def test_freq_down_read_nah_phonon(self):
        npt.assert_array_equal(self.NaH_phonon.freq_down,
                               self.NaH_phonon.expected_freq_down)

    def test_eigenvecs_read_nah_phonon(self):
        npt.assert_array_equal(self.NaH_phonon.eigenvecs,
                               self.NaH_phonon.expected_eigenvecs)

    def test_fermi_read_nah_phonon(self):
        npt.assert_array_equal(self.NaH_phonon.fermi,
                               self.NaH_phonon.expected_fermi)


class TestReadInputFileFeBands(unittest.TestCase):

    def setUp(self):
        # Create trivial function object so attributes can be assigned to it
        Fe_bands = lambda:0
        # Need to use actual files here rather than simulating their content
        # with StringIO, in order to test the way the read_input_file function
        # searches for missing data (e.g. ion_pos) in other files
        Fe_bands_file = 'test/Fe.bands'
        units = 'hartree'
        up = False
        down = False

        with open(Fe_bands_file, 'r') as f:
            (Fe_bands.cell_vec, Fe_bands.ion_pos, Fe_bands.ion_type,
                Fe_bands.kpts, Fe_bands.weights, Fe_bands.freqs,
                Fe_bands.freq_down, Fe_bands.eigenvecs,
                Fe_bands.fermi) = disp.read_input_file(
                    f, units, up, down)

        Fe_bands.expected_cell_vec = [[-2.708355,  2.708355,  2.708355],
                                      [ 2.708355, -2.708355,  2.708355],
                                      [ 2.708355,  2.708355, -2.708355]]

        Fe_bands.expected_ion_pos = []
        Fe_bands.expected_ion_type = []
        Fe_bands.expected_kpts = [[-0.37500000, -0.45833333,  0.29166667],
                                  [-0.37500000, -0.37500000,  0.29166667]]
        Fe_bands.expected_weights = [0.01388889, 0.01388889]
        Fe_bands.expected_freqs = [[0.02278248, 0.02644693, 0.12383402,
                                    0.15398152, 0.17125020, 0.43252010],
                                   [0.02760952, 0.02644911, 0.12442671,
                                    0.14597457, 0.16728951, 0.35463529]]
        Fe_bands.expected_freq_down = [[0.08112495, 0.08345039, 0.19185076,
                                        0.22763689, 0.24912308, 0.46511567],
                                       [0.08778721, 0.08033338, 0.19288937,
                                        0.21817779, 0.24476910, 0.39214129]]
        Fe_bands.expected_fermi = [0.173319, 0.173319]
        self.Fe_bands = Fe_bands

    def test_cell_vec_read_fe_bands(self):
        npt.assert_array_equal(self.Fe_bands.cell_vec,
                               self.Fe_bands.expected_cell_vec)

    def test_ion_pos_read_fe_bands(self):
        npt.assert_array_equal(self.Fe_bands.ion_pos,
                               self.Fe_bands.expected_ion_pos)

    def test_ion_type_read_fe_bands(self):
        npt.assert_array_equal(self.Fe_bands.ion_type,
                               self.Fe_bands.expected_ion_type)

    def test_kpts_read_fe_bands(self):
        npt.assert_array_equal(self.Fe_bands.kpts,
                               self.Fe_bands.expected_kpts)

    def test_weights_read_fe_bands(self):
        npt.assert_array_equal(self.Fe_bands.weights,
                               self.Fe_bands.expected_weights)

    def test_freqs_read_fe_bands(self):
        npt.assert_array_equal(self.Fe_bands.freqs,
                               self.Fe_bands.expected_freqs)

    def test_freq_down_read_fe_bands(self):
        npt.assert_array_equal(self.Fe_bands.freq_down,
                               self.Fe_bands.expected_freq_down)

    def test_eigenvecs_read_fe_bands(self):
        npt.assert_array_equal(self.Fe_bands.eigenvecs, [])

    def test_fermi_read_fe_bands(self):
        npt.assert_array_equal(self.Fe_bands.fermi,
                               self.Fe_bands.expected_fermi)


class TestReadDotPhononAndHeader(unittest.TestCase):

    def setUp(self):
        # Create trivial function object so attributes can be assigned to it
        NaH = lambda:0
        NaH.content = '\n'.join([
           u' BEGIN header',
            ' Number of ions         2',
            ' Number of branches     6',
            ' Number of wavevectors  2',
            ' Frequencies in         cm-1',
            ' IR intensities in      (D/A)**2/amu',
            ' Raman activities in    A**4 amu**(-1)',
            ' Unit cell vectors (A)',
            '    0.000000    2.399500    2.399500',
            '    2.399500    0.000000    2.399500',
            '    2.399500    2.399500    0.000000',
            ' Fractional Co-ordinates',
            '     1     0.500000    0.500000    0.500000   H         1.007940',
            '     2     0.000000    0.000000    0.000000   Na       22.989770',
            ' END header',
            '     q-pt=    1   -0.250000 -0.250000 -0.250000      0.1250000000',
            '       1      91.847109',
            '       2      91.847109',
            '       3     166.053018',
            '       4     564.508299',
            '       5     564.508299',
            '       6     884.068976',
            '                        Phonon Eigenvectors',
            'Mode Ion                X                                   Y                                   Z',
            '   1   1 -0.061613336996 -0.060761142686     -0.005526816216 -0.006379010526      0.067140153211  0.067140153211',
            '   1   2  0.666530886823 -0.004641603630      0.064846864124  0.004641603630     -0.731377750947  0.000000000000',
            '   2   1 -0.043088481348 -0.041294487960      0.074981829953  0.073187836565     -0.031893348605 -0.031893348605',
            '   2   2  0.459604449490 -0.009771253020     -0.807028225834  0.009771253020      0.347423776344  0.000000000000',
            '   3   1 -0.062303354995 -0.062303354995     -0.062303354995 -0.062303354995     -0.062303354995 -0.062303354995',
            '   3   2  0.570587344099 -0.000000000000      0.570587344099 -0.000000000000      0.570587344099  0.000000000000',
            '   4   1  0.286272749085  0.286272749085      0.286272749085  0.286272749085     -0.572545498170 -0.572545498170',
            '   4   2  0.052559422840 -0.000000000000      0.052559422840  0.000000000000     -0.105118845679  0.000000000000',
            '   5   1 -0.459591797004  0.529611084985      0.459591797004 -0.529611084985      0.000000000000 -0.000000000000',
            '   5   2  0.006427739587  0.090808385909     -0.006427739587 -0.090808385909      0.000000000000  0.000000000000',
            '   6   1 -0.403466180272 -0.403466180272     -0.403466180272 -0.403466180272     -0.403466180272 -0.403466180272',
            '   6   2 -0.088110249616 -0.000000000000     -0.088110249616 -0.000000000000     -0.088110249616  0.000000000000',
            '     q-pt=    2   -0.250000 -0.500000 -0.500000      0.3750000000',
            '       1     132.031513',
            '       2     154.825631',
            '       3     206.213940',
            '       4     642.513551',
            '       5     690.303338',
            '       6     832.120011',
            '                        Phonon Eigenvectors',
            'Mode Ion                X                                   Y                                   Z',
            '   1   1  0.000000000000  0.000000000000      0.031866260273 -0.031866260273     -0.031866260273  0.031866260273',
            '   1   2 -0.000000000000 -0.000000000000     -0.705669244698  0.000000000000      0.705669244698  0.000000000000',
            '   2   1 -0.001780156891  0.001780156891     -0.012680513033  0.012680513033     -0.012680513033  0.012680513033',
            '   2   2 -0.582237273385  0.000000000000      0.574608665929 -0.000000000000      0.574608665929  0.000000000000',
            '   3   1 -0.021184502078  0.021184502078     -0.011544287510  0.011544287510     -0.011544287510  0.011544287510',
            '   3   2  0.812686635458 -0.000000000000      0.411162853378  0.000000000000      0.411162853378  0.000000000000',
            '   4   1  0.000000000000  0.000000000000     -0.498983508201  0.498983508201      0.498983508201 -0.498983508201',
            '   4   2  0.000000000000  0.000000000000     -0.045065697460 -0.000000000000      0.045065697460  0.000000000000',
            '   5   1  0.400389305548 -0.400389305548     -0.412005183792  0.412005183792     -0.412005183792  0.412005183792',
            '   5   2  0.009657696420 -0.000000000000     -0.012050954709  0.000000000000     -0.012050954709  0.000000000000',
            '   6   1 -0.582440084400  0.582440084400     -0.282767859813  0.282767859813     -0.282767859813  0.282767859813',
            '   6   2 -0.021140457173  0.000000000000     -0.024995270201 -0.000000000000     -0.024995270201  0.000000000000'
            ])
        NaH.expected_n_ions = 2
        NaH.expected_n_branches = 6
        NaH.expected_n_qpts = 2
        NaH.expected_cell_vec = [[0.000000, 2.399500, 2.399500],
                                 [2.399500, 0.000000, 2.399500],
                                 [2.399500, 2.399500, 0.000000]]
        NaH.expected_ion_pos = [[0.500000, 0.500000, 0.500000],
                                 [0.000000, 0.000000, 0.000000]]
        NaH.expected_ion_type = ['H', 'Na']
        NaH.expected_qpts = [[-0.250000, -0.250000, -0.250000],
                             [-0.250000, -0.500000, -0.500000]]
        NaH.expected_weights = [0.125, 0.375]
        NaH.expected_freqs = [[91.847109, 91.847109, 166.053018,
                               564.508299, 564.508299, 884.068976],
                              [132.031513, 154.825631, 206.213940,
                               642.513551, 690.303338, 832.120011]]
        NaH.expected_eigenvecs = [[[-0.061613336996 - 0.060761142686*1j,
                                    -0.005526816216 - 0.006379010526*1j,
                                     0.067140153211 + 0.067140153211*1j],
                                   [ 0.666530886823 - 0.004641603630*1j,
                                     0.064846864124 + 0.004641603630*1j,
                                    -0.731377750947 + 0.000000000000*1j],
                                   [-0.043088481348 - 0.041294487960*1j,
                                     0.074981829953 + 0.073187836565*1j,
                                    -0.031893348605 - 0.031893348605*1j],
                                   [ 0.459604449490 - 0.009771253020*1j,
                                    -0.807028225834 + 0.009771253020*1j,
                                     0.347423776344 + 0.000000000000*1j],
                                   [-0.062303354995 - 0.062303354995*1j,
                                    -0.062303354995 - 0.062303354995*1j,
                                    -0.062303354995 - 0.062303354995*1j],
                                   [ 0.570587344099 - 0.000000000000*1j,
                                     0.570587344099 - 0.000000000000*1j,
                                     0.570587344099 + 0.000000000000*1j],
                                   [ 0.286272749085 + 0.286272749085*1j,
                                     0.286272749085 + 0.286272749085*1j,
                                    -0.572545498170 - 0.572545498170*1j],
                                   [ 0.052559422840 - 0.000000000000*1j,
                                     0.052559422840 + 0.000000000000*1j,
                                    -0.105118845679 + 0.000000000000*1j],
                                   [-0.459591797004 + 0.529611084985*1j,
                                     0.459591797004 - 0.529611084985*1j,
                                     0.000000000000 - 0.000000000000*1j],
                                   [ 0.006427739587 + 0.090808385909*1j,
                                    -0.006427739587 - 0.090808385909*1j,
                                     0.000000000000 + 0.000000000000*1j],
                                   [-0.403466180272 - 0.403466180272*1j,
                                    -0.403466180272 - 0.403466180272*1j,
                                    -0.403466180272 - 0.403466180272*1j],
                                   [-0.088110249616 - 0.000000000000*1j,
                                    -0.088110249616 - 0.000000000000*1j,
                                    -0.088110249616 + 0.000000000000*1j]],
                                  [[ 0.000000000000 + 0.000000000000*1j,
                                     0.031866260273 - 0.031866260273*1j,
                                    -0.031866260273 + 0.031866260273*1j],
                                   [-0.000000000000 - 0.000000000000*1j,
                                    -0.705669244698 + 0.000000000000*1j,
                                     0.705669244698 + 0.000000000000*1j],
                                   [-0.001780156891 + 0.001780156891*1j,
                                    -0.012680513033 + 0.012680513033*1j,
                                    -0.012680513033 + 0.012680513033*1j],
                                   [-0.582237273385 + 0.000000000000*1j,
                                     0.574608665929 - 0.000000000000*1j,
                                     0.574608665929 + 0.000000000000*1j],
                                   [-0.021184502078 + 0.021184502078*1j,
                                    -0.011544287510 + 0.011544287510*1j,
                                    -0.011544287510 + 0.011544287510*1j],
                                   [ 0.812686635458 - 0.000000000000*1j,
                                     0.411162853378 + 0.000000000000*1j,
                                     0.411162853378 + 0.000000000000*1j],
                                   [ 0.000000000000 + 0.000000000000*1j,
                                    -0.498983508201 + 0.498983508201*1j,
                                     0.498983508201 - 0.498983508201*1j],
                                   [ 0.000000000000 + 0.000000000000*1j,
                                    -0.045065697460 - 0.000000000000*1j,
                                     0.045065697460 + 0.000000000000*1j],
                                   [ 0.400389305548 - 0.400389305548*1j,
                                    -0.412005183792 + 0.412005183792*1j,
                                    -0.412005183792 + 0.412005183792*1j],
                                   [ 0.009657696420 - 0.000000000000*1j,
                                    -0.012050954709 + 0.000000000000*1j,
                                    -0.012050954709 + 0.000000000000*1j],
                                   [-0.582440084400 + 0.582440084400*1j,
                                    -0.282767859813 + 0.282767859813*1j,
                                    -0.282767859813 + 0.282767859813*1j],
                                   [-0.021140457173 + 0.000000000000*1j,
                                    -0.024995270201 - 0.000000000000*1j,
                                    -0.024995270201 + 0.000000000000*1j]]]
        (NaH.n_ions, NaH.n_branches, NaH.n_qpts, NaH.cell_vec, NaH.ion_pos,
            NaH.ion_type) = disp.read_dot_phonon_header(StringIO(NaH.content))
        (NaH.cell_vec_file, NaH.ion_pos_file, NaH.ion_type_file, NaH.qpts,
            NaH.weights, NaH.freqs, NaH.eigenvecs) = disp.read_dot_phonon(
                StringIO(NaH.content), read_eigenvecs=True)
        self.NaH = NaH

    def test_n_ions_read_nah(self):
        self.assertEqual(self.NaH.n_ions, self.NaH.expected_n_ions)

    def test_n_branches_read_nah(self):
        self.assertEqual(self.NaH.n_branches, self.NaH.expected_n_branches)

    def test_cell_vec_read_nah(self):
        npt.assert_array_equal(self.NaH.cell_vec, self.NaH.expected_cell_vec)

    def test_ion_pos_read_nah(self):
        npt.assert_array_equal(self.NaH.ion_pos, self.NaH.expected_ion_pos)

    def test_ion_type_read_nah(self):
        npt.assert_array_equal(self.NaH.ion_type, self.NaH.expected_ion_type)

    def test_n_qpts_read_nah(self):
        self.assertEqual(self.NaH.n_qpts, self.NaH.expected_n_qpts)

    def test_qpts_read_nah(self):
        npt.assert_array_equal(self.NaH.qpts, self.NaH.expected_qpts)

    def test_weights_read_nah(self):
        npt.assert_array_equal(self.NaH.weights, self.NaH.expected_weights)

    def test_freqs_read_nah(self):
        npt.assert_array_equal(self.NaH.freqs, self.NaH.expected_freqs)

    def test_eigenvecs_read_nah(self):
        npt.assert_array_equal(self.NaH.eigenvecs, self.NaH.expected_eigenvecs)

    def test_cell_vec_file_read_nah(self):
        npt.assert_array_equal(self.NaH.cell_vec_file, self.NaH.expected_cell_vec)

    def test_ion_pos_file_read_nah(self):
        npt.assert_array_equal(self.NaH.ion_pos_file, self.NaH.expected_ion_pos)

    def test_ion_type_file_read_nah(self):
        npt.assert_array_equal(self.NaH.ion_type_file, self.NaH.expected_ion_type)


class TestReadDotBands(unittest.TestCase):

    def setUp(self):
        # Create trivial function object so attributes can be assigned to it
        iron = lambda:0
        iron.content = '\n'.join([
           u'Number of k-points   2',
            'Number of spin components 2',
            'Number of electrons  4.500     3.500',
            'Number of eigenvalues      6     6',
            'Fermi energies (in atomic units)     0.173319    0.173319',
            'Unit cell vectors',
            '   -2.708355    2.708355    2.708355',
            '    2.708355   -2.708355    2.708355',
            '    2.708355    2.708355   -2.708355',
            'K-point    1 -0.37500000 -0.45833333  0.29166667  0.01388889',
            'Spin component 1',
            '    0.02278248',
            '    0.02644693',
            '    0.12383402',
            '    0.15398152',
            '    0.17125020',
            '    0.43252010',
            'Spin component 2',
            '    0.08112495',
            '    0.08345039',
            '    0.19185076',
            '    0.22763689',
            '    0.24912308',
            '    0.46511567',
            'K-point    2 -0.37500000 -0.37500000  0.29166667  0.01388889',
            'Spin component 1',
            '    0.02760952',
            '    0.02644911',
            '    0.12442671',
            '    0.14597457',
            '    0.16728951',
            '    0.35463529',
            'Spin component 2',
            '    0.08778721',
            '    0.08033338',
            '    0.19288937',
            '    0.21817779',
            '    0.24476910',
            '    0.39214129'
        ])
        iron.expected_fermi = [0.173319, 0.173319]
        iron.expected_cell_vec = [[-2.708355,  2.708355,  2.708355],
                                  [ 2.708355, -2.708355,  2.708355],
                                  [ 2.708355,  2.708355, -2.708355]]
        iron.expected_kpts = [[-0.37500000, -0.45833333,  0.29166667],
                              [-0.37500000, -0.37500000,  0.29166667]]
        iron.expected_weights = [0.01388889, 0.01388889]
        iron.expected_freq_up = [[0.02278248, 0.02644693, 0.12383402,
                                  0.15398152, 0.17125020, 0.43252010],
                                 [0.02760952, 0.02644911, 0.12442671,
                                  0.14597457, 0.16728951, 0.35463529]]
        iron.expected_freq_down = [[0.08112495, 0.08345039, 0.19185076,
                                    0.22763689, 0.24912308, 0.46511567],
                                   [0.08778721, 0.08033338, 0.19288937,
                                    0.21817779, 0.24476910, 0.39214129]]
        (iron.fermi, iron.cell_vec, iron.kpts, iron.weights, iron.freq_up,
            iron.freq_down) = disp.read_dot_bands(
                StringIO(iron.content), False, False, units='hartree')
        self.iron = iron

    def test_freq_up_read_iron(self):
        npt.assert_array_equal(self.iron.freq_up, self.iron.expected_freq_up)

    def test_freq_down_read_iron(self):
        npt.assert_array_equal(self.iron.freq_down, self.iron.expected_freq_down)

    def test_kpts_read_iron(self):
        npt.assert_array_equal(self.iron.kpts, self.iron.expected_kpts)

    def test_fermi_read_iron(self):
        npt.assert_array_equal(self.iron.fermi, self.iron.expected_fermi)

    def test_weights_read_iron(self):
        npt.assert_array_equal(self.iron.weights, self.iron.expected_weights)

    def test_cell_vec_read_iron(self):
        npt.assert_array_equal(self.iron.cell_vec, self.iron.expected_cell_vec)

    def test_up_arg_freq_up_read_iron(self):
        freq_up = disp.read_dot_bands(
            StringIO(self.iron.content), True, False, units='hartree')[4]
        npt.assert_array_equal(freq_up, self.iron.expected_freq_up)

    def test_up_arg_freq_down_read_iron(self):
        freq_down = disp.read_dot_bands(
            StringIO(self.iron.content), True, False, units='hartree')[5]
        self.assertEqual(freq_down.size, 0)

    def test_down_arg_freq_up_read_iron(self):
        freq_up = disp.read_dot_bands(
            StringIO(self.iron.content), False, True, units='hartree')[4]
        self.assertEqual(freq_up.size, 0)

    def test_down_arg_freq_down_read_iron(self):
        freq_down = disp.read_dot_bands(
            StringIO(self.iron.content), False, True, units='hartree')[5]
        npt.assert_array_equal(freq_down, self.iron.expected_freq_down)

    def test_freq_up_cm_units_iron(self):
        freq_up_cm = disp.read_dot_bands(
            StringIO(self.iron.content), units='1/cm')[4]
        expected_freq_up_cm = [[5000.17594, 5804.429679, 27178.423392,
                                33795.034234, 37585.071062, 94927.180782],
                               [6059.588667, 5804.908134, 27308.5038,
                                32037.711995, 36715.800165, 77833.44239]]
        npt.assert_allclose(freq_up_cm, expected_freq_up_cm)

    def test_freq_down_cm_units_iron(self):
        freq_down_cm = disp.read_dot_bands(
            StringIO(self.iron.content), units='1/cm')[5]
        expected_freq_down_cm = [[17804.86686, 18315.2419, 42106.370959,
                                  49960.517927, 54676.191123, 102081.080835],
                                 [19267.063783, 17631.137342, 42334.319485,
                                  47884.485632, 53720.603056, 86065.057157]]
        npt.assert_allclose(freq_down_cm, expected_freq_down_cm)


class TestReorderFreqs(unittest.TestCase):

    def test_reorder_freqs_NaH(self):
        # This test reads eigenvector data from file. Not good practice but
        # testing reordering requires at least 3 q-points and it's too
        # cumbersome to explicity specify all the eigenvectors
        filename = 'test/NaH-reorder-test.phonon'
        with open(filename, 'r') as f:
            (cell_vec, ion_pos, ion_type, qpts, weights, freqs,
                eigenvecs) = disp.read_dot_phonon(f, read_eigenvecs=True)
        expected_reordered_freqs = [[ 91.847109,  91.847109, 166.053018,
                                     564.508299, 564.508299, 884.068976],
                                    [154.825631, 132.031513, 206.21394,
                                     642.513551, 690.303338, 832.120011],
                                    [106.414367, 106.414367, 166.512415,
                                     621.498613, 621.498613, 861.71391],
                                    [-4.05580000e-02, -4.05580000e-02,
                                      1.23103200e+00,  5.30573108e+02,
                                      5.30573108e+02,  8.90673361e+02],
                                    [139.375186, 139.375186, 207.564309,
                                     686.675791, 686.675791, 833.291584],
                                    [123.623059, 152.926351, 196.644517,
                                     586.674239, 692.696132, 841.62725],
                                    [154.308477, 181.239973, 181.239973,
                                     688.50786,  761.918164, 761.918164],
                                    [124.976823, 124.976823, 238.903818,
                                     593.189877, 593.189877, 873.903056]]
        npt.assert_array_equal(disp.reorder_freqs(freqs, qpts, eigenvecs),
                                expected_reordered_freqs)


    def test_reorder_freqs_LZO(self):
        # This test reads eigenvector data from file. Not good practice but
        # testing reordering requires at least 3 q-points and it's too
        # cumbersome to explicity specify all the eigenvectors
        filename = 'test/La2Zr2O7.phonon'
        with open(filename, 'r') as f:
            (cell_vec, ion_pos, ion_type, qpts, weights, freqs,
                eigenvecs) = disp.read_dot_phonon(f, read_eigenvecs=True)

        expected_reordered_freqs =\
        [[65.062447,65.062447,70.408176,76.847761,76.847761,85.664054,
         109.121893,109.121893,117.920003,119.363588,128.637195,128.637195,
         155.905812,155.905812,160.906969,170.885818,172.820917,174.026075,
         178.344487,183.364621,183.364621,199.25343,199.25343,222.992334,
         225.274444,231.641854,253.012884,265.452117,270.044891,272.376357,
         272.376357,275.75891,299.890562,299.890562,315.067652,315.067652,
         319.909059,338.929562,338.929562,339.067304,340.308461,349.793091,
         376.784786,391.288446,391.288446,396.109935,408.179774,408.179774,
         410.991152,421.254131,456.215732,456.215732,503.360953,532.789756,
         532.789756,545.400861,548.704226,552.622463,552.622463,557.488238,
         560.761581,560.761581,618.721858,734.650232,739.200593,739.200593],
        [62.001197,62.001197,67.432601,70.911126,70.911126,87.435181,
         109.893289,109.893289,110.930712,114.6143,129.226412,129.226412,
         150.593502,150.593502,148.065107,165.856823,168.794942,167.154743,
         169.819174,187.349434,187.349434,202.003734,202.003734,221.329787,
         231.797486,228.999412,259.308314,264.453017,279.078288,270.15176,
         270.15176,278.861064,300.349651,300.349651,311.929653,311.929653,
         318.251662,334.967743,334.967743,340.747776,338.357732,356.048074,
         372.658152,395.526156,395.526156,398.356528,406.398552,406.398552,
         407.216469,421.122741,460.527859,460.527859,486.346855,533.694179,
         533.694179,544.93361,549.252501,550.733812,550.733812,558.006939,
         559.641583,559.641583,591.170512,739.589673,738.563124,738.563124],
        [55.889266,55.889266,64.492348,66.375741,66.375741,88.940906,
         109.388591,109.388591,100.956751,109.379914,130.017598,130.017598,
         145.579207,145.579207,134.563651,161.166842,164.427227,159.401681,
         161.563336,190.735683,190.735683,205.550607,205.550607,219.351563,
         238.204625,226.878861,265.686284,263.148071,287.722953,267.983859,
         267.983859,281.041577,299.480498,299.480498,308.176127,308.176127,
         318.101514,332.930623,332.930623,344.002317,335.480119,361.930637,
         368.350971,399.050499,399.050499,399.241143,404.639113,404.639113,
         400.809087,420.335936,465.504468,465.504468,470.205579,534.544778,
         534.544778,544.501022,549.755212,548.80696,548.80696,556.193672,
         558.101279,558.101279,565.776342,741.372005,737.860626,737.860626],
        [46.935517,46.935517,61.690137,63.177342,63.177342,90.180632,
         107.721223,107.721223,86.944159,104.159787,130.879196,130.879196,
         141.295304,141.295304,122.536218,157.146893,160.037586,151.613374,
         153.750028,193.160653,193.160653,209.882364,209.882364,215.936117,
         244.178665,225.432553,272.052764,261.655838,295.533954,265.906764,
         265.906764,282.006965,295.142911,295.142911,307.16826,307.16826,
         319.295877,332.071847,332.071847,348.814514,332.065989,367.152249,
         364.288189,400.773283,400.773283,399.790407,404.068253,404.068253,
         387.165977,418.829125,470.716023,470.716023,460.278318,535.223077,
         535.223077,544.111882,550.193478,547.016352,547.016352,552.362689,
         556.261571,556.261571,543.678775,740.965394,737.162508,737.162508],
        [36.367201,36.367201,59.168434,60.36167,60.36167,91.154677,
         105.37576,105.37576,68.755044,99.446481,131.658334,131.658334,
         138.017877,138.017877,113.14576,153.975056,156.016054,144.576942,
         146.47047,194.581347,194.581347,214.716315,214.716315,210.473211,
         249.235088,224.769091,278.102009,260.171794,302.032435,263.72796,
         263.72796,282.018114,289.408098,289.408098,308.097577,308.097577,
         321.241146,331.659808,331.659808,353.492915,328.675778,371.468173,
         362.406897,399.901709,399.901709,399.179346,405.625572,405.625572,
         368.236337,416.52493,475.665346,475.665346,458.944007,535.667484,
         535.667484,543.78033,550.551048,545.494533,545.494533,547.179463,
         554.338811,554.338811,524.846465,739.380608,736.536495,736.536495],
        [24.785718,24.785718,57.117299,57.830885,57.830885,91.859898,
         103.047316,103.047316,47.456331,95.691927,132.248074,132.248074,
         135.79383,135.79383,106.389552,151.718169,152.772977,138.984268,
         139.88209,195.244028,195.244028,219.466615,219.466615,203.707835,
         252.993107,224.615517,283.248783,258.912028,306.841458,261.246129,
         261.246129,281.584343,284.696598,284.696598,309.37963,309.37963,
         323.205545,331.373295,331.373295,353.088149,326.000428,374.686778,
         367.331006,398.738183,398.738183,398.433921,407.157219,407.157219,
         349.637392,413.438689,479.806857,479.806857,463.608166,535.889622,
         535.889622,543.524255,550.815232,544.325882,544.325882,541.757933,
         552.630089,552.630089,508.677347,737.533584,736.042236,736.042236],
        [12.555025,12.555025,55.757043,55.972359,55.972359,92.288749,
         101.380298,101.380298,24.214202,93.270077,132.593517,132.593517,
         134.540163,134.540163,102.211134,150.378051,150.665566,135.38769,
         134.747421,195.473725,195.473725,223.210107,223.210107,197.85154,
         255.276828,224.659659,286.758828,258.068085,309.776254,258.846604,
         258.846604,281.147316,281.874849,281.874849,310.385381,310.385381,
         324.609898,331.158402,331.158402,351.072968,324.818103,376.67194,
         374.186388,397.950964,397.950964,397.878833,408.114477,408.114477,
         336.37863,410.112489,482.591747,482.591747,471.735469,535.964134,
         535.964134,543.361599,550.977172,543.571634,543.571634,537.566668,
         551.451065,551.451065,494.626062,736.133131,735.72609,735.72609,],
        [-1.96210000e-02,-1.96210000e-02,5.52779270e+01,5.52779270e+01,
          5.52779270e+01,9.24329110e+01,1.00780857e+02,1.00780857e+02,
         -1.96210000e-02,9.24329110e+01,1.32696363e+02,1.32696363e+02,
          1.34147102e+02,1.34147102e+02,1.00780857e+02,1.49934817e+02,
          1.49934817e+02,1.34147102e+02,1.32696363e+02,1.95519690e+02,
          1.95519690e+02,2.24698049e+02,2.24698049e+02,1.95519690e+02,
          2.56039866e+02,2.24698049e+02,2.88011070e+02,2.57771213e+02,
          3.10763767e+02,2.57771213e+02,2.57771213e+02,2.80972846e+02,
          2.80972846e+02,2.80972846e+02,3.10763767e+02,3.10763767e+02,
          3.25114540e+02,3.31073494e+02,3.31073494e+02,3.50234619e+02,
          3.25114540e+02,3.77342620e+02,3.77342620e+02,3.97677533e+02,
          3.97677533e+02,3.97677533e+02,4.08435923e+02,4.08435923e+02,
          3.31073494e+02,4.08435923e+02,4.83578389e+02,4.83578389e+02,
          4.80948578e+02,5.35976810e+02,5.35976810e+02,5.43305729e+02,
          5.51031712e+02,5.43305729e+02,5.43305729e+02,5.35976810e+02,
          5.51031712e+02,5.51031712e+02,4.83578389e+02,7.35617369e+02,
          7.35617369e+02,7.35617369e+02],
        [12.555025,12.555025,55.757043,55.972359,55.972359,92.288749,
         101.380298,101.380298,24.214202,93.270077,132.593517,132.593517,
         134.540163,134.540163,102.211134,150.378051,150.665566,135.38769,
         134.747421,195.473725,195.473725,223.210107,223.210107,197.85154,
         255.276828,224.659659,286.758828,258.068085,309.776254,258.846604,
         258.846604,281.147316,281.874849,281.874849,310.385381,310.385381,
         324.609898,331.158402,331.158402,351.072968,324.818103,376.67194,
         374.186388,397.950964,397.950964,397.878833,408.114477,408.114477,
         336.37863,410.112489,482.591747,482.591747,471.735469,535.964134,
         535.964134,543.361599,550.977172,543.571634,543.571634,537.566668,
         551.451065,551.451065,494.626062,736.133131,735.72609,735.72609,],
        [24.785718,24.785718,57.117299,57.830885,57.830885,91.859898,
         103.047316,103.047316,47.456331,95.691927,132.248074,132.248074,
         135.79383,135.79383,106.389552,151.718169,152.772977,138.984268,
         139.88209,195.244028,195.244028,219.466615,219.466615,203.707835,
         252.993107,224.615517,283.248783,258.912028,306.841458,261.246129,
         261.246129,281.584343,284.696598,284.696598,309.37963,309.37963,
         323.205545,331.373295,331.373295,353.088149,326.000428,374.686778,
         367.331006,398.738183,398.738183,398.433921,407.157219,407.157219,
         349.637392,413.438689,479.806857,479.806857,463.608166,535.889622,
         535.889622,543.524255,550.815232,544.325882,544.325882,541.757933,
         552.630089,552.630089,508.677347,737.533584,736.042236,736.042236]]
        npt.assert_array_equal(disp.reorder_freqs(freqs, qpts, eigenvecs),
                                expected_reordered_freqs)

class TestCalcAbscissa(unittest.TestCase):

    def test_iron(self):
        recip = [[0., 1.15996339, 1.15996339],
                 [1.15996339, 0., 1.15996339],
                 [1.15996339, 1.15996339, 0.]]
        qpts = [[-0.37500000, -0.45833333,  0.29166667],
                [-0.37500000, -0.37500000,  0.29166667],
                [-0.37500000, -0.37500000,  0.37500000],
                [-0.37500000,  0.45833333, -0.20833333],
                [-0.29166667, -0.45833333,  0.20833333],
                [-0.29166667, -0.45833333,  0.29166667],
                [-0.29166667, -0.37500000, -0.29166667],
                [-0.29166667, -0.37500000,  0.04166667],
                [-0.29166667, -0.37500000,  0.12500000],
                [-0.29166667, -0.37500000,  0.29166667]]
        expected_abscissa = [0., 0.13670299, 0.27340598, 1.48844879,
                             2.75618022, 2.89288323, 3.78930474,
                             4.33611674, 4.47281973, 4.74622573]
        npt.assert_allclose(disp.calc_abscissa(qpts, recip),
                            expected_abscissa)


class TestRecipSpaceLabels(unittest.TestCase):

    def setUp(self):
        # Create trivial function object so attributes can be assigned to it
        NaH = lambda:0
        NaH.cell_vec = [[0.0, 2.3995, 2.3995],
                        [2.3995, 0.0, 2.3995],
                        [2.3995, 2.3995, 0.0]]
        NaH.ion_pos = [[0.5, 0.5, 0.5],
                       [0.0, 0.0, 0.0]]
        NaH.ion_type = ['H', 'Na']
        NaH.qpts = np.array([[-0.25, -0.25, -0.25],
                             [-0.25, -0.50, -0.50],
                             [ 0.00, -0.25, -0.25],
                             [ 0.00,  0.00,  0.00],
                             [ 0.00, -0.50, -0.50],
                             [ 0.25,  0.00, -0.25],
                             [ 0.25, -0.50, -0.25],
                             [-0.50, -0.50, -0.50]])
        NaH.expected_labels = ['', '', '', 'X', '', 'W_2', 'L']
        NaH.expected_qpts_with_labels = [0, 1, 2, 4, 5, 6, 7]
        (NaH.labels, NaH.qpts_with_labels) = disp.recip_space_labels(
            NaH.qpts, NaH.cell_vec, NaH.ion_pos, NaH.ion_type)
        self.NaH = NaH

    def test_labels_nah(self):
        npt.assert_equal(self.NaH.labels, self.NaH.expected_labels)

    def test_qpts_with_labels_nah(self):
        npt.assert_equal(self.NaH.qpts_with_labels,
                         self.NaH.expected_qpts_with_labels)


class TestGenericQptLabels(unittest.TestCase):

    def setUp(self):
        self.generic_dict = disp.generic_qpt_labels()

    def test_returns_dict(self):
        self.assertIsInstance(self.generic_dict, dict)

    def test_gamma_point(self):
        key = '0 0 0'
        expected_value = [0., 0., 0.]
        npt.assert_array_equal(self.generic_dict[key], expected_value)

    def test_mixed_point(self):
        key = '5/8 1/3 3/8'
        expected_value = [0.625, 1./3., 0.375]
        npt.assert_allclose(self.generic_dict[key], expected_value)


class TestGetQptLabel(unittest.TestCase):

    def setUp(self):
        # Create trivial function object so attributes can be assigned to it
        NaH = lambda:0
        cell_vec = [[0.0, 2.3995, 2.3995],
                    [2.3995, 0.0, 2.3995],
                    [2.3995, 2.3995, 0.0]]
        ion_pos = [[0.5, 0.5, 0.5],
                   [0.0, 0.0, 0.0]]
        ion_num = [1, 2]
        cell = (cell_vec, ion_pos, ion_num)
        NaH.point_labels = seekpath.get_path(cell)["point_coords"]
        self.NaH = NaH

    def test_gamma_pt_nah(self):
        gamma_pt = [0.0, 0.0, 0.0]
        expected_label = 'GAMMA'
        self.assertEqual(disp.get_qpt_label(gamma_pt, self.NaH.point_labels),
                         expected_label)

    def test_x_pt_nah(self):
        x_pt = [0.0, -0.5, -0.5]
        expected_label = 'X'
        self.assertEqual(disp.get_qpt_label(x_pt, self.NaH.point_labels),
                         expected_label)

    def test_w2_pt_nah(self):
        w2_pt = [0.25, -0.5, -0.25]
        expected_label = 'W_2'
        self.assertEqual(disp.get_qpt_label(w2_pt, self.NaH.point_labels),
                         expected_label)


class TestDirectionChanged(unittest.TestCase):

    def test_direction_changed_nah(self):
        qpts = [[-0.25, -0.25, -0.25],
                [-0.25, -0.50, -0.50],
                [ 0.00, -0.25, -0.25],
                [ 0.00,  0.00,  0.00],
                [ 0.00, -0.50, -0.50],
                [ 0.25,  0.00, -0.25],
                [ 0.25, -0.50, -0.25],
                [-0.50, -0.50, -0.50]]
        expected_direction_changed = [True, True, False, True, True, True]
        npt.assert_equal(disp.direction_changed(qpts),
                         expected_direction_changed)


class TestReciprocalLattice(unittest.TestCase):

    def test_identity(self):
        recip = disp.reciprocal_lattice([[1., 0., 0.],
                                         [0., 1., 0.],
                                         [0., 0., 1.]])
        expected_recip = [[2*math.pi, 0., 0.],
                          [0., 2*math.pi, 0.],
                          [0., 0., 2*math.pi]]
        npt.assert_allclose(recip, expected_recip)

    def test_graphite(self):
        recip = disp.reciprocal_lattice([[ 4.025915, -2.324363,  0.000000],
                                         [-0.000000,  4.648726,  0.000000],
                                         [ 0.000000,  0.000000, 12.850138]])
        expected_recip = [[1.56068503860106, 0., 0.],
                          [0.780342519300529, 1.3515929541082, 0.],
                          [0., 0., 0.488958586061845]]

        npt.assert_allclose(recip, expected_recip)

    def test_iron(self):
        recip = disp.reciprocal_lattice([[-2.708355,  2.708355,  2.708355],
                                         [ 2.708355, -2.708355,  2.708355],
                                         [ 2.708355,  2.708355, -2.708355]])
        expected_recip = [[0., 1.15996339, 1.15996339],
                          [1.15996339, 0., 1.15996339],
                          [1.15996339, 1.15996339, 0.]]
        npt.assert_allclose(recip, expected_recip)
