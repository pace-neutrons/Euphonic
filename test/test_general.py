import unittest
import math
import seekpath
import matplotlib
# Need to set non-interactive backend before importing casteppy to avoid
# DISPLAY requirement when testing plotting functions
matplotlib.use('Agg')
import numpy as np
import numpy.testing as npt
from io import StringIO
from pint import UnitRegistry
from matplotlib import figure
import casteppy.general as cpy


class TestSetUpUnitRegistry(unittest.TestCase):

    def test_returns_unit_registry(self):
        self.assertIsInstance(cpy.set_up_unit_registry(),
                              type(UnitRegistry()))

    def test_has_rydberg_units(self):
        ureg = cpy.set_up_unit_registry()
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
        ureg = UnitRegistry()

        with open(NaH_bands_file, 'r') as f:
            (NaH_bands.cell_vec, NaH_bands.ion_pos, NaH_bands.ion_type,
                NaH_bands.kpts, NaH_bands.weights, NaH_bands.freqs,
                NaH_bands.freq_down, NaH_bands.i_intens, NaH_bands.r_intens,
                NaH_bands.eigenvecs, NaH_bands.fermi) = cpy.read_input_file(
                    f, ureg, units, up, down)

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
        ir = False
        raman = False
        read_eigenvecs = True
        ureg = UnitRegistry()

        with open(NaH_phonon_file, 'r') as f:
            (NaH_phonon.cell_vec, NaH_phonon.ion_pos, NaH_phonon.ion_type,
                NaH_phonon.kpts, NaH_phonon.weights, NaH_phonon.freqs,
                NaH_phonon.freq_down, NaH_phonon.i_intens,
                NaH_phonon.r_intens, NaH_phonon.eigenvecs,
                NaH_phonon.fermi) = cpy.read_input_file(
                    f, ureg, units, up, down, ir, raman, read_eigenvecs)

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
        ureg = UnitRegistry()

        with open(Fe_bands_file, 'r') as f:
            (Fe_bands.cell_vec, Fe_bands.ion_pos, Fe_bands.ion_type,
                Fe_bands.kpts, Fe_bands.weights, Fe_bands.freqs,
                Fe_bands.freq_down, Fe_bands.i_intens, Fe_bands.r_intens,
                Fe_bands.eigenvecs, Fe_bands.fermi) = cpy.read_input_file(
                    f, ureg, units, up, down)

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
        self.ureg = UnitRegistry()
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
            NaH.ion_type) = cpy.read_dot_phonon_header(StringIO(NaH.content))
        (NaH.cell_vec_file, NaH.ion_pos_file, NaH.ion_type_file, NaH.qpts,
            NaH.weights, NaH.freqs, NaH.i_intens, NaH.r_intens,
            NaH.eigenvecs) = cpy.read_dot_phonon(
                StringIO(NaH.content), self.ureg, read_eigenvecs=True)
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
        self.ureg = UnitRegistry()
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
            iron.freq_down) = cpy.read_dot_bands(
                StringIO(iron.content), self.ureg, False, False,
                units='hartree')
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
        freq_up = cpy.read_dot_bands(
            StringIO(self.iron.content), self.ureg, True, False,
            units='hartree')[4]
        npt.assert_array_equal(freq_up, self.iron.expected_freq_up)

    def test_up_arg_freq_down_read_iron(self):
        freq_down = cpy.read_dot_bands(
            StringIO(self.iron.content), self.ureg, True, False,
            units='hartree')[5]
        self.assertEqual(freq_down.size, 0)

    def test_down_arg_freq_up_read_iron(self):
        freq_up = cpy.read_dot_bands(
            StringIO(self.iron.content), self.ureg, False, True,
            units='hartree')[4]
        self.assertEqual(freq_up.size, 0)

    def test_down_arg_freq_down_read_iron(self):
        freq_down = cpy.read_dot_bands(
            StringIO(self.iron.content), self.ureg, False, True,
            units='hartree')[5]
        npt.assert_array_equal(freq_down, self.iron.expected_freq_down)

    def test_freq_up_cm_units_iron(self):
        freq_up_cm = cpy.read_dot_bands(
            StringIO(self.iron.content), self.ureg, units='1/cm')[4]
        expected_freq_up_cm = [[5000.17594, 5804.429679, 27178.423392,
                                33795.034234, 37585.071062, 94927.180782],
                               [6059.588667, 5804.908134, 27308.5038,
                                32037.711995, 36715.800165, 77833.44239]]
        npt.assert_allclose(freq_up_cm, expected_freq_up_cm)

    def test_freq_down_cm_units_iron(self):
        freq_down_cm = cpy.read_dot_bands(
            StringIO(self.iron.content), self.ureg, units='1/cm')[5]
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
        ureg = UnitRegistry()
        filename = 'test/NaH-reorder-test.phonon'
        with open(filename, 'r') as f:
            (cell_vec, ion_pos, ion_type, qpts, weights, freqs, i_intens,
                r_intens, eigenvecs) = cpy.read_dot_phonon(
                    f, ureg, read_eigenvecs=True)
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
        npt.assert_array_equal(cpy.reorder_freqs(freqs, qpts, eigenvecs),
                                expected_reordered_freqs)


    def test_reorder_freqs_LZO(self):
        # This test reads eigenvector data from file. Not good practice but
        # testing reordering requires at least 3 q-points and it's too
        # cumbersome to explicity specify all the eigenvectors
        ureg = UnitRegistry()
        filename = 'test/La2Zr2O7.phonon'
        with open(filename, 'r') as f:
            (cell_vec, ion_pos, ion_type, qpts, weights, freqs, i_intens,
                r_intens, eigenvecs) = cpy.read_dot_phonon(
                    f, ureg, read_eigenvecs=True)

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
        npt.assert_array_equal(cpy.reorder_freqs(freqs, qpts, eigenvecs),
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
        npt.assert_allclose(cpy.calc_abscissa(qpts, recip),
                            expected_abscissa)


class TestCalculateDos(unittest.TestCase):

    def setUp(self):
        # Create trivial function object so attributes can be assigned to it
        iron = lambda:0
        iron.freqs = [[0.61994279, 0.71965754, 3.36969496,
                       4.19005014, 4.6599548, 11.76947015],
                      [0.75129323, 0.71971687, 3.38582288,
                       3.97216995, 4.55217895, 9.65011675],
                      [0.6597925, 0.57831182, 3.93864281,
                       3.93873641, 4.29865403, 9.30585629],
                      [1.91623131, 0.68094255, 2.32844159,
                       4.60528713, 3.54545356, 6.92907803],
                      [1.31309482, 0.83883714, 3.63675863,
                       4.72910328, 2.62995079, 9.91107037],
                      [0.94849194, 0.70136796, 3.60521438,
                       4.76928334, 3.30697109, 9.84672389],
                      [4.09669468, 9.87219429, 2.03982814,
                       4.2007423, 1.66974108, 1.08526588],
                      [4.21879869, 10.03489224, 1.14830785,
                       3.98517346, 1.64518417, 1.969158],
                      [4.44163652, 9.09726117, 1.03032365,
                       3.73380313, 2.10271909, 1.6011567],
                      [4.38372743, 7.62127765, 0.80563191,
                       3.47965126, 3.32296295, 1.06940083]]
        iron.freq_down = [[2.2075221, 2.27080053, 5.22052453,
                           6.19431463, 6.77898358, 12.6564407],
                          [2.38881141, 2.18598238, 5.24878655,
                           5.93691943, 6.66050576, 10.67070688],
                          [2.31515727, 2.03424572, 5.88946115,
                           5.88693158, 6.39890777, 10.28803347],
                          [3.44111831, 8.40982789, 4.00526716,
                           6.7185675,  5.55451977, 1.92028853],
                          [2.82460361, 2.36408714, 5.5877661,
                           6.85656976, 4.39400517, 10.95945204],
                          [2.4933306,  2.2438063,  5.50581874,
                           6.91045619, 5.17588562, 10.86601713],
                          [11.13142744, 6.42438252, 3.42254246,
                           6.29084184, 3.13545502, 2.57103189],
                          [11.23283365, 6.10962655, 2.61564386,
                           6.36839401, 3.21634358, 3.39038677],
                          [10.33929618, 5.76769511, 2.47484074,
                           6.5454329,  3.77890382, 3.08910234],
                          [8.88541321, 5.37048973, 2.08710084,
                           6.46498163, 5.18913947, 2.76060135]]
        iron.weights = [0.01388889, 0.01388889, 0.00347222, 0.02777778,
                        0.01388889, 0.01388889, 0.01388889, 0.01388889,
                        0.02777778, 0.01388889]
        iron.bwidth = 0.05
        iron.gwidth = 0.1
        iron.expected_dos_gauss =\
        [2.20437428e-01, 4.48772892e-01, 7.52538648e-01, 5.32423726e-01,
         3.02050060e-01, 2.13306464e-01, 1.39663554e-01, 1.63610442e-01,
         2.69621331e-01, 4.72982611e-01, 3.91689064e-01, 2.20437459e-01,
         8.23129014e-02, 7.36543547e-02, 1.30734252e-01, 6.52407077e-02,
         8.15883166e-03, 7.68497930e-04, 1.68213446e-02, 1.46787090e-01,
         3.91432231e-01, 3.91432239e-01, 1.46791072e-01, 1.73310220e-02,
         1.70781781e-02, 1.38638237e-01, 3.26452354e-01, 2.69619336e-01,
         1.63096763e-01, 2.69619336e-01, 3.26452354e-01, 1.38638229e-01,
         1.70741962e-02, 1.68213446e-02, 1.30481396e-01, 2.60954828e-01,
         1.30477414e-01, 1.63116672e-02, 7.64516076e-04, 8.15882000e-03,
         6.52387129e-02, 1.30477410e-01, 6.52387051e-02, 8.15483814e-03,
         2.54838692e-04, 1.99092728e-06, 3.88852984e-09, 0.00000000e+00,
         3.79739243e-12, 7.77895839e-09, 3.98574499e-06, 5.11672200e-04,
         1.65665059e-02, 1.38887091e-01, 3.34350362e-01, 3.26452350e-01,
         2.20692302e-01, 1.55708432e-01, 2.77778164e-01, 3.99847888e-01,
         3.34860036e-01, 2.28590311e-01, 2.04382628e-01, 2.69371473e-01,
         1.31118497e-01, 2.87987680e-02, 1.06524545e-01, 2.61213606e-01,
         2.36496263e-01, 1.43474182e-01, 1.63479519e-01, 2.61276359e-01,
         4.01629772e-01, 2.12539443e-01, 6.57503538e-02, 9.86245646e-02,
         2.63258326e-01, 3.34923753e-01, 2.20435974e-01, 2.85678156e-01,
         3.92453576e-01, 2.85423321e-01, 2.12280638e-01, 2.69621327e-01,
         1.30736235e-01, 1.63116750e-02, 5.09681272e-04, 3.98185456e-06,
         7.77705969e-09, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 3.79739243e-12, 7.77705969e-09, 3.98185456e-06,
         5.09677384e-04, 1.63096763e-02, 1.30477410e-01, 2.60954821e-01,
         1.30477410e-01, 1.63096763e-02, 5.09677384e-04, 3.98185456e-06,
         7.77705969e-09, 0.00000000e+00, 1.89869621e-12, 3.88852984e-09,
         1.99092728e-06, 2.54838692e-04, 8.15483814e-03, 6.52387051e-02,
         1.30477410e-01, 6.52387051e-02, 8.15483814e-03, 2.54838692e-04,
         1.99092728e-06, 3.88852984e-09, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         3.79739243e-12, 7.77705969e-09, 3.98185456e-06, 5.09677384e-04,
         1.63096763e-02, 1.30477411e-01, 2.60955318e-01, 1.30541120e-01,
         1.83483843e-02, 1.68193419e-02, 3.26233109e-02, 1.63096723e-02,
         2.03871196e-03, 6.57005544e-05, 2.55336423e-04, 8.15483912e-03,
         6.52387129e-02, 1.30481396e-01, 6.57503734e-02, 2.47193531e-02,
         1.38887091e-01, 3.26197507e-01, 2.61209663e-01, 8.97032196e-02,
         7.39032207e-02, 1.30736231e-01, 6.52407038e-02, 8.15484203e-03,
         2.54838692e-04, 1.99092728e-06, 3.88852984e-09, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 1.89869621e-12, 3.88852984e-09, 1.99092728e-06,
         2.54838692e-04, 8.15483814e-03, 6.52387051e-02, 1.30477410e-01,
         6.52387051e-02, 8.15483814e-03, 2.54838692e-04, 1.99092728e-06,
         3.88852984e-09, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00]
        iron.expected_dos_down_gauss =\
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         3.79739243e-12, 7.77705969e-09, 3.98185456e-06, 5.09677384e-04,
         1.63096773e-02, 1.30477912e-01, 2.61020521e-01, 1.32770965e-01,
         4.07781686e-02, 9.88813718e-02, 1.63610912e-01, 2.14130212e-01,
         4.01946324e-01, 4.16410894e-01, 2.53313623e-01, 2.45411647e-01,
         3.37157561e-01, 4.08064445e-01, 3.42758550e-01, 2.77523322e-01,
         2.04639455e-01, 8.20620445e-02, 8.18052188e-02, 1.95972949e-01,
         1.95718110e-01, 7.33935549e-02, 8.41366258e-03, 7.68501819e-04,
         1.65685007e-02, 1.38887091e-01, 3.34348364e-01, 3.26193533e-01,
         2.12029781e-01, 7.44168799e-02, 2.52330124e-02, 1.47043924e-01,
         3.91434226e-01, 3.91432235e-01, 1.46787087e-01, 1.68193614e-02,
         5.17641093e-04, 5.13667015e-04, 1.63096841e-02, 1.30477418e-01,
         2.60958802e-01, 1.30987088e-01, 3.26193526e-02, 1.30987088e-01,
         2.60958802e-01, 1.30477418e-01, 1.63096763e-02, 5.09681272e-04,
         5.97278184e-06, 2.54846469e-04, 8.15483814e-03, 6.52387051e-02,
         1.30477410e-01, 6.52387051e-02, 8.15483814e-03, 2.54838692e-04,
         1.99092728e-06, 3.88852984e-09, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 1.89869621e-12, 3.89232724e-09, 1.99870624e-06,
         2.58824435e-04, 8.66650645e-03, 8.18032240e-02, 2.69111650e-01,
         3.91687069e-01, 2.77264501e-01, 1.47043924e-01, 1.39402741e-01,
         7.41640360e-02, 8.99600570e-02, 2.69364509e-01, 3.91438204e-01,
         2.69619340e-01, 9.81128983e-02, 1.39144916e-01, 2.61343055e-01,
         1.34811664e-01, 5.70838513e-02, 1.30989032e-01, 1.63355564e-01,
         7.74729581e-02, 7.37758051e-02, 1.38890077e-01, 1.30736238e-01,
         1.47044422e-01, 1.39207626e-01, 2.14321337e-01, 2.85678148e-01,
         3.02240660e-01, 2.36747136e-01, 2.06425312e-01, 2.78094721e-01,
         2.12285118e-01, 2.77523326e-01, 3.35114875e-01, 2.12284612e-01,
         2.20435476e-01, 2.61466489e-01, 2.03874939e-01, 7.36483897e-02,
         8.41166776e-03, 2.56833508e-04, 1.99481581e-06, 3.88852984e-09,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 3.79739243e-12, 7.77705969e-09,
         3.98185456e-06, 5.09677384e-04, 1.63096763e-02, 1.30477410e-01,
         2.60954821e-01, 1.30477410e-01, 1.63096763e-02, 5.09677384e-04,
         3.98185646e-06, 1.16655895e-08, 1.99092728e-06, 2.54838692e-04,
         8.15483814e-03, 6.52387051e-02, 1.30477410e-01, 6.52387051e-02,
         8.15483814e-03, 2.54838692e-04, 1.99092728e-06, 3.88852984e-09,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         4.74673712e-13, 9.75929154e-10, 5.05508521e-07, 6.76914817e-05,
         2.54838545e-03, 3.26193408e-02, 1.63096739e-01, 2.77264485e-01,
         1.32516122e-01, 1.63753768e-02, 7.65013807e-04, 8.15882097e-03,
         6.52387168e-02, 1.30479401e-01, 6.54935477e-02, 1.63116672e-02,
         6.57483825e-02, 1.38634239e-01, 1.30477418e-01, 1.38634239e-01,
         6.57483864e-02, 1.63136581e-02, 6.57483864e-02, 1.38634239e-01,
         1.30477414e-01, 1.38632248e-01, 6.54935438e-02, 8.15682907e-03,
         2.54842580e-04, 1.99092728e-06, 3.88852984e-09, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.89869621e-12,
         3.88852984e-09, 1.99092728e-06, 2.54838692e-04, 8.15483814e-03,
         6.52387051e-02, 1.30477410e-01]
        iron.expected_dos_lorentz =\
        [2.36213697e-01, 3.40960260e-01, 5.59791268e-01, 4.05306050e-01,
         3.00598745e-01, 2.43092104e-01, 1.84501806e-01, 2.12440899e-01,
         2.41422709e-01, 3.74553522e-01, 3.00602608e-01, 2.18331410e-01,
         1.26875982e-01, 1.06125833e-01, 1.33009500e-01, 8.27211557e-02,
         5.67620854e-02, 5.60551098e-02, 7.60557996e-02, 1.44770764e-01,
         2.86200343e-01, 2.88029390e-01, 1.51215080e-01, 9.03030819e-02,
         8.84900554e-02, 1.43118025e-01, 2.59738838e-01, 2.28740492e-01,
         1.74253600e-01, 2.27089806e-01, 2.56264802e-01, 1.37057751e-01,
         7.82387532e-02, 7.22717763e-02, 1.14916280e-01, 1.98181334e-01,
         1.07339937e-01, 5.39137068e-02, 3.84507277e-02, 3.90891380e-02,
         6.14510617e-02, 1.03358784e-01, 5.78728658e-02, 3.07348683e-02,
         2.17869020e-02, 1.84978042e-02, 1.75391620e-02, 1.79809140e-02,
         1.96814765e-02, 2.29887110e-02, 2.89082558e-02, 4.02850972e-02,
         6.52972607e-02, 1.34070480e-01, 2.63217893e-01, 2.55190061e-01,
         2.22274470e-01, 1.85256660e-01, 2.55123153e-01, 3.23141606e-01,
         2.80154419e-01, 2.33922425e-01, 1.95579381e-01, 2.43000590e-01,
         1.45495058e-01, 1.03245617e-01, 1.33509038e-01, 2.23492540e-01,
         2.11145578e-01, 1.62820603e-01, 1.93941715e-01, 2.17413233e-01,
         3.23354158e-01, 1.93497505e-01, 1.37824072e-01, 1.45823628e-01,
         2.32192550e-01, 2.86377875e-01, 2.16187833e-01, 2.60758613e-01,
         3.15208920e-01, 2.51521079e-01, 1.92696540e-01, 2.29726552e-01,
         1.21024085e-01, 5.79789945e-02, 3.46176173e-02, 2.37294585e-02,
         1.76658441e-02, 1.38712402e-02, 1.13365964e-02, 9.46142550e-03,
         8.08900071e-03, 7.03126965e-03, 6.19545426e-03, 5.52169446e-03,
         4.96976375e-03, 4.51171417e-03, 4.12758862e-03, 3.80281147e-03,
         3.52654240e-03, 3.29060629e-03, 3.08878053e-03, 2.91631187e-03,
         2.69887842e-03, 2.54258423e-03, 2.40860930e-03, 2.33097726e-03,
         2.23691766e-03, 2.12783765e-03, 2.07647651e-03, 2.04752802e-03,
         2.07890953e-03, 2.06717075e-03, 2.16253029e-03, 2.30037279e-03,
         2.49420576e-03, 2.71171131e-03, 3.05843646e-03, 3.60532593e-03,
         4.44493556e-03, 5.68551364e-03, 7.60565706e-03, 1.12217284e-02,
         1.85265449e-02, 3.62439128e-02, 8.93055286e-02, 1.77715984e-01,
         8.93806665e-02, 3.64040780e-02, 1.87980192e-02, 1.17207806e-02,
         8.40712866e-03, 6.72839778e-03, 6.11665105e-03, 6.39258212e-03,
         7.67847107e-03, 1.09303720e-02, 1.94891231e-02, 4.58401335e-02,
         8.98934531e-02, 4.55661961e-02, 1.89525340e-02, 1.00461964e-02,
         6.36008876e-03, 4.53045250e-03, 3.50418856e-03, 2.88031312e-03,
         2.48161009e-03, 2.22067560e-03, 2.05082161e-03, 1.94572894e-03,
         1.89018541e-03, 1.87557669e-03, 1.89761743e-03, 1.95525225e-03,
         2.05026174e-03, 2.18740247e-03, 2.37509553e-03, 2.62686080e-03,
         2.96397172e-03, 3.42034456e-03, 4.05186639e-03, 4.95524738e-03,
         6.30914282e-03, 8.47301283e-03, 1.22562202e-02, 1.97770682e-02,
         3.77849753e-02, 9.12980790e-02, 1.80428367e-01, 9.32495061e-02,
         4.28345215e-02, 3.23803443e-02, 3.69192859e-02, 2.33552857e-02,
         1.63350706e-02, 1.54633814e-02, 1.85894814e-02, 2.82176395e-02,
         5.69305330e-02, 1.05577010e-01, 7.05254413e-02, 6.56961296e-02,
         1.19566254e-01, 2.32519327e-01, 1.90028371e-01, 1.00499075e-01,
         8.21126732e-02, 1.09723491e-01, 5.78987561e-02, 2.73020731e-02,
         1.60177253e-02, 1.07927734e-02, 7.90434641e-03, 6.11284612e-03,
         4.91247225e-03, 4.06322774e-03, 3.43802517e-03, 2.96384639e-03,
         2.59605395e-03, 2.30602449e-03, 2.07472037e-03, 1.88913924e-03,
         1.74025907e-03, 1.62180973e-03, 1.52952958e-03, 1.46072816e-03,
         1.41406280e-03, 1.38949045e-03, 1.38839771e-03, 1.41395751e-03,
         1.47182970e-03, 1.57144956e-03, 1.72841012e-03, 1.96903223e-03,
         2.33964680e-03, 2.92692643e-03, 3.90692971e-03, 5.67896039e-03,
         9.22305166e-03, 1.80438414e-02, 4.45502384e-02, 8.87420594e-02,
         4.45070028e-02, 1.79662533e-02, 9.11048774e-03, 5.45685864e-03,
         3.64453449e-03, 2.62239101e-03, 1.99069950e-03, 1.53756987e-03,
         1.24786384e-03, 1.03781587e-03, 8.80373037e-04, 6.88366229e-04,
         5.60123022e-04, 4.87189378e-04, 4.28047121e-04, 3.44044435e-04,
         3.04894551e-04, 2.72059754e-04]
        iron.expected_dos_down_lorentz =\
        [0.0013475,  0.00146787, 0.00159829, 0.00170474, 0.00182271,
         0.00195399, 0.00217147, 0.00241002, 0.00260242, 0.00282077,
         0.00307032, 0.0033578, 0.00369198, 0.00408453, 0.00462199,
         0.00518793, 0.00588205, 0.00675157, 0.00794075, 0.00943199,
         0.01150628, 0.01457372, 0.0195215,  0.02859401, 0.04875881,
         0.105446,   0.19968201, 0.12170387, 0.09154094, 0.12484921,
         0.17948006, 0.20193574, 0.3240029,  0.33821893, 0.24954929,
         0.26075907, 0.29592108, 0.33989225, 0.29051919, 0.25064571,
         0.20443418, 0.12634374, 0.11741981, 0.17151742, 0.16472999,
         0.09227176, 0.05935657, 0.0554398, 0.07256366, 0.13562152,
         0.25907784, 0.24308362, 0.19462211, 0.11651152, 0.09850056,
         0.15311752, 0.28742319, 0.28408118, 0.14141214, 0.07127905,
         0.04923873, 0.04553691, 0.05820503, 0.10952556, 0.19917575,
         0.11646775, 0.08006111, 0.11475898, 0.19549973, 0.10344664,
         0.04861393, 0.03045644, 0.02403997, 0.02340451, 0.02981211,
         0.05485597, 0.0980809,  0.05330216, 0.02648495, 0.01757166,
         0.01404505, 0.01261522, 0.0121577,  0.01241502, 0.01339631,
         0.01514719, 0.01807406, 0.02300105, 0.03208878, 0.05128866,
         0.10110371, 0.21041985, 0.2868535,  0.22804927, 0.1535098,
         0.15898331, 0.11597735, 0.13079622, 0.22629805, 0.29521889,
         0.22940054, 0.14053834, 0.14700392, 0.22122407, 0.13342144,
         0.09972378, 0.13109371, 0.14992302, 0.10012457, 0.09932044,
         0.14448793, 0.12925554, 0.16084467, 0.14957955, 0.20665545,
         0.24246127, 0.25827098, 0.23121046, 0.20098413, 0.26324916,
         0.20533169, 0.24776663, 0.28393675, 0.19780771, 0.20992422,
         0.21170163, 0.17377596, 0.08797637, 0.04513846, 0.0283027,
         0.02004941, 0.01529163, 0.01224536, 0.01015221, 0.00864108,
         0.00751084, 0.00664392, 0.0059676,  0.00543508, 0.00501547,
         0.00468831, 0.00440512, 0.0041598,  0.00402182, 0.004001,
         0.00408715, 0.0042393,  0.00451851, 0.00494137, 0.00560175,
         0.00674536, 0.00875912, 0.01233736, 0.01954766, 0.03725946,
         0.09040613, 0.17891059, 0.09065215, 0.03784843, 0.0205432,
         0.01382251, 0.01124858, 0.01097745, 0.01340217, 0.02138911,
         0.04731746, 0.09108457, 0.04655124, 0.01981425, 0.01074552,
         0.00699312, 0.0050785,  0.00394609, 0.00329972, 0.00285159,
         0.00255002, 0.00234778, 0.00225378, 0.00221641, 0.0022279,
         0.00228543, 0.00239015, 0.00254718, 0.00276628, 0.00306359,
         0.00346482, 0.00401131, 0.00477179, 0.00586637, 0.00752012,
         0.01019561, 0.01500473, 0.02484254, 0.04962022, 0.11436168,
         0.19264527, 0.09899253, 0.0460507, 0.03181838, 0.03397179,
         0.05824292, 0.10293166, 0.06234208, 0.04600056, 0.06731815,
         0.11718641, 0.0987186,  0.11756247, 0.06834638, 0.04825988,
         0.06737867, 0.11547569, 0.09513789, 0.11122093, 0.05712155,
         0.02622802, 0.01502387, 0.00997969, 0.00725769, 0.00560479,
         0.00451746, 0.0037611, 0.00321387, 0.00280723, 0.00250029,
         0.00226771, 0.00209352, 0.00196792, 0.00188558, 0.00184495,
         0.00184827, 0.00190246, 0.00202124, 0.00222951, 0.00257251,
         0.00313605, 0.00409552, 0.0058497,  0.00944888, 0.0182532,
         0.0447449,  0.08892355]

        self.iron = iron


    def test_dos_iron_gauss_both(self):
        iron = self.iron
        dos = cpy.calculate_dos(
            iron.freqs, iron.freq_down, iron.weights, iron.bwidth,
            iron.gwidth, lorentz=False)[0]
        npt.assert_allclose(dos, iron.expected_dos_gauss)

    def test_dos_iron_gauss_both_ir(self):
        iron = self.iron
        ir_factor = 2.0
        ir = np.full(np.array(iron.freqs).shape, ir_factor)
        dos = cpy.calculate_dos(
            iron.freqs, iron.freq_down, iron.weights, iron.bwidth,
            iron.gwidth, lorentz=False, intensities=ir)[0]
        npt.assert_allclose(dos/ir_factor, iron.expected_dos_gauss)

    def test_dos_iron_gauss_down(self):
        iron = self.iron
        dos = cpy.calculate_dos(
            [], iron.freq_down, iron.weights, iron.bwidth,
            iron.gwidth, lorentz=False)[0]
        self.assertEqual(dos.size, 0)

    def test_dos_down_iron_gauss_both(self):
        iron = self.iron
        dos_down = cpy.calculate_dos(
            iron.freqs, iron.freq_down, iron.weights, iron.bwidth,
            iron.gwidth, lorentz=False)[1]
        npt.assert_allclose(dos_down, iron.expected_dos_down_gauss)

    def test_dos_down_iron_gauss_both_ir(self):
        iron = self.iron
        ir_factor = 2.0
        ir = np.full(np.array(iron.freq_down).shape, ir_factor)
        dos_down = cpy.calculate_dos(
            iron.freqs, iron.freq_down, iron.weights, iron.bwidth,
            iron.gwidth, lorentz=False, intensities=ir)[1]
        npt.assert_allclose(dos_down/ir_factor, iron.expected_dos_down_gauss)

    def test_dos_down_iron_gauss_up(self):
        iron = self.iron
        dos_down = cpy.calculate_dos(
            iron.freqs, [], iron.weights, iron.bwidth,
            iron.gwidth, lorentz=False)[1]
        self.assertEqual(dos_down.size, 0)

    def test_dos_iron_lorentz_both(self):
        iron = self.iron
        dos = cpy.calculate_dos(
            iron.freqs, iron.freq_down, iron.weights, iron.bwidth,
            iron.gwidth, lorentz=True)[0]
        npt.assert_allclose(dos, iron.expected_dos_lorentz)

    def test_dos_iron_lorentz_both_ir(self):
        iron = self.iron
        ir_factor = 2.0
        ir = np.full(np.array(iron.freq_down).shape, ir_factor)
        dos = cpy.calculate_dos(
            iron.freqs, iron.freq_down, iron.weights, iron.bwidth,
            iron.gwidth, lorentz=True, intensities=ir)[0]
        npt.assert_allclose(dos/ir_factor, iron.expected_dos_lorentz)

    def test_dos_iron_lorentz_down(self):
        iron = self.iron
        dos = cpy.calculate_dos(
            [], iron.freq_down, iron.weights, iron.bwidth,
            iron.gwidth, lorentz=True)[0]
        self.assertEqual(dos.size, 0)

    def test_dos_down_iron_lorentz_both(self):
        iron = self.iron
        dos_down = cpy.calculate_dos(
            iron.freqs, iron.freq_down, iron.weights, iron.bwidth,
            iron.gwidth, lorentz=True)[1]
        npt.assert_allclose(
            dos_down, iron.expected_dos_down_lorentz, atol=1e-8)

    def test_dos_down_iron_lorentz_both_ir(self):
        iron = self.iron
        ir_factor = 2.0
        ir = np.full(np.array(iron.freq_down).shape, ir_factor)
        dos_down = cpy.calculate_dos(
            iron.freqs, iron.freq_down, iron.weights, iron.bwidth,
            iron.gwidth, lorentz=True, intensities=ir)[1]
        npt.assert_allclose(
            dos_down/ir_factor, iron.expected_dos_down_lorentz, atol=1e-8)

    def test_dos_down_iron_lorentz_up(self):
        iron = self.iron
        dos_down = cpy.calculate_dos(
            iron.freqs, [], iron.weights, iron.bwidth,
            iron.gwidth, lorentz=True)[1]
        self.assertEqual(dos_down.size, 0)


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
        (NaH.labels, NaH.qpts_with_labels) = cpy.recip_space_labels(
            NaH.qpts, NaH.cell_vec, NaH.ion_pos, NaH.ion_type)
        self.NaH = NaH

    def test_labels_nah(self):
        npt.assert_equal(self.NaH.labels, self.NaH.expected_labels)

    def test_qpts_with_labels_nah(self):
        npt.assert_equal(self.NaH.qpts_with_labels,
                         self.NaH.expected_qpts_with_labels)


class TestGenericQptLabels(unittest.TestCase):

    def setUp(self):
        self.generic_dict = cpy.generic_qpt_labels()

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
        self.assertEqual(cpy.get_qpt_label(gamma_pt, self.NaH.point_labels),
                         expected_label)

    def test_x_pt_nah(self):
        x_pt = [0.0, -0.5, -0.5]
        expected_label = 'X'
        self.assertEqual(cpy.get_qpt_label(x_pt, self.NaH.point_labels),
                         expected_label)

    def test_w2_pt_nah(self):
        w2_pt = [0.25, -0.5, -0.25]
        expected_label = 'W_2'
        self.assertEqual(cpy.get_qpt_label(w2_pt, self.NaH.point_labels),
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
        npt.assert_equal(cpy.direction_changed(qpts),
                         expected_direction_changed)


class TestReciprocalLattice(unittest.TestCase):

    def test_identity(self):
        recip = cpy.reciprocal_lattice([[1., 0., 0.],
                                         [0., 1., 0.],
                                         [0., 0., 1.]])
        expected_recip = [[2*math.pi, 0., 0.],
                          [0., 2*math.pi, 0.],
                          [0., 0., 2*math.pi]]
        npt.assert_allclose(recip, expected_recip)

    def test_graphite(self):
        recip = cpy.reciprocal_lattice([[ 4.025915, -2.324363,  0.000000],
                                         [-0.000000,  4.648726,  0.000000],
                                         [ 0.000000,  0.000000, 12.850138]])
        expected_recip = [[1.56068503860106, 0., 0.],
                          [0.780342519300529, 1.3515929541082, 0.],
                          [0., 0., 0.488958586061845]]

        npt.assert_allclose(recip, expected_recip)

    def test_iron(self):
        recip = cpy.reciprocal_lattice([[-2.708355,  2.708355,  2.708355],
                                         [ 2.708355, -2.708355,  2.708355],
                                         [ 2.708355,  2.708355, -2.708355]])
        expected_recip = [[0., 1.15996339, 1.15996339],
                          [1.15996339, 0., 1.15996339],
                          [1.15996339, 1.15996339, 0.]]
        npt.assert_allclose(recip, expected_recip)

class TestPlotDispersion(unittest.TestCase):

    def setUp(self):
        # Input values
        self.abscissa = [0.0, 0.13, 0.27, 1.48, 2.75, 2.89]
        self.freq_up = [[0.61, 0.71, 3.36, 4.19, 4.65, 11.76],
                        [0.75, 0.71, 3.38, 3.97, 4.55, 9.65],
                        [0.65, 0.57, 3.93, 3.93, 4.29, 9.30],
                        [1.91, 0.68, 2.32, 4.60, 3.54, 6.92],
                        [1.31, 0.83, 3.63, 4.72, 2.62, 9.91],
                        [0.94, 0.70, 3.60, 4.76, 3.30, 9.84]]
        self.freq_down = [[2.20, 2.27, 5.22, 6.19, 6.77, 12.65],
                          [2.38, 2.18, 5.24, 5.93, 6.66, 10.67],
                          [2.31, 2.03, 5.88, 5.88, 6.39, 10.28],
                          [3.44, 8.40, 4.00, 6.71, 5.55, 1.92],
                          [2.82, 2.36, 5.58, 6.85, 4.39, 10.95],
                          [2.49, 2.24, 5.50, 6.91, 5.17, 10.86]]
        self.units = 'eV'
        self.title = 'Iron'
        self.xticks = [0.0, 0.13, 0.27, 1.48, 2.75, 2.89]
        self.xlabels = ['', '', '5/8 5/8 3/8', '', '', '']
        self.fermi = [4.71, 4.71]

        # Results
        self.fig = cpy.plot_dispersion(
            self.abscissa, self.freq_up, self.freq_down, self.units,
            self.title, self.xticks, self.xlabels, self.fermi)
        self.ax = self.fig.axes[0]

    def test_returns_fig(self):
        self.assertIsInstance(self.fig, figure.Figure)

    def test_n_series(self):
        n_series = (len(self.freq_up[0])
                  + len(self.freq_down[0])
                  + len(self.fermi))
        self.assertEqual(len(self.ax.get_lines()), n_series)

    def test_freq_xaxis(self):
        n_correct_x = 0
        for line in self.ax.get_lines():
            if np.array_equal(line.get_data()[0], self.abscissa):
                n_correct_x += 1
        # Check that there are as many lines with abscissa for the x-axis
        # values as there are both freq_up and freq_down branches
        self.assertEqual(n_correct_x,
                         len(self.freq_up[0]) + len(self.freq_down[0]))

    def test_freq_yaxis(self):
        all_freq_branches = np.vstack((np.transpose(self.freq_up),
                                       np.transpose(self.freq_down)))
        n_correct_y = 0
        for branch in all_freq_branches:
            for line in self.ax.get_lines():
                if np.array_equal(line.get_data()[1], branch):
                    n_correct_y += 1
                    break
        # Check that every branch has a matching y-axis line
        self.assertEqual(n_correct_y, len(all_freq_branches))

    def test_fermi_yaxis(self):
        n_correct_y = 0
        for ef in self.fermi:
            for line in self.ax.get_lines():
                if np.all(np.array(line.get_data()[1]) == ef):
                    n_correct_y += 1
                    break
        self.assertEqual(n_correct_y, len(self.fermi))

    def test_xaxis_tick_locs(self):
        npt.assert_array_equal(self.ax.get_xticks(), self.xticks)

    def test_xaxis_tick_labels(self):
        ticklabels = [x.get_text() for x in self.ax.get_xticklabels()]
        npt.assert_array_equal(ticklabels, self.xlabels)

    def test_freq_up_empty(self):
        # Test freq down is still plotted when freq_up is empty
        fig = cpy.plot_dispersion(
            self.abscissa, [], self.freq_down, self.units,
            self.title, self.xticks, self.xlabels, self.fermi)
        n_correct_y = 0
        for branch in np.transpose(self.freq_down):
            for line in fig.axes[0].get_lines():
                if np.array_equal(line.get_data()[1], branch):
                    n_correct_y += 1
                    break
        # Check that every freq down branch has a matching y-axis line
        self.assertEqual(n_correct_y, len(self.freq_down[0]))

    def test_freq_down_empty(self):
        # Test freq up is still plotted when freq_down is empty
        fig = cpy.plot_dispersion(
            self.abscissa, self.freq_up, [], self.units,
            self.title, self.xticks, self.xlabels, self.fermi)
        n_correct_y = 0
        for branch in np.transpose(self.freq_up):
            for line in fig.axes[0].get_lines():
                if np.array_equal(line.get_data()[1], branch):
                    n_correct_y += 1
                    break
        # Check that every freq up branch has a matching y-axis line
        self.assertEqual(n_correct_y, len(self.freq_up[0]))

class TestPlotDispersionWithBreak(unittest.TestCase):

    def setUp(self):
        # Input values
        self.abscissa = [1.63, 1.76, 1.88, 2.01, 3.43, 3.55, 3.66, 3.78]
        self.freq_up = [[0.98, 0.98, 2.65, 4.34, 4.34, 12.14],
                        [0.51, 0.51, 3.55, 4.50, 4.50, 12.95],
                        [0.24, 0.24, 4.39, 4.66, 4.66, 13.79],
                        [0.09, 0.09, 4.64, 4.64, 4.64, 14.19],
                        [0.08, 1.62, 3.90, 4.22, 5.01,  5.12],
                        [0.03, 1.65, 3.81, 4.01, 4.43,  4.82],
                        [-0.2, 1.78, 3.10, 3.90, 4.00,  4.43],
                        [-0.9, 1.99, 2.49, 3.76, 3.91,  3.92]]
        self.freq_down = [[2.52, 2.52, 4.29, 6.28, 6.28, 13.14],
                          [2.07, 2.07, 5.35, 6.47, 6.47, 13.77],
                          [1.80, 1.80, 6.34, 6.67, 6.67, 14.40],
                          [1.66, 1.66, 6.67, 6.67, 6.67, 14.68],
                          [1.30, 3.22, 5.51, 6.00, 6.45,  7.07],
                          [1.16, 3.26, 4.98, 5.99, 6.25,  6.86],
                          [0.65, 3.41, 4.22, 5.95, 6.22,  6.41],
                          [-0.35, 3.65, 4.00, 5.82, 5.85, 6.13]]
        self.units = 'eV'
        self.title = 'Iron'
        self.xticks = [0.00, 2.01, 3.43, 4.25, 5.55, 6.13]
        self.xlabels = ['0 0 0', '1/2 1/2 1/2', '1/2 0 0', '0 0 0',
                        '3/4 1/4 3/4', '1/2 0 0']
        self.fermi = [0.17, 0.17]
        #Index at which the abscissa/frequencies are split into subplots
        self.breakpoint = 4

        # Results
        self.fig = cpy.plot_dispersion(
            self.abscissa, self.freq_up, self.freq_down, self.units,
            self.title, self.xticks, self.xlabels, self.fermi)
        self.subplots = self.fig.axes

    def test_returns_fig(self):
        self.assertIsInstance(self.fig, figure.Figure)

    def test_has_2_subplots(self):
        self.assertEqual(len(self.subplots), 2)

    def test_freq_xaxis(self):
        bp = self.breakpoint
        # Check x-axis for each subplot separately
        n_correct_x = np.array([0, 0])
        for line in self.subplots[0].get_lines():
            # Check x-axis for first plot, abscissa[0:4]
            if np.array_equal(line.get_data()[0], self.abscissa[:bp]):
                n_correct_x[0] += 1
        for line in self.subplots[1].get_lines():
            # Check x-axis for second plot, abscissa[4:]
            if np.array_equal(line.get_data()[0], self.abscissa[bp:]):
                n_correct_x[1] += 1
        # Check that there are as many lines with abscissa for the x-axis
        # values as there are both freq_up and freq_down branches
        self.assertTrue(np.all(
            n_correct_x == (len(self.freq_up[0]) + len(self.freq_down[0]))))

    def test_freq_yaxis(self):
        bp = self.breakpoint
        all_freq_branches = np.vstack((np.transpose(self.freq_up),
                                       np.transpose(self.freq_down)))
        # Check y-axis for each subplot separately
        n_correct_y = np.array([0, 0])
        # Subplot 0
        for branch in all_freq_branches[:, :bp]:
            for line in self.subplots[0].get_lines():
                if np.array_equal(line.get_data()[1], branch):
                    n_correct_y[0] += 1
                    break
        # Subplot 1
        for branch in all_freq_branches[:, bp:]:
            for line in self.subplots[1].get_lines():
                if np.array_equal(line.get_data()[1], branch):
                    n_correct_y[1] += 1
                    break
        # Check that every branch has a matching y-axis line
        self.assertTrue(np.all(n_correct_y == len(all_freq_branches)))

    def test_fermi_yaxis(self):
        n_correct_y = np.array([0, 0])
        for ef in self.fermi:
            # Subplot 0
            for line in self.subplots[0].get_lines():
                if np.all(np.array(line.get_data()[1]) == ef):
                    n_correct_y[0] += 1
                    break
            # Subplot 1
            for line in self.subplots[0].get_lines():
                if np.all(np.array(line.get_data()[1]) == ef):
                    n_correct_y[1] += 1
                    break
        self.assertTrue(np.all(n_correct_y == len(self.fermi)))

    def test_xaxis_tick_locs(self):
        for subplot in self.subplots:
            npt.assert_array_equal(subplot.get_xticks(), self.xticks)

    def test_xaxis_tick_labels(self):
        ticklabels = [[xlabel.get_text()
                         for xlabel in subplot.get_xticklabels()]
                         for subplot in self.subplots]
        for i, subplot in enumerate(self.subplots):
            npt.assert_array_equal(ticklabels[i], self.xlabels)


class TestPlotDos(unittest.TestCase):

    def setUp(self):
        # Input values
        self.dos = [2.30e-01, 1.82e-01, 8.35e-02, 3.95e-02, 2.68e-02, 3.89e-02,
                    6.15e-02, 6.75e-02, 6.55e-02, 5.12e-02, 3.60e-02, 2.80e-02,
                    5.22e-02, 1.12e-01, 1.52e-01, 1.37e-01, 9.30e-02, 6.32e-02,
                    7.92e-02, 1.32e-01, 1.53e-01, 8.88e-02, 2.26e-02, 2.43e-03,
                    1.08e-04, 2.00e-06, 8.11e-07, 4.32e-05, 9.63e-04, 8.85e-03,
                    3.35e-02, 5.22e-02, 3.35e-02, 8.85e-03, 9.63e-04, 4.32e-05,
                    7.96e-07, 6.81e-09, 9.96e-08, 5.40e-06, 1.21e-04, 1.13e-03,
                    4.71e-03, 1.19e-02, 2.98e-02, 6.07e-02, 6.91e-02, 3.79e-02,
                    9.33e-03, 9.85e-04, 4.40e-05, 2.24e-05, 4.82e-04, 4.43e-03,
                    1.67e-02, 2.61e-02, 1.67e-02, 4.43e-03, 4.82e-04, 2.16e-05,
                    3.98e-07]
        self.dos_down = [6.05e-09, 7.97e-07, 4.33e-05, 9.71e-04, 9.08e-03,
                         3.72e-02, 8.06e-02, 1.37e-01, 1.84e-01, 1.47e-01,
                         7.37e-02, 3.84e-02, 2.67e-02, 3.80e-02, 5.36e-02,
                         4.24e-02, 4.28e-02, 5.76e-02, 5.03e-02, 3.55e-02,
                         2.32e-02, 3.15e-02, 7.39e-02, 1.24e-01, 1.40e-01,
                         1.11e-01, 7.48e-02, 5.04e-02, 5.22e-02, 8.75e-02,
                         1.37e-01, 1.30e-01, 6.37e-02, 1.47e-02, 1.51e-03,
                         1.09e-04, 9.64e-04, 8.85e-03, 3.35e-02, 5.22e-02,
                         3.35e-02, 8.85e-03, 9.63e-04, 4.33e-05, 6.19e-06,
                         1.21e-04, 1.13e-03, 4.71e-03, 1.19e-02, 2.98e-02,
                         6.07e-02, 6.91e-02, 3.79e-02, 9.33e-03, 9.85e-04,
                         4.40e-05, 2.24e-05, 4.82e-04, 4.43e-03, 1.67e-02,
                         2.61e-02]
        self.bins = [ 0.58,  0.78,  0.98,  1.18,  1.38,  1.58,  1.78,  1.98,
                      2.18,  2.38,  2.58,  2.78,  2.98,  3.18,  3.38,  3.58,
                      3.78,  3.98,  4.18,  4.38,  4.58,  4.78,  4.98,  5.18,
                      5.38,  5.58,  5.78,  5.98,  6.18,  6.38,  6.58,  6.78,
                      6.98,  7.18,  7.38,  7.58,  7.78,  7.98,  8.18,  8.38,
                      8.58,  8.78,  8.98,  9.18,  9.38,  9.58,  9.78,  9.98,
                      10.18, 10.38, 10.58, 10.78, 10.98, 11.18, 11.38, 11.58,
                      11.78, 11.98, 12.18, 12.38, 12.58, 12.78]
        self.units = 'eV'
        self.title = 'Iron'
        self.fermi = [4.71, 4.71]
        self.mirror = False

        # Results
        self.fig = cpy.plot_dos(
            self.dos, self.dos_down, self.bins, self.units,
            self.title, self.fermi, self.mirror)
        self.ax = self.fig.axes[0]

    def test_returns_fig(self):
        self.assertIsInstance(self.fig, figure.Figure)

    def test_n_series(self):
        # 2 series, 1 for dos, 1 for dos_down
        n_series = 2 + len(self.fermi)
        self.assertEqual(len(self.ax.get_lines()), n_series)

    def test_dos_xaxis(self):
        bin_centres = np.array(self.bins[:-1]) + (self.bins[1]
                                                - self.bins[0])/2
        n_correct_x = 0
        for line in self.ax.get_lines():
            if np.array_equal(line.get_data()[0], bin_centres):
                n_correct_x += 1
        # Check there are exactly 2 lines with bin centres for the x-axis
        # (1 for dos, 1 for dos_down)
        self.assertEqual(n_correct_x, 2)

    def test_dos_yaxis(self):
        match = False
        for line in self.ax.get_lines():
            if np.array_equal(line.get_data()[1], self.dos):
                match = True
        self.assertTrue(match)

    def test_dos_down_yaxis(self):
        match = False
        for line in self.ax.get_lines():
            if np.array_equal(line.get_data()[1], self.dos_down):
                match = True
        self.assertTrue(match)

    def test_fermi_xaxis(self):
        n_correct_x = 0
        for ef in self.fermi:
            for line in self.ax.get_lines():
                if np.all(np.array(line.get_data()[0]) == ef):
                    n_correct_x += 1
                    break
        self.assertEqual(n_correct_x, len(self.fermi))

    def test_mirror_true(self):
        fig = cpy.plot_dos(
            self.dos, self.dos_down, self.bins, self.units,
            self.title, self.fermi, mirror=True)
        for line in fig.axes[0].get_lines():
            if np.array_equal(line.get_data()[1], np.negative(self.dos_down)):
                match = True
        self.assertTrue(match)

    def test_empty_dos(self):
        # Test that dos_down is still plotted when dos is empty
        fig = cpy.plot_dos(
            [], self.dos_down, self.bins, self.units,
            self.title, self.fermi, self.mirror)
        match = False
        for line in fig.axes[0].get_lines():
            if np.array_equal(line.get_data()[1], self.dos_down):
                match = True
        self.assertTrue(match)

    def test_empty_dos_down(self):
        # Test that dos is still plotted when dos is empty
        fig = cpy.plot_dos(
            self.dos, [], self.bins, self.units,
            self.title, self.fermi, self.mirror)
        match = False
        for line in fig.axes[0].get_lines():
            if np.array_equal(line.get_data()[1], self.dos):
                match = True
        self.assertTrue(match)
