import unittest
import math
import numpy.testing as npt
from io import StringIO
from pint import UnitRegistry
from casteppy import dispersion as disp

class TestReciprocalLatticeCalculation(unittest.TestCase):

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


class TestBandsFileRead(unittest.TestCase):

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
        iron.expected_freq_up = [[0.02278248, 0.02644693, 0.12383402,
                                  0.15398152, 0.17125020, 0.43252010],
                                 [0.02760952, 0.02644911, 0.12442671,
                                  0.14597457, 0.16728951, 0.35463529]]
        iron.expected_freq_down = [[0.08112495, 0.08345039, 0.19185076,
                                    0.22763689, 0.24912308, 0.46511567],
                                   [0.08778721, 0.08033338, 0.19288937,
                                    0.21817779, 0.24476910, 0.39214129]]
        iron.expected_kpts = [[-0.37500000, -0.45833333,  0.29166667],
                              [-0.37500000, -0.37500000,  0.29166667]]
        iron.expected_fermi = [0.173319, 0.173319]
        iron.expected_weights = [0.01388889, 0.01388889]
        iron.expected_cell = [[-2.708355,  2.708355,  2.708355],
                              [ 2.708355, -2.708355,  2.708355],
                              [ 2.708355,  2.708355, -2.708355]]
        (iron.freq_up, iron.freq_down, iron.kpts, iron.fermi, iron.weights,
            iron.cell) = disp.read_dot_bands(
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

    def test_cell_read_iron(self):
        npt.assert_array_equal(self.iron.cell, self.iron.expected_cell)

    def test_up_arg_freq_up_read_iron(self):
        freq_up = disp.read_dot_bands(
            StringIO(self.iron.content), True, False, units='hartree')[0]
        npt.assert_array_equal(freq_up, self.iron.expected_freq_up)

    def test_up_arg_freq_down_read_iron(self):
        freq_down = disp.read_dot_bands(
            StringIO(self.iron.content), True, False, units='hartree')[1]
        self.assertEqual(freq_down.size, 0)

    def test_down_arg_freq_up_read_iron(self):
        freq_up = disp.read_dot_bands(
            StringIO(self.iron.content), False, True, units='hartree')[0]
        self.assertEqual(freq_up.size, 0)

    def test_down_arg_freq_down_read_iron(self):
        freq_down = disp.read_dot_bands(
            StringIO(self.iron.content), False, True, units='hartree')[1]
        npt.assert_array_equal(freq_down, self.iron.expected_freq_down)

    def test_freq_up_cm_units_iron(self):
        freq_up_cm = disp.read_dot_bands(
            StringIO(self.iron.content), units='1/cm')[0]
        expected_freq_up_cm = [[5000.17594, 5804.429679, 27178.423392,
                                33795.034234, 37585.071062, 94927.180782],
                               [6059.588667, 5804.908134, 27308.5038,
                                32037.711995, 36715.800165, 77833.44239]]
        npt.assert_allclose(freq_up_cm, expected_freq_up_cm)

    def test_freq_down_cm_units_iron(self):
        freq_down_cm = disp.read_dot_bands(
            StringIO(self.iron.content), units='1/cm')[1]
        expected_freq_down_cm = [[17804.86686, 18315.2419, 42106.370959,
                                  49960.517927, 54676.191123, 102081.080835],
                                 [19267.063783, 17631.137342, 42334.319485,
                                  47884.485632, 53720.603056, 86065.057157]]
        npt.assert_allclose(freq_down_cm, expected_freq_down_cm)


class TestAbscissaCalculation(unittest.TestCase):

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


class TestUnitRegistrySetup(unittest.TestCase):

    def test_returns_unit_registry(self):
        self.assertIsInstance(disp.set_up_unit_registry(),
                              type(UnitRegistry()))

    def test_has_rydberg_units(self):
        ureg = disp.set_up_unit_registry()
        test_ev = 1 * ureg.Ry
        test_ev.ito(ureg.eV)
        self.assertEqual(test_ev.magnitude, 13.605693009)
