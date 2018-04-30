import unittest
import math
import numpy.testing as npt
from casteppy import dispersion as disp

class TestDispersion(unittest.TestCase):

    def test_reciprocal_lattice_calc_identity(self):
        recip = disp.reciprocal_lattice([[1., 0., 0.],
                                         [0., 1., 0.],
                                         [0., 0., 1.]])
        expected_recip = [[2*math.pi, 0., 0.],
                          [0., 2*math.pi, 0.],
                          [0., 0., 2*math.pi]]
        npt.assert_allclose(recip, expected_recip)

    def test_reciprocal_lattice_calc_graphite(self):
        recip = disp.reciprocal_lattice([[ 4.025915, -2.324363,  0.000000],
                                         [-0.000000,  4.648726,  0.000000],
                                         [ 0.000000,  0.000000, 12.850138]])
        expected_recip = [[1.560685, 0.000000, 0.000000],
                          [0.780342, 1.351592, 0.000000],
                          [0.000000, 0.000000, 0.488958]]
        npt.assert_allclose(recip, expected_recip)
