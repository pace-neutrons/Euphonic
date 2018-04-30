import unittest
import math
import numpy.testing as npt
from casteppy import dispersion as disp

class TestDispersion(unittest.TestCase):

    def test_reciprocal_lattice_calculation(self):
        recip = disp.reciprocal_lattice([[1, 0, 0],
                                         [0, 1, 0],
                                         [0, 0, 1]])
        expected_recip = [[2*math.pi, 0, 0],
                          [0, 2*math.pi, 0],
                          [0, 0, 2*math.pi]] 
        npt.assert_allclose(recip, expected_recip)
