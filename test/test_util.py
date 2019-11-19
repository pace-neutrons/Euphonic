import os
import unittest
import math
import numpy as np
import numpy.testing as npt
from euphonic.util import reciprocal_lattice, direction_changed, mp_grid


class TestReciprocalLattice(unittest.TestCase):

    def test_identity(self):
        recip = reciprocal_lattice([[1., 0., 0.],
                                    [0., 1., 0.],
                                    [0., 0., 1.]])
        expected_recip = [[2*math.pi, 0., 0.],
                          [0., 2*math.pi, 0.],
                          [0., 0., 2*math.pi]]
        npt.assert_allclose(recip, expected_recip)

    def test_graphite(self):
        recip = reciprocal_lattice([[4.025915, -2.324363, 0.000000],
                                    [-0.000000, 4.648726, 0.000000],
                                    [0.000000, 0.000000, 12.850138]])
        expected_recip = [[1.56068503860106, 0., 0.],
                          [0.780342519300529, 1.3515929541082, 0.],
                          [0., 0., 0.488958586061845]]

        npt.assert_allclose(recip, expected_recip)

    def test_iron(self):
        recip = reciprocal_lattice([[-2.708355, 2.708355, 2.708355],
                                    [2.708355, -2.708355, 2.708355],
                                    [2.708355, 2.708355, -2.708355]])
        expected_recip = [[0., 1.15996339, 1.15996339],
                          [1.15996339, 0., 1.15996339],
                          [1.15996339, 1.15996339, 0.]]
        npt.assert_allclose(recip, expected_recip)


class TestDirectionChanged(unittest.TestCase):

    def test_direction_changed_nah(self):
        qpts = [[-0.25, -0.25, -0.25],
                [-0.25, -0.50, -0.50],
                [0.00, -0.25, -0.25],
                [0.00, 0.00, 0.00],
                [0.00, -0.50, -0.50],
                [0.25, 0.00, -0.25],
                [0.25, -0.50, -0.25],
                [-0.50, -0.50, -0.50]]
        expected_direction_changed = [True, True, False, True, True, True]
        npt.assert_equal(direction_changed(qpts),
                         expected_direction_changed)

class TestMPGrid(unittest.TestCase):

    def test_444_grid(self):
        qpts = mp_grid([4,4,4])
        expected_qpts = np.loadtxt(os.path.join('data','qgrid_444.txt'))
        npt.assert_equal(qpts, expected_qpts)
