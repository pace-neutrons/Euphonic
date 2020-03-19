import os
import unittest
import numpy as np
import numpy.testing as npt
from euphonic import ureg, Crystal
from euphonic.util import direction_changed, mp_grid
from helpers import mock_crystal
from .utils import get_data_path


class TestCrystal(unittest.TestCase):

    def test_identity(self):
        cell_vectors = np.array([[1., 0., 0.],
                                 [0., 1., 0.],
                                 [0., 0., 1.]])*ureg('angstrom')
        expected_reciprocal_cell = np.array([
            [2*np.pi, 0., 0.],
            [0., 2*np.pi, 0.],
            [0., 0., 2*np.pi]])*ureg('1/angstrom')
        expected_cell_volume = 1.0*ureg('angstrom**3')
        crystal = mock_crystal(cell_vectors)
        npt.assert_allclose(crystal.reciprocal_cell().magnitude,
                            expected_reciprocal_cell.magnitude)
        npt.assert_allclose(crystal.cell_volume().magnitude,
                            expected_cell_volume.magnitude)

    def test_graphite(self):
        cell_vectors = np.array([
            [4.025915, -2.324363, 0.000000],
            [-0.000000, 4.648726, 0.000000],
            [0.000000, 0.000000, 12.850138]])*ureg('bohr')
        expected_reciprocal_cell = np.array([
            [1.56068503860106, 0., 0.],
            [0.780342519300529, 1.3515929541082, 0.],
            [0., 0., 0.488958586061845]])*ureg('1/bohr')
        expected_cell_volume = 240.49516090747784*ureg('bohr**3')
        crystal = mock_crystal(cell_vectors)
        npt.assert_allclose(crystal.reciprocal_cell().magnitude,
                            expected_reciprocal_cell.magnitude)
        npt.assert_allclose(crystal.cell_volume().magnitude,
                            expected_cell_volume.magnitude)

    def test_iron(self):
        cell_vectors = np.array([
            [-2.708355, 2.708355, 2.708355],
            [2.708355, -2.708355, 2.708355],
            [2.708355, 2.708355, -2.708355]])*ureg('angstrom')
        expected_reciprocal_cell = np.array([
            [0., 1.15996339, 1.15996339],
            [1.15996339, 0., 1.15996339],
            [1.15996339, 1.15996339, 0.]])*ureg('1/angstrom')
        expected_cell_volume = 79.46515944812734*ureg('angstrom**3')
        crystal = mock_crystal(cell_vectors)
        npt.assert_allclose(crystal.reciprocal_cell().magnitude,
                            expected_reciprocal_cell.magnitude)
        npt.assert_allclose(crystal.cell_volume().magnitude,
                            expected_cell_volume.magnitude)


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
        expected_qpts = np.loadtxt(os.path.join(get_data_path(), 'qgrid_444.txt'))
        npt.assert_equal(qpts, expected_qpts)
