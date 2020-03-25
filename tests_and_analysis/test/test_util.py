import os
import unittest
import numpy as np
import numpy.testing as npt
from euphonic import ureg, Crystal
from euphonic.util import direction_changed, mp_grid
from .utils import get_data_path


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
