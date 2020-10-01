import os

import pytest
import numpy as np
import numpy.testing as npt

from euphonic.util import direction_changed, mp_grid, get_qpoint_labels
from tests_and_analysis.test.utils import get_data_path
from tests_and_analysis.test.euphonic_test.test_crystal import get_crystal


@pytest.mark.unit
class TestDirectionChanged:

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


@pytest.mark.unit
class TestMPGrid:

    def test_444_grid(self):
        qpts = mp_grid([4,4,4])
        expected_qpts = np.loadtxt(
            os.path.join(get_data_path(), 'util', 'qgrid_444.txt'))
        npt.assert_equal(qpts, expected_qpts)


@pytest.mark.unit
class TestGetQptLabels:

    @pytest.mark.parametrize('qpts, kwargs, expected_labels', [
        (np.array([[0.5, 0.5, 0.5], [0.0, 0.0, 0.5]]),
         {'cell': get_crystal('quartz').to_spglib_cell()},
         [(0, ''), (1, 'A')]),
        (np.array([[0.5, 0.5, 0.5], [0.4, 0.4, 0.5], [0.3, 0.3, 0.5],
                   [0.2, 0.2, 0.5], [0.1, 0.1, 0.5], [0.0, 0.0, 0.5]]),
         {'cell': get_crystal('quartz').to_spglib_cell()},
         [(0, ''), (5, 'A')]),
        (np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.2],
                   [0.0, 0.0, 0.3], [0.0, 0.0, 0.4], [0.0, 0.0, 0.5],
                   [0.125, 0.25, 0.5], [0.25, 0.5, 0.5], [0.375, 0.75, 0.5]]),
         {},
         [(0, '0 0 0'), (5, '0 0 1/2'), (8, '3/8 3/4 1/2')])])
    def test_get_qpt_labels(self, qpts, kwargs, expected_labels):
        labels = get_qpoint_labels(qpts, **kwargs)
        assert labels == expected_labels
