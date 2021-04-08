import os

import pytest
import numpy as np
import numpy.testing as npt

from euphonic import ureg
from euphonic.util import (direction_changed, mp_grid, get_qpoint_labels,
                           mode_gradients_to_widths)
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
         [(0, '0 0 0'), (5, '0 0 1/2'), (8, '3/8 3/4 1/2')]),
        (np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.2],
                   [0.0, 0.0, 0.3], [0.0, 0.0, 0.4], [0.0, 0.0, 0.5],
                   [0.125, 0.25, 0.5], [0.25, 0.5, 0.5], [0.375, 0.75, 0.5]]),
         {'cell': get_crystal('quartz_cv_only').to_spglib_cell()},
         [(0, '0 0 0'), (5, '0 0 1/2'), (8, '3/8 3/4 1/2')])])
    def test_get_qpt_labels(self, qpts, kwargs, expected_labels):
        labels = get_qpoint_labels(qpts, **kwargs)
        assert labels == expected_labels

@pytest.mark.unit
class TestModeGradientsToWidths:

    @pytest.mark.parametrize('mode_grads, cell_vecs, expected_mode_widths', [
        (np.array([[42.150888, 23.297914, 46.008871],
                   [25.679001, 55.025284,  4.599791],
                   [23.106633, 25.714757, 21.541563],
                   [15.468629, 15.307626, 20.417814],
                   [41.257245, 43.774049, 35.250878]])*ureg('meV*angstrom'),
        np.array([[2.426176, -4.20226, 0.000000],
                  [2.426176,  4.20226, 0.000000],
                  [0.000000,  0.00000, 5.350305]])*ureg('angstrom'),
        np.array([[10.317524,  5.702769, 11.261865],
                  [ 6.285602, 13.468866,  1.125918],
                  [ 5.655948,  6.294354,  5.272857],
                  [ 3.786348,  3.746939,  4.997790],
                  [10.098781, 10.714834,  8.628567]])*ureg('meV'))
        ])
    def test_mode_gradients_to_widths(self, mode_grads, cell_vecs,
            expected_mode_widths):

        mode_widths = mode_gradients_to_widths(mode_grads, cell_vecs)
        assert mode_widths.units == expected_mode_widths.units
        npt.assert_allclose(mode_widths.magnitude,
                            expected_mode_widths.magnitude, atol=1e-5)
