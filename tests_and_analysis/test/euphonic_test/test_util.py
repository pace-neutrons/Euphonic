import json
import os

import pytest
import numpy as np
import numpy.testing as npt

from euphonic import ureg
from euphonic.util import (direction_changed, mp_grid, get_qpoint_labels,
                           mode_gradients_to_widths)
from tests_and_analysis.test.utils import get_data_path
from tests_and_analysis.test.euphonic_test.test_crystal import get_crystal
from tests_and_analysis.test.euphonic_test.test_force_constants import (
    get_fc_dir)


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


class TestMPGrid:

    def test_444_grid(self):
        qpts = mp_grid([4,4,4])
        expected_qpts = np.loadtxt(
            os.path.join(get_data_path(), 'util', 'qgrid_444.txt'))
        npt.assert_equal(qpts, expected_qpts)


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


def get_modg(mode_gradients_file):
    with open(os.path.join(get_fc_dir(), mode_gradients_file), 'r') as fp:
        modg_dict = json.load(fp)
    modg = modg_dict['mode_gradients']*ureg(
        modg_dict['mode_gradients_unit'])
    return modg


def get_modg_norm(mode_gradients_file):
    modg = get_modg(mode_gradients_file)
    return np.linalg.norm(modg.magnitude, axis=-1)*modg.units


class TestModeGradientsToWidths:

    @pytest.mark.parametrize('mode_grads, cell_vecs, expected_mode_widths_file', [
        (get_modg('quartz_554_full_mode_gradients.json'),
         get_crystal('quartz').cell_vectors,
         'quartz_554_full_mode_widths.json'),
        (get_modg_norm('quartz_554_full_mode_gradients.json'),
         get_crystal('quartz').cell_vectors,
         'quartz_554_full_mode_widths.json'),
        (get_modg('lzo_222_full_mode_gradients.json'),
         get_crystal('LZO').cell_vectors,
         'lzo_222_full_mode_widths.json'),
        (get_modg_norm('lzo_222_full_mode_gradients.json'),
         get_crystal('LZO').cell_vectors,
         'lzo_222_full_mode_widths.json')
        ])
    def test_mode_gradients_to_widths(self, mode_grads, cell_vecs,
                                      expected_mode_widths_file):
        mode_widths = mode_gradients_to_widths(mode_grads, cell_vecs)
        with open(os.path.join(get_fc_dir(), expected_mode_widths_file), 'r') as fp:
            modw_dict = json.load(fp)
        expected_mode_widths = modw_dict['mode_widths']*ureg(
            modw_dict['mode_widths_unit'])
        assert mode_widths.units == expected_mode_widths.units
        npt.assert_allclose(mode_widths.magnitude,
                            expected_mode_widths.magnitude, atol=3e-4)
