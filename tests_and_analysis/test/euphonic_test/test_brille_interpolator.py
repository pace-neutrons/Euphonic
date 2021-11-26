import json
import os

import pytest
import numpy as np
import numpy.testing as npt
import spglib as spg
import brille as br

from euphonic import BrilleInterpolator, ForceConstants, Crystal, ureg
from tests_and_analysis.test.euphonic_test.test_force_constants import (
    get_fc)
from tests_and_analysis.test.euphonic_test.test_crystal import (
    get_crystal)


def get_grid(material, grid_type='trellis', fill=True):
    fc = get_fc(material)
    crystal = fc.crystal
    cell_vectors = crystal._cell_vectors
    cell = crystal.to_spglib_cell()

    dataset = spg.get_symmetry_dataset(cell)
    rotations = dataset['rotations']  # in fractional
    translations = dataset['translations']

    symmetry = br.Symmetry(rotations, translations)
    direct = br.Direct(*cell)
    direct.spacegroup = symmetry
    bz = br.BrillouinZone(direct.star)
    n_grid_points = 10
    if grid_type == 'trellis':
        vol = bz.ir_polyhedron.volume
        grid_kwargs = {
            'node_volume_fraction': vol/n_grid_points}
        grid = br.BZTrellisQdc(bz, **grid_kwargs)
    elif grid_type == 'mesh':
        grid_kwargs = {
            'max_size': bz.ir_polyhedron.volume/n_grid_points,
            'max_points': n_grid_points}
        grid = br.BZMeshQdc(bz, **grid_kwargs)
    elif grid_type == 'nest':
        grid_kwargs = {'number_density': n_grid_points}
        grid = br.BZNestQdc(bz, **grid_kwargs)

    if fill:
        n_qpts = len(grid.rlu)
        n_atoms = crystal.n_atoms
        vals = np.random.rand(len(grid.rlu), 3*n_atoms, 1)
        vec_real = np.random.rand(len(grid.rlu), 3*n_atoms, 3*n_atoms)
        vecs = vec_real + vec_real*1j
        grid.fill(vals, (1,), (1., 0., 0.), vecs,
                  (0., 3*n_atoms , 0, 3, 0, 0), (0., 1., 0.))
    return grid


class TestBrilleInterpolatorCreation:

    @pytest.fixture(params=['quartz', 'LZO'])
    def create_from_constructor(self, request):
        material = request.param
        crystal = get_crystal(material)
        grid = get_grid(material)
        bri = BrilleInterpolator(crystal, grid)
        return bri

    @pytest.fixture(params=[
        ('LZO', {'n_grid_points': 10}),
        ('quartz', {'n_grid_points': 10})])
    def create_from_force_constants(self, request):
        material, kwargs = request.param
        fc = get_fc(material)
        bri = BrilleInterpolator.from_force_constants(fc, **kwargs)
        return bri

    @pytest.mark.parametrize(('bri_creator'), [
        pytest.lazy_fixture('create_from_constructor'),
        pytest.lazy_fixture('create_from_force_constants')])
    def test_correct_object_creation(self, bri_creator):
        bri = bri_creator

    @pytest.fixture(params=[
        ('crystal',
         get_fc('quartz'),
         TypeError),
        ('grid',
         get_grid('quartz', fill=False),
         ValueError),
        ('grid',
         get_grid('LZO'),
         ValueError)])
    def inject_faulty_elements(self, request):
        faulty_arg, faulty_value, expected_exception = request.param
        args = {'crystal': get_crystal('quartz'),
                'grid': get_grid('quartz')}
        args[faulty_arg] = faulty_value
        # Inject the faulty value and get a tuple of constructor arguments
        #args, kwargs = expected_fc.to_constructor_args(**{faulty_arg: faulty_value})
        #return args, kwargs, expected_exception
        return (), args, expected_exception

    def test_faulty_object_creation(self, inject_faulty_elements):
        faulty_args, faulty_kwargs, expected_exception = inject_faulty_elements
        with pytest.raises(expected_exception):
            BrilleInterpolator(*faulty_args, **faulty_kwargs)
