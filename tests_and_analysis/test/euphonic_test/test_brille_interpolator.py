import pytest
import numpy as np
import spglib as spg

from euphonic import Crystal, ureg
from tests_and_analysis.test.euphonic_test.test_force_constants import (
    get_fc)
from tests_and_analysis.test.euphonic_test.test_crystal import (
    get_crystal)

# Allow tests with brille marker to be collected and
# deselected if brille isn't installed
pytestmark = pytest.mark.brille
try:
    import brille as br
    from euphonic.brille import BrilleInterpolator
except ModuleNotFoundError:
    pass


def test_import__without_brille_raises_err(
        mocker):
    # Mock import of brille to raise ModuleNotFoundError
    import builtins
    from importlib import reload
    import euphonic.brille
    real_import = builtins.__import__
    def mocked_import(name, *args, **kwargs):
        if name == 'brille':
            raise ModuleNotFoundError
        return real_import(name, *args, **kwargs)
    mocker.patch('builtins.__import__', side_effect=mocked_import)
    with pytest.raises(ModuleNotFoundError) as mnf_error:
        reload(euphonic.brille)
    assert "Cannot import Brille" in mnf_error.value.args[0]

@pytest.fixture
def grid(grid_args):
    # Currently can only use 1 fixture argument in indirectly
    # parametrized fixtures, so work around it
    material, kwargs = grid_args
    fill = kwargs.get('fill', True)
    grid_type = kwargs.get('grid_type', 'trellis')

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
        vals = np.random.rand(n_qpts, 3*n_atoms, 1)
        vec_real = np.random.rand(n_qpts, 3*n_atoms, 3*n_atoms)
        vecs = vec_real + vec_real*1j
        grid.fill(vals, (1,), (1., 0., 0.), vecs,
                  (0., 3*n_atoms , 0, 3, 0, 0), (0., 1., 0.))
    return grid

class TestBrilleInterpolatorCreation:

    @pytest.mark.parametrize('material, grid_args', [
        ('quartz', ('quartz', {})),
        ('LZO', ('LZO', {}))])
    def test_create_from_constructor(self, material, grid):
        crystal = get_crystal(material)
        bri = BrilleInterpolator(crystal, grid)

    @pytest.mark.parametrize('material, kwargs', [
        ('LZO', {'n_grid_points': 10}),
        ('quartz', {'n_grid_points': 10})])
    def test_create_from_force_constants(self, material, kwargs):
        fc = get_fc(material)
        bri = BrilleInterpolator.from_force_constants(fc, **kwargs)

    @pytest.mark.parametrize('faulty_crystal, expected_exception',
        [(get_fc('quartz'), TypeError)])
    # This uses implicit indirect parametrization - material is passed to the
    # grid fixture
    @pytest.mark.parametrize('grid_args', [('quartz', {})])
    def test_faulty_crystal_object_creation(self, faulty_crystal, expected_exception, grid):
        with pytest.raises(expected_exception):
            BrilleInterpolator(faulty_crystal, grid)

    # Deliberately create a grid incompatible with the quartz crystal
    # This uses implicit indirect parametrization - material and fill
    # are passed to the grid fixture
    @pytest.mark.parametrize('expected_exception, grid_args',
        [(ValueError, ('LZO', {})),
         (ValueError, ('quartz', {'fill': False}))])
    def test_faulty_grid_object_creation(self, expected_exception, grid):
        with pytest.raises(expected_exception):
            BrilleInterpolator(get_crystal('quartz'), grid)
