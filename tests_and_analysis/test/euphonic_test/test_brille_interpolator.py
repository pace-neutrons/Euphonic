from multiprocessing import cpu_count

import pytest
import numpy as np
import spglib as spg

from euphonic import ForceConstants, QpointPhononModes
from tests_and_analysis.test.utils import get_test_qpts
from tests_and_analysis.test.euphonic_test.test_force_constants import (
    get_fc)
from tests_and_analysis.test.euphonic_test.test_structure_factor import (
    get_sf, check_structure_factor)
from tests_and_analysis.test.euphonic_test.test_qpoint_phonon_modes import (
    get_qpt_ph_modes_from_json)
from tests_and_analysis.test.euphonic_test.test_qpoint_frequencies import (
    get_qpt_freqs, check_qpt_freqs)
from tests_and_analysis.test.euphonic_test.test_crystal import (
    get_crystal, check_crystal)

# Allow tests with brille marker to be collected and
# deselected if brille isn't installed
pytestmark = pytest.mark.brille
try:
    import brille as br
    from euphonic.brille import BrilleInterpolator
except ModuleNotFoundError:
    pass


# Use this function until we can read Brille grids from a HDF5 file
def get_grid(material, grid_type='trellis', fill=True):
    fc = get_fc(material)
    crystal = fc.crystal
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
        br_grid = br.BZTrellisQdc(bz, **grid_kwargs)
    elif grid_type == 'mesh':
        grid_kwargs = {
            'max_size': bz.ir_polyhedron.volume/n_grid_points,
            'max_points': n_grid_points}
        br_grid = br.BZMeshQdc(bz, **grid_kwargs)
    elif grid_type == 'nest':
        grid_kwargs = {'number_density': n_grid_points}
        br_grid = br.BZNestQdc(bz, **grid_kwargs)

    if fill:
        n_qpts = len(br_grid.rlu)
        n_atoms = crystal.n_atoms
        vals = np.random.rand(n_qpts, 3*n_atoms, 1)
        vec_real = np.random.rand(n_qpts, 3*n_atoms, 3*n_atoms)
        vecs = vec_real + vec_real*1j
        br_grid.fill(vals, (1,), (1., 0., 0.), vecs,
                     (0., 3*n_atoms , 0, 3, 0, 0), (0., 1., 0.))
    return br_grid


# Use this fixture until we can read Brille grids from a HDF5 file
@pytest.fixture
def grid(grid_args):
    # Currently can only use 1 fixture argument in indirectly
    # parametrized fixtures, so work around it
    material, kwargs = grid_args
    return get_grid(material, **kwargs)


def test_import_without_brille_raises_err(
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


class TestBrilleInterpolatorCreation:

    @pytest.mark.parametrize('material, grid_args', [
        ('quartz', ('quartz', {})),
        ('LZO', ('LZO', {}))])
    def test_create_from_constructor(self, material, grid):
        crystal = get_crystal(material)
        bri = BrilleInterpolator(crystal, grid)
        check_crystal(crystal, bri.crystal)
        assert grid == bri._grid

    @pytest.mark.parametrize('material, kwargs', [
        ('LZO', {'n_grid_points': 10,
                 'grid_type': 'trellis'}),
        ('quartz', {'n_grid_points': 10,
                    'grid_type': 'mesh'}),
        ('NaCl', {'n_grid_points': 10,
                  'grid_type': 'nest'})
        ])
    def test_create_from_force_constants(self, material, kwargs):
        fc = get_fc(material)
        bri = BrilleInterpolator.from_force_constants(fc, **kwargs)

    @pytest.mark.parametrize(
        'kwargs, expected_grid_type, expected_grid_kwargs', [
            ({'n_grid_points': 10,
              'grid_kwargs': {'node_volume_fraction': 0.04}},
             'brille.BZTrellisQdc',
             {'node_volume_fraction': 0.04}),
            ({'n_grid_points': 10, 'grid_type': 'trellis'},
             'brille.BZTrellisQdc',
             {'node_volume_fraction': 0.013193727}),
            ({'grid_type': 'mesh', 'n_grid_points': 50},
             'brille.BZMeshQdc',
             {'max_size': 0.00263874557, 'max_points': 50}),
            ({'grid_type': 'nest', 'n_grid_points': 100},
             'brille.BZNestQdc',
             {'number_density': 100}),
            ({},
             'brille.BZTrellisQdc',
             {'node_volume_fraction': 0.00013193727})])
    def test_from_force_constants_correct_grid_kwargs_passed_to_brille(
            self, mocker, kwargs, expected_grid_type, expected_grid_kwargs):
        # Patch __init__ to avoid type checks - BZTrellisQdc is now a mock
        # object, so can't be used as the 2nd argument in isinstance checks
        mocker.patch.object(BrilleInterpolator, '__init__', return_value=None)
        mock_grid = mocker.patch(expected_grid_type)
        mock_grid().rlu = np.ones((5, 3))  # Note mock_grid() parentheses

        fc = get_fc('Si2-sc-skew')
        BrilleInterpolator.from_force_constants(fc, **kwargs)
        for called_key, called_val in mock_grid.call_args[1].items():
            expected_val = expected_grid_kwargs.pop(called_key)
            assert expected_val == pytest.approx(called_val)
        # Make sure all expected kwargs have been passed
        assert not expected_grid_kwargs

    @pytest.mark.parametrize(
        'kwargs', [
            ({}),
            ({'grid_type': 'nest',
              'grid_kwargs': {'number_density': 50},
              'interpolation_kwargs': {'insert_gamma': True}}),
            ({'interpolation_kwargs': {'reduce_qpts': True,
                                       'eta_scale': 0.5,
                                       'n_threads': 2}})
        ])
    def test_from_force_constants_correct_interpolation_kwargs_passed(
            self, mocker, kwargs):
        # Patch __init__ to avoid type checks - we're mocking
        # calculate_qpoint_phonon_modes so don't care about the return value
        mocker.patch.object(BrilleInterpolator, '__init__', return_value=None)
        mock_qpm = mocker.patch.object(
            ForceConstants, 'calculate_qpoint_phonon_modes',
            return_value=get_qpt_ph_modes_from_json(
                'quartz', 'quartz_reciprocal_qpoint_phonon_modes.json'))

        fc = get_fc('quartz')
        BrilleInterpolator.from_force_constants(fc, **kwargs)

        # default_kwargs should be set no matter what the user provides,
        # as the input and output qpts of calculate_qpoint_phonon_modes
        # must be the same
        default_kwargs = {'insert_gamma': False, 'reduce_qpts': False}
        expected_interp_kwargs = {**kwargs.get('interpolation_kwargs', {}),
                                  **default_kwargs}
        assert expected_interp_kwargs == mock_qpm.call_args[1]

    def test_from_fc_with_invalid_grid_type_raises_value_error(self):
        with pytest.raises(ValueError):
            BrilleInterpolator.from_force_constants(
                get_fc('quartz'), grid_type='unknown')

    @pytest.mark.parametrize('faulty_crystal, expected_exception',
        [(get_fc('quartz'), TypeError)])
    # This uses implicit indirect parametrization - material is passed to the
    # grid fixture
    @pytest.mark.parametrize('grid_args', [('quartz', {})])
    def test_faulty_crystal_object_creation(
            self, faulty_crystal, expected_exception, grid):
        with pytest.raises(expected_exception):
            BrilleInterpolator(faulty_crystal, grid)

    # Deliberately create a grid incompatible with the quartz crystal
    # and check that it raises an error. This uses implicit indirect
    # parametrization - material and fill are passed to the grid
    # fixture
    @pytest.mark.parametrize('expected_exception, grid_args',
        [(ValueError, ('LZO', {})),
         (ValueError, ('quartz', {'fill': False}))])
    def test_faulty_grid_object_creation(self, expected_exception, grid):
        with pytest.raises(expected_exception):
            BrilleInterpolator(get_crystal('quartz'), grid)


class TestBrilleInterpolatorCalculateQpointPhononModes:

    @pytest.mark.parametrize(
        'material, from_fc_kwargs, expected_sf_file', [
            ('LZO', {'n_grid_points': 100},
             'La2Zr2O7_trellis_100_structure_factor.json'),
            ('CaHgO2', {'grid_type': 'mesh', 'n_grid_points': 50},
             'CaHgO2_mesh_50_structure_factor.json')
        ])
    def test_calculate_qpoint_phonon_modes(
            self, material, from_fc_kwargs, expected_sf_file):
        fc = get_fc(material)
        bri = BrilleInterpolator.from_force_constants(fc, **from_fc_kwargs)

        qpts = get_test_qpts()
        qpm = bri.calculate_qpoint_phonon_modes(qpts)

        # Calculate structure factor to test eigenvectors
        sf = qpm.calculate_structure_factor()
        expected_sf = get_sf(material, expected_sf_file)
        check_structure_factor(sf, expected_sf, freq_rtol=1e-4,
                               sf_rtol=1e-3)

    def test_brille_qpoint_phonon_modes_similar_to_those_from_fc(self):
        fc = get_fc('graphite')
        bri = BrilleInterpolator.from_force_constants(fc, n_grid_points=5000)
        qpts = np.array([[-0.20, 0.55, 0.55],
                         [ 0.35, 0.07, 0.02],
                         [ 0.00, 0.50, 0.00],
                         [ 0.65, 0.05, 0.25],
                         [ 1.80, 0.55, 2.55]])
        qpm_brille = bri.calculate_qpoint_phonon_modes(qpts)
        qpm_fc = fc.calculate_qpoint_phonon_modes(qpts)

        # Calculate structure factor to test eigenvectors
        sf_brille = qpm_brille.calculate_structure_factor()
        sf_fc = qpm_fc.calculate_structure_factor()
        # Tolerances are quite generous, but that is required unless
        # we have a very dense grid (expensive to test)
        check_structure_factor(sf_brille, sf_fc, freq_rtol=1e-3,
                               sf_rtol=0.01, sf_atol=0.035)

    def test_calculate_qpoint_phonon_modes_single_qpt(self):
        fc = get_fc('graphite')
        bri = BrilleInterpolator.from_force_constants(fc, n_grid_points=10)
        qpm = bri.calculate_qpoint_phonon_modes(np.array([[0.5, 0.5, 0.5]]))
        assert isinstance(qpm, QpointPhononModes)
        assert qpm.qpts.shape == (1, 3)
        assert qpm.frequencies.shape == (1, 12)
        assert qpm.eigenvectors.shape == (1, 12, 4, 3)

    @pytest.mark.parametrize(
        'grid_args, material, kwargs', [
            (('LZO', {}), 'LZO', {}),
            (('quartz', {}), 'quartz', {'useparallel': False}),
            (('LZO', {}), 'LZO', {'kwarg_one': 'one', 'kwarg_two': 2}),
        ])
    def test_calculate_qpoint_phonon_modes_correct_kwargs_passed(
            self, mocker, grid, material, kwargs):
        qpts = np.ones((10, 3))
        crystal = get_crystal(material)
        bri = BrilleInterpolator(crystal, grid)
        n_atoms = crystal.n_atoms
        mock_interpolate = mocker.patch.object(
            grid, 'ir_interpolate_at',
            return_value=(np.ones((len(qpts), 3*n_atoms, 1)),
                          np.ones((len(qpts), 3*n_atoms, n_atoms, 3),
                                  dtype=np.complex128)))

        bri.calculate_qpoint_phonon_modes(qpts, **kwargs)
        if kwargs:
            assert mock_interpolate.call_args[1] == kwargs
        else:
            default_kwargs = {'useparallel': True, 'threads': cpu_count()}
            assert mock_interpolate.call_args[1] == default_kwargs


class TestBrilleInterpolatorCalculateQpointFrequencies:

    @pytest.mark.parametrize(
        'material, from_fc_kwargs, expected_qpf_file', [
            ('NaCl', {'n_grid_points': 10},
             'NaCl_trellis_10_qpoint_frequencies.json'),
            ('quartz', {'grid_type': 'mesh', 'n_grid_points': 200},
             'quartz_mesh_200_qpoint_frequencies.json'),
        ])
    def test_calculate_qpoint_frequencies(
            self, material, from_fc_kwargs, expected_qpf_file):
        fc = get_fc(material)
        bri = BrilleInterpolator.from_force_constants(fc, **from_fc_kwargs)

        qpts = get_test_qpts()
        qpf = bri.calculate_qpoint_frequencies(qpts)
        expected_qpf = get_qpt_freqs(material, expected_qpf_file)
        check_qpt_freqs(qpf, expected_qpf, frequencies_rtol=0.01,
                        frequencies_atol=0.8)

    def test_brille_qpoint_frequencies_similar_to_those_from_fc(self):
        fc = get_fc('graphite')
        bri = BrilleInterpolator.from_force_constants(fc, n_grid_points=100)
        qpts = np.array([[-0.2     ,  0.55    ,  0.55    ],
                         [ 0.35    ,  0.07    ,  0.02    ],
                         [ 0.00    ,  0.5    ,  0.00    ],
                         [ 0.65    ,  0.05    ,  0.25    ],
                         [ 1.8     ,  0.55    ,  2.55    ]])
        qpf_brille = bri.calculate_qpoint_frequencies(qpts)
        qpf_fc = fc.calculate_qpoint_frequencies(qpts)
        # Tolerances are quite generous, but that is required unless
        # we have a very dense grid (expensive to test)
        check_qpt_freqs(qpf_brille, qpf_fc, frequencies_rtol=0.01)
