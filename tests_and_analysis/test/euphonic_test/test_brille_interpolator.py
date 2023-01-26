import sys
from multiprocessing import cpu_count

import pytest
import numpy as np
import numpy.testing as npt
import spglib as spg

from euphonic import ForceConstants, QpointPhononModes, ureg
from tests_and_analysis.test.utils import get_data_path, get_test_qpts
from tests_and_analysis.test.euphonic_test.test_force_constants import (
    get_fc)
from tests_and_analysis.test.euphonic_test.test_qpoint_phonon_modes import (
    get_qpt_ph_modes_from_json)
from tests_and_analysis.test.euphonic_test.test_qpoint_frequencies import (
    get_qpt_freqs, check_qpt_freqs)
from tests_and_analysis.test.euphonic_test.test_crystal import (
    get_crystal, check_crystal)
from tests_and_analysis.test.euphonic_test.test_spectrum1d import (
    get_spectrum1d, check_spectrum1d)

# Allow tests with brille marker to be collected and
# deselected if brille isn't installed
pytestmark = pytest.mark.brille
try:
    import brille as br
    from brille import BZTrellisQdc, BZMeshQdc, BZNestQdc, ApproxConfig
    from euphonic.brille import BrilleInterpolator
except ModuleNotFoundError:
    pass


def get_brille_grid(grid_file):
    filepath = get_data_path('brille_grid', grid_file)
    if 'trellis' in grid_file:
        return BZTrellisQdc.from_file(filepath)
    elif 'mesh' in grid_file:
        return BZMeshQdc.from_file(filepath)
    else:
        return BZNestQdc.from_file(filepath)


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

    @pytest.mark.parametrize('material, grid_file', [
        ('quartz', 'quartz_mesh_10.hdf5'),
        ('LZO', 'lzo_trellis_10.hdf5')])
    def test_create_from_constructor(self, material, grid_file):
        crystal = get_crystal(material)
        grid = get_brille_grid(grid_file)
        bri = BrilleInterpolator(crystal, grid)
        check_crystal(crystal, bri.crystal)
        assert grid == bri._grid

    @pytest.mark.parametrize('material, kwargs, expected_grid_file', [
        ('LZO', {'grid_npts': 10,
                 'grid_type': 'trellis'},
         'lzo_trellis_10.hdf5'),
        ('quartz', {'grid_npts': 10,
                    'grid_type': 'mesh'},
         'quartz_mesh_10.hdf5')
        ])
    def test_create_from_force_constants(
            self, material, kwargs, expected_grid_file):
        fc = get_fc(material)
        bri = BrilleInterpolator.from_force_constants(fc, **kwargs)
        expected_grid = get_brille_grid(expected_grid_file)
        check_crystal(fc.crystal, bri.crystal)
        assert type(expected_grid) is type(bri._grid)
        assert expected_grid.rlu.shape == bri._grid.rlu.shape

    @pytest.mark.parametrize(
        'kwargs, expected_grid_type, expected_grid_kwargs', [
            ({'grid_npts': 10,
              'grid_kwargs': {'node_volume_fraction': 0.04}},
             'brille.BZTrellisQdc',
             {'node_volume_fraction': 0.04}),
            ({'grid_npts': 10, 'grid_type': 'trellis'},
             'brille.BZTrellisQdc',
             {'node_volume_fraction': 0.037894368,
              'always_triangulate': False,
              'approx_config': None}),
            # Note: approx_config should actually be brille.ApproxConfig
            # but we can't import it on the top level here due to optional
            # extras
            ({'grid_type': 'mesh', 'grid_npts': 50},
             'brille.BZMeshQdc',
             {'max_size': 0.00757887361, 'max_points': 50}),
            ({'grid_type': 'nest', 'grid_npts': 100},
             'brille.BZNestQdc',
             {'number_density': 100}),
            ({},
             'brille.BZTrellisQdc',
             {'node_volume_fraction': 0.00037894368,
              'always_triangulate': False,
              'approx_config': None})])
    def test_from_force_constants_correct_grid_kwargs_passed_to_brille(
            self, mocker, kwargs, expected_grid_type, expected_grid_kwargs):
        # Patch __init__ to avoid type checks - BZTrellisQdc is now a mock
        # object, so can't be used as the 2nd argument in isinstance checks
        mocker.patch.object(BrilleInterpolator, '__init__', return_value=None)
        mock_grid = mocker.patch(expected_grid_type)
        mock_grid().rlu = np.ones((5, 3))  # Note mock_grid() parentheses

        fc = get_fc('quartz')
        BrilleInterpolator.from_force_constants(fc, **kwargs)
        for called_key, called_val in mock_grid.call_args[1].items():
            expected_val = expected_grid_kwargs.pop(called_key)
            if called_key == 'approx_config':
                assert isinstance(called_val, ApproxConfig)
            else:
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
                                       'dipole_parameter': 0.5,
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

    @pytest.mark.parametrize('faulty_crystal, grid_file, expected_exception',
        [(get_fc('quartz'), 'lzo_trellis_10.hdf5', TypeError)])
    def test_faulty_crystal_object_creation(
            self, faulty_crystal, grid_file, expected_exception):
        grid = get_brille_grid(grid_file)
        with pytest.raises(expected_exception):
            BrilleInterpolator(faulty_crystal, grid)

    # Deliberately create a grid incompatible with the quartz crystal
    # and check that it raises an error
    @pytest.mark.parametrize('expected_exception, grid_file',
        [(ValueError, 'lzo_trellis_10.hdf5'),
         (ValueError, 'quartz_mesh_10_unfilled.hdf5')])
    def test_faulty_grid_object_creation(self, expected_exception, grid_file):
        grid = get_brille_grid(grid_file)
        with pytest.raises(expected_exception):
            BrilleInterpolator(get_crystal('quartz'), grid)


class TestBrilleInterpolatorCalculateQpointPhononModes:

    @pytest.mark.parametrize(
        'material, from_fc_kwargs, emax, expected_sf1d_file', [
            ('LZO', {'grid_npts': 100}, 100,
             'La2Zr2O7_trellis_100_sf_1d_average.json'),
            ('CaHgO2', {'grid_type': 'trellis', 'grid_npts': 500}, 90,
             'CaHgO2_trellis_500_sf_1d_average.json')
        ])
    def test_calculate_qpoint_phonon_modes(
            self, material, from_fc_kwargs, emax, expected_sf1d_file):
        fc = get_fc(material)
        bri = BrilleInterpolator.from_force_constants(fc, **from_fc_kwargs)

        qpts = get_test_qpts()
        qpm = bri.calculate_qpoint_phonon_modes(qpts)
        # Replace frequencies to ensure same binning
        qpm_eu = fc.calculate_qpoint_phonon_modes(qpts)
        qpm.frequencies = qpm_eu.frequencies

        # Calculate structure factor 1D to test eigenvectors
        sf = qpm.calculate_structure_factor()
        ebins = np.arange(5, 100)*ureg('meV')
        sf1d = sf.calculate_1d_average(ebins)
        expected_sf1d = get_spectrum1d(expected_sf1d_file)
        check_spectrum1d(sf1d, expected_sf1d, y_rtol=0.01, y_atol=3e-3)

    def test_brille_qpoint_phonon_modes_similar_to_those_from_fc(self):
        fc = get_fc('graphite')
        bri = BrilleInterpolator.from_force_constants(fc, grid_npts=5000)
        qpts = np.array([[-0.20, 0.55, 0.55],
                         [ 0.35, 0.07, 0.02],
                         [ 0.00, 0.50, 0.00],
                         [ 0.65, 0.05, 0.25],
                         [ 1.80, 0.55, 2.55]])
        qpm_fc = fc.calculate_qpoint_phonon_modes(qpts)
        qpm_brille = bri.calculate_qpoint_phonon_modes(qpts)
        # Replace frequencies to ensure same binning
        qpm_brille.frequencies = qpm_fc.frequencies
        # Calculate structure factor 1D to test eigenvectors
        ebins = np.arange(5, 190)*ureg('meV')
        sf_fc = qpm_fc.calculate_structure_factor()
        sf1d_fc = sf_fc.calculate_1d_average(ebins)
        sf_brille = qpm_brille.calculate_structure_factor()
        sf1d_brille = sf_brille.calculate_1d_average(ebins)
        # Tolerances are quite generous, but that is required unless
        # we have a very dense grid (expensive to test)
        check_spectrum1d(sf1d_brille, sf1d_fc,
                         y_rtol=0.01, y_atol=5e-3)

    def test_calculate_qpoint_phonon_modes_single_qpt(self):
        fc = get_fc('graphite')
        bri = BrilleInterpolator.from_force_constants(fc, grid_npts=10)
        qpm = bri.calculate_qpoint_phonon_modes(np.array([[0.5, 0.5, 0.5]]))
        assert isinstance(qpm, QpointPhononModes)
        assert qpm.qpts.shape == (1, 3)
        assert qpm.frequencies.shape == (1, 12)
        assert qpm.eigenvectors.shape == (1, 12, 4, 3)

    @pytest.mark.parametrize(
        'material, grid_file, kwargs', [
            ('LZO', 'lzo_trellis_10.hdf5', {}),
            ('quartz', 'quartz_mesh_10.hdf5', {'useparallel': False}),
            ('LZO', 'lzo_trellis_10.hdf5', {'kwarg_one': 'one', 'kwarg_two': 2}),
        ])
    def test_calculate_qpoint_phonon_modes_correct_kwargs_passed(
            self, mocker, material, grid_file, kwargs):
        qpts = np.ones((10, 3))
        crystal = get_crystal(material)
        grid = get_brille_grid(grid_file)
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
        'material, from_fc_kwargs, expected_qpf_file, rtol', [
            ('NaCl', {'grid_npts': 10},
             'NaCl_trellis_10_qpoint_frequencies.json', 0.01),
            # Higher rtol on quartz due to degenerate modes
            ('quartz', {'grid_type': 'mesh', 'grid_npts': 200},
             'quartz_mesh_200_qpoint_frequencies.json', 0.07),
        ])
    def test_calculate_qpoint_frequencies(
            self, material, from_fc_kwargs, expected_qpf_file, rtol):
        fc = get_fc(material)
        bri = BrilleInterpolator.from_force_constants(fc, **from_fc_kwargs)

        qpts = get_test_qpts()
        qpf = bri.calculate_qpoint_frequencies(qpts)
        expected_qpf = get_qpt_freqs(material, expected_qpf_file)
        check_qpt_freqs(qpf, expected_qpf, frequencies_rtol=rtol,
                        frequencies_atol=0.8)

    def test_brille_qpoint_frequencies_similar_to_those_from_fc(self):
        fc = get_fc('graphite')
        bri = BrilleInterpolator.from_force_constants(fc, grid_npts=100)
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
