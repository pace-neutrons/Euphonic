import json

import numpy as np
import numpy.testing as npt
from pint import DimensionalityError
import pytest
from pytest_lazy_fixtures import lf as lazy_fixture
import spglib

from euphonic import Crystal, Quantity, ureg
from tests_and_analysis.test.utils import (
    check_json_metadata,
    check_property_setters,
    check_unit_conversion,
    get_data_path,
)


class ExpectedCrystal:

    def __init__(self, crystal_json):
        self.data = crystal_json

    @property
    def cell_vectors(self) -> Quantity:
        return np.array(self.data['cell_vectors'])*ureg(
            self.data['cell_vectors_unit'])

    @property
    def n_atoms(self) -> int:
        return self.data['n_atoms']

    @property
    def atom_r(self) -> np.array:
        if self.n_atoms == 0 and len(self.data['atom_r']) == 0:
            return np.zeros((0, 3), dtype=np.float64)
        return np.array(self.data['atom_r'])

    @property
    def atom_type(self) -> np.array:
        return np.array(
            self.data['atom_type'],
            dtype='<U2')

    @property
    def atom_mass(self) -> Quantity:
        return np.array(self.data['atom_mass'])*ureg(
            self.data['atom_mass_unit'])

    def to_dict(self):
        return {
            'cell_vectors': self.cell_vectors.magnitude,
            'cell_vectors_unit': str(self.cell_vectors.units),
            'n_atoms': self.n_atoms,
            'atom_r': self.atom_r,
            'atom_type': self.atom_type,
            'atom_mass': self.atom_mass.magnitude,
            'atom_mass_unit': str(self.atom_mass.units)}

    def to_constructor_args(self, cell_vectors=None, atom_r=None,
                            atom_type=None, atom_mass=None):
        if cell_vectors is None:
            cell_vectors = self.cell_vectors
        if atom_r is None:
            atom_r = self.atom_r
        if atom_type is None:
            atom_type = self.atom_type
        if atom_mass is None:
            atom_mass = self.atom_mass
        return (cell_vectors, atom_r, atom_type, atom_mass)


def get_crystal_path(*subpaths):
    return get_data_path('crystal', *subpaths)


def get_json_file(crystal_name):
    return get_crystal_path(f'crystal_{crystal_name}.json')


def get_expected_crystal(crystal_name):
    return ExpectedCrystal(json.load(open(get_json_file(crystal_name))))


def get_crystal_from_json_file(filepath):
    return Crystal.from_json_file(filepath)


def get_crystal(crystal):
    return get_crystal_from_json_file(
        get_json_file(crystal))


def check_crystal(crystal, expected_crystal):
    assert crystal.n_atoms == expected_crystal.n_atoms

    assert (crystal.cell_vectors.units
            == expected_crystal.cell_vectors.units)
    npt.assert_allclose(
        crystal.cell_vectors.magnitude,
        expected_crystal.cell_vectors.magnitude,
        atol=np.finfo(np.float64).eps)

    npt.assert_allclose(
        crystal.atom_r,
        expected_crystal.atom_r,
        atol=np.finfo(np.float64).eps)

    npt.assert_equal(
        crystal.atom_type,
        expected_crystal.atom_type)

    assert (crystal.atom_mass.units
            == expected_crystal.atom_mass.units)
    npt.assert_allclose(
        crystal.atom_mass.magnitude,
        expected_crystal.atom_mass.magnitude,
        atol=np.finfo(np.float64).eps)


class TestCrystalCreation:

    @pytest.fixture(params=[get_expected_crystal('quartz'),
                            get_expected_crystal('quartz_cv_only'),
                            get_expected_crystal('LZO')])
    def create_from_constructor(self, request):
        expected_crystal = request.param
        crystal = Crystal(*expected_crystal.to_constructor_args())
        return crystal, expected_crystal

    @pytest.fixture(params=[get_expected_crystal('quartz'),
                            get_expected_crystal('quartz_cv_only'),
                            get_expected_crystal('LZO')])
    def create_from_dict(self, request):
        expected_crystal = request.param
        d = expected_crystal.to_dict()
        crystal = Crystal.from_dict(d)
        return crystal, expected_crystal

    @pytest.fixture(params=[
        (get_json_file('quartz'),
         get_expected_crystal('quartz')),
        (get_json_file('quartz_cv_only'),
         get_expected_crystal('quartz_cv_only')),
        (get_json_file('LZO'),
         get_expected_crystal('LZO')),
    ])
    def create_from_json_file(self, request):
        filename, expected_crystal = request.param
        return get_crystal_from_json_file(filename), expected_crystal

    @pytest.fixture(params=[
        (Quantity(np.array([[2.426176, -4.20226, 0.000000],
                            [2.426176,  4.20226, 0.000000],
                            [0.000000,  0.00000, 5.350304]]), 'angstrom'),
         get_expected_crystal('quartz_cv_only'))])
    def create_from_cell_vectors(self, request):
        cell_vectors, expected_crystal = request.param
        return Crystal.from_cell_vectors(cell_vectors), expected_crystal

    @pytest.mark.parametrize('crystal_creator', [
        lazy_fixture('create_from_constructor'),
        lazy_fixture('create_from_json_file'),
        lazy_fixture('create_from_dict'),
        lazy_fixture('create_from_cell_vectors'),
    ])
    def test_create(self, crystal_creator):
        crystal, expected_crystal = crystal_creator
        check_crystal(crystal, expected_crystal)

    faulty_elements = [
        ('cell_vectors',
         np.array([[1.23, 2.45, 0.0],
                   [3.45, 5.66, 7.22]])*ureg('angstrom'),
         ValueError),
        ('cell_vectors',
         get_expected_crystal('quartz').cell_vectors.magnitude*ureg('kg'),
         DimensionalityError),
        ('cell_vectors',
         get_expected_crystal('quartz').cell_vectors.magnitude*ureg(''),
         DimensionalityError),
        ('atom_r',
         np.array([[0.125, 0.125, 0.125],
                   [0.875, 0.875, 0.875]]),
         ValueError),
        ('atom_mass',
         np.array(
             [15.999399987607514, 15.999399987607514, 91.2239999293416])*ureg(
                 'amu'),
         ValueError),
        ('atom_mass',
         get_expected_crystal('quartz').atom_mass.magnitude*ureg('angstrom'),
         DimensionalityError),
        ('atom_mass',
         get_expected_crystal('quartz').atom_mass.magnitude*ureg(''),
         DimensionalityError),
        ('atom_type', np.array(['O', 'Zr', 'La']), ValueError),
    ]

    @pytest.fixture(params=faulty_elements)
    def inject_faulty_elements(self, request):
        faulty_arg, faulty_value, expected_exception = request.param
        crystal = get_expected_crystal('quartz')
        # Inject the faulty value and get a tuple of constructor arguments
        args = crystal.to_constructor_args(**{faulty_arg: faulty_value})
        return args, expected_exception

    def test_faulty_creation(self, inject_faulty_elements):
        faulty_args, expected_exception = inject_faulty_elements
        with pytest.raises(expected_exception):
            Crystal(*faulty_args)


class TestCrystalSerialisation:

    @pytest.mark.parametrize('crystal', [
        get_crystal('quartz'),
        get_crystal('quartz_cv_only'),
        get_crystal('LZO')])
    def test_serialise_to_json_file(self, crystal, tmp_path):
        # Serialise
        output_file = tmp_path / 'tmp.test'
        crystal.to_json_file(output_file)
        # Test file metadata
        check_json_metadata(output_file, 'Crystal')
        # Deserialise
        deserialised_crystal = Crystal.from_json_file(output_file)
        return check_crystal(crystal, deserialised_crystal)

    @pytest.fixture(params=[
        (get_crystal('quartz'),
         get_expected_crystal('quartz')),
        (get_crystal('quartz_cv_only'),
         get_expected_crystal('quartz_cv_only')),
        (get_crystal('LZO'),
         get_expected_crystal('LZO')),
    ])
    def serialise_to_dict(self, request):
        crystal, expected_crystal = request.param
        # Convert to dict, then back to object to test
        crystal_dict = crystal.to_dict()
        crystal_from_dict = Crystal.from_dict(crystal_dict)
        return crystal_from_dict, expected_crystal

    def test_serialise_to_dict(self, serialise_to_dict):
        crystal, expected_crystal = serialise_to_dict
        check_crystal(crystal, expected_crystal)


class TestCrystalUnitConversion:

    @pytest.mark.parametrize('material, attr, unit_val', [
        ('quartz', 'cell_vectors', 'bohr'),
        ('quartz', 'atom_mass', 'kg')])
    def test_correct_unit_conversion(self, material, attr,
                                     unit_val):
        crystal = get_crystal(material)
        check_unit_conversion(crystal, attr, unit_val)

    @pytest.mark.parametrize('material, unit_attr, unit_val, err', [
        ('quartz', 'cell_vectors_unit', 'kg', ValueError),
        ('quartz', 'atom_mass_unit', 'bohr', ValueError)])
    def test_incorrect_unit_conversion(self, material, unit_attr,
                                       unit_val, err):
        crystal = get_crystal(material)
        with pytest.raises(err):
            setattr(crystal, unit_attr, unit_val)


class TestCrystalSetters:

    @pytest.mark.parametrize('material, attr, unit, scale', [
        ('quartz', 'cell_vectors', 'bohr', 2.),
        ('quartz', 'cell_vectors', 'angstrom', 3.),
        ('quartz', 'atom_mass', 'kg', 2.),
        ('quartz', 'atom_mass', 'm_e', 0.5),
        ])
    def test_setter_correct_units(self, material, attr,
                                  unit, scale):
        crystal = get_crystal(material)
        check_property_setters(crystal, attr, unit, scale)

    @pytest.mark.parametrize('material, attr, unit, err', [
        ('quartz', 'cell_vectors', 'kg', ValueError),
        ('quartz', 'atom_mass', 'bohr', ValueError)])
    def test_incorrect_unit_conversion(self, material, attr,
                                       unit, err):
        crystal = get_crystal(material)
        new_attr = getattr(crystal, attr).magnitude*ureg(unit)
        with pytest.raises(err):
            setattr(crystal, attr, new_attr)


class TestCrystalMethods:

    quartz_reciprocal_cell = np.array([
        [1.29487418, -0.74759597, 0.],
        [1.29487418, 0.74759597, 0.],
        [0., 0., 1.17436043],
    ])*ureg('1/angstrom')

    lzo_reciprocal_cell = np.array([
        [8.28488599e-01, 0.00000000e+00, -5.85829906e-01],
        [-2.01146673e-33, 8.28488599e-01, 5.85829906e-01],
        [2.01146673e-33, -8.28488599e-01, 5.85829906e-01],
    ])*ureg('1/angstrom')

    @pytest.mark.parametrize('crystal,expected_recip', [
        (get_crystal('quartz'), quartz_reciprocal_cell),
        (get_crystal('LZO'), lzo_reciprocal_cell),
    ])
    def test_reciprocal_cell(self, crystal, expected_recip):
        recip = crystal.reciprocal_cell()
        npt.assert_allclose(
            recip.to('1/angstrom').magnitude,
            expected_recip.to('1/angstrom').magnitude,
        )

    quartz_cell_volume = 109.09721804482547*ureg('angstrom**3')

    lzo_cell_volume = 308.4359549515967*ureg('angstrom**3')

    @pytest.mark.parametrize('crystal,expected_vol', [
        (get_crystal('quartz'), quartz_cell_volume),
        (get_crystal('LZO'), lzo_cell_volume),
    ])
    def test_cell_volume(self, crystal, expected_vol):
        vol = crystal.cell_volume()
        npt.assert_allclose(
            vol.to('angstrom**3').magnitude,
            expected_vol.to('angstrom**3').magnitude,
        )

    quartz_spglib_cell = ([[2.426176, -4.20226, 0.000000],
                           [2.426176,  4.20226, 0.000000],
                           [0.000000,  0.00000, 5.350304]],
                          [[0.411708, 0.275682, 0.279408],
                           [0.724318, 0.136026, 0.946074],
                           [0.863974, 0.588292, 0.612741],
                           [0.136026, 0.724318, 0.053926],
                           [0.588292, 0.863974, 0.387259],
                           [0.275682, 0.411708, 0.720592],
                           [0.464960, 0.000000, 0.166667],
                           [0.000000, 0.464960, 0.833333],
                           [0.535040, 0.535040, 0.500000]],
                           [0, 0, 0, 0, 0, 0, 1, 1, 1])

    quartz_cv_only_spglib_cell = ([[2.426176, -4.20226, 0.000000],
                                   [2.426176,  4.20226, 0.000000],
                                   [0.000000,  0.00000, 5.350304]],
                                   [],
                                   [])

    lzo_spglib_cell = ([[7.58391282,  0.00000000, 0.00000000],
                        [3.79195641,  3.79195641, 5.36263619],
                        [3.79195641, -3.79195641, 5.36263619]],
                       [[0.12500000, 0.12500000, 0.12500000],
                        [0.87500000, 0.87500000, 0.87500000],
                        [0.41861943, 0.41861943, 0.83138057],
                        [0.83138057, 0.83138057, 0.41861943],
                        [0.41861943, 0.83138057, 0.41861943],
                        [0.83138057, 0.41861943, 0.83138057],
                        [0.83138057, 0.41861943, 0.41861943],
                        [0.41861943, 0.83138057, 0.83138057],
                        [0.16861943, 0.58138057, 0.16861943],
                        [0.58138057, 0.16861943, 0.58138057],
                        [0.16861943, 0.16861943, 0.58138057],
                        [0.58138057, 0.58138057, 0.16861943],
                        [0.16861943, 0.58138057, 0.58138057],
                        [0.58138057, 0.16861943, 0.16861943],
                        [0.50000000, 0.50000000, 0.50000000],
                        [0.00000000, 0.50000000, 0.50000000],
                        [0.50000000, 0.00000000, 0.50000000],
                        [0.50000000, 0.50000000, 0.00000000],
                        [0.00000000, 0.00000000, 0.00000000],
                        [0.50000000, 0.00000000, 0.00000000],
                        [0.00000000, 0.50000000, 0.00000000],
                        [0.00000000, 0.00000000, 0.50000000]],
                       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        2, 2, 2, 2,
                        0, 0, 0, 0])

    @pytest.mark.parametrize('crystal,expected_spglib_cell', [
        (get_crystal('quartz'), quartz_spglib_cell),
        (get_crystal('quartz_cv_only'), quartz_cv_only_spglib_cell),
        (get_crystal('LZO'), lzo_spglib_cell)])
    def test_to_spglib_cell(self, crystal, expected_spglib_cell):
        spglib_cell = crystal.to_spglib_cell()
        assert isinstance(spglib_cell, tuple)
        npt.assert_allclose(spglib_cell[0], expected_spglib_cell[0],
                            atol=np.finfo(np.float64).eps)
        npt.assert_allclose(spglib_cell[1], expected_spglib_cell[1],
                            atol=np.finfo(np.float64).eps)
        assert spglib_cell[2] == expected_spglib_cell[2]

    @pytest.mark.parametrize('crystal,kwargs,expected', [
        (get_crystal('quartz'), {}, (15, 15, 12)),
        (get_crystal('quartz'), {'spacing': 0.05 * ureg('1/angstrom')},
         (30, 30, 24)),
        (get_crystal('quartz'), {'spacing': 0.05 * ureg('1/bohr')},
         (16, 16, 13)),
        (get_crystal('LZO'), {'spacing': 0.5 * ureg('1/angstrom')},
         (3, 3, 3))])
    def test_get_mp_grid_spec(self, crystal, kwargs, expected):
        assert crystal.get_mp_grid_spec(**kwargs) == expected

    @pytest.mark.parametrize('crystal, expected_spec_idx', [
        (get_crystal('quartz'),
         {'O': np.arange(6), 'Si': np.arange(6, 9)}),
        (get_crystal('LZO'),
         {'O': np.arange(14), 'Zr': np.arange(14, 18),
          'La': np.arange(18, 22)}),
        ])
    def test_get_species_idx(self, crystal, expected_spec_idx):
        spec_idx = crystal.get_species_idx()
        assert spec_idx.keys() == expected_spec_idx.keys()
        for key in expected_spec_idx:
            npt.assert_equal(spec_idx[key], expected_spec_idx[key])

    @pytest.mark.parametrize('crystal, kwargs, expected_res_file', [
        (get_crystal('quartz'), {}, 'quartz_equiv_atoms.json'),
        (get_crystal('LZO'), {}, 'lzo_equiv_atoms.json'),
        ])
    def test_get_symmetry_equivalent_atoms(
            self, crystal, kwargs, expected_res_file):
        rot, trans, equiv_atoms = crystal.get_symmetry_equivalent_atoms(
            **kwargs)
        with open(get_crystal_path(expected_res_file)) as fp:
            res = json.load(fp)
        exp_rot = np.array(res['rotations'])
        exp_trans = np.array(res['translations'])
        exp_equiv_atoms = np.array(res['equivalent_atoms'])
        # In older versions of Numpy/spglib, sometimes the operations
        # are in a different order
        if not np.array_equal(rot, exp_rot):
            for i, rot_i in enumerate(rot):
                match_found = False
                for j, exp_rot_j in enumerate(exp_rot):
                    if np.array_equal(rot_i, exp_rot_j):
                        npt.assert_equal(rot_i, exp_rot_j)
                        # The translations may also be in different cells
                        trans_diff = np.absolute(trans[i] - exp_trans[j])
                        assert np.sum(trans_diff - np.rint(trans_diff)) < 1e-10
                        npt.assert_equal(equiv_atoms[i], exp_equiv_atoms[j])
                        match_found = True
                        break
                if not match_found:
                    pytest.fail(f'No matching rotations found for {rot_i}')
        else:
            npt.assert_equal(rot, exp_rot)
            npt.assert_allclose(trans, exp_trans, atol=1e-10)
            npt.assert_equal(equiv_atoms, exp_equiv_atoms)

    @pytest.mark.parametrize('crystal, kwargs, expected_symprec', [
        (get_crystal('LZO'), {'tol': 1e-12*ureg('angstrom')}, 1e-12),
        (get_crystal('LZO'), {'tol': 1.8897261246e-12*ureg('bohr')}, 1e-12),
        ])
    def test_get_symmetry_equivalent_atoms_with_tol(
            self, mocker, crystal, kwargs, expected_symprec):
        # Ensure tol gets passed correctly to spglib - we don't care about
        # the details of what spglib does
        mock = mocker.patch('spglib.get_symmetry', wraps=spglib.get_symmetry)
        rot, trans, equiv_atoms = crystal.get_symmetry_equivalent_atoms(
            **kwargs)
        npt.assert_almost_equal(mock.call_args[1]['symprec'],
                                expected_symprec)
        npt.assert_equal(rot.shape, (4, 3, 3))
        npt.assert_equal(trans.shape, (4, 3))
        npt.assert_equal(equiv_atoms.shape, (4, 22))

    def test_get_symmetry_equivalent_atoms_errors(self, mocker):
        mocker.patch('spglib.get_symmetry',
                     wraps=spglib.get_symmetry,
                     return_value=None)
        crystal = get_crystal('LZO')
        with pytest.raises(RuntimeError,
                           match='spglib.get_symmetry returned None'):
            crystal.get_symmetry_equivalent_atoms()
