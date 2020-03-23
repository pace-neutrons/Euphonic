import os
import unittest
import numpy.testing as npt
import numpy as np
import pytest
from euphonic import ureg, Crystal


def quartz_attrs():
    # Create trivial function object so attributes can be assigned to it
    expected_crystal = type('', (), {})()
    expected_crystal.cell_vectors = np.array([
        [2.426176, -4.20226,   0.      ],
        [2.426176,  4.20226,   0.      ],
        [0.,        0.,        5.350304]])*ureg('angstrom')
    expected_crystal.n_atoms = 9
    expected_crystal.atom_r = np.array([
        [0.411708, 0.275682, 0.279408],
        [0.724318, 0.136026, 0.946074],
        [0.863974, 0.588292, 0.612741],
        [0.136026, 0.724318, 0.053926],
        [0.588292, 0.863974, 0.387259],
        [0.275682, 0.411708, 0.720592],
        [0.46496 , 0.      , 0.166667],
        [0.      , 0.46496 , 0.833333],
        [0.53504 , 0.53504 , 0.5     ]])
    expected_crystal.atom_type = np.array(
        ['O', 'O', 'O', 'O', 'O', 'O', 'Si', 'Si', 'Si'], dtype='<U2')
    expected_crystal.atom_mass = np.array(
        [15.999400000000001, 15.999400000000001, 15.999400000000001,
         15.999400000000001, 15.999400000000001, 15.999400000000001,
         28.085500000000003, 28.085500000000003, 28.085500000000003])*ureg(
            'amu')
    return expected_crystal

def get_filepath(filename):
    return os.path.join(os.path.dirname(__file__), 'test_files', 'crystal',
                        filename)


################################################################################
# Test object creation
################################################################################
def crystal_from_constructor(crystal_attrs):
        crystal = Crystal(crystal_attrs.cell_vectors,
                          crystal_attrs.n_atoms,
                          crystal_attrs.atom_r,
                          crystal_attrs.atom_type,
                          crystal_attrs.atom_mass)
        return crystal


def crystal_from_json_file(filename):
    filepath = get_filepath(filename)
    crystal = Crystal.from_json_file(filepath)
    return crystal


def crystal_from_dict(crystal_attrs):
    d = {}
    d['cell_vectors'] = crystal_attrs.cell_vectors.magnitude
    d['cell_vectors_unit'] = str(crystal_attrs.cell_vectors.units)
    d['n_atoms'] = crystal_attrs.n_atoms
    d['atom_r'] = crystal_attrs.atom_r
    d['atom_type'] = crystal_attrs.atom_type
    d['atom_mass'] = crystal_attrs.atom_mass.magnitude
    d['atom_mass_unit'] = str(crystal_attrs.atom_mass.units)
    crystal = Crystal.from_dict(d)
    return crystal


create_crystal_tests = [
    pytest.param(crystal_from_constructor(quartz_attrs()), quartz_attrs(),
                 id='quartz_from_constructor'),
    pytest.param(crystal_from_dict(quartz_attrs()), quartz_attrs(),
                 id='quartz_from_dict'),
    pytest.param(crystal_from_json_file('crystal_quartz.json'), quartz_attrs(),
                 id='quartz_from_json_file')
]


@pytest.mark.parametrize("crystal,expected_crystal_attrs", create_crystal_tests)
def test_cell_vectors(crystal, expected_crystal_attrs):
    npt.assert_allclose(crystal.cell_vectors.to('angstrom').magnitude,
                        expected_crystal_attrs.cell_vectors.magnitude)


@pytest.mark.parametrize("crystal,expected_crystal_attrs", create_crystal_tests)
def test_n_atoms(crystal, expected_crystal_attrs):
    assert crystal.n_atoms == expected_crystal_attrs.n_atoms


@pytest.mark.parametrize("crystal,expected_crystal_attrs", create_crystal_tests)
def test_atom_r(crystal, expected_crystal_attrs):
    npt.assert_equal(crystal.atom_r,
                     expected_crystal_attrs.atom_r)


@pytest.mark.parametrize("crystal,expected_crystal_attrs", create_crystal_tests)
def test_atom_type(crystal, expected_crystal_attrs):
    npt.assert_equal(crystal.atom_type,
                     expected_crystal_attrs.atom_type)


@pytest.mark.parametrize("crystal,expected_crystal_attrs", create_crystal_tests)
def test_atom_mass(crystal, expected_crystal_attrs):
    npt.assert_allclose(crystal.atom_mass.to('amu').magnitude,
                        expected_crystal_attrs.atom_mass.magnitude)


################################################################################
# Test object serialisation
################################################################################
def crystal_to_dict():
    pass


def test_to_dict():
    pass


def test_to_json_file():
    pass


################################################################################
# Test object methods
################################################################################
def test_reciprocal_cell():
    pass


def test_cell_volume():
    pass