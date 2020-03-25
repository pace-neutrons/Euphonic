import json
import os
import unittest
import numpy.testing as npt
import numpy as np
import pytest
from euphonic import ureg, Crystal
from .utils import get_data_path


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

quartz_json_file = 'crystal_quartz.json'

def get_filepath(filename):
    return os.path.join(get_data_path(), 'crystal',
                        filename)


def dict_from_attrs(crystal_attrs):
    d = {}
    d['cell_vectors'] = crystal_attrs.cell_vectors.magnitude
    d['cell_vectors_unit'] = str(crystal_attrs.cell_vectors.units)
    d['n_atoms'] = crystal_attrs.n_atoms
    d['atom_r'] = crystal_attrs.atom_r
    d['atom_type'] = crystal_attrs.atom_type
    d['atom_mass'] = crystal_attrs.atom_mass.magnitude
    d['atom_mass_unit'] = str(crystal_attrs.atom_mass.units)
    return d


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
    d = dict_from_attrs(crystal_attrs)
    crystal = Crystal.from_dict(d)
    return crystal


def check_crystal_attrs(crystal, expected_crystal):
    npt.assert_allclose(crystal.cell_vectors.to('angstrom').magnitude,
                        expected_crystal.cell_vectors.magnitude)
    assert crystal.n_atoms == expected_crystal.n_atoms
    npt.assert_equal(crystal.atom_r, expected_crystal.atom_r)
    npt.assert_equal(crystal.atom_type, expected_crystal.atom_type)
    npt.assert_allclose(crystal.atom_mass.to('amu').magnitude,
                        expected_crystal.atom_mass.magnitude)


crystal_create_tests = [
    pytest.param(crystal_from_constructor(quartz_attrs()), quartz_attrs(),
                 id='quartz_from_constructor'),
    pytest.param(crystal_from_dict(quartz_attrs()), quartz_attrs(),
                 id='quartz_from_dict'),
    pytest.param(crystal_from_json_file(quartz_json_file), quartz_attrs(),
                 id='quartz_from_json_file')]


@pytest.mark.parametrize("crystal,expected_crystal",
                         crystal_create_tests)
def test_crystal_create(crystal, expected_crystal):
    check_crystal_attrs(crystal, expected_crystal)


################################################################################
# Test object serialisation
################################################################################
def get_quartz_crystal():
    return crystal_from_json_file(quartz_json_file)

@pytest.fixture(
    params=[
    # Params: tuple of (Crystal, filetype)
    (get_quartz_crystal(),'json')],
    ids=[
        'quartz_to_json'])
def crystal_to_file(request, tmpdir):
    output_file = str(tmpdir.join('tmp.test'))
    if request.param[1] == 'json':
        request.param[0].to_json_file(output_file)
        output_crystal = Crystal.from_json_file(output_file)
    return request.param[0], output_crystal

def check_crystal_dict(cdict, expected_cdict):
    npt.assert_allclose(cdict['cell_vectors'],
                        expected_cdict['cell_vectors'])
    assert ureg(cdict['cell_vectors_unit']) == ureg(
        expected_cdict['cell_vectors_unit'])
    assert cdict['n_atoms'] == cdict['n_atoms']
    npt.assert_equal(cdict['atom_r'], expected_cdict['atom_r'])
    npt.assert_equal(cdict['atom_type'], expected_cdict['atom_type'])
    npt.assert_allclose(cdict['atom_mass'], expected_cdict['atom_mass'])
    assert ureg(cdict['atom_mass_unit']) == ureg(
        expected_cdict['atom_mass_unit'])


crystal_to_dict_tests = [
    pytest.param(get_quartz_crystal().to_dict(),
                 dict_from_attrs(quartz_attrs()),
                 id='quartz_to_dict')]

@pytest.mark.parametrize("cdict,expected_cdict", crystal_to_dict_tests)
def test_crystal_to_dict(cdict, expected_cdict):
    check_crystal_dict(cdict, expected_cdict)


def test_crystal_to_file(crystal_to_file):
    check_crystal_attrs(crystal_to_file[0], crystal_to_file[1])


################################################################################
# Test object methods
################################################################################
quartz_reciprocal_cell = np.array([
    [1.29487418, -0.74759597, 0.        ],
    [1.29487418,  0.74759597, 0.        ],
    [0.,          0.,         1.17436043]])*ureg('1/angstrom')

@pytest.mark.parametrize("crystal,expected_recip", [
    pytest.param(get_quartz_crystal(), quartz_reciprocal_cell,
                 id='quartz_reciprocal_cell')
])
def test_reciprocal_cell(crystal, expected_recip):
    recip = crystal.reciprocal_cell()
    npt.assert_allclose(recip.to('1/angstrom').magnitude,
                        expected_recip.to('1/angstrom').magnitude)


quartz_cell_volume = 109.09721804482547*ureg('angstrom**3')

@pytest.mark.parametrize("crystal,expected_vol", [
    pytest.param(get_quartz_crystal(), quartz_cell_volume,
                 id='quartz_cell_volume')
])
def test_cell_volume(crystal, expected_vol):
    vol = crystal.cell_volume()
    npt.assert_allclose(vol.to('angstrom**3').magnitude,
                        expected_vol.to('angstrom**3').magnitude)