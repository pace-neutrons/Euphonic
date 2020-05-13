import json
import os
import unittest
import numpy.testing as npt
import numpy as np
import pytest
from euphonic import ureg, Crystal
from ..utils import get_data_path


class ExpectedCrystal:

    @property
    def cell_vectors(self) -> np.array:
        return np.array([
            [2.426176, -4.20226,   0.      ],
            [2.426176,  4.20226,   0.      ],
            [0.,        0.,        5.350304]
        ])*ureg('angstrom')

    @property
    def n_atoms(self) -> int:
        return 9

    @property
    def atom_r(self) -> np.array:
        return np.array([
            [0.411708, 0.275682, 0.279408],
            [0.724318, 0.136026, 0.946074],
            [0.863974, 0.588292, 0.612741],
            [0.136026, 0.724318, 0.053926],
            [0.588292, 0.863974, 0.387259],
            [0.275682, 0.411708, 0.720592],
            [0.46496 , 0.      , 0.166667],
            [0.      , 0.46496 , 0.833333],
            [0.53504 , 0.53504 , 0.5     ]
        ])

    @property
    def atom_type(self) -> np.array:
        return np.array(
            ['O', 'O', 'O', 'O', 'O', 'O', 'Si', 'Si', 'Si'],
            dtype='<U2'
        )

    @property
    def atom_mass(self) -> np.array:
        return np.array([
            15.999400000000001, 15.999400000000001, 15.999400000000001,
            15.999400000000001, 15.999400000000001, 15.999400000000001,
            28.085500000000003, 28.085500000000003, 28.085500000000003
        ])*ureg('amu')


class SharedCode:

    @staticmethod
    def quartz_attrs():
        return ExpectedCrystal()

    @classmethod
    def get_quartz_json_file(cls):
        return cls.get_filepath('crystal_quartz.json')

    @classmethod
    def get_filepath(cls, filename):
        return os.path.join(get_data_path(), 'crystal', filename)

    @staticmethod
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

    @staticmethod
    def crystal_from_json_file(filename):
        filepath = SharedCode.get_filepath(filename)
        crystal = Crystal.from_json_file(filepath)
        return crystal


@pytest.mark.unit
class TestObjectCreation:

    @pytest.fixture(params=[SharedCode.quartz_attrs()])
    def crystal_from_constructor(self, request):
        crystal_attrs = request.param
        crystal = Crystal(
            crystal_attrs.cell_vectors,
            crystal_attrs.atom_r,
            crystal_attrs.atom_type,
            crystal_attrs.atom_mass
        )
        return crystal, crystal_attrs

    @pytest.fixture(params=[SharedCode.quartz_attrs()])
    def crystal_from_dict(self, request):
        crystal_attrs = request.param
        d = SharedCode.dict_from_attrs(crystal_attrs)
        crystal = Crystal.from_dict(d)
        return crystal, crystal_attrs

    @pytest.fixture(params=[
        (SharedCode.get_quartz_json_file(), SharedCode.quartz_attrs())
    ])
    def crystal_from_dict(self, request):
        filename, crystal_attrs = request.param
        return SharedCode.crystal_from_json_file(filename), crystal_attrs

    @pytest.mark.parametrize('crystal_creator', [
        pytest.lazy_fixture('crystal_from_constructor'),
        pytest.lazy_fixture('crystal_from_json_file'),
        pytest.lazy_fixture('crystal_from_dict')
    ])
    def test_crystal_create(self, crystal_creator):
        crystal, expected_crystal = crystal_creator
        npt.assert_allclose(crystal.cell_vectors.to('angstrom').magnitude,
                            expected_crystal.cell_vectors.magnitude)
        assert crystal.n_atoms == expected_crystal.n_atoms
        npt.assert_equal(crystal.atom_r, expected_crystal.atom_r)
        npt.assert_equal(crystal.atom_type, expected_crystal.atom_type)
        npt.assert_allclose(crystal.atom_mass.to('amu').magnitude,
                            expected_crystal.atom_mass.magnitude)



################################################################################
# Test object serialisation
################################################################################

# @pytest.mark.unit
# class TestObjectSerialisation:
#
#     def get_quartz_crystal(self):
#         return SharedCode.crystal_from_json_file(SharedCode.get_quartz_json_file())
#
#     @pytest.fixture(
#         params=[
#             # Params: tuple of (Crystal, filetype)
#             (get_quartz_crystal(),'json')],
#         ids=[
#             'quartz_to_json'])
#     def crystal_to_file(self, request, tmpdir):
#         output_file = str(tmpdir.join('tmp.test'))
#         if request.param[1] == 'json':
#             request.param[0].to_json_file(output_file)
#             output_crystal = Crystal.from_json_file(output_file)
#         return request.param[0], output_crystal
#
#     def check_crystal_dict(self, cdict, expected_cdict):
#         npt.assert_allclose(cdict['cell_vectors'],
#                             expected_cdict['cell_vectors'])
#         assert ureg(cdict['cell_vectors_unit']) == ureg(
#             expected_cdict['cell_vectors_unit'])
#         assert cdict['n_atoms'] == cdict['n_atoms']
#         npt.assert_equal(cdict['atom_r'], expected_cdict['atom_r'])
#         npt.assert_equal(cdict['atom_type'], expected_cdict['atom_type'])
#         npt.assert_allclose(cdict['atom_mass'], expected_cdict['atom_mass'])
#         assert ureg(cdict['atom_mass_unit']) == ureg(
#             expected_cdict['atom_mass_unit'])
#
#
#     crystal_to_dict_tests = [
#         pytest.param(get_quartz_crystal().to_dict(),
#                      dict_from_attrs(quartz_attrs()),
#                      id='quartz_to_dict')]
#
#     @pytest.mark.parametrize("crystal_serialiser", [
#         pytest.lazy_fixture(crystal_to_dict_tests)
#     ])
#     def test_crystal_to_dict(self, crystal_serialiser):
#         cdict, expected_cdict = crystal_serialiser
#         check_crystal_dict(cdict, expected_cdict)
#
#
#     def test_crystal_to_file(self, crystal_to_file):
#         check_crystal_attrs(crystal_to_file[0], crystal_to_file[1])
#
#
# ################################################################################
# # Test object methods
# ################################################################################
# quartz_reciprocal_cell = np.array([
#     [1.29487418, -0.74759597, 0.        ],
#     [1.29487418,  0.74759597, 0.        ],
#     [0.,          0.,         1.17436043]])*ureg('1/angstrom')
#
# @pytest.mark.parametrize("crystal,expected_recip", [
#     pytest.param(get_quartz_crystal(), quartz_reciprocal_cell,
#                  id='quartz_reciprocal_cell')
# ])
# def test_reciprocal_cell(crystal, expected_recip):
#     recip = crystal.reciprocal_cell()
#     npt.assert_allclose(recip.to('1/angstrom').magnitude,
#                         expected_recip.to('1/angstrom').magnitude)
#
#
# quartz_cell_volume = 109.09721804482547*ureg('angstrom**3')
#
# @pytest.mark.parametrize("crystal,expected_vol", [
#     pytest.param(get_quartz_crystal(), quartz_cell_volume,
#                  id='quartz_cell_volume')
# ])
# def test_cell_volume(crystal, expected_vol):
#     vol = crystal.cell_volume()
#     npt.assert_allclose(vol.to('angstrom**3').magnitude,
#                         expected_vol.to('angstrom**3').magnitude)