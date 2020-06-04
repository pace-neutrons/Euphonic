import pytest
import os
from tests_and_analysis.test.utils import get_data_path
from euphonic import ForceConstants, ureg
import numpy as np
import numpy.testing as npt
from tests_and_analysis.test.euphonic_test.force_constant_utils import (
    check_force_constant_attrs, ExpectedForceConstants, quartz_attrs,
    ExpectedData
)
from slugify import slugify


@pytest.mark.unit
class TestObjectCreation:

    @pytest.fixture(params=[
        ('LZO', 'La2Zr2O7.castep_bin'),
        ('graphite', 'graphite.castep_bin'),
        ('quartz', 'quartz.castep_bin')
    ])
    def create_from_castep(self, request):
        castep_bin_dir, castep_bin_file = request.param
        dirpath = os.path.join(
            get_data_path(), 'force_constants', castep_bin_dir
        )
        json_filepath = os.path.join(
            dirpath, castep_bin_dir.lower() + "_force_constants.json"
        )
        castep_filepath = os.path.join(dirpath,  castep_bin_file)
        expected_fc = ExpectedForceConstants(json_filepath)
        fc = ForceConstants.from_castep(castep_filepath)
        return fc, expected_fc

    @pytest.fixture(params=['LZO', 'graphite', 'quartz'])
    def create_from_json(self, request):
        material_name = request.param
        json_file = os.path.join(
            get_data_path(), 'force_constants',
            material_name, material_name.lower() + "_force_constants.json"
        )
        expected_fc = ExpectedForceConstants(json_file)
        fc = ForceConstants.from_json_file(json_file)
        return fc, expected_fc

    @pytest.fixture(params=['LZO', 'graphite', 'quartz'])
    def create_from_dict(self, request):
        material_name = request.param
        json_file = os.path.join(
            get_data_path(), 'force_constants',
            material_name, material_name.lower() + "_force_constants.json"
        )
        expected_fc = ExpectedForceConstants(json_file)
        fc = ForceConstants.from_dict(expected_fc.to_dict())
        return fc, expected_fc

    @pytest.fixture(params=[
        {"summary_name": "phonopy.yaml"},
        {"summary_name": "phonopy_prim.yaml"},
        {"summary_name": "phonopy_nofc.yaml", "fc_name": "FORCE_CONSTANTS"},
        {"summary_name": "phonopy_nofc.yaml","fc_name": "FULL_FORCE_CONSTANTS"},
        {
            "summary_name": "phonopy_prim_nofc.yaml",
            "fc_name": "PRIMITIVE_FORCE_CONSTANTS"
        },
        {
            "summary_name": "phonopy_nofc.yaml",
            "fc_name": "full_force_constants.hdf5"
        },
        {
            "summary_name": "phonopy_nofc.yaml",
            "fc_name": "force_constants.hdf5"
        },
        {"summary_name": "phonopy_nofc_noborn.yaml", "born_name": "BORN"},
        {
            "summary_name": "phonopy_prim_nofc.yaml",
            "fc_name": "primitive_force_constants.hdf5"
        }
    ])
    def create_from_phonopy(self, request):
        phonopy_args = request.param
        test_data_file = slugify(
            "-".join(phonopy_args.values())
        ) + ".json"
        phonopy_args["path"] = os.path.join(
            get_data_path(), 'phonopy_data', 'NaCl', 'force_constants'
        )
        expected_filepath = os.path.join(phonopy_args["path"], test_data_file)
        expected_fc = ExpectedForceConstants(expected_filepath)
        fc = ForceConstants.from_phonopy(**phonopy_args)
        return fc, expected_fc

    @pytest.fixture(params=[
        (
            {
                "born": quartz_attrs().to_constructor_args().born,
                "dielectric": None
            }
        ),
        (
            {
                "born": None,
                "dielectric": quartz_attrs().to_constructor_args().dielectric
            }
        )
    ])
    def inject_elements(self, request):
        injected_args = request.param
        test_data_file = os.path.join(
            get_data_path(), "force_constants",
            "quartz", "quartz_force_constants.json"
        )
        expected_fc = ExpectedForceConstants(test_data_file)
        # Inject the faulty value and get a tuple of constructor arguments
        args = expected_fc.to_constructor_args(**injected_args)
        expected_fc = ExpectedForceConstants(test_data_file, args._asdict())
        fc = ForceConstants(*args)
        return fc, expected_fc

    @pytest.mark.parametrize(("force_constants_creator"), [
        pytest.lazy_fixture("inject_elements"),
        pytest.lazy_fixture("create_from_dict"),
        pytest.lazy_fixture("create_from_json"),
        pytest.lazy_fixture("create_from_phonopy"),
        pytest.lazy_fixture("create_from_castep")
    ])
    def test_correct_object_creation(self, force_constants_creator):
        force_constants, expected_force_constants = force_constants_creator
        check_force_constant_attrs(force_constants, expected_force_constants)

    @pytest.fixture(params=[
        (
            {
                "sc_matrix": quartz_attrs().sc_matrix[:2]
            },
            ValueError
        ),
        (
            {
                "cell_origins": quartz_attrs().sc_matrix
            },
            ValueError
        ),
        (
            {
                "force_constants": quartz_attrs().force_constants[:2]
            },
            ValueError
        ),
        (
            {
                "born": quartz_attrs().born[:2]
            },
            ValueError
        ),
        (
            {
                "dielectric": quartz_attrs().dielectric[:2]
            },
            ValueError
        ),
        (
            {
                "sc_matrix": list(quartz_attrs().sc_matrix)
            },
            TypeError
        ),
        (
            {
                "cell_origins": quartz_attrs().sc_matrix * ureg.meter
            },
            TypeError
        ),
        (
            {
                "crystal": quartz_attrs().crystal.to_dict()
            },
            TypeError
        ),
        (
            {
                "force_constants": quartz_attrs().force_constants.magnitude
            },
            TypeError
        ),
        (
            {
                "born": list(quartz_attrs().born.magnitude)
            },
            TypeError
        ),
        (
            {
                "dielectric": quartz_attrs().dielectric.shape
            },
            TypeError
        ),
    ])
    def inject_faulty_elements(self, request):
        faulty_args, expected_exception = request.param
        test_data_dir = os.path.join(
            get_data_path(), "force_constants",
            "quartz", "quartz_force_constants.json"
        )
        expected_fc = ExpectedForceConstants(test_data_dir)
        # Inject the faulty value and get a tuple of constructor arguments
        args = expected_fc.to_constructor_args(**faulty_args)
        return args, expected_exception

    def test_faulty_object_creation(self, inject_faulty_elements):
        faulty_args, expected_exception = inject_faulty_elements
        with pytest.raises(expected_exception):
            ForceConstants(*faulty_args)

    @pytest.fixture(params=[
        (
            'LZO', 'La2Zr2O7.castep_bin',
            np.array([
                [1.26492555e-01, -2.31204635e-31, -1.16997352e-13],
                [-2.31204635e-31, 1.26492555e-01, 3.15544362e-30],
                [-1.16997352e-13, 1.05181454e-30, 1.26492555e-01]
            ]) * ureg('hartree/bohr**2'),
            [3, 30, 33, 15, 18],
            np.array([
                [-8.32394989e-04, -2.03285211e-03, 3.55359333e-04],
                [-2.22156212e-03, -6.29315975e-04, 1.21568713e-03],
                [7.33617499e-05, 1.16282999e-03, 5.22410338e-05]
            ]) * ureg('hartree/bohr**2')
        ),
        (
            'graphite', 'graphite.castep_bin',
            np.array([
                [6.35111387e-01, 2.76471554e-18, 0.00000000e+00],
                [2.05998413e-18, 6.35111387e-01, 0.00000000e+00],
                [0.00000000e+00, 0.00000000e+00, 1.52513691e-01]
            ]) * ureg('hartree/bohr**2'),
            [10, 6, 9, 9, 12],
            np.array([
                [-8.16784177e-05, -1.31832252e-05, -1.11904290e-05],
                [-1.31832252e-05, 5.42461675e-05, 3.73913780e-06],
                [-1.11904290e-05, 3.73913780e-06, -2.99013850e-05]
            ]) * ureg('hartree/bohr**2')
        ),
        (
            'quartz', 'quartz.castep_bin',
            np.array([
                [0.22324308, 0.29855096, 0.31240272],
                [0.29855096, 0.43968813, 0.34437270],
                [0.31240272, 0.34437270, 0.33448280]
            ]) * ureg('hartree/bohr**2'),
            [4, 6, 9, 21, 24],
            np.array([
                [6.37874988e-05, -2.73116279e-04, -1.73624413e-04],
                [1.99404810e-08, 4.40511248e-04, 2.33322837e-04],
                [-1.85830030e-05, -9.05395392e-05, 2.17526152e-05]
            ]) * ureg('hartree/bohr**2')
        )
    ])
    def fc_mat(self, request):
        directory, filename, fc_mat_cell0_i0_j0, celln, fc_mat_celln = \
            request.param
        path = os.path.join(get_data_path(), 'force_constants', directory)
        expected_data = ExpectedData(
            fc_mat_cell0_i0_j0=fc_mat_cell0_i0_j0,
            fc_mat_celln=fc_mat_celln
        )
        filepath = os.path.join(path, filename)
        fc = ForceConstants.from_castep(filepath)
        return fc, celln, expected_data

    def test_fc_mat_cells(self, fc_mat):
        fc, celln, expected_data = fc_mat
        npt.assert_allclose(
            fc.force_constants[0, 0:3, 0:3].magnitude,
            expected_data.fc_mat_cell0_i0_j0.magnitude
        )
        npt.assert_allclose(
            fc.force_constants[
                celln[0], celln[1]:celln[2], celln[3]:celln[4]
            ].magnitude,
            expected_data.fc_mat_celln.magnitude
        )
