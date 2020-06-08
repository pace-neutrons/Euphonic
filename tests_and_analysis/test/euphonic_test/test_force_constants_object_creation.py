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
