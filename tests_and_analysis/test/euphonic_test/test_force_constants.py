from euphonic import ForceConstants, Crystal
import pytest
import os
from ..utils import get_data_path
import numpy.testing as npt
import numpy as np
import json
from tests_and_analysis.test.euphonic_test.test_crystal import (
    check_crystal_attrs
)
from slugify import slugify
from collections import namedtuple
from euphonic import ureg
from contextlib import contextmanager

ConstructorArgs = namedtuple(
    "ConstructorArgs",
    [
        "crystal", "force_constants", "sc_matrix",
        "cell_origins","born", "dielectric"
    ]
)


class ExpectedForceConstants:

    def __init__(self, dirpath: str, with_material: bool = True):
        """
        Collect data from files for comparison to real ForceConstants files.

        Parameters
        ----------
        dirpath : str
            The directory path containing the files with the force constants
            data in.
        with_material : bool
            True if the material is a prefix to the filename.
        """
        self.dirpath = dirpath
        self.with_material = with_material
        # Set up force constants
        self._force_constants_unit = json.load(
            open(self._get_file_from_dir_and_property("properties", "json"))
        )["force_constants_unit"]
        self._force_constants_mag = np.load(
            self._get_file_from_dir_and_property("force_constants", "npy"),
            allow_pickle=True
        )
        # Set up born
        self._born_unit = json.load(
            open(self._get_file_from_dir_and_property("properties", "json"))
        )["born_unit"]
        self._born_mag = np.load(
            self._get_file_from_dir_and_property("born", "npy"),
            allow_pickle=True
        )
        # Set up dielectric
        self._dielectric_unit = json.load(
            open(self._get_file_from_dir_and_property("properties", "json"))
        )["dielectric_unit"]
        self._dielectric_mag = np.load(
            self._get_file_from_dir_and_property("dielectric", "npy"),
            allow_pickle=True
        )
        # Set up sc_matrix
        self.sc_matrix = np.load(
            self._get_file_from_dir_and_property("sc_matrix", "npy"),
            allow_pickle=True
        )
        # Set up cell origins
        self.cell_origins = np.load(
            self._get_file_from_dir_and_property("cell_origins", "npy"),
            allow_pickle=True
        )
        # Set up crystal
        self.crystal = Crystal.from_json_file(
            self._get_file_from_dir_and_property("crystal", "json")
        )
        # Set up n_cells_in_sc
        self.n_cells_in_sc = json.load(
            open(self._get_file_from_dir_and_property("properties", "json"))
        )["n_cells_in_sc"]

    def _get_file_from_dir_and_property(self, property_name: str,
                                        extension: str) -> str:
        material = os.path.split(self.dirpath)[-1].lower()
        if self.with_material:
            filepath = os.path.join(
                self.dirpath, material + "_" + property_name + "." + extension
            )
        else:
            filepath = os.path.join(
                self.dirpath, property_name + "." + extension
            )
        for f in os.listdir(self.dirpath):
            f_path = os.path.join(self.dirpath, f)
            if filepath == f_path:
                return f_path
        else:
            raise FileNotFoundError("Could not find " + filepath)

    @property
    def born(self):
        if self._born_mag.shape != ():
            return self._born_mag * ureg(self._born_unit)
        else:
            return None

    @property
    def force_constants(self):
        if self._force_constants_mag.shape != ():
            return self._force_constants_mag * ureg(self._force_constants_unit)
        else:
            return None

    @property
    def dielectric(self):
        if self._dielectric_mag.shape != ():
            return self._dielectric_mag * ureg(self._dielectric_unit)
        else:
            return None

    def to_dict(self):
        d = {
            'crystal': self.crystal.to_dict(),
            'force_constants': self.force_constants.magnitude,
            'force_constants_unit': str(self.force_constants.units),
            'n_cells_in_sc': self.n_cells_in_sc,
            'sc_matrix': self.sc_matrix,
            'cell_origins': self.cell_origins,
        }
        if self.born is not None:
            d["born"] = self.born.magnitude
            d["born_unit"] = str(self.born.units)
        if self.dielectric is not None:
            d["dielectric"] = self.dielectric.magnitude
            d["dielectric_unit"] = str(self.dielectric.units)
        return d

    def to_constructor_args(self, crystal=None, force_constants=None,
                            sc_matrix=None, cell_origins=None,
                            born=None, dielectric=None):
        if crystal is None:
            crystal = self.crystal
        if force_constants is None:
            force_constants = self.force_constants
        if sc_matrix is None:
            sc_matrix = self.sc_matrix
        if cell_origins is None:
            cell_origins = self.cell_origins
        if born is None:
            born = self.born
        if dielectric is None:
            dielectric = self.dielectric
        return ConstructorArgs(
            crystal, force_constants, sc_matrix, cell_origins, born, dielectric
        )


def check_force_constant_attrs(actual_force_constants, expected_force_constants,
                               crystal=None, force_constants=None,
                               sc_matrix=None, cell_origins=None,
                               born=None, born_unit=None,
                               dielectric=None, dielectric_unit=None,
                               n_cells_in_sc=None, force_constants_unit=None):
    # Force constants
    if force_constants is None:
        force_constants = expected_force_constants.force_constants
    npt.assert_allclose(
        actual_force_constants.force_constants.magnitude,
        force_constants.magnitude
    )
    if force_constants_unit is None:
        force_constants_unit = expected_force_constants.force_constants.units
    assert actual_force_constants.force_constants_unit == force_constants_unit
    # sc matrix
    if sc_matrix is None:
        sc_matrix = expected_force_constants.sc_matrix
    npt.assert_allclose(actual_force_constants.sc_matrix, sc_matrix)
    # Cell origins
    if cell_origins is None:
        cell_origins = expected_force_constants.cell_origins
    npt.assert_allclose(actual_force_constants.cell_origins, cell_origins)
    # Born
    if born is None:
        born = expected_force_constants.born
    if actual_force_constants.born is not None:
        npt.assert_allclose(
            actual_force_constants.born.magnitude,
            born.magnitude
        )
        if born_unit is None:
            born_unit = born.units
            print(born_unit)
        else:
            print("Is not none: " + str(born_unit))
        assert actual_force_constants.born_unit == born_unit
    else:
        assert born is None
    # Dielectric
    if dielectric is None:
        dielectric = expected_force_constants.dielectric
    if actual_force_constants.dielectric is not None:
        npt.assert_allclose(
            actual_force_constants.dielectric.magnitude, dielectric.magnitude
        )
        if dielectric_unit is None:
            dielectric_unit = expected_force_constants.dielectric.units
        assert actual_force_constants.dielectric_unit == dielectric_unit
    else:
        assert dielectric is None
    # n_cells_in_sc
    if n_cells_in_sc is None:
        n_cells_in_sc = expected_force_constants.n_cells_in_sc
    assert actual_force_constants.n_cells_in_sc == n_cells_in_sc
    # Crystal
    if crystal is None:
        crystal = expected_force_constants.crystal
    check_crystal_attrs(
        actual_force_constants.crystal, crystal
    )


def quartz_attrs():
    test_data_dir = os.path.join(
        get_data_path(), "interpolation", "quartz"
    )
    return ExpectedForceConstants(test_data_dir)


@contextmanager
def no_exception():
    try:
        yield
    except Exception as e:
        raise pytest.fail("Exception raised: " + str(e))


@pytest.mark.unit
class TestObjectCreation:

    @pytest.mark.parametrize(("castep_bin_dir", "castep_bin_file"), [
        ('LZO', 'La2Zr2O7.castep_bin'),
        ('graphite', 'graphite.castep_bin'),
        ('quartz', 'quartz.castep_bin')
    ])
    def test_creation_from_castep(self, castep_bin_dir, castep_bin_file):
        dirpath = os.path.join(
            get_data_path(), 'interpolation', castep_bin_dir
        )
        filepath = os.path.join(dirpath,  castep_bin_file)
        fc = ForceConstants.from_castep(filepath)
        check_force_constant_attrs(fc, ExpectedForceConstants(dirpath))

    @pytest.mark.parametrize(("json_dir", "json_file"), [
        ('LZO', 'lzo_force_constants.json'),
        ('graphite', 'graphite_force_constants.json'),
        ('quartz', 'quartz_force_constants.json')
    ])
    def test_creation_from_json(self, json_dir, json_file):
        dirpath = os.path.join(
            get_data_path(), 'interpolation', json_dir
        )
        filepath = os.path.join(dirpath, json_file)
        fc = ForceConstants.from_json_file(filepath)
        check_force_constant_attrs(fc, ExpectedForceConstants(dirpath))

    @pytest.mark.parametrize(("json_dir", "json_file"), [
        ('LZO', 'lzo_force_constants.json'),
        ('graphite', 'graphite_force_constants.json'),
        ('quartz', 'quartz_force_constants.json')
    ])
    def test_creation_from_dict(self, json_dir, json_file):
        dirpath = os.path.join(
            get_data_path(), 'interpolation', json_dir
        )
        expected_fc = ExpectedForceConstants(dirpath)
        fc = ForceConstants.from_dict(expected_fc.to_dict())
        check_force_constant_attrs(fc, expected_fc)

    @pytest.mark.parametrize("phonopy_args", [
        {"summary_name": "phonopy.yaml"},
        {
            "summary_name": "phonopy_nofc.yaml", "fc_name": "FORCE_CONSTANTS"
        },
        {
            "summary_name": "phonopy_nofc.yaml",
            "fc_name": "FULL_FORCE_CONSTANTS"
        },
        {
            "summary_name": "phonopy_prim.yaml",
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
        {
            "summary_name": "phonopy_nofc_noborn.yaml",
            "born_name": "BORN"
        },
        {
            "summary_name": "phonopy_prim_nofc.yaml",
            "fc_name": "primitive_force_constants.hdf5"
        }
    ])
    def test_creation_from_phonopy(self, phonopy_args):
        test_data_dir = slugify(
            "-".join(phonopy_args.keys()) + "-"
            + "-".join(phonopy_args.values())
        )
        phonopy_args["path"] = os.path.join(
            get_data_path(), 'phonopy_data', 'NaCl', 'interpolation'
        )
        fc = ForceConstants.from_phonopy(**phonopy_args)
        expected_dirpath = os.path.join(phonopy_args["path"], test_data_dir)
        check_force_constant_attrs(
            fc, ExpectedForceConstants(expected_dirpath, with_material=False)
        )

    correct_elements = [
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
    ]

    @pytest.fixture(params=correct_elements)
    def inject_elements(self, request):
        injected_args = request.param
        test_data_dir = os.path.join(
            get_data_path(), "interpolation", "quartz"
        )
        expected_fc = ExpectedForceConstants(test_data_dir)
        # Inject the faulty value and get a tuple of constructor arguments
        args = expected_fc.to_constructor_args(**injected_args)
        return args, expected_fc

    def test_correct_object_creation(self, inject_elements):
        args, expected_fc = inject_elements
        fc = ForceConstants(*args)
        check_force_constant_attrs(fc, expected_fc, **args._asdict())

    faulty_elements = [
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
    ]

    @pytest.fixture(params=faulty_elements)
    def inject_faulty_elements(self, request):
        faulty_args, expected_exception = request.param
        test_data_dir = os.path.join(
            get_data_path(), "interpolation", "quartz"
        )
        expected_fc = ExpectedForceConstants(test_data_dir)
        # Inject the faulty value and get a tuple of constructor arguments
        args = expected_fc.to_constructor_args(**faulty_args)
        return args, expected_exception

    def test_faulty_object_creation(self, inject_faulty_elements):
        faulty_args, expected_exception = inject_faulty_elements
        with pytest.raises(expected_exception):
            ForceConstants(*faulty_args)

    @pytest.fixture
    def lzo_fc(self):
        ExpectedData = namedtuple(
            "ExpectedData", ["fc_mat_cell0_i0_j0", "fc_mat_cell3_i10_j5"]
        )
        fc_mat_cell0_i0_j0 = np.array(
            [[1.26492555e-01, -2.31204635e-31, -1.16997352e-13],
             [-2.31204635e-31, 1.26492555e-01, 3.15544362e-30],
             [-1.16997352e-13, 1.05181454e-30, 1.26492555e-01]]
        ) * ureg('hartree/bohr**2')
        fc_mat_cell3_i10_j5 = np.array(
            [[-8.32394989e-04, -2.03285211e-03, 3.55359333e-04],
             [-2.22156212e-03, -6.29315975e-04, 1.21568713e-03],
             [7.33617499e-05, 1.16282999e-03, 5.22410338e-05]]
        ) * ureg('hartree/bohr**2')
        path = os.path.join(get_data_path(), 'interpolation', 'LZO')
        expected_data = ExpectedData(
            fc_mat_cell0_i0_j0=fc_mat_cell0_i0_j0,
            fc_mat_cell3_i10_j5=fc_mat_cell3_i10_j5
        )
        filename = os.path.join(path, 'La2Zr2O7.castep_bin')
        fc = ForceConstants.from_castep(filename)
        return fc, expected_data

    def test_fc_mat_cell0_i0_j0_read(self, lzo_fc):
        fc, expected_data = lzo_fc
        npt.assert_allclose(
            fc.force_constants[0, 0:3, 0:3].magnitude,
            expected_data.fc_mat_cell0_i0_j0.magnitude
        )

    def test_fc_mat_cell3_i10_j5_read(self, lzo_fc):
        fc, expected_data = lzo_fc
        npt.assert_allclose(
            fc.force_constants[3, 30:33, 15:18].magnitude,
            expected_data.fc_mat_cell3_i10_j5.magnitude
        )
