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

ExpectedData = namedtuple(
    "ExpectedData", ["fc_mat_cell0_i0_j0", "fc_mat_celln"]
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
    check_fc_force_constants(
        actual_force_constants, expected_force_constants,
        force_constants, force_constants_unit
    )
    check_fc_sc_matrix(
        actual_force_constants, expected_force_constants, sc_matrix
    )
    check_fc_cell_origins(
        actual_force_constants, expected_force_constants, cell_origins
    )
    check_fc_born(
        actual_force_constants, expected_force_constants,
        born, born_unit
    )
    check_fc_dielectric(
        actual_force_constants, expected_force_constants,
        dielectric, dielectric_unit
    )
    check_fc_n_cells_in_sc(
        actual_force_constants, expected_force_constants, n_cells_in_sc
    )
    check_fc_crystal(actual_force_constants, expected_force_constants, crystal)


def check_fc_force_constants(actual_force_constants, expected_force_constants,
                             overriding_force_constants=None,
                             overriding_force_constants_unit=None):
    # Allow override of expected force constants
    force_constants = overriding_force_constants if \
        overriding_force_constants is not None \
        else expected_force_constants.force_constants
    # Test values are sufficiently close
    npt.assert_allclose(
        actual_force_constants.force_constants.magnitude,
        force_constants.magnitude
    )
    # Allow override of units
    force_constants_unit = overriding_force_constants_unit if \
        overriding_force_constants_unit is not None \
        else expected_force_constants.force_constants.units
    # Test unit matches
    assert actual_force_constants.force_constants_unit == force_constants_unit


def check_fc_sc_matrix(actual_force_constants, expected_force_constants,
                       overriding_sc_matrix=None):
    # Allow override
    sc_matrix = overriding_sc_matrix if overriding_sc_matrix is not None \
        else expected_force_constants.sc_matrix
    # Test values are sufficiently close
    npt.assert_allclose(actual_force_constants.sc_matrix, sc_matrix)


def check_fc_cell_origins(actual_force_constants, expected_force_constants,
                          overriding_cell_origins=None):
    # Allow override
    cell_origins = overriding_cell_origins if \
        overriding_cell_origins is not None \
        else expected_force_constants.cell_origins
    # Test values are sufficiently close
    npt.assert_allclose(actual_force_constants.cell_origins, cell_origins)


def check_fc_crystal(actual_force_constants, expected_force_constants,
                     overriding_crystal=None):
    # Allow override
    crystal = overriding_crystal if overriding_crystal is not None \
        else expected_force_constants.crystal
    # Defer testing of Crystal
    check_crystal_attrs(
        actual_force_constants.crystal, crystal
    )


def check_fc_n_cells_in_sc(actual_force_constants, expected_force_constants,
                           overriding_n_cells_in_sc=None):
    # Allow override
    n_cells_in_sc = overriding_n_cells_in_sc if \
        overriding_n_cells_in_sc is not None \
        else expected_force_constants.n_cells_in_sc
    # Test
    assert actual_force_constants.n_cells_in_sc == n_cells_in_sc


def check_fc_dielectric(actual_force_constants, expected_force_constants,
                        overriding_dielectric=None,
                        overriding_dielectric_unit=None):
    # Allow override
    dielectric = overriding_dielectric if overriding_dielectric is not None \
        else expected_force_constants.dielectric
    # dielectric is optional, detect if option has data in
    if actual_force_constants.dielectric is not None:
        npt.assert_allclose(
            actual_force_constants.dielectric.magnitude, dielectric.magnitude
        )
        # Allow override
        dielectric_unit = overriding_dielectric_unit if \
            overriding_dielectric_unit is not None \
            else expected_force_constants.dielectric.units
        assert actual_force_constants.dielectric_unit == dielectric_unit
    else:
        # Assert that dielectric also is none in expected
        assert dielectric is None


def check_fc_born(actual_force_constants, expected_force_constants,
                  overriding_born=None, overriding_born_unit=None):
    # Allow override
    born = overriding_born if overriding_born is not None \
        else expected_force_constants.born
    # born is optional, detect if option has data in
    if actual_force_constants.born is not None:
        npt.assert_allclose(
            actual_force_constants.born.magnitude,
            born.magnitude
        )
        # Allow override
        born_unit = overriding_born_unit if overriding_born_unit is not None \
            else born.units
        assert actual_force_constants.born_unit == born_unit
    else:
        # Actual is none, check expected is the same
        assert born is None


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
        path = os.path.join(get_data_path(), 'interpolation', directory)
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
