from collections import namedtuple
import json
import numpy as np
import numpy.testing as npt
from euphonic import Crystal, ureg
import os
from tests_and_analysis.test.euphonic_test.test_crystal import (
    check_crystal_attrs
)
from tests_and_analysis.test.utils import get_data_path
import pytest
from contextlib import contextmanager


ConstructorArgs = namedtuple(
    "ConstructorArgs",
    [
        "crystal", "force_constants", "sc_matrix",
        "cell_origins", "born", "dielectric"
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
