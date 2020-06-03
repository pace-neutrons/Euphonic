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
from typing import Dict


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

    def __init__(self, dirpath: str, with_material: bool = True,
                 kwargs: Dict = {}):
        """
        Collect data from files for comparison to real ForceConstants files.

        Parameters
        ----------
        dirpath : str
            The directory path containing the files with the force constants
            data in.
        with_material : bool
            True if the material is a prefix to the filename.
        kwargs : Dict
            A dictionary of values to override values such as force constants
            from the file
        """
        self.dirpath = dirpath
        self.with_material = with_material
        # Set up force constants
        if "force_constants" in kwargs:
            self._force_constants = kwargs["force_constants"]
        else:
            self._force_constants_unit = json.load(
                open(self._get_file_from_dir_and_property("properties", "json"))
            )["force_constants_unit"]
            self._force_constants_mag = np.load(
                self._get_file_from_dir_and_property("force_constants", "npy"),
                allow_pickle=True
            )
        # Set up born
        if "born" in kwargs:
            self._born = kwargs["born"]
        else:
            self._born_unit = json.load(
                open(self._get_file_from_dir_and_property("properties", "json"))
            )["born_unit"]
            self._born_mag = np.load(
                self._get_file_from_dir_and_property("born", "npy"),
                allow_pickle=True
            )
        # Set up dielectric
        if "dielectric" in kwargs:
            self._dielectric = kwargs["dielectric"]
        else:
            self._dielectric_unit = json.load(
                open(self._get_file_from_dir_and_property("properties", "json"))
            )["dielectric_unit"]
            self._dielectric_mag = np.load(
                self._get_file_from_dir_and_property("dielectric", "npy"),
                allow_pickle=True
            )
        # Set up sc_matrix
        if "sc_matrix" in kwargs:
            self.sc_matrix = kwargs["sc_matrix"]
        else:
            self.sc_matrix = np.load(
                self._get_file_from_dir_and_property("sc_matrix", "npy"),
                allow_pickle=True
            )
        # Set up cell origins
        if "cell_origins" in kwargs:
            self.cell_origins = kwargs["cell_origins"]
        else:
            self.cell_origins = np.load(
                self._get_file_from_dir_and_property("cell_origins", "npy"),
                allow_pickle=True
            )
        # Set up crystal
        if "crystal" in kwargs:
            self.crystal = kwargs["crystal"]
        else:
            self.crystal = Crystal.from_json_file(
                self._get_file_from_dir_and_property("crystal", "json")
            )
        # Set up n_cells_in_sc
        if "n_cells_in_sc" in kwargs:
            self.n_cells_in_sc = kwargs["n_cells_in_sc"]
        else:
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
        if hasattr(self, "_born"):
            return self._born
        else:
            if self._born_mag.shape != ():
                return self._born_mag * ureg(self._born_unit)
            else:
                return None

    @property
    def force_constants(self):
        if hasattr(self, "_force_constants"):
            return self._force_constants
        else:
            if self._force_constants_mag.shape != ():
                return self._force_constants_mag * \
                    ureg(self._force_constants_unit)
            else:
                return None

    @property
    def dielectric(self):
        if hasattr(self, "_dielectric"):
            return self._dielectric
        else:
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


def check_force_constant_attrs(
        actual_force_constants, expected_force_constants):
    # Test force constant values are sufficiently close
    npt.assert_allclose(
        actual_force_constants.force_constants.magnitude,
        expected_force_constants.force_constants.magnitude
    )
    # Test force constant units match
    assert actual_force_constants.force_constants_unit == \
           expected_force_constants.force_constants.units
    # Test sc_matrix values are sufficiently close
    npt.assert_allclose(
        actual_force_constants.sc_matrix, expected_force_constants.sc_matrix
    )
    # Test cell_origins values are sufficiently close
    npt.assert_allclose(
        actual_force_constants.cell_origins,
        expected_force_constants.cell_origins
    )
    check_fc_born(
        actual_force_constants, expected_force_constants
    )
    check_fc_dielectric(
        actual_force_constants, expected_force_constants
    )
    assert actual_force_constants.n_cells_in_sc == \
        expected_force_constants.n_cells_in_sc
    # Defer testing of Crystal
    check_crystal_attrs(
        actual_force_constants.crystal, expected_force_constants.crystal
    )


def check_fc_dielectric(actual_force_constants, expected_force_constants,
                        overriding_dielectric_unit=None):
    # dielectric is optional, detect if option has data in
    if actual_force_constants.dielectric is not None:
        npt.assert_allclose(
            actual_force_constants.dielectric.magnitude,
            expected_force_constants.dielectric.magnitude
        )
        assert actual_force_constants.dielectric_unit == \
            expected_force_constants.dielectric.units
    else:
        # Assert that dielectric also is none in expected
        assert expected_force_constants.dielectric is None


def check_fc_born(actual_force_constants, expected_force_constants,
                  overriding_born=None, overriding_born_unit=None):
    # born is optional, detect if option has data in
    if actual_force_constants.born is not None:
        npt.assert_allclose(
            actual_force_constants.born.magnitude,
            expected_force_constants.born.magnitude
        )
        assert actual_force_constants.born_unit == \
            expected_force_constants.born.units
    else:
        # Actual is none, check expected is the same
        assert expected_force_constants.born is None


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
