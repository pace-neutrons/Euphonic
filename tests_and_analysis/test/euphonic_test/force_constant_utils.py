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

    def __init__(self, force_constants_json_file: str, kwargs: Dict = {}):
        """
        Collect data from files for comparison to real ForceConstants files.

        Parameters
        ----------
        force_constants_json_file : str
            The json file containing all the information for the force constants
            object.
        kwargs : Dict
            A dictionary of values to override values such as force constants
            from the file
        """
        self._data = json.load(open(force_constants_json_file))
        for attr_name, attr_value in kwargs.items():
            setattr(self, "_" + attr_name, attr_value)
        # Create crystal and sanitise the input lists into arrays and quantities
        crystal_dict = self._data["crystal"]
        cell_vectors = np.array(crystal_dict["cell_vectors"]) * \
            ureg(crystal_dict["cell_vectors_unit"])
        atom_r = np.array(crystal_dict["atom_r"])
        atom_type = np.array(crystal_dict["atom_type"])
        atom_mass = np.array(crystal_dict["atom_mass"]) * \
            ureg(crystal_dict["atom_mass_unit"])
        self.crystal = Crystal(cell_vectors, atom_r, atom_type, atom_mass)

    @property
    def sc_matrix(self):
        if hasattr(self, "_sc_matrix"):
            return self._sc_matrix
        else:
            return np.array(self._data["sc_matrix"])

    @property
    def cell_origins(self):
        if hasattr(self, "_cell_origins"):
            return self._cell_origins
        else:
            return np.array(self._data["cell_origins"])

    @property
    def born(self):
        if hasattr(self, "_born"):
            return self._born
        else:
            if self._data["born"] is not None:
                return np.array(self._data["born"]) * \
                       ureg(self._data["born_unit"])
            else:
                return None

    @property
    def force_constants(self):
        if hasattr(self, "_force_constants"):
            return self._force_constants
        else:
            return np.array(self._data["force_constants"]) * \
                   ureg(self._data["force_constants_unit"])

    @property
    def dielectric(self):
        if hasattr(self, "_dielectric"):
            return self._dielectric
        else:
            if self._data["dielectric"] is not None:
                return np.array(self._data["dielectric"]) * \
                       ureg(self._data["dielectric_unit"])
            else:
                return None

    @property
    def n_cells_in_sc(self):
        if hasattr(self, "_n_cells_in_sc"):
            return self._n_cells_in_sc
        else:
            return self._data["n_cells_in_sc"]

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


def check_fc_dielectric(actual_force_constants, expected_force_constants):
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


def check_fc_born(actual_force_constants, expected_force_constants):
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
    test_data_file = os.path.join(
        get_data_path(), "interpolation",
        "quartz", "quartz_force_constants.json"
    )
    return ExpectedForceConstants(test_data_file)


@contextmanager
def no_exception():
    try:
        yield
    except Exception as e:
        raise pytest.fail("Exception raised: " + str(e))
