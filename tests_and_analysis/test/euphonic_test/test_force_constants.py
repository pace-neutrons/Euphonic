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


class ExpectedForceConstants:

    def __init__(self, dirpath):
        """
        Collect data from files for comparison to real ForceConstants files.

        Parameters
        ----------
        dirpath : str
            The directory path containing the files with the force constants
            data in.
        """
        self.dirpath = dirpath

    def _get_file_from_dir_and_property(self, property_name: str) -> str:
        material = os.path.split(self.dirpath)[-1].lower()
        filepath = os.path.join(self.dirpath, material + "_" + property_name)
        for f in os.listdir(self.dirpath):
            f_path = os.path.join(self.dirpath, f)
            if filepath in f_path:
                return f_path
        else:
            raise FileNotFoundError("Could not find " + filepath)

    @property
    def force_constants(self):
        return np.load(
            self._get_file_from_dir_and_property("force_constants"),
            allow_pickle=True
        )

    @property
    def sc_matrix(self):
        return np.load(
            self._get_file_from_dir_and_property("sc_matrix"),
            allow_pickle=True
        )

    @property
    def cell_origins(self):
        return np.load(
            self._get_file_from_dir_and_property("cell_origins"),
            allow_pickle=True
        )

    @property
    def born(self):
        return np.load(
            self._get_file_from_dir_and_property("born"),
            allow_pickle=True
        )

    @property
    def dielectric(self):
        return np.load(
            self._get_file_from_dir_and_property("dielectric"),
            allow_pickle=True
        )

    @property
    def crystal(self):
        return Crystal.from_json_file(
            self._get_file_from_dir_and_property("crystal")
        )

    @property
    def n_cells_in_sc(self):
        return json.load(
            open(self._get_file_from_dir_and_property("properties"))
        )["n_cells_in_sc"]

    @property
    def force_constants_unit(self):
        return json.load(
            open(self._get_file_from_dir_and_property("properties"))
        )["force_constants_unit"]

    @property
    def born_unit(self):
        return json.load(
            open(self._get_file_from_dir_and_property("properties"))
        )["born_unit"]

    @property
    def dielectric_unit(self):
        return json.load(
            open(self._get_file_from_dir_and_property("properties"))
        )["dielectric_unit"]


def check_force_constant_attrs(force_constants, expected_force_constants):
    npt.assert_allclose(
        force_constants.force_constants.magnitude,
        expected_force_constants.force_constants
    )
    npt.assert_allclose(
        force_constants.sc_matrix,
        expected_force_constants.sc_matrix
    )
    npt.assert_allclose(
        force_constants.cell_origins,
        expected_force_constants.cell_origins
    )
    if force_constants.born is not None:
        npt.assert_allclose(
            force_constants.born.magnitude,
            expected_force_constants.born
        )
    else:
        assert expected_force_constants.born.item() is None
    if force_constants.dielectric is not None:
        npt.assert_allclose(
            force_constants.dielectric.magnitude,
            expected_force_constants.dielectric
        )
    else:
        assert expected_force_constants.dielectric.item() is None
    assert force_constants.n_cells_in_sc == \
        expected_force_constants.n_cells_in_sc
    assert force_constants.force_constants_unit == \
        expected_force_constants.force_constants_unit
    assert force_constants.born_unit == expected_force_constants.born_unit
    assert force_constants.dielectric_unit == \
        expected_force_constants.dielectric_unit
    check_crystal_attrs(
        force_constants.crystal, expected_force_constants.crystal
    )


@pytest.mark.unit
class TestObjectCreation:

    @pytest.mark.parametrize("castep_bin_file", [
        os.path.join('LZO', 'La2Zr2O7.castep_bin'),
        os.path.join('graphite', 'graphite.castep_bin'),
        os.path.join('quartz', 'quartz.castep_bin')
    ])
    def test_creation_from_castep(self, castep_bin_file):
        inter_dirpath = os.path.join(get_data_path(), 'interpolation')
        dirpath = os.path.join(inter_dirpath, os.path.split(castep_bin_file)[0])
        filepath = os.path.join(inter_dirpath, castep_bin_file)
        fc = ForceConstants.from_castep(filepath)
        check_force_constant_attrs(fc, ExpectedForceConstants(dirpath))

    @pytest.mark.parametrize("phonopy_args", [
        {"fc_name": "FORCE_CONSTANTS"},
        {"fc_name": "FULL_FORCE_CONSTANTS"},
        {"fc_name": "full_force_constants.hdf5"},
        {"summary_name": "phonopy_nofc_noborn.yaml"}
    ])
    def test_creation_from_phonopy(self, phonopy_args):
        phonopy_args["path"] = os.path.join(
            get_data_path(), 'phonopy_data', 'NaCl', 'interpolation'
        )
        fc = ForceConstants.from_phonopy(**phonopy_args)
        # check_force_constant_attrs(fc)


