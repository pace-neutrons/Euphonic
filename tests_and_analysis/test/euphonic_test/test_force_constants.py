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
    def force_constants(self):
        return np.load(
            self._get_file_from_dir_and_property("force_constants", "npy"),
            allow_pickle=True
        )

    @property
    def sc_matrix(self):
        return np.load(
            self._get_file_from_dir_and_property("sc_matrix", "npy"),
            allow_pickle=True
        )

    @property
    def cell_origins(self):
        return np.load(
            self._get_file_from_dir_and_property("cell_origins", "npy"),
            allow_pickle=True
        )

    @property
    def born(self):
        return np.load(
            self._get_file_from_dir_and_property("born", "npy"),
            allow_pickle=True
        )

    @property
    def dielectric(self):
        return np.load(
            self._get_file_from_dir_and_property("dielectric", "npy"),
            allow_pickle=True
        )

    @property
    def crystal(self):
        return Crystal.from_json_file(
            self._get_file_from_dir_and_property("crystal", "json")
        )

    @property
    def n_cells_in_sc(self):
        return json.load(
            open(self._get_file_from_dir_and_property("properties", "json"))
        )["n_cells_in_sc"]

    @property
    def force_constants_unit(self):
        return json.load(
            open(self._get_file_from_dir_and_property("properties", "json"))
        )["force_constants_unit"]

    @property
    def born_unit(self):
        return json.load(
            open(self._get_file_from_dir_and_property("properties", "json"))
        )["born_unit"]

    @property
    def dielectric_unit(self):
        return json.load(
            open(self._get_file_from_dir_and_property("properties", "json"))
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
        dir = slugify(
            "-".join(phonopy_args.keys()) + "-"
            + "-".join(phonopy_args.values())
        )
        phonopy_args["path"] = os.path.join(
            get_data_path(), 'phonopy_data', 'NaCl', 'interpolation'
        )
        fc = ForceConstants.from_phonopy(**phonopy_args)
        expected_dirpath = os.path.join(phonopy_args["path"], dir)
        check_force_constant_attrs(
            fc, ExpectedForceConstants(expected_dirpath, with_material=False)
        )
