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
import itertools

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


@pytest.mark.unit
class TestInterpolatePhonons:

    path = os.path.join(get_data_path(), 'interpolation')
    unique_sc_offsets = [[] for _ in range(3)]
    unique_cell_origins = [[] for _ in range(3)]

    graphite_unique_sc_i = np.loadtxt(os.path.join(
        path, 'graphite', 'graphite_unique_sc_i.txt'), dtype=np.int32)
    graphite_unique_cell_i = np.loadtxt(os.path.join(
        path, 'graphite', 'graphite_unique_cell_i.txt'), dtype=np.int32)
    lzo_unique_sc_i = np.loadtxt(os.path.join(
        path, 'LZO', 'lzo_unique_sc_i.txt'), dtype=np.int32)
    lzo_unique_cell_i = np.loadtxt(os.path.join(
        path, 'LZO', 'lzo_unique_cell_i.txt'), dtype=np.int32)

    @pytest.fixture(params=[
        (
            'LZO', 'La2Zr2O7.castep_bin',
            np.array([-1, 9.35, 3.35]),
            np.loadtxt(
                os.path.join(path, 'LZO', 'lzo_sc_phases.txt'),
                dtype=np.complex128
            ),
            np.loadtxt(
                os.path.join(path, 'LZO', 'lzo_cell_phases.txt'),
                dtype=np.complex128
            )
        ),
        (
            'LZO', 'La2Zr2O7.castep_bin',
            np.array([0.0, 0.0, 0.0]),  # At gamma point
            np.full(
                len(lzo_unique_sc_i), 1.0 + 0.0 * 1j, dtype=np.complex128
            ),
            np.full(
                len(lzo_unique_cell_i), 1.0 + 0.0 * 1j, dtype=np.complex128
            )
        ),
        (
            'graphite', 'graphite.castep_bin',
            np.array([0.001949, 0.001949, 0.0]),
            np.loadtxt(os.path.join(
                path, 'graphite', 'graphite_sc_phases.txt'
            ), dtype=np.complex128),
            np.loadtxt(os.path.join(
                path, 'graphite', 'graphite_cell_phases.txt'
            ), dtype=np.complex128)
        ),
        (
            'graphite', 'graphite.castep_bin',
            np.array([0.0, 0.0, 0.0]),  # At gamma point
            np.full(
                len(graphite_unique_sc_i), 1.0 + 0.0 * 1j, dtype=np.complex128
            ),
            np.full(
                len(graphite_unique_cell_i), 1.0 + 0.0 * 1j, dtype=np.complex128
            )
        )
    ])
    def calculate_phases(self, request):
        material_name, castep_bin_file, qpt, expected_sc_phases, \
            expected_cell_phases = request.param
        lower_material = material_name.lower()
        filename = os.path.join(self.path, material_name, castep_bin_file)
        self.unique_sc_i = np.loadtxt(
            os.path.join(
                self.path, material_name,
                '{}_unique_sc_i.txt'.format(lower_material)
            ), dtype=np.int32
        )
        self.unique_cell_i = np.loadtxt(
            os.path.join(
                self.path, material_name,
                '{}_unique_cell_i.txt'.format(lower_material)
            ), dtype=np.int32
        )
        unique_offsets_filepath = os.path.join(
            self.path, material_name,
            '{}_unique_sc_offsets.txt'.format(lower_material)
        )
        with open(unique_offsets_filepath) as f:
            for i in range(3):
                self.unique_sc_offsets[i] = [int(x)
                                             for x in f.readline().split()]
        unique_cell_origins_filepath = os.path.join(
            self.path, material_name,
            '{}_unique_cell_origins.txt'.format(lower_material)
        )
        with open(unique_cell_origins_filepath) as f:
            for i in range(3):
                self.unique_cell_origins[i] = [int(x)
                                               for x in f.readline().split()]
        return ForceConstants.from_castep(filename), qpt, \
            expected_sc_phases, expected_cell_phases

    def test_calculate_phases(self, calculate_phases):
        fc, qpt, expected_sc_phases, expected_cell_phases = calculate_phases
        sc_phases, cell_phases = fc._calculate_phases(
            qpt, self.unique_sc_offsets, self.unique_sc_i,
            self.unique_cell_origins, self.unique_cell_i
        )
        npt.assert_allclose(sc_phases, expected_sc_phases)
        npt.assert_allclose(cell_phases, expected_cell_phases)

    # Supercell image calculation limit - 2 supercells in each direction
    lim = 2
    n_atoms = 4
    n_cells_in_sc = 98

    def get_ijn_from_image_data(self, image_data):
        i = image_data[:, 0].astype(int)
        j = image_data[:, 1].astype(int)
        n = image_data[:, 2].astype(int)
        return i, j, n

    @pytest.fixture
    def lzo_calculate_supercell_images_n_sc_images(self):
        image_data = np.loadtxt(
            os.path.join(self.path, 'LZO', 'lzo_n_sc_images.txt')
        )
        i, j, n = self.get_ijn_from_image_data(image_data)
        expected_n_images = np.zeros((22, 88))
        expected_n_images[i, j] = n
        # After refactoring where the shape of n_sc_images was changed from
        # (n_atoms, n_cells_in_sc*n_atoms) to (n_cells_in_sc, n_atoms, n_atoms),
        # expected_n_images must be reshaped to ensure tests still pass
        expected_n_images = np.transpose(np.reshape(
            expected_n_images, (22, 4, 22)), axes=[1, 0, 2]
        )
        filename = os.path.join(self.path, 'LZO', 'La2Zr2O7.castep_bin')
        return ForceConstants.from_castep(filename), "_n_sc_images", \
            expected_n_images

    @pytest.fixture
    def graphite_calculate_supercell_images_n_sc_images(self):
        image_data = np.loadtxt(
            os.path.join(self.path, 'graphite', 'graphite_n_sc_images.txt')
        )
        i, j, n = self.get_ijn_from_image_data(image_data)
        expected_n_images = np.zeros(
            (self.n_atoms, self.n_atoms * self.n_cells_in_sc))
        expected_n_images[i, j] = n
        # After refactoring where the shape of n_sc_images was changed from
        # (n_atoms, n_cells_in_sc*n_atoms) to (n_cells_in_sc, n_atoms, n_atoms),
        # expctc_n_images must be reshaped to ensure tests still pass
        expected_n_images = np.transpose(
            np.reshape(
                expected_n_images,
                (self.n_atoms, self.n_cells_in_sc, self.n_atoms)
            ), axes=[1, 0, 2]
        )
        filename = os.path.join(self.path, 'graphite', 'graphite.castep_bin')
        return ForceConstants.from_castep(filename), "_n_sc_images", \
            expected_n_images

    @pytest.fixture
    def lzo_calculate_supercell_images_sc_image_i(self):
        image_data = np.loadtxt(
            os.path.join(self.path, 'LZO', 'lzo_sc_image_i.txt')
        )
        i, j, n = self.get_ijn_from_image_data(image_data)
        sc_i = image_data[:, 3].astype(int)
        max_n = np.max(n) + 1
        # size = n_atoms X n_atoms*n_cells_in_sc X max supercell images
        expected_sc_image_i = np.full((22, 88, max_n), -1)
        expected_sc_image_i[i, j, n] = sc_i
        # After refactoring where the shape of sc_image_i was changed from
        # (n_atoms, n_cells_in_sc*n_atoms, (2*lim + 1)**3) to
        # (n_cells_in_sc, n_atoms, n_atoms, (2*lim + 1)**3),
        # expctd_image_i must be reshaped to ensure tests still pass
        expected_sc_image_i = np.transpose(
            np.reshape(expected_sc_image_i, (22, 4, 22, max_n)),
            axes=[1, 0, 2, 3]
        )
        filename = os.path.join(self.path, 'LZO', 'La2Zr2O7.castep_bin')
        return ForceConstants.from_castep(filename), "_sc_image_i", \
            expected_sc_image_i

    @pytest.fixture
    def graphite_calculate_supercell_images_sc_image_i(self):
        image_data = np.loadtxt(
            os.path.join(self.path, 'graphite', 'graphite_sc_image_i.txt')
        )
        i, j, n = self.get_ijn_from_image_data(image_data)
        sc_i = image_data[:, 3].astype(int)
        max_n = np.max(n) + 1
        # size = n_atoms X n_atoms*n_cells_in_sc X max supercell images
        expected_sc_image_i = np.full(
            (self.n_atoms, self.n_atoms * self.n_cells_in_sc, max_n), -1)
        expected_sc_image_i[i, j, n] = sc_i
        # After refactoring where the shape of sc_image_i was changed from
        # (n_atoms, n_cells_in_sc*n_atoms, (2*lim + 1)**3) to
        # (n_cells_in_sc, n_atoms, n_atoms, (2*lim + 1)**3),
        # expctc_image_i must be reshaped to ensure tests still pass
        expected_sc_image_i = np.transpose(
            np.reshape(
                expected_sc_image_i,
                (self.n_atoms, self.n_cells_in_sc,self.n_atoms, max_n)
            ), axes=[1, 0, 2, 3]
        )
        filename = os.path.join(self.path, 'graphite', 'graphite.castep_bin')
        return ForceConstants.from_castep(filename), "_sc_image_i", \
            expected_sc_image_i

    @pytest.mark.parametrize('calculate_supercell_images', [
        pytest.lazy_fixture('lzo_calculate_supercell_images_n_sc_images'),
        pytest.lazy_fixture('lzo_calculate_supercell_images_sc_image_i'),
        pytest.lazy_fixture('graphite_calculate_supercell_images_n_sc_images'),
        pytest.lazy_fixture('graphite_calculate_supercell_images_sc_image_i')
    ])
    def test_calculate_supercell_images(self, calculate_supercell_images):
        fc, testing_attribute, expected_attribute_data = \
            calculate_supercell_images
        fc._calculate_supercell_images(self.lim)
        npt.assert_equal(
            getattr(fc, testing_attribute), expected_attribute_data
        )


@pytest.mark.integration
class TestCalculateQPointPhononModes:

    path = os.path.join(get_data_path(), 'interpolation')

    expected_freqs = {
        "LZO": {
            "asr": np.array([
                [0.0002964449, 0.0002964449, 0.0003208033,
                 0.0003501419, 0.0003501419, 0.0003903141,
                 0.0004972138, 0.0004972138, 0.0005372809,
                 0.0005438643, 0.0005861166, 0.0005861166,
                 0.0007103804, 0.0007103804, 0.0007331639,
                 0.0007786131, 0.0007874376, 0.0007929211,
                 0.0008126016, 0.0008354861, 0.0008354861,
                 0.0009078731, 0.0009078731, 0.0010160378,
                 0.0010264374, 0.0010554444, 0.0011528145,
                 0.0012094888, 0.0012304278, 0.0012410548,
                 0.0012410548, 0.0012564500, 0.0013664070,
                 0.0013664070, 0.0014355566, 0.0014355566,
                 0.0014576129, 0.0015442745, 0.0015442745,
                 0.0015449039, 0.0015505652, 0.0015937746,
                 0.0017167608, 0.0017828465, 0.0017828465,
                 0.0018048096, 0.0018598080, 0.0018598080,
                 0.0018726170, 0.0019193824, 0.0020786777,
                 0.0020786777, 0.0022934801, 0.0024275754,
                 0.0024275754, 0.0024850292, 0.0025000804,
                 0.0025179345, 0.0025179345, 0.0025401087,
                 0.0025550191, 0.0025550191, 0.0028191070,
                 0.0033473173, 0.0033680501, 0.0033680501],
                [-1.2527213902e-09, -1.2524650945e-09, -1.2522509615e-09,
                 2.5186476888e-04, 2.5186476888e-04, 2.5186476888e-04,
                 4.2115533128e-04, 4.2115533128e-04, 4.5919137201e-04,
                 4.5919137201e-04, 4.5919137201e-04, 6.0460911494e-04,
                 6.0460911494e-04, 6.0460911494e-04, 6.1121916807e-04,
                 6.1121916807e-04, 6.1121916807e-04, 6.8315329115e-04,
                 6.8315329115e-04, 8.9085325717e-04, 8.9085325717e-04,
                 8.9085325717e-04, 1.0237996415e-03, 1.0237996415e-03,
                 1.0237996415e-03, 1.1666034640e-03, 1.1744920636e-03,
                 1.1744920636e-03, 1.1744920636e-03, 1.2802064794e-03,
                 1.2802064794e-03, 1.2802064794e-03, 1.3122749877e-03,
                 1.4159439055e-03, 1.4159439055e-03, 1.4159439055e-03,
                 1.4813308344e-03, 1.4813308344e-03, 1.5084818310e-03,
                 1.5084818310e-03, 1.5084818310e-03, 1.5957863393e-03,
                 1.7192994802e-03, 1.7192994802e-03, 1.8119521571e-03,
                 1.8119521571e-03, 1.8119521571e-03, 1.8609709896e-03,
                 1.8609709896e-03, 1.8609709896e-03, 2.1913629570e-03,
                 2.2033452584e-03, 2.2033452584e-03, 2.2033452584e-03,
                 2.4420900293e-03, 2.4420900293e-03, 2.4420900293e-03,
                 2.4754830417e-03, 2.4754830417e-03, 2.4754830417e-03,
                 2.5106852083e-03, 2.5106852083e-03, 2.5106852083e-03,
                 3.3517193438e-03, 3.3517193438e-03, 3.3517193438e-03]
            ]) * ureg('hartree'),
            "no_asr": np.array([
                [0.0002964623, 0.0002964623, 0.0003208033, 0.000350174,
                 0.000350174, 0.0003903141, 0.0004972179, 0.0004972179,
                 0.0005372886, 0.0005438642, 0.0005861163, 0.0005861163,
                 0.0007103807, 0.0007103807, 0.0007331935, 0.0007786131,
                 0.0007874315, 0.0007929211, 0.0008126019, 0.0008354958,
                 0.0008354958, 0.000907874, 0.000907874, 0.0010160402,
                 0.0010264376, 0.0010554468, 0.0011528125, 0.0012094888,
                 0.0012304238, 0.0012410492, 0.0012410492, 0.00125645,
                 0.0013664066, 0.0013664066, 0.0014355603, 0.0014355603,
                 0.0014576129, 0.0015442837, 0.0015442837, 0.0015449061,
                 0.0015505634, 0.0015937746, 0.0017167601, 0.0017828448,
                 0.0017828448, 0.001804811, 0.0018598084, 0.0018598084,
                 0.0018726182, 0.0019193824, 0.0020786769, 0.0020786769,
                 0.002293487, 0.0024275755, 0.0024275755, 0.0024850292,
                 0.0025000804, 0.002517934, 0.002517934, 0.0025401063,
                 0.0025550198, 0.0025550198, 0.0028191102, 0.0033473155,
                 0.0033680501, 0.0033680501],
                [1.2522582708e-05, 1.2522582708e-05, 1.2522582708e-05,
                 2.5186476888e-04, 2.5186476888e-04, 2.5186476888e-04,
                 4.2115533128e-04, 4.2115533128e-04, 4.5920462007e-04,
                 4.5920462007e-04, 4.5920462007e-04, 6.0462274991e-04,
                 6.0462274991e-04, 6.0462274991e-04, 6.1121916807e-04,
                 6.1121916807e-04, 6.1121916807e-04, 6.8315329115e-04,
                 6.8315329115e-04, 8.9089400855e-04, 8.9089400855e-04,
                 8.9089400855e-04, 1.0238000223e-03, 1.0238000223e-03,
                 1.0238000223e-03, 1.1666034640e-03, 1.1744920636e-03,
                 1.1744920636e-03, 1.1744920636e-03, 1.2802064794e-03,
                 1.2802064794e-03, 1.2802064794e-03, 1.3122749877e-03,
                 1.4159439055e-03, 1.4159439055e-03, 1.4159439055e-03,
                 1.4813308344e-03, 1.4813308344e-03, 1.5085078032e-03,
                 1.5085078032e-03, 1.5085078032e-03, 1.5957863393e-03,
                 1.7192994802e-03, 1.7192994802e-03, 1.8119544413e-03,
                 1.8119544413e-03, 1.8119544413e-03, 1.8609709896e-03,
                 1.8609709896e-03, 1.8609709896e-03, 2.1913629570e-03,
                 2.2033465408e-03, 2.2033465408e-03, 2.2033465408e-03,
                 2.4420900293e-03, 2.4420900293e-03, 2.4420900293e-03,
                 2.4754830417e-03, 2.4754830417e-03, 2.4754830417e-03,
                 2.5106852083e-03, 2.5106852083e-03, 2.5106852083e-03,
                 3.3517193438e-03, 3.3517193438e-03, 3.3517193438e-03]
            ]) * ureg('hartree')
        },
        "graphite": {
            "asr": np.array([
                [-1.4954416353e-09, -1.4920078435e-09, -1.4801148666e-09,
                 1.8361944616e-04, 1.8361944616e-04, 4.3310694904e-04,
                 4.0094292783e-03, 4.0298656188e-03, 7.0857269505e-03,
                 7.0857269505e-03, 7.1044827434e-03, 7.1044827434e-03],
                [5.1653999633e-06, 3.2069627188e-05, 5.1670788680e-05,
                 1.8632833975e-04, 1.9060943285e-04, 4.3314392040e-04,
                 4.0093920383e-03, 4.0298240624e-03, 7.0856315683e-03,
                 7.0859645028e-03, 7.1043776174e-03, 7.1046906727e-03],
                [2.1681099868e-03, 2.2111725605e-03, 2.8193685159e-03,
                 2.8197548439e-03, 2.9049709255e-03, 2.9141514127e-03,
                 6.0277000188e-03, 6.0316368591e-03, 6.0977873315e-03,
                 6.1104625371e-03, 6.3230408313e-03, 6.3262959952e-03],
                [7.6871083387e-04, 8.8171407452e-04, 2.1005257455e-03,
                 2.1043480758e-03, 3.5723508265e-03, 3.5925792804e-03,
                 3.7679565502e-03, 3.7804526590e-03, 6.6594311463e-03,
                 6.6651464071e-03, 7.2007012704e-03, 7.2132353835e-03],
                [1.4065426444e-04, 1.4065426444e-04, 1.4065426444e-04,
                 1.4065426444e-04, 3.2305904460e-04, 3.2305904460e-04,
                 4.0222682768e-03, 4.0222682768e-03, 7.1591503492e-03,
                 7.1591503492e-03, 7.1591503492e-03, 7.1591503492e-03]
            ]) * ureg('hartree'),
            "no_asr": np.array([
                [-1.4947813885e-05, -1.4947813885e-05, 1.7871973126e-05,
                 1.8361944616e-04, 1.8361944616e-04, 4.3310694904e-04,
                 4.0094292783e-03, 4.0298656188e-03, 7.0857269505e-03,
                 7.0857269505e-03, 7.1044827434e-03, 7.1044827434e-03],
                [1.8594596548e-05, 2.8377223801e-05, 4.9463914519e-05,
                 1.8632830014e-04, 1.9060940094e-04, 4.3314392073e-04,
                 4.0093920384e-03, 4.0298240625e-03, 7.0856315682e-03,
                 7.0859645027e-03, 7.1043776174e-03, 7.1046906727e-03],
                [2.1681107372e-03, 2.2111728203e-03, 2.8193681109e-03,
                 2.8197547098e-03, 2.9049711018e-03, 2.9141516000e-03,
                 6.0276999661e-03, 6.0316367945e-03, 6.0977872708e-03,
                 6.1104623404e-03, 6.3230407709e-03, 6.3262959354e-03],
                [7.6869781440e-04, 8.8171407111e-04, 2.1005260482e-03,
                 2.1043510112e-03, 3.5723508344e-03, 3.5925812023e-03,
                 3.7679565852e-03, 3.7804526859e-03, 6.6594311404e-03,
                 6.6651464171e-03, 7.2007012512e-03, 7.2132353897e-03],
                [1.4065426444e-04, 1.4065426444e-04, 1.4065426444e-04,
                 1.4065426444e-04, 3.2305904460e-04, 3.2305904460e-04,
                 4.0222682768e-03, 4.0222682768e-03, 7.1591503492e-03,
                 7.1591503492e-03, 7.1591503492e-03, 7.1591503492e-03]
            ]) * ureg('hartree')
        }
    }

    materials = [
        (
            "LZO", 'La2Zr2O7.castep_bin',
            np.array([  # Qpoints
                [-1.00, 9.35, 3.35],
                [-1.00, 9.00, 3.00]
            ])
        ),
        (
            "graphite", "graphite.castep_bin",
            np.array([  # Qpoints
                [0.00, 0.00, 0.00],
                [0.001949, 0.001949, 0.00],
                [0.50, 0.00, 0.00],
                [0.25, 0.00, 0.00],
                [0.00, 0.00, 0.50]
            ])
        )
    ]

    kwargs = [
        {"use_c": False, "fall_back_on_python": True, "n_threads": 1},
        {"use_c": True, "fall_back_on_python": False, "n_threads": 1},
        {"use_c": True, "fall_back_on_python": False, "n_threads": 2},
        {
            "use_c": False, "fall_back_on_python": True,
            "n_threads": 1, "asr": "realspace"
        },
        {
            "use_c": True, "fall_back_on_python": False,
            "n_threads": 1, "asr": "realspace"
        },
        {
            "use_c": True, "fall_back_on_python": False,
            "n_threads": 2, "asr": "realspace"
        }
    ]

    @pytest.fixture(params=list(itertools.product(materials, kwargs)))
    def calculate_qpoint_phonon_modes(self, request):
        material, kwargs = request.param
        material_name, castep_bin_file, qpts = material
        filename = os.path.join(self.path, material_name, castep_bin_file)
        return ForceConstants.from_castep(filename), qpts, material_name, kwargs

    def test_calculate_qpoint_phonon_modes(self, calculate_qpoint_phonon_modes):
        fc, qpts, material_name, kwargs = calculate_qpoint_phonon_modes
        print(material_name)
        qpoint_phonon_modes = fc.calculate_qpoint_phonon_modes(
            qpts, **kwargs
        )
        if "asr" in kwargs:
            test_expected_freqs = self.expected_freqs[material_name]["asr"]
        else:
            test_expected_freqs = self.expected_freqs[material_name]["no_asr"]
        npt.assert_allclose(
            qpoint_phonon_modes.frequencies.to('hartree').magnitude,
            test_expected_freqs.to('hartree').magnitude,
            atol=1e-10
        )
