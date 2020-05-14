import json
import os

import numpy as np
import numpy.testing as npt
import pytest

from euphonic import Crystal, ureg

from ..utils import get_data_path


class ExpectedCrystal:

    def __init__(self, crystal_json_file):
        self.data = json.load(open(crystal_json_file))

    @property
    def cell_vectors(self) -> np.array:
        return np.array(self.data["cell_vectors"]) * \
               ureg(self.data["cell_vectors_unit"])

    @property
    def n_atoms(self) -> int:
        return self.data["n_atoms"]

    @property
    def atom_r(self) -> np.array:
        return np.array(self.data["atom_r"])

    @property
    def atom_type(self) -> np.array:
        return np.array(
            self.data["atom_type"],
            dtype='<U2'
        )

    @property
    def atom_mass(self) -> np.array:
        return np.array(self.data["atom_mass"]) * \
               ureg(self.data["atom_mass_unit"])

    def to_dict(self):
        d = {}
        d['cell_vectors'] = self.cell_vectors.magnitude
        d['cell_vectors_unit'] = str(self.cell_vectors.units)
        d['n_atoms'] = self.n_atoms
        d['atom_r'] = self.atom_r
        d['atom_type'] = self.atom_type
        d['atom_mass'] = self.atom_mass.magnitude
        d['atom_mass_unit'] = str(self.atom_mass.units)
        return d

    def from_dict(self, dict):
        self.data = dict


def quartz_attrs():
    return ExpectedCrystal(get_quartz_json_file())


def lzo_attrs():
    return ExpectedCrystal(get_lzo_json_file())


def get_quartz_json_file():
    return get_filepath('crystal_quartz.json')


def get_lzo_json_file():
    return get_filepath('crystal_lzo.json')


def get_filepath(filename):
    return os.path.join(get_data_path(), 'crystal', filename)


def crystal_from_json_file(filename):
    filepath = get_filepath(filename)
    crystal = Crystal.from_json_file(filepath)
    return crystal


def check_crystal_attrs(crystal, expected_crystal):
    npt.assert_allclose(
        crystal.cell_vectors.to('angstrom').magnitude,
        expected_crystal.cell_vectors.magnitude
    )
    npt.assert_equal(
        crystal.atom_r,
        expected_crystal.atom_r
    )
    npt.assert_equal(
        crystal.atom_type,
        expected_crystal.atom_type
    )
    npt.assert_allclose(
        crystal.atom_mass.to('amu').magnitude,
        expected_crystal.atom_mass.magnitude
    )
    assert crystal.n_atoms == expected_crystal.n_atoms


def get_quartz_crystal():
    return crystal_from_json_file(
        get_quartz_json_file()
    )


def get_lzo_crystal():
    return crystal_from_json_file(
        get_lzo_json_file()
    )


@pytest.mark.unit
class TestObjectCreation:

    @pytest.fixture(params=[quartz_attrs(), lzo_attrs()])
    def crystal_from_constructor(self, request):
        crystal_attrs = request.param
        crystal = Crystal(
            crystal_attrs.cell_vectors,
            crystal_attrs.atom_r,
            crystal_attrs.atom_type,
            crystal_attrs.atom_mass
        )
        return crystal, crystal_attrs

    @pytest.fixture(params=[quartz_attrs(), lzo_attrs()])
    def crystal_from_dict(self, request):
        crystal_attrs = request.param
        d = crystal_attrs.to_dict()
        crystal = Crystal.from_dict(d)
        return crystal, crystal_attrs

    @pytest.fixture(params=[
        (get_quartz_json_file(), quartz_attrs()),
        (get_lzo_json_file(), lzo_attrs())
    ])
    def crystal_from_json_file(self, request):
        filename, crystal_attrs = request.param
        return crystal_from_json_file(filename), crystal_attrs

    @pytest.fixture(params=[quartz_attrs(), lzo_attrs()])
    def crystal_from_dict(self, request):
        crystal_attrs = request.param
        d = crystal_attrs.to_dict()
        crystal = Crystal.from_dict(d)
        return crystal, crystal_attrs

    @pytest.mark.parametrize('crystal_creator', [
        pytest.lazy_fixture('crystal_from_constructor'),
        pytest.lazy_fixture('crystal_from_json_file'),
        pytest.lazy_fixture('crystal_from_dict')
    ])
    def test_crystal_create(self, crystal_creator):
        crystal, expected_crystal = crystal_creator
        check_crystal_attrs(crystal, expected_crystal)

    @pytest.fixture(params=[
        (
            "cell_vectors",
            np.array([[1.23, 2.45, 0.0], [3.45, 5.66, 7.22], [0.001, 4.55]]),
            ValueError
        ),
        (
                "cell_vectors",
                np.array([
                    [1.23, 2.45, 0.0],
                    [3.45, "5.66", 7.22],
                    [0.001, 4.55, "5.64"]
                ]),
                TypeError
        ),
        (
            "atom_r",
            np.array([[0.125, 0.125, 0.125], [0.875, 0.875, 0.875]]),
            ValueError
        ),
        (
            "atom_mass",
            np.array([15.999399987607514, 15.999399987607514, 91.2239999293416]),
            ValueError
        ),
        ("atom_type", np.array(["O", "Zr", "La"]), ValueError),
        ("atom_mass_unit", "angstrom", TypeError),
        ("cell_vectors_unit", "kg", TypeError),
        ("atom_mass_unit", "", TypeError),
        ("cell_vectors_unit", "", TypeError)
    ])
    def inject_faulty_dict_elements(self, request):
        dict_key, dict_value, exception_type = request.param
        d = quartz_attrs().to_dict()
        d[dict_key] = dict_value
        return d, exception_type

    def test_faulty_dict_creation(self, inject_faulty_dict_elements):
        faulty_dict, exception_type = inject_faulty_dict_elements
        with pytest.raises(exception_type):
            Crystal.from_dict(faulty_dict)


@pytest.mark.unit
class TestObjectSerialisation:

    @pytest.fixture(params=[get_quartz_crystal(), get_lzo_crystal()])
    def crystal_to_json_file(self, request, tmpdir):
        crystal = request.param
        # Serialise
        output_file = str(tmpdir.join('tmp.test'))
        crystal.to_json_file(output_file)
        # Deserialise
        deserialised_crystal = Crystal.from_json_file(output_file)
        return crystal, deserialised_crystal

    def test_crystal_to_file(self, crystal_to_json_file):
        crystal, deserialised_crystal = crystal_to_json_file
        check_crystal_attrs(crystal, deserialised_crystal)

    @pytest.fixture(params=[
        (get_quartz_crystal(), quartz_attrs()),
        (get_lzo_crystal(), lzo_attrs())
    ])
    def crystal_to_dict(self, request):
        crystal, quartz_attributes = request.param
        serialised_crystal = crystal.to_dict()
        return serialised_crystal, quartz_attributes.to_dict()

    def check_crystal_dict(self, cdict, expected_cdict):
        npt.assert_allclose(
            cdict['cell_vectors'],
            expected_cdict['cell_vectors']
        )
        npt.assert_equal(
            cdict['atom_r'],
            expected_cdict['atom_r']
        )
        npt.assert_equal(
            cdict['atom_type'],
            expected_cdict['atom_type']
        )
        npt.assert_allclose(
            cdict['atom_mass'],
            expected_cdict['atom_mass']
        )
        assert ureg(cdict['cell_vectors_unit']) == \
            ureg(expected_cdict['cell_vectors_unit'])
        assert cdict['n_atoms'] == cdict['n_atoms']
        assert ureg(cdict['atom_mass_unit']) == \
            ureg(expected_cdict['atom_mass_unit'])

    def test_crystal_to_dict(self, crystal_to_dict):
        cdict, expected_cdict = crystal_to_dict
        self.check_crystal_dict(cdict, expected_cdict)


@pytest.mark.unit
class TestObjectMethods:

    quartz_reciprocal_cell = np.array([
        [1.29487418, -0.74759597, 0.],
        [1.29487418, 0.74759597, 0.],
        [0., 0., 1.17436043]
    ]) * ureg('1/angstrom')

    lzo_reciprocal_cell = np.array([
        [8.28488599e-01, 0.00000000e+00, -5.85829906e-01],
        [-2.01146673e-33, 8.28488599e-01, 5.85829906e-01],
        [2.01146673e-33, -8.28488599e-01, 5.85829906e-01]
    ]) * ureg('1/angstrom')

    @pytest.mark.parametrize("crystal,expected_recip", [
        (get_quartz_crystal(), quartz_reciprocal_cell),
        (get_lzo_crystal(), lzo_reciprocal_cell)
    ])
    def test_reciprocal_cell(self, crystal, expected_recip):
        recip = crystal.reciprocal_cell()
        npt.assert_allclose(
            recip.to('1/angstrom').magnitude,
            expected_recip.to('1/angstrom').magnitude
        )

    quartz_cell_volume = 109.09721804482547 * ureg('angstrom**3')

    lzo_cell_volume = 308.4359549515967 * ureg('angstrom**3')

    @pytest.mark.parametrize("crystal,expected_vol", [
        (get_quartz_crystal(), quartz_cell_volume),
        (get_lzo_crystal(), lzo_cell_volume)
    ])
    def test_cell_volume(self, crystal, expected_vol):
        vol = crystal.cell_volume()
        npt.assert_allclose(
            vol.to('angstrom**3').magnitude,
            expected_vol.to('angstrom**3').magnitude
        )
