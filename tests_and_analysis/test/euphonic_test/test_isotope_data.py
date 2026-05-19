from dataclasses import dataclass

import numpy as np
from numpy.testing import assert_allclose
import pytest

from euphonic.data.isotopes import (
    AtomTypeDictData,
    AtomTypeShallowDictData,
    IsotopeData,
    LegacyJsonData,
    Structure,
)
from euphonic.ureg import Quantity


def _compare_quantity(a: Quantity, b: Quantity) -> None:
    assert a.units == b.units
    assert_allclose(a.magnitude, b.magnitude)


@dataclass
class TestStructure:
    atom_type: np.ndarray
    atom_mass: Quantity


@pytest.fixture
def structure() -> TestStructure:
    return TestStructure(
        atom_type=np.array(['Na', 'Cl']),
        atom_mass=Quantity([22.99, 35.45], 'amu'),
    )


@pytest.fixture
def bad_structure() -> TestStructure:
    return TestStructure(
        atom_type=np.array(['X', 'M']),
        atom_mass=Quantity([-1.0, 200.0], 'amu'),
    )


def test_atom_type_dict_data(structure) -> None:
    isotope_data = AtomTypeDictData(
        {'key1': {'Na': Quantity(1.0, 'barn'), 'Cl': Quantity(2.0, 'barn')}},
    )

    assert isotope_data.get_item('Na', 0.0) == {'key1': Quantity(1.0, 'barn')}

    assert isotope_data.get_value('Na', 0.0, 'key1') == Quantity(1.0, 'barn')

    _compare_quantity(
        isotope_data.get_array(structure, 'key1'),
        Quantity([1.0, 2.0], 'barn'),
    )

    with pytest.raises(KeyError):
        isotope_data.get_item('K', 0.0)

    with pytest.raises(KeyError, match="Property 'key2' not found in dict."):
        isotope_data.get_value('Na', 0.0, 'key2')

    with pytest.raises(KeyError):
        isotope_data.get_value('K', 0.0, 'key1')


def test_atom_type_shallow_dict_data(structure) -> None:
    isotope_data = AtomTypeShallowDictData(
        {'Na': Quantity(1.0, 'barn'), 'Cl': Quantity(2.0, 'barn')}
    )

    assert isotope_data.get_item('Na', 0.0) == {'': Quantity(1.0, 'barn')}

    assert isotope_data.get_value('Na', 0.0, 'dummy') == Quantity(1.0, 'barn')

    _compare_quantity(
        isotope_data.get_array(structure, 'dummy'),
        Quantity([1.0, 2.0], 'barn'),
    )

    with pytest.raises(KeyError):
        isotope_data.get_item('K', 0.0)

    with pytest.raises(KeyError):
        isotope_data.get_value('K', 0.0, 'key1')


def test_missing_atom_type_shallow_dict_data(structure) -> None:
    isotope_data = AtomTypeDictData(
        {'K': Quantity(1.0, 'barn'), 'Cl': Quantity(2.0, 'barn')}
    )

    with pytest.raises(KeyError):
        isotope_data.get_array(structure, '')


def test_legacy_json_data(structure, bad_structure) -> None:
    isotope_data = LegacyJsonData('Sears1992')

    assert isotope_data.get_item('Na', 0.0) == {
        'coherent_scattering_length': Quantity(3.63, 'fm'),
    }

    assert isotope_data.get_value('Na', 0.0, 'coherent_scattering_length') == (
        Quantity(3.63, 'fm')
    )

    _compare_quantity(
        isotope_data.get_array(structure, 'coherent_scattering_length'),
        Quantity([3.63, 9.5770], 'fm'),
    )

    with pytest.raises(KeyError):
        isotope_data.get_item('X', 0.0)

    with pytest.raises(KeyError):
        isotope_data.get_value('Na', 0.0, 'missing')

    with pytest.raises(
        KeyError, match="Property 'missing' not found in 'Sears1992'"
    ):
        isotope_data.get_array(structure, 'missing')

    with pytest.raises(KeyError):
        isotope_data.get_array(bad_structure, 'coherent_scattering_length')


def test_protocol_get_value() -> None:
    """get_value implemented on protocol but not used in dict-based classes"""
    dummy = 'dummy'

    class TestIsotopeData(IsotopeData):
        def get_item(self, symbol: str, mass: float) -> dict[str, Quantity]:
            assert symbol == dummy

            return {'key1': Quantity(1.0, 'barn'), 'key2': Quantity(2.0, 'kg')}

        def get_array(self, structure: Structure, key: str) -> Quantity:
            raise NotImplementedError

    isotope_data = TestIsotopeData()

    assert isotope_data.get_value(dummy, 0.0, 'key2') == Quantity(2.0, 'kg')
