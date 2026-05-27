from dataclasses import dataclass
from math import isnan

import numpy as np
from numpy.testing import assert_allclose
import pytest

from euphonic.data.isotopes import (
    AtomTypeDictData,
    AtomTypeShallowDictData,
    IsotopeData,
    LegacyJsonData,
    Structure,
    sears_1992,
)
from euphonic.ureg import Quantity, ureg


def _compare_quantity(a: Quantity, b: Quantity) -> None:
    assert a.units == b.units
    assert_allclose(a.magnitude, b.magnitude)


@dataclass
class SomeStructure:
    atom_type: np.ndarray
    atom_mass: Quantity


@pytest.fixture
def structure() -> SomeStructure:
    return SomeStructure(
        atom_type=np.array(['Na', 'Cl']),
        atom_mass=Quantity([22.99, 35.45], 'amu'),
    )


@pytest.fixture
def bad_structure() -> SomeStructure:
    return SomeStructure(
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


def _assert_equal_quantity_dict(
    item_1: dict[str, Quantity], item_2: dict[str, Quantity]
) -> None:
    assert item_1.keys() == item_2.keys()

    for key, value in item_1.items():
        if isnan(value.magnitude):
            assert isnan(item_2[key].magnitude)
        else:
            assert value == item_2[key]


class TestSears1992CSV:
    def test_internals(self) -> None:
        table, units = sears_1992._table_and_units

        assert table.dtype == [
            ('symbol', object),
            ('z_number', '<i8'),
            ('a_number', '<i8'),
            ('mass', '<f8'),
            ('spin', object),
            ('abundance', '<f8'),
            ('half_life', '<f8'),
            ('coherent_scattering_length', '<c16'),
            ('incoherent_scattering_length', '<c16'),
            ('coherent_cross_section', '<f8'),
            ('incoherent_cross_section', '<f8'),
            ('scattering_cross_section', '<f8'),
            ('absorption_cross_section', '<f8'),
        ]

        assert None in units
        assert ureg.Unit('fermi') in units

        assert sears_1992._unit_map['mass'] == ureg.Unit('amu')
        assert 'spin' not in sears_1992._unit_map

    def test_get_item(self) -> None:
        # Monisotopic element
        # Au,79,197,196.966570103,3/2(+),100.0,,(7.63+0j),(-1.84+0j),7.32,0.43,7.75,98.65
        _assert_equal_quantity_dict(
            sears_1992.get_item('Au', mass=197.0),
            {
                'z_number': Quantity(79, 'dimensionless'),
                'a_number': Quantity(197, 'dimensionless'),
                'mass': Quantity(196.966570103, 'amu'),
                'abundance': Quantity(100.0, 'percent'),
                'half_life': Quantity(float('nan'), 'year'),
                'coherent_scattering_length': Quantity(7.63 + 0j, 'fermi'),
                'incoherent_scattering_length': Quantity(-1.84 + 0j, 'fermi'),
                'coherent_cross_section': Quantity(7.32, 'barn'),
                'incoherent_cross_section': Quantity(0.43, 'barn'),
                'scattering_cross_section': Quantity(7.75, 'barn'),
                'absorption_cross_section': Quantity(98.65, 'barn'),
            },
        )

        # Isotopic mixture
        # Hg,80,,200.592,,,,(12.692+0j),,20.24,6.6,26.8,372.3
        _assert_equal_quantity_dict(
            sears_1992.get_item('Hg', mass=200.7),
            {
                'z_number': Quantity(80, 'dimensionless'),
                'a_number': Quantity(0, 'dimensionless'),
                'mass': Quantity(200.592, 'amu'),
                'abundance': Quantity(float('nan'), 'percent'),
                'half_life': Quantity(float('nan'), 'year'),
                'coherent_scattering_length': Quantity(12.692 + 0j, 'fermi'),
                'incoherent_scattering_length': Quantity(
                    float('nan'), 'fermi'
                ),
                'coherent_cross_section': Quantity(20.24, 'barn'),
                'incoherent_cross_section': Quantity(6.6, 'barn'),
                'scattering_cross_section': Quantity(26.8, 'barn'),
                'absorption_cross_section': Quantity(372.3, 'barn'),
            },
        )

    def test_get_value(self) -> None:
        assert sears_1992.get_value(
            'Hg', mass=200.7, key='coherent_scattering_length'
             ) == Quantity(12.692+0j, 'fermi')
