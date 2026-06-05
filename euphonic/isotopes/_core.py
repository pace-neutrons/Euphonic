from abc import abstractmethod
import builtins
from collections.abc import Collection
from dataclasses import dataclass
from importlib.resources import files
import json
from pathlib import Path
import re
from typing import Protocol

import numpy as np
from pint import UndefinedUnitError
from typing_extensions import Self

from euphonic.ureg import Quantity, ureg
from euphonic.util import comma_join, format_error

from . import data as isotope_data


def _validate_key(
    key: str, *, valid_keys: Collection[str], location: str
) -> None:
    if key not in valid_keys:
        msg = format_error(
            f'Property {key!r} not found in {location}.',
            fix=f'Available keys: {comma_join(valid_keys)}.',
        )
        raise KeyError(msg)


class Structure(Protocol):
    """Structure data like euphonic.crystal.Crystal

    Specified as a more general protocol here, so that additional structure
    types may be added for cluster, slab etc. without breaking the IsotopeData
    interface (which could plausibly be implemented in external code.)
    """

    atom_type: np.ndarray
    atom_mass: Quantity


class IsotopeData(Protocol):
    """Interface for data such as neutron scattering length, cross-section

    Implementations may ignore mass or use it to distinguish between
    isotopes.
    """

    def get_value(self, symbol: str, mass: float, key: str) -> Quantity:
        """Get a property value for specified species"""
        item = self.get_item(symbol, mass)
        _validate_key(key, valid_keys=item.keys(), location='data')
        return item[key]

    @abstractmethod
    def get_item(self, symbol: str, mass: float) -> dict[str, Quantity]:
        """Get available data for specified species"""

    @abstractmethod
    def get_array(self, structure: Structure, key: str) -> Quantity:
        """Get a Quantity array of property corresponding to structure"""


class ArrayFromValuesMixin:
    """Basic implementation of IsotopeData.get_array using get_value method"""

    def get_array(self, structure: Structure, key: str) -> Quantity:
        """Get a Quantity array of property corresponding to structure"""

        items = [
            self.get_value(symbol, mass, key)
            for symbol, mass in zip(
                structure.atom_type, structure.atom_mass, strict=True
            )
        ]
        return self._items_to_quantity(items)

    @staticmethod
    def _items_to_quantity(items: list[Quantity]) -> Quantity:
        # Quantity.from_list chokes on mixed float/complex items;
        # instead create bare numpy array of appropriate type then add units.

        dtype = (
            complex
            if any(isinstance(item.magnitude, complex) for item in items)
            else float
        )
        units = items[0].units

        magnitude = np.fromiter(
            (item.to(units).magnitude for item in items),
            dtype=dtype,
        )
        return Quantity(magnitude, units)


class AtomTypeDictData(ArrayFromValuesMixin, IsotopeData):
    def __init__(self, data: dict[str, dict[str, Quantity]]) -> None:
        """Property data from simple dict

        Dict must have structure::

          { key: {atom_type_1: value_1, atom_type_2: value_2, ...}, ...}

        Where *key* is a string key passed to corresponding argument of
        .get_array() method (e.g. "coherent_cross_section"),
        and atom types correspond to Structure.atom_type array (e.g. "Si").

        Only the atom_type attribute is used to select data from the structure:
        this is not very robust in case of isotopic substitutions.
        """
        self._data = data

    def get_item(
        self,
        symbol: str,
        mass: float,  # noqa: ARG002
    ) -> dict[str, Quantity]:
        return {key: value[symbol] for key, value in self._data.items()}

    def get_value(
        self,
        symbol: str,
        mass: float,  # noqa: ARG002
        key: str,
    ) -> Quantity:

        _validate_key(key, valid_keys=self._data.keys(), location='dict')
        return self._data[key][symbol]


class AtomTypeShallowDictData(ArrayFromValuesMixin, IsotopeData):
    def __init__(self, data: dict[str, Quantity]) -> None:
        """Property data from very simple dict

        Dict must have structure::

          { atom_type_1: value_1, atom_type_2: value_2, ...}

        As these data only contains a single property, the usual *key* will be
        ignored. This is intended for convenient user input of weights at the
        time they are used, and not for development of a longer-term reusable
        dataset.

        Only the atom_type attribute is used to select data from the structure:
        this is not very robust in case of isotopic substitutions.
        """
        self._data = data

    def get_item(
        self,
        symbol: str,
        mass: float,  # noqa: ARG002
    ) -> dict[str, Quantity]:
        """Get all properties for specified species

        In this case, mass is ignored and sole property has empty-string key ''
        """
        return {'': self._data[symbol]}

    def get_value(
        self,
        symbol: str,
        mass: float,  # noqa: ARG002
        key: str,  # noqa: ARG002
    ) -> Quantity:
        """Get property value for specified species

        In this case, mass and key are ignored: only one value is available
        for each symbol.
        """
        return self._data[symbol]


class LegacyJsonData(AtomTypeDictData):
    """Property data from JSON file

    This provides an interface for the legacy reference data.

    Only the atom_type attribute is used to select data from the structure:
    this is not very robust in case of isotopic substitutions.
    """

    def __init__(self, collection: str):
        self._collection = collection

        data = _get_all_dicts_from_json(collection)
        super().__init__(data)

    def get_array(self, structure: Structure, key: str) -> Quantity:
        """Get a Quantity array of property corresponding to structure"""
        _validate_key(
            key, valid_keys=self._data.keys(), location=repr(self._collection)
        )
        return super().get_array(structure, key)


@dataclass
class CsvColumnInfo:
    name: str
    dtype: type
    unit: ureg.Unit | None

    @classmethod
    def from_raw(
        cls, raw_name: str, raw_dtype: str, name_map: dict[str, str]
    ) -> Self:
        """Convert items from zipped CSV headers

        Parameters
        ----------

        raw_name:
            e.g. 'b_inc (fermi)'

        raw_dtype:
            e.g. 'complex'

        name_map:
            mapping from CSV column header names (without unit) and names used
            in resulting data objects. Empty dict {} is an acceptable value;
            missing items will not be changed.

            Note that this is the inverse of the CsvData ``property_map``.

            e.g. {'b_inc': 'incoherent_scattering_length'}

        """
        name, unit = cls._split_unit(raw_name)
        name = name_map.get(name, name)

        dtype = getattr(builtins, raw_dtype)

        if unit is None:
            return cls(name, dtype, None)
        return cls(name, dtype, ureg.Unit(unit))

    @staticmethod
    def _split_unit(col_header: str) -> tuple[str, str | None]:
        """Split unit from column name if present

        e.g.::
            'name  '       -> 'name', None
            'name (unit)'  -> 'name', '(unit)'
        """

        if match := re.fullmatch(
            r"""
              (?P<name>\w+)  # Mandatory NAME; any word characters
                              #
              \s*             # Any amount of whitespace
                              #
              (?:             # Begin non-capturing (UNIT) group
                              #
                 \(           # literal (
                              #
                 (?P<unit>.+) # capture anything between parens as UNIT
                              #
                              #
                 \)           # literal )
                              #
              )?              # end optional group
            """,
            col_header,
            re.VERBOSE,
        ):
            name, unit = match.group('name', 'unit')
            return name.strip(), unit.strip() if unit else None

        msg = format_error(
            f'Could not interpret column header {col_header!r}.',
            fix='Format should be "name (unit)" or "name".',
        )
        raise ValueError(msg)

    def _apply_unit(
        self, value: int | float | complex | np.ndarray | str
    ) -> Quantity | str:
        """Apply unit to number or make Dimensionless"""
        match value, self.unit:
            case str(s), None:
                return s
            case str(), _:
                msg = format_error(
                    'Cannot apply units to a string.',
                    fix='Check header format in CSV data.',
                )
                raise TypeError(msg)
            case x, None:
                return Quantity(x, 'dimensionless')
            case x, unit:
                return Quantity(value, unit)


def _get_all_dicts_from_json(
    collection: str = 'Sears1992',
) -> dict[str, dict[str, Quantity]]:
    _reference_data_files = {
        'Sears1992': 'sears-1992.json',
        'BlueBook': 'bluebook.json',
    }

    def custom_decode(dct):
        if '__complex__' in dct:
            return complex(dct['real'], dct['imag'])
        return dct

    if filename := _reference_data_files.get(collection):
        file_path = files(isotope_data) / filename
    else:
        file_path = Path(collection)

    if not file_path.is_file():
        msg = format_error(
            f'No data files known for collection "{collection}".',
            fix=f'Available collections: {comma_join(_reference_data_files)}.',
        )
        raise ValueError(msg)

    with file_path.open() as fd:
        file_data = json.load(fd, object_hook=custom_decode)

    if 'physical_property' not in file_data:
        msg = format_error(
            'Data file does not contain required key "physical_property".',
            fix='Ensure file is formatted correctly.',
        )
        raise AttributeError(msg)

    result: dict[str[dict[str, Quantity]]] = {}
    for physical_property, data in file_data['physical_property'].items():
        unit_str = data.get('__units__')

        no_units_msg = format_error(
            f'No units in file ({file_path.name}).',
            fix='Ensure file specifies dimensions with "__units__" metadata.',
        )
        if unit_str is None:
            raise ValueError(no_units_msg)

        try:
            unit = ureg(unit_str)
        except UndefinedUnitError as exc:
            msg = format_error(
                f'Unsupported units ({unit_str}) from data file '
                f'"{file_path.name}".',
                fix='Ensure units are supported by Euphonic unit register.',
            )
            raise ValueError(msg) from exc

        result[physical_property] = {
            key: Quantity(value, unit)
            for key, value in data.items()
            if isinstance(value, (float, complex))
        }

    return result
