from abc import abstractmethod
from functools import cached_property
from importlib.resources import files
import json
from math import isnan
from pathlib import Path
import re
from typing import Any, Protocol

import numpy as np
from pint import UndefinedUnitError
from toolz.dicttoolz import valfilter

import euphonic.data
from euphonic.ureg import Quantity, ureg
from euphonic.util import comma_join, format_error, zips


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
        return self.get_item(symbol, mass)[key]

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

        if key not in self._data:
            msg = format_error(
                f'Property {key!r} not found in dict.',
                fix=(f'Available keys: {comma_join(self._data.keys())}.'),
            )
            raise KeyError(msg)

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
        if key not in self._data:
            msg = format_error(
                f'Property {key!r} not found in {self._collection!r}',
                fix=(f'Available keys: {comma_join(self._data.keys())}.'),
            )
            raise KeyError(msg)
        return super().get_array(structure, key)


class CsvData(IsotopeData):
    def __init__(self, csv_file: Path, property_map: dict[str, str]) -> None:
        """Isotope data collection from Sears-like CSV table

        The format consists of comma-separated fields. The first line should
        be a header with column labels, including units in parentheses as
        appropriate, e.g.:

          symbol,z_number,a_number,mass (amu),b_c (fm)

        The next row should indicate native python data types, e.g.:

          str,int,int,float,complex

        Standard element mixtures should have no a_number. Monoisotopic
        elements (or elements that lack data for other isotopes) should be
        treated as isotopes, including all three properties.  Elements without
        a standard isotopic mixture (such as Tc) should be expressed as a set
        of isotopes.

        Internally, this implementation creates a numpy "record array" and
        queries the data with numpy comparison/indexing features.

        Parameters
        ----------

        csv_file: reference data set, such as sears-1992.csv
        property_columns: mapping of property names to column headers in CSV.
            The property names *must* include symbol, mass and a_number; these
            are used to select and organise the data.

        property_map: mapping of keys used to access data to column names in
            original CSV table. e.g. {"coherent_scattering_length": "b_c"} By
            default the original column names are used (with units stripped),
            so {} is an acceptable value given the right CSV headers.
        """
        self._csv_file = csv_file
        self._property_map = property_map
        self._units: dict[str, ureg.Unit] = {}

    def _get_nearest_row(self, symbol: str, mass: float) -> np.recarray:
        table, _ = self._table_and_units

        symbol_rows = table[table['symbol'] == symbol]
        if len(symbol_rows) < 1:
            if ':' in symbol:  # Variant notation e.g. H:D; try the stem as el
                return self.get_item(
                    self, symbol=(symbol.split(':', maxsplit=1)[0])
                )

            msg = format_error(
                f'No data found for {symbol!r} in {self._csv_file}.',
                fix='Ensure symbol corresponds to an element in the table.',
            )
            raise ValueError(msg)

        return symbol_rows[np.argmin(np.abs(symbol_rows['mass'] - mass))]

    def _validate_raw_value(self, row: np.recarray, key: str) -> None:
        if isnan(raw_value := row[key]):
            if row.a_number == 0:
                summary = (
                    f'Isotopic mixture {row.symbol} has '
                    f'invalid value for {key!r}'
                )
            else:
                summary = (
                    f'Isotope {row.symbol}-{row.a_number} has invalid value '
                    f'for {key!r}'
                )
            msg = format_error(
                summary,
                fix='Ensure key corresponds to a column in the table.',
            )
            raise ValueError(msg)

        if isinstance(raw_value, str):
            msg = format_error(
                'This method cannot return str values.',
                fix=(
                    'Choose another property, or read '
                    f'{self._csv_file} by other means.'
                ),
            )
            raise TypeError(msg)

    def get_value(self, symbol: str, mass: float, key: str) -> Quantity:
        table, units = self._table_and_units
        if key not in table.dtype.names:
            msg = format_error(
                f'No data found for {key!r} in {self._csv_file}.',
                fix='Ensure key corresponds to a column in the table.',
            )
            raise AttributeError(msg)
        unit = units[table.dtype.names.index(key)]

        row = self._get_nearest_row(symbol, mass)
        self._validate_raw_value(row, key)

        return self._apply_unit(row[key], unit)

    def get_array(self, structure: Structure, key: str) -> Quantity:
        targets = list(zips(structure.atom_type, structure.atom_mass))

        # WIP
        _ = {
            (symbol, mass): self.get_value(symbol, mass, key)
            for (symbol, mass) in targets
        }
        return NotImplemented

    def get_item(self, symbol: str, mass: float) -> dict[str, Quantity]:
        """Get available data for specified species"""
        table, units = self._table_and_units

        symbol_rows = table[table['symbol'] == symbol]
        if len(symbol_rows) < 1:
            if ':' in symbol:  # Variant notation e.g. H:D; try the stem as el
                return self.get_item(
                    self, symbol=(symbol.split(':', maxsplit=1)[0])
                )

            msg = format_error(
                f'No data found for {symbol!r} in {self._csv_file}.',
                fix='Ensure symbol corresponds to an element in the table.',
            )
            raise ValueError(msg)

        nearest_row = symbol_rows[
            np.argmin(np.abs(symbol_rows['mass'] - mass))
        ]

        values = (
            self._apply_unit(item, unit)
            for item, unit in zips(nearest_row, units)
        )

        return {
            key: val
            for (key, val) in zips(table.dtype.names, values)
            if val is not None
        }

    @staticmethod
    def _apply_unit(
        value: int | float | complex | str, unit: ureg.Unit | None
    ) -> Quantity | str:
        """Apply unit to number or make Dimensionless"""
        match value, unit:
            case str(), _:
                return None
            case _, None:
                return value * ureg.Unit('dimensionless')
            case _:
                return value * unit

    @staticmethod
    def _split_unit(col_header: str) -> tuple[str, str | None]:
        if match := re.match(
            r"""(.+?)      # Mandatory NAME; any characters, non-greedy
                               #
                   \s*         # Any amount of whitespace between NAME and UNIT
                               #
                   (           # Begin unit group
                               #
                   \(.+\)    # At least one character surrounded by parens
                               #
                   )?$         # Optional group must complete the input string;
                               # this ensures whole line is used despite
                               # non-greedy NAME.  Otherwise we can get
                               # 'NAME (UNIT)' -> ('N', None)
                """,
            col_header,
            re.VERBOSE,
        ):
            name, unit = match.groups()
            return name.strip(), unit.strip() if unit else None

        msg = format_error(
            f'Could not interpret column header {col_header!r}.',
            fix='Format should be "name (unit)" or "name".',
        )
        raise ValueError(msg)

    @staticmethod
    def _check_valid_types(
        types_line: list[str], types_map: dict[str, type]
    ) -> None:
        if unknown_types := set(types_line) - set(types_map):
            msg = format_error(
                'Not all types from second line of CSV were recognised.',
                reason=f'Could not interpret types {unknown_types}.',
                fix=(
                    'Second line of CSV should contain comma-separated '
                    '"int", "complex" and "str"'
                ),
            )
            raise ValueError(msg)

    @staticmethod
    def _normalise_record(line: str, types: list[type]) -> tuple:
        record = line.strip().split(',')

        # We can't deal with uncertain values, these should become NaN
        bad_complex = re.compile(r'\(?[±<>].*$')
        bad_float = re.compile(r'[±<>].*$')

        def _cast(item: str, item_type: type) -> int | float | complex | str:
            import builtins

            match item, item_type:
                case '', builtins.int:
                    return 0
                case value, builtins.float if not value or bad_float.match(
                    value
                ):
                    return float('NaN')
                case value, builtins.complex if not value or bad_complex.match(
                    value
                ):
                    return complex(float('NaN'), float('NaN'))
                case _:
                    try:
                        return item_type(item)
                    except ValueError as err:
                        raise ValueError(item, item_type) from err

        return tuple(map(_cast, record, types))

    @property
    def _table(self) -> np.recarray:
        return self._table_and_units[0]

    @cached_property
    def _unit_map(self) -> dict[str, ureg.Unit]:
        def is_not_none(a: Any) -> bool:
            """In operator from Python 3.14"""
            return a is not None

        table, units = self._table_and_units
        unit_map = dict(zips(table.dtype.names, units))
        return valfilter(is_not_none, unit_map)

    @cached_property
    def _table_and_units(self) -> tuple[np.recarray, list[ureg.Unit | None]]:
        types_map = {
            'int': int,
            'float': float,
            'complex': complex,
            'str': str,
        }

        # Map of CSV columns to rename: inverse of user-provided map
        name_map = {value: key for key, value in self._property_map}

        with self._csv_file.open('rt') as fd:
            col_names = next(fd).strip().split(',')
            types_line = next(fd).strip().split(',')

            self._check_valid_types(types_line, types_map)
            types = [types_map[t] for t in types_line]

            col_names, col_units = zips(*map(self._split_unit, col_names))

            col_names = [name_map.get(name, name) for name in col_names]

            records = [self._normalise_record(record, types) for record in fd]

            # Store strings a pointers to avoid truncation
            types = [(object if t is str else t) for t in types]

            table = np.rec.fromrecords(
                records,
                dtype=list(zips(col_names, types)),
            )

        units = [ureg.Unit(unit) if unit else None for unit in col_units]

        return table, units


sears_1992 = CsvData(files(euphonic.data) / 'sears-1992.csv', {})

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
        file_path = files(euphonic.data) / filename
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
            f'No units in file ({filename}).',
            fix='Ensure file specifies dimensions with "__units__" metadata.',
        )
        if unit_str is None:
            raise ValueError(no_units_msg)

        try:
            unit = ureg(unit_str)
        except UndefinedUnitError as exc:
            msg = format_error(
                f'Unsupported units ({unit_str}) from data file "{filename}".',
                fix='Ensure units are supported by Euphonic unit register.',
            )
            raise ValueError(msg) from exc

        result[physical_property] = {
            key: Quantity(value, unit)
            for key, value in data.items()
            if isinstance(value, (float, complex))
        }

    return result
