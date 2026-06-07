import builtins
import csv
from dataclasses import dataclass
from functools import cached_property, partial
from math import isnan
from numbers import Complex, Integral, Real
from pathlib import Path
import re
from typing import Any, ClassVar

import numpy as np
from toolz.dicttoolz import valfilter
from toolz.itertoolz import first
from typing_extensions import Self

from euphonic.ureg import Quantity, ureg
from euphonic.util import format_error, zips

from ._core import IsotopeData, Structure


class NotQuantityError(TypeError): ...


class MissingValueError(KeyError): ...


class NoMatchingIsotopeError(KeyError): ...


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

        Symbols are permitted to include comments/variations after a colon
        character, e.g. {symbol='H:2', mass=2.0} might conveniently indicate
        a deuterium nucleus.

        Internally, this implementation creates a numpy "record array" and
        queries the data with numpy comparison/indexing features.

        Numpy-friendly sentinel values are used for missing data; these are
        stored in the MISSING attribute.

        The MASS_MATCH_THRESHOLD attribute sets the tolerance for mismatch
        between input masses and matched data rows.

        Parameters
        ----------

        csv_file:
            reference data set, such as sears-1992.csv
        property_columns:
            mapping of property names to column headers in CSV.
            The property names *must* include symbol, mass and a_number; these
            are used to select and organise the data.

        property_map:
            mapping of keys used to access data to column names in
            original CSV table. e.g. {"coherent_scattering_length": "b_c"} By
            default the original column names are used (with units stripped),
            so {} is an acceptable value given the right CSV headers.
        """
        self._csv_file = csv_file
        self._property_map = property_map

    MISSING: ClassVar[dict[str, Any]] = {
        int: -(2**31),
        float: float('-Inf'),
        complex: complex(float('-Inf'), float('-Inf')),
    }
    INVALID_FLOAT: ClassVar[float] = float('NaN')
    MASS_MATCH_THRESHOLD: ClassVar[float] = 0.3

    # Regular expressions: identify uncertain values which should become NaN
    _bad_complex = re.compile(r'\(?[±<>].*$')
    _bad_float = re.compile(r'[±<>].*$')

    @classmethod
    def _is_missing(cls, value: Any) -> bool:
        """Check if value is None or matches the sentinel of its type

        Empty strings are not considered empty
        """
        # Numpy classes may need mapping back to their native equivalent
        match value:
            case Integral():
                dtype = int
            case Real():
                dtype = float
            case Complex():
                dtype = complex
            case _:
                dtype = type(value)

        return value == cls.MISSING.get(dtype)

    def _get_symbol_rows(self, symbol: str) -> np.recarray:
        symbol_rows = self._table[self._table['symbol'] == symbol]
        if len(symbol_rows) < 1:
            if ':' in symbol:  # Variant notation e.g. H:D; try the stem as el
                return self._get_symbol_rows(symbol.split(':', maxsplit=1)[0])

            msg = format_error(
                f'No data found for {symbol!r} in {self._csv_file}.',
                fix='Ensure symbol corresponds to an element in the table.',
            )
            raise KeyError(msg)
        return symbol_rows

    @classmethod
    def _format_row_name(cls, row: np.record) -> str:
        if cls._is_missing(row.a_number):
            return f'{row.symbol}'
        return f'{row.symbol}-{row.a_number}'

    def _get_nearest_row(self, symbol: str, mass: float) -> np.recarray:
        symbol_rows = self._get_symbol_rows(symbol)
        nearest = symbol_rows[np.argmin(np.abs(symbol_rows['mass'] - mass))]

        if not np.isclose(
            nearest['mass'], mass, atol=self.MASS_MATCH_THRESHOLD
        ):
            msg = format_error(
                f'Could not find a satisfactory match in {self._csv_file} '
                f'for {symbol} with mass {mass}: best match was '
                f'{self._format_row_name(nearest)} with mass {mass}.',
                fix=(
                    'Correct input symbol and mass '
                    'or use a different data source.'
                ),
            )
            raise NoMatchingIsotopeError(msg)

        print(
            f'Found reference data for {self._format_row_name(nearest)} '
            f'to match input symbol {symbol} with mass {mass}.'
        )

        return nearest

    def get_value(self, symbol: str, mass: float, key: str) -> Quantity:
        """Get specific column data for specified species

        This will be validated, raising errors if the data is missing or
        otherwise unusable.

        Parameters
        ----------
        symbol:
            element symbol e.g. 'Hg'
        mass:
            mass in a.m.u. e.g. 200.6. This is used to identify the isotope or
            standard isotope mixture (i.e. row) from reference data table.
        key:
            property name e.g. 'coherent_cross_section'
        """
        item = self._get_item(symbol, mass)

        if key not in item:
            msg = format_error(
                f'Column {key!r} was not found in data.',
                fix=(
                    'Check key is spelled correctly and present '
                    f'in {self._csv_file} for a numerical data type.'
                ),
            )
            raise KeyError(msg)
        value = item[key]

        if isinstance(value, str):
            msg = format_error(
                'This method cannot return str values.',
                fix=(
                    'Choose another property, or read '
                    f'{self._csv_file} by other means.'
                ),
            )
            raise NotQuantityError(msg)

        if self._is_missing(value.magnitude):
            msg = format_error(
                f'Value is missing for {key} in row '
                f'{symbol}{item["a_number"]}',
                fix='Check isotope is correct, or use another data source.',
            )
            raise MissingValueError(msg)

        if isnan(value.magnitude):
            if self._is_missing(item['a_number'].magnitude):
                summary = (
                    f'Isotopic mixture {symbol} has invalid value for {key!r}'
                )
            else:
                summary = (
                    f'Isotope {symbol}-{item["a_number"].magnitude} has '
                    f'invalid value for {key!r}'
                )
            msg = format_error(
                summary,
                fix='Check isotope is correct, or use another data source.',
            )
            raise ValueError(msg)

        return value

    def get_array(self, structure: Structure, key: str) -> Quantity:
        """Get a Quantity array of property corresponding to structure"""
        targets = list(
            zips(structure.atom_type, structure.atom_mass.to('amu').magnitude)
        )

        value_table = {
            (symbol, mass): self.get_value(symbol, mass, key).magnitude
            for (symbol, mass) in targets
        }

        result_raw = np.empty_like(
            structure.atom_mass,
            dtype=type(  # int, real or complex
                first(value_table.values())
            ),
        )

        for i, target in enumerate(targets):
            result_raw[i] = value_table[target]

        col_info = self._column_headers[key]
        return col_info._apply_unit(result_raw)

    @classmethod
    def _allowed_value(cls, value: str | Quantity) -> bool:
        return not (isinstance(value, str) or cls._is_missing(value.magnitude))

    def get_item(self, symbol: str, mass: float) -> dict[str, Quantity]:
        """Get available data for specified species

        This may include invalid (NaN) values, but *missing* values will be
        omitted from the dict.
        """
        item = self._get_item(symbol, mass)
        return valfilter(self._allowed_value, item)

    def _get_item(self, symbol: str, mass: float) -> dict[str, Quantity]:
        """Get available data for specified species

        This may include invalid (NaN) values and missing values
        (represented by sentry values from CsvData.MISSING).
        """
        nearest_row = self._get_nearest_row(symbol, mass)

        return {
            col_info.name: col_info._apply_unit(item)
            for col_info, item in zips(
                self._column_headers.values(), nearest_row
            )
        }

    @classmethod
    def _normalise_record(cls, record: list[str], types: list[type]) -> tuple:

        def _cast(item: str, item_type: type) -> int | float | complex | str:
            # str stored as object(pointer) but should still be cast to str
            item_type = str if item_type is object else item_type

            match item, item_type:
                case '', builtins.int | builtins.float | builtins.complex:
                    return cls.MISSING[item_type]

                case value, builtins.float if (
                    not value or cls._bad_float.match(value)
                ):
                    return cls.INVALID_FLOAT

                case value, builtins.complex if (
                    not value or cls._bad_complex.match(value)
                ):
                    return complex(cls.INVALID_FLOAT, cls.INVALID_FLOAT)
                case _:
                    try:
                        return item_type(item)
                    except ValueError as err:
                        raise ValueError(item, item_type) from err

        return tuple(map(_cast, record, types))

    @property
    def _table(self) -> np.recarray:
        return self._table_and_headers[0]

    @property
    def _column_headers(self) -> dict[str, CsvColumnInfo]:
        return self._table_and_headers[1]

    @cached_property
    def _table_and_headers(
        self,
    ) -> tuple[np.recarray, dict[str, CsvColumnInfo]]:
        # Map of CSV columns to rename: inverse of user-provided map
        name_map = {value: key for key, value in self._property_map.items()}

        with self._csv_file.open('rt', encoding='utf-8') as fd:
            reader = csv.reader(fd)
            col_names = next(reader)
            col_types = next(reader)
            records = list(filter(None, reader))

        # Column header info is stored on class for later use
        build_info = partial(CsvColumnInfo.from_raw, name_map=name_map)
        column_headers = {
            col_info.name: col_info
            for col_info in map(build_info, col_names, col_types)
        }

        # Use cleaned-up names for recarray columns
        col_names = (col_info.name for col_info in column_headers.values())

        # Store strings as pointers in recarray to avoid truncation
        record_types = [
            (object if col_info.dtype is str else col_info.dtype)
            for col_info in column_headers.values()
        ]

        # Handle some missing/invalid data scenarios
        records = [
            self._normalise_record(record, record_types) for record in records
        ]

        return (
            np.rec.fromrecords(
                records,
                dtype=list(zips(col_names, record_types)),
            ),
            column_headers,
        )
