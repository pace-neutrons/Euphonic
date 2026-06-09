from importlib.resources import files

from . import data as isotope_data
from ._core import (
    ArrayFromValuesMixin,
    AtomTypeDictData,
    AtomTypeShallowDictData,
    IsotopeData,
    LegacyJsonData,
    Structure,
)
from ._csv import (
    CsvColumnInfo,
    CsvData,
    MissingValueError,
    NoMatchingIsotopeError,
    NotQuantityError,
)

sears_1992 = CsvData(files(isotope_data) / 'sears-1992.csv', {})

__all__ = [
    'ArrayFromValuesMixin',
    'AtomTypeDictData',
    'AtomTypeShallowDictData',
    'CsvColumnInfo',
    'CsvData',
    'IsotopeData',
    'LegacyJsonData',
    'MissingValueError',
    'NoMatchingIsotopeError',
    'NotQuantityError',
    'Structure',
    'sears_1992',
]


def pointless_add(a: int, b: int) -> int:
    return a + b
