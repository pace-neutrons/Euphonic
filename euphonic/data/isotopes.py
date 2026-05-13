from abc import abstractmethod
from importlib.resources import files
import json
from pathlib import Path
from typing import Protocol

import numpy as np
from pint import UndefinedUnitError

import euphonic.data
from euphonic.ureg import Quantity, ureg


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
        from euphonic.util import comma_join, format_error

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
        from euphonic.util import comma_join, format_error

        if key not in self._data:
            msg = format_error(
                f'Property {key!r} not found in {self._collection!r}',
                fix=(f'Available keys: {comma_join(self._data.keys())}.'),
            )
            raise KeyError(msg)
        return super().get_array(structure, key)


def get_reference_data(
    collection: str = 'Sears1992',
    physical_property: str = 'coherent_scattering_length',
) -> dict[str, Quantity]:
    """
    Get physical data as a dict of (possibly-complex) floats from reference
    data.

    Each "collection" refers to a JSON file which may contain any set of
    properties, indexed by physical_property.

    Properties are stored in JSON files, encoding a single dictionary with the
    structure::

      {"metadata1": "metadata1 text", "metadata2": ...,
       "physical_properties": {"property1": {"__units__": "unit_str",
                                             "H": H_property1_value,
                                             "He": He_property1_value,
                                             "Li": {"__complex__": true,
                                                    "real": Li_property1_real,
                                                    "imag": Li_property1_imag},
                                             "Nh": None,
                                             ...},
                               "property2": ...}}

    Parameters
    ----------
    collection
        Identifier of data file; this may be an inbuilt data set ("Sears1992"
        or "BlueBook") or a path to a JSON file (e.g. "./my_custom_data.json").

    physical_property
        The name of the property for which data should be extracted. This must
        match an entry of "physical_properties" in the data file.

    Returns
    -------
    dict[str, Quantity]
        Requested data as a dict with string keys and (possibly-complex)
        float Quantity values. String or None items of the original data file
        will be omitted.

    """

    # Avoid import loop; the problem will go away when this function is removed
    from euphonic.util import _deprecation_warn, comma_join, format_error

    _deprecation_warn(
        'get_reference_data', 'IsotopeData.get_array', stacklevel=3
    )

    data = _get_all_dicts_from_json(collection)

    if physical_property not in data:
        msg = format_error(
            (f'No such collection {collection!r} '
             f'with property {physical_property!r}.'),
            fix=('Available properties for this collection:'
                 f'{comma_join(data)}.'),
        )
        raise ValueError(msg)

    return data[physical_property]


def _get_all_dicts_from_json(
    collection: str = 'Sears1992',
) -> dict[str, dict[str, Quantity]]:
    from euphonic.util import comma_join, format_error  # Avoid import loop

    _reference_data_files = {
        'Sears1992': 'sears-1992.json',
        'BlueBook': 'bluebook.json',
    }

    def custom_decode(dct):
        if '__complex__' in dct:
            return complex(dct['real'], dct['imag'])
        return dct

    if filename := _reference_data_files.get(collection):

        with open(files(euphonic.data) / filename) as fd:
            file_data = json.load(fd, object_hook=custom_decode)

    elif (filename := Path(collection)).is_file():
        with open(filename) as fd:
            file_data = json.load(fd, object_hook=custom_decode)
    else:
        msg = format_error(
            f'No data files known for collection "{collection}".',
            fix=f'Available collections: {comma_join(_reference_data_files)}.',
        )
        raise ValueError(msg)

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
