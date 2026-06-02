"""This private module contains the implementation of get_reference_data

That function is publicly available to import from euphonic.util, but is
deprecated and will be removed in a future version of Euphonic.

The code is kept in its own file to simplify implementation-sharing with
euphonic.data.isotopes while avoiding import loops with euphonic.util.

"""

from euphonic.ureg import Quantity


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
    from euphonic.data.isotopes import _get_all_dicts_from_json
    from euphonic.util import (
        _deprecation_warn,
        comma_join,
        format_error,
    )

    _deprecation_warn(
        'get_reference_data', 'IsotopeData.get_array', stacklevel=3
    )

    data = _get_all_dicts_from_json(collection)

    if physical_property not in data:
        msg = format_error(
            (
                f'No such collection {collection!r} '
                f'with property {physical_property!r}.'
            ),
            fix=(
                f'Available properties for this collection: {comma_join(data)}.'
            ),
        )
        raise ValueError(msg)

    return data[physical_property]
