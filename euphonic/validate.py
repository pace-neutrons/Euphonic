from typing import List, Any, Union, Type, Tuple, Optional

import numpy as np
from pint import DimensionalityError

from euphonic import ureg


def _check_constructor_inputs(
        objs: List[object],
        types: List[Union[Type, List[Type]]],
        shapes: List[Union[Tuple[int, ...], List[Tuple[int, ...]]]],
        names: List[str]) -> None:
    """
    Make sure all the inputs are all the expected type, and if they are
    an array, the correct shape

    Parameters
    ----------
    objs
        The objects to check
    types
        The expected class of each input. If multiple types are
        accepted, the expected class can be a list of types. e.g.
        types=[[list, np.ndarray], int]
    shapes
        The expected shape of each object (if the object has a shape
        attribute). If the shape of some dimensions don't matter,
        provide -1 for those dimensions, or if none of the dimensions
        matter, provide an empty tuple (). If multiple shapes are
        accepted, the expected shapes can be a list of tuples. e.g.
        shapes=[[(n, 3), (n + 1, 3)], (3,)]
    names
        The name of each input variable

    Raises
    ------
    TypeError
        If one of the items in objs isn't the correct type
    ValueError
        If an array shape don't match the expected shape
    """
    for obj, typ, shape, name in zip(objs, types, shapes, names):
        if not isinstance(typ, list):
            typ = [typ]
        if not any(isinstance(obj, t) for t in typ):
            raise TypeError((f'The type of {name} {type(obj)} doesn\'t '
                             f'match the expected type(s) {typ}'))
        if hasattr(obj, 'shape') and shape:
            if not isinstance(shape, list):
                shape = [shape]
            if not any(obj.shape == _replace_dim(s, obj.shape) for s in shape):
                raise ValueError((
                    f'The shape of {name} {obj.shape} doesn\'t match '
                    f'the expected shape(s) {shape}'))


def _check_unit_conversion(obj: object, attr_name: str, attr_value: Any,
                           unit_attrs: List[str]) -> None:
    """
    If setting an attribute on an object that relates to the units of a
    Quantity (e.g. 'frequencies_unit' in QpointPhononModes) check that
    the unit conversion is valid before allowing the value to be set

    Parameters
    ----------
    obj
        The object to check
    attr_name
        The name of the attribute that is being set
    attr_value
        The new value of the attribute
    unit_attrs
        Only check the unit conversion if the attribute is one of
        unit_attrs

    Raises
    ------
    ValueError
        If the unit conversion is not valid
    """
    if hasattr(obj, attr_name):
        if attr_name in unit_attrs:
            try:
                _ = ureg(getattr(obj, attr_name)).to(attr_value)
            except DimensionalityError:
                raise ValueError((
                    f'"{attr_value}" is not a known dimensionally-consistent '
                    f'unit for "{attr_name}"'))


def _replace_dim(expected_shape: Tuple[int, ...],
                 obj_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    # Allow -1 dimensions to be any size
    idx = np.where(np.array(expected_shape) == -1)[0]
    if len(idx) == 0 or len(expected_shape) != len(obj_shape):
        return expected_shape
    else:
        expected_shape_replaced = np.array(expected_shape)
        expected_shape_replaced[idx] = np.array(obj_shape)[idx]
        return tuple(expected_shape_replaced)


def _ensure_contiguous_attrs(obj: object, required_attrs: List[str],
                             opt_attrs: List[str] = []) -> None:
    """
    Make sure all listed attributes of obj are C Contiguous and of the
    correct type (int32, float64, complex128). This should only be used
    internally, and called before any calls to Euphonic C extension
    functions

    Parameters
    ----------
    obj
        The object that will have it's attributes checked
    required_attrs
        The attributes of obj to be checked. They should all be Numpy
        arrays
    opt_attrs
        The attributes of obj to be checked, but if they don't exist
        will not throw an error. e.g. Depending on the material
        ForceConstants objects may or may not have 'born' defined
    """
    for attr_name in required_attrs:
        attr = getattr(obj, attr_name)
        attr = attr.astype(_get_dtype(attr), order='C', copy=False)
        setattr(obj, attr_name, attr)

    for attr_name in opt_attrs:
        try:
            attr = getattr(obj, attr_name)
            attr = attr.astype(_get_dtype(attr), order='C', copy=False)
            setattr(obj, attr_name, attr)
        except AttributeError:
            pass


def _ensure_contiguous_args(*args: np.ndarray) -> Tuple[np.ndarray, ...]:
    """
    Make sure all arguments are C Contiguous and of the correct type
    (int32, float64, complex128). This should only be used internally,
    and called before any calls to Euphonic C extension functions
    Example use: arr1, arr2 = _ensure_contiguous_args(arr1, arr2)

    Parameters
    ----------
    *args
        The Numpy arrays to be checked

    Returns
    -------
    contiguous_args
        The same as the provided args, but all contiguous.
    """
    return tuple(arg.astype(_get_dtype(arg), order='C', copy=False)
                 for arg in args)


def _get_dtype(arr: np.ndarray) -> Optional[Type]:
    """
    Get the Numpy dtype that should be used for the input array

    Parameters
    ----------
    arr
        The Numpy array to get the type of

    Returns
    -------
    dtype
        The type the array should be
    """
    if np.issubdtype(arr.dtype, np.integer):
        return np.int32
    elif np.issubdtype(arr.dtype, np.floating):
        return np.float64
    elif np.issubdtype(arr.dtype, np.complexfloating):
        return np.complex128
    return None
