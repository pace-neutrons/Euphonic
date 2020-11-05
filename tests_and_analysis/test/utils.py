import os
import json
from typing import Optional

import numpy as np
import numpy.testing as npt

from euphonic import ureg, __version__

def get_data_path() -> str:
    """
    Get the path to the data for the tests.

    Returns
    -------
    str: The path to the test data.
    """
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), "data")


def get_test_qpts(qpts_option: Optional[str] = 'default') -> np.ndarray:
    """
    Returns 'standard' sets of qpoints for testing

    Parameters
    ----------
    qpts_option
        The set of q-points to return, one of {'default', 'split',
        'split_insert_gamma'}

    Returns
    -------
    test_qpts
        A 2D (n, 3) array of q-points
    """
    if qpts_option == 'split':
        return np.array([
            [0.00, 0.00, 0.00],
            [0.00, 0.00, 0.50],
            [0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00],
            [-0.25, 0.50, 0.50],
            [0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00],
            [-0.151515, 0.575758, 0.5],
            [0.00, 0.00, 0.00]])
    elif qpts_option == 'split_insert_gamma':
        return np.array([
            [0.00, 0.00, 0.00],
            [0.00, 0.00, 0.50],
            [0.00, 0.00, 0.00],
            [-0.25, 0.50, 0.50],
            [0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00],
            [-0.151515, 0.575758, 0.5],
            [0.00, 0.00, 0.00]])
    else:
        return np.array([
            [0.00, 0.00, 0.00],
            [0.00, 0.00, 0.50],
            [-0.25, 0.50, 0.50],
            [-0.151515, 0.575758, 0.5],
            [0.30, 0.00, 0.00],
            [0.00, 0.40, 0.00],
            [0.60, 0.00, 0.20],
            [2.00, 2.00, 0.5],
            [1.75, 0.50, 2.50]])


def check_mode_values_at_qpts(qpts, values, expected_values, atol, rtol,
                              gamma_atol=None, gamma_operator=np.less):
    """
    Utility function for comparing difficult to test values (e.g.
    frequencies, structure factors) which are often unstable at the
    gamma point

    Parameters
    ----------
    qpts : (n_qpts, 3) float ndarray
        Q-points at which values have been calculated
    values : (n_qpts, Any) float ndarray
        Values to test
    expected_values : (n_qpts, Any) float ndarray
        What the values should be
    atol : float
        Absolute tolerance (is passed to npt.assert_allclose)
    rtol : float
        Relative tolerance (is passed to npt.assert_allclose)
    gamma_atol : float, optional default None
        Absolute tolerance to test gamma point acoustic mode values
        against
    gamma_operator : numpy.ufunc, optional, default numpy.less
        The function to compare the test values and expected values at
        the gamma point. Should be numpy.less or numpy.greater
    """
    values_to_test = np.ones(values.shape, dtype=bool)

    if gamma_atol:
        gamma_points = (np.sum(np.absolute(qpts - np.rint(qpts)), axis=-1)
                        < 1e-10)
        values_to_test[gamma_points, :3] = False
        assert np.all(gamma_operator(
            np.absolute(values[~values_to_test]),
            gamma_atol))

    npt.assert_allclose(
        values[values_to_test],
        expected_values[values_to_test],
        atol=atol, rtol=rtol)


def check_unit_conversion(obj, attr, unit):
    """
    Utility function to check unit conversions in Euphonic objects

    Parameters
    ----------
    obj : Object
        The object to check
    attr : str
        The name of the attribute to change the units of
    unit : str
        The unit to change attr to
    """
    unit_attr = attr + '_unit'
    original_attr_value = np.copy(
        getattr(obj, attr).magnitude)*getattr(obj, attr).units
    setattr(obj, unit_attr, unit)

    # Check unit value (e.g. 'frequencies_unit') has changed
    assert getattr(obj, unit_attr) == unit
    # Check when returning the attrbute (e.g. 'frequencies') it is in
    # the desired unit and has the correct magnitude
    if attr == 'temperature':
        assert str(getattr(obj, attr).units) == str(ureg(unit).units)
    else:
        assert getattr(obj, attr).units == ureg(unit)
    npt.assert_allclose(getattr(obj, attr).magnitude,
                        original_attr_value.to(unit).magnitude)


def check_json_metadata(json_file: str, class_name: str):
    """
    Utility function to check that .json file metadata has been output
    correctly

    Parameters
    ----------
    json_file
        Path and name of the .json file
    class_name
        The name of the class that should've been written to the file.
        Using a string rather than cls.__name__ so it is more explicit
        for testing
    """
    with open(json_file, 'r') as f:
        data = json.loads(f.read())
    assert data['__euphonic_class__'] == class_name
    assert data['__euphonic_version__'] == __version__
