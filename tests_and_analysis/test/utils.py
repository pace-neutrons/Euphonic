import os
import json
from typing import Optional

import numpy as np
import numpy.testing as npt

from euphonic import ureg, __version__, Spectrum1D, Spectrum1DCollection


def get_data_path() -> str:
    """
    Get the path to the data for the tests.

    Returns
    -------
    str: The path to the test data.
    """
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), "data")


def get_phonopy_path(material: str, filename: str) -> str:
    """
    Get the path to a Phonopy data file

    Returns
    -------
    str: The path to the Phonopy data file
    """
    return os.path.join(get_data_path(), 'phonopy_files', material, filename)


def get_castep_path(material: str, filename: str) -> str:
    """
    Get the path to a CASTEP data file

    Returns
    -------
    str: The path to the CASTEP data file
    """
    return os.path.join(get_data_path(), 'castep_files', material, filename)


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
            [1.00, 1.00, 1.00],
            [0.00, 0.00, 0.50],
            [0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00],
            [-0.25, 0.50, 0.50],
            [2.00, 1.00, 1.00],
            [2.00, 1.00, 1.00],
            [0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00],
            [-1.00, 1.00, 1.00],
            [-1.00, 1.00, 1.00],
            [-0.151515, 0.575758, 0.5],
            [-1.00, 1.00, 1.00]])
    elif qpts_option == 'split_insert_gamma':
        return np.array([
            [1.00, 1.00, 1.00],
            [0.00, 0.00, 0.50],
            [0.00, 0.00, 0.00],
            [-0.25, 0.50, 0.50],
            [2.00, 1.00, 1.00],
            [0.00, 0.00, 0.00],
            [-1.00, 1.00, 1.00],
            [-0.151515, 0.575758, 0.5],
            [-1.00, 1.00, 1.00]])
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


def check_frequencies_at_qpts(qpts, freqs, expected_freqs, atol, rtol,
                              gamma_atol=None):
    check_mode_values_at_qpts(
            qpts, freqs, expected_freqs, atol, rtol, gamma_atol,
            gamma_operator=np.less, gamma_idx_limit=3)


def check_structure_factors_at_qpts(qpts, freqs, expected_freqs, atol, rtol,
                                    gamma_atol=None):
    check_mode_values_at_qpts(
            qpts, freqs, expected_freqs, atol, rtol, gamma_atol,
            gamma_operator=np.greater, gamma_idx_limit=1)


def check_mode_values_at_qpts(qpts, values, expected_values, atol, rtol,
                              gamma_atol=None, gamma_operator=np.less,
                              gamma_idx_limit=None):
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
    gamma_idx_limit : int, optional default 1
        Required if gamma_atol is set. Describes which gamma-point values
        to test with gamma_atol e.g. gamma_idx_limit = 3 will test the
        first 3 values (i.e. the acoustic modes) at each gamma qpoint
    """
    values_to_test = np.ones(values.shape, dtype=bool)

    if gamma_atol:
        gamma_points = (np.sum(np.absolute(qpts - np.rint(qpts)), axis=-1)
                        < 1e-10)
        values_to_test[gamma_points, :gamma_idx_limit] = False
        # Don't use the gamma_atol method where values == 0 e.g. structure
        # factors at q=[0., 0., 0.] should be 0. so would fail if testing
        # that they are more than gamma_atol
        values_to_test[np.where(values == 0.)] = True
        assert np.all(gamma_operator(
            np.absolute(values[~values_to_test]),
            gamma_atol))

    npt.assert_allclose(
        values[values_to_test],
        expected_values[values_to_test],
        atol=atol, rtol=rtol)


def check_unit_conversion(obj: object, attr: str, unit: str) -> None:
    """
    Utility function to check unit conversions in Euphonic objects

    Parameters
    ----------
    obj
        The object to check
    attr
        The name of the attribute to change the units of
    unit
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


def check_property_setters(obj: object, attr: str, unit: str,
                           scale: float) -> None:
    """
    Utility function to check setters for Quantity properties in
    Euphonic objects

    Parameters
    ----------
    obj
        The object to check
    attr
        The name of the attribute to set
    unit
        The unit the attribute should be set to
    scale
        What to scale the attribute by, so we can check its been changed
    """
    new_attr = scale*getattr(obj, attr).to(unit)
    setattr(obj, attr, new_attr)
    set_attr = getattr(obj, attr)
    assert set_attr.units == new_attr.units
    npt.assert_allclose(set_attr.magnitude, new_attr.magnitude)


def check_json_metadata(json_file: str, class_name: str) -> None:
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


def sum_at_degenerate_modes(values_to_sum, frequencies, TOL=0.05):
    """
    For degenerate frequency modes, eigenvectors are an arbitrary
    admixture, so derived quantities (e.g. structure factors) can
    only be compared when summed over degenerate modes. This function
    performs that summation

    Parameters
    ----------
    values_to_sum (n_qpts, n_branches, ...) float ndarray
        Magnitude of values to be summed, can be 2D or 3D
    frequencies (n_qpts, n_branches) float ndarray
        The plain frequency magnitudes

    Returns
    -------
    value_sum (n_qpts, n_branches, ...) float ndarray
        The summed values. As there will be different numbers
        of summed values for each q-point depending on the number of
        degenerate modes, the last few entries for some q-points will be
        zero
    """
    value_sum = np.zeros(values_to_sum.shape)
    if value_sum.ndim == 2:
        value_sum = np.expand_dims(value_sum, -1)
        values_to_sum = np.expand_dims(values_to_sum, -1)
    for i in range(len(frequencies)):
        diff = np.append(TOL + 1, np.diff(frequencies[i]))
        unique_index = np.where(diff > TOL)[0]
        x = np.zeros(len(frequencies[0]), dtype=np.int32)
        x[unique_index] = 1
        unique_modes = np.cumsum(x) - 1
        for j in range(value_sum.shape[-1]):
            value_sum[i, :len(unique_index), j] = np.bincount(
                unique_modes, values_to_sum[i, :, j])
    return value_sum.squeeze()


def get_spectrum_from_text(text_filename, is_collection=True):
    """
    Reads a Spectrum1DCollection or Spectrum1D from a text file,
    for testing to_text_file method. The text file
    should consist of a header, followed by data columns.
    The first column is x_data, subsequent columns are
    each y_data spectrum.

    Header example:
    # Generated by Euphonic 0.6.2
    # x_data in (millielectron_volt)
    # y_data in (millibarn / millielectron_volt)
    # Common metadata: {"weighting": "coherent"}
    # Column 1: x_data
    # Column 2: y_data[0] {"index": 0, "species": "O"}
    # Column 3: y_data[1] {"index": 1, "species": "O"}
    # Column 4: y_data[2] {"index": 2, "species": "Si"}
    """
    metadata = {}
    data = np.loadtxt(text_filename)
    x_data = data[:, 0]
    y_data = data[:, 1:]
    with open(text_filename, 'r') as fp:
        fp.readline()  # Skip 'Generated by Euphonic...' line
        line = fp.readline()  # 'x_data in...' line
        x_data_unit = line[line.index('('):]
        line = fp.readline()  # 'y_data in...' line
        y_data_unit = line[line.index('('):]
        line = fp.readline()  # 'Common metadata...' line
        metadata.update(json.loads(line[line.index('{'):]))
        fp.readline()  # Skip 'Column 1: x_data' line
        for idx in range(len(y_data[0])):
            line = fp.readline()  # 'Column i: y_data[i + 1]...' line
            line_data = json.loads(line[line.index('{'):])
            if line_data:
                if not 'line_data' in metadata:
                    metadata['line_data'] = [
                        {} for i in range(len(y_data[0]))]
                metadata['line_data'][idx] = line_data
    if is_collection:
        cls = Spectrum1DCollection
        y_data = y_data.transpose()
    else:
        cls = Spectrum1D
        y_data = y_data.squeeze()
    return cls(x_data*ureg(x_data_unit),
               y_data*ureg(y_data_unit),
               metadata=metadata)


def check_spectrum_text_header(text_filename):
    """
    Checks parts of the header that aren't required to
    create a spectrum object, but are important for human
    readability
    """
    with open(text_filename, 'r') as fp:
        line = fp.readline()
        assert '# Generated by Euphonic ' in line
        version = line.split()[-1]
        assert version == __version__
        assert '# x_data in (' in fp.readline()
        assert '# y_data in (' in fp.readline()
        assert '# Common metadata: {' in fp.readline()
        assert '# Column 1: x_data' in fp.readline()
        for idx, line in enumerate(fp):
            if idx > 0 and line[0] != '#':  # Ensure at least 1 y_data line
                break
            assert f'# Column {idx + 2}: y_data[{idx}] {{' in line
