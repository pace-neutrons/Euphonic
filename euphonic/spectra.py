import numpy as np
from euphonic import ureg

class Spectrum1D(object):
    """
    For storing generic 1D spectra e.g. density of states

    x_data : (n_x_data,) or (n_x_data + 1,) float Quantity
        The x_data points (if size (n_x_data,)) or x_data bin edges (if
        size (n_x_data + 1,))
    y_data : (n_x_data,) float Quantity
        The plot data in y
    x_tick_labels : list (int, string) tuples or None
        Special tick labels e.g. for high-symmetry points. The int refers to the
        index in x_data the label should be applied to
    """
    def __init__(self, x_data, y_data, x_tick_labels=None):
        if isinstance(x_data, np.ndarray):
            self._x_data = x_data
            self.x_data_unit = 'dimensionless'
            self._internal_x_data_unit = 'dimensionless'
        else:
            self._x_data = x_data.magnitude
            self.x_data_unit = str(x_data.units)
            self._internal_x_data_unit = str(x_data.units)
        
        if isinstance(y_data, np.ndarray):
            self._y_data = y_data
            self.y_data_unit = 'dimensionless'
            self._internal_y_data_unit = 'dimensionless'
        else:
            self._y_data = y_data.magnitude
            self.y_data_unit = str(y_data.units)
            self._internal_y_data_unit = str(y_data.units)
        self.x_tick_labels = x_tick_labels

    @property
    def x_data(self):
        return self._x_data*ureg(self._internal_x_data_unit).to(
            self.x_data_unit)

    @property
    def y_data(self):
        return self._y_data*ureg(self._internal_y_data_unit).to(
            self.y_data_unit)


class Spectrum2D(object):
    """
    For storing generic 2D spectra e.g. S(Q,w)

    x_data : (n_x_data,) or (n_x_data + 1,) float Quantity
        The x_data points (if size (n_x_data,)) or x_data bin edges (if
        size (n_x_data + 1,))
    y_data : (n_y_data + 1,) float Quantity
        The y_data bin edges
    z_data : (n_x_data, n_y_data) float Quantity
        The plot data in z
    x_tick_labels : list (int, string) tuples or None
        Special tick labels e.g. for high-symmetry points. The int refers to the
        index in x_data the label should be applied to
    """
    def __init__(self, x_data, y_data, z_data, x_tick_labels=None):
        if isinstance(x_data, np.ndarray):
            self._x_data = x_data
            self.x_data_unit = 'dimensionless'
            self._internal_x_data_unit = 'dimensionless'
        else:
            self._x_data = x_data.magnitude
            self.x_data_unit = str(x_data.units)
            self._internal_x_data_unit = str(x_data.units)

        if isinstance(y_data, np.ndarray):
            self._y_data = y_data
            self.y_data_unit = 'dimensionless'
            self._internal_y_data_unit = 'dimensionless'
        else:
            self._y_data = y_data.magnitude
            self.y_data_unit = str(y_data.units)
            self._internal_y_data_unit = str(y_data.units)

        if isinstance(z_data, np.ndarray):
            self._z_data = z_data
            self.z_data_unit = 'dimensionless'
            self._internal_z_data_unit = 'dimensionless'
        else:
            self._z_data = z_data.magnitude
            self.z_data_unit = str(z_data.units)
            self._internal_z_data_unit = str(z_data.units)
        self.x_tick_labels = x_tick_labels

    @property
    def x_data(self):
        return self._x_data*ureg(self._internal_x_data_unit).to(
            self.x_data_unit)

    @property
    def y_data(self):
        return self._y_data*ureg(self._internal_y_data_unit).to(
            self.y_data_unit)

    @property
    def z_data(self):
        return self._z_data*ureg(self._internal_z_data_unit).to(
            self.z_data_unit)