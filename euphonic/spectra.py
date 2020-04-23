import numpy as np
from euphonic import ureg

class Spectrum1D(object):
    """
    For storing generic 1D spectra e.g. density of states

    x_bins : (n_x_bins + 1,) float Quantity
        The x-bin edges
    y_data : (n_x_bins,) float Quantity
        The plot data in y
    x_tick_labels : list (int, string) tuples or None
        Special tick labels e.g. for high-symmetry points. The int refers to the
        bin that the label should be applied to, note this is the bin midpoint
    """
    def __init__(self, x_bins, y_data, x_tick_labels=None):
        if isinstance(x_bins, np.ndarray):
            self._x_bins = x_bins
            self.x_bins_unit = 'dimensionless'
            self._internal_x_bins_unit = 'dimensionless'
        else:
            self._x_bins = x_bins.magnitude
            self.x_bins_unit = str(x_bins.units)
            self._internal_x_bins_unit = str(x_bins.units)
        
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
    def x_bins(self):
        return self._x_bins*ureg(self._internal_x_bins_unit).to(self.x_bins_unit)

    @property
    def y_data(self):
        return self._y_data*ureg(self._internal_y_data_unit).to(
            self.y_data_unit)
