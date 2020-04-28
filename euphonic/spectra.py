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
        self._set_data(x_data, 'x')
        self._set_data(y_data, 'y')
        self.x_tick_labels = x_tick_labels

    @property
    def x_data(self):
        return self._x_data*ureg(self._internal_x_data_unit).to(
            self.x_data_unit)

    @property
    def y_data(self):
        return self._y_data*ureg(self._internal_y_data_unit).to(
            self.y_data_unit)

    def _set_data(self, data, attr_name):
        if isinstance(data, np.ndarray):
            setattr(self, f'_{attr_name}_data', data)
            setattr(self, f'{attr_name}_data_unit', 'dimensionless')
            setattr(self, f'_internal_{attr_name}_data_unit', 'dimensionless')
        else:
            setattr(self, f'_{attr_name}_data', data.magnitude)
            setattr(self, f'{attr_name}_data_unit', str(data.units))
            setattr(self, f'_internal_{attr_name}_data_unit', str(data.units))

    def _is_bin_edge(self, bin_ax, data_ax='y'):
        enum = {'x': 0, 'y': 1}
        bins = getattr(self, f'{bin_ax}_data')
        data = getattr(self, f'{data_ax}_data')
        if len(bins) == data.shape[enum[bin_ax]] + 1:
            return True, bins
        return False, bins

    def _get_bin_edges(self, bin_ax='x'):
        is_bin_edge, bins = self._is_bin_edge(bin_ax)
        if is_bin_edge:
            return bins
        else:
            return np.concatenate((
                [bins[0]], (bins[1:] + bins[:-1])/2, [bins[-1]]))

    def _get_bin_centres(self, bin_ax='x'):
        is_bin_edge, bins = self._is_bin_edge(bin_ax)
        if is_bin_edge:
            return bins[:-1] + 0.5*np.diff(bins)
        else:
            return bins


class Spectrum2D(Spectrum1D):
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
        super().__init__(x_data, y_data, x_tick_labels)
        self._set_data(z_data, 'z')

    @property
    def z_data(self):
        return self._z_data*ureg(self._internal_z_data_unit).to(
            self.z_data_unit)

    def _is_bin_edge(self, bin_ax, data_ax='z'):
        return super()._is_bin_edge(bin_ax, data_ax)