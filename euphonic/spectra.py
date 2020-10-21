import math
from copy import deepcopy

from scipy import signal
import numpy as np

from euphonic.validate import _check_constructor_inputs, _check_unit_conversion
from euphonic.io import (_obj_to_json_file, _obj_from_json_file,
                         _obj_to_dict, _process_dict)
from euphonic import ureg, Quantity


class Spectrum1D(object):
    """
    For storing generic 1D spectra e.g. density of states

    Attributes
    ----------
    x_data : (n_x_data,) or (n_x_data + 1,) float Quantity
        The x_data points (if size (n_x_data,)) or x_data bin edges (if
        size (n_x_data + 1,))
    y_data : (n_x_data,) float Quantity
        The plot data in y
    x_tick_labels : list (int, string) tuples or None
        Special tick labels e.g. for high-symmetry points. The int
        refers to the index in x_data the label should be applied to
    """
    def __init__(self, x_data, y_data, x_tick_labels=None):
        """
        Parameters
        ----------
        x_data : (n_x_data,) or (n_x_data + 1,) float Quantity
            The x_data points (if size (n_x_data,)) or x_data bin edges
            (if size (n_x_data + 1,))
        y_data : (n_x_data,) float Quantity
            The plot data in y
        x_tick_labels : list (int, string) tuples or None
            Special tick labels e.g. for high-symmetry points. The int
            refers to the index in x_data the label should be applied to
        """
        _check_constructor_inputs(
            [y_data, x_tick_labels],
            [Quantity, [list, type(None)]],
            [(-1,), ()],
            ['y_data', 'x_tick_labels'])
        ny = len(y_data)
        _check_constructor_inputs(
            [x_data], [Quantity],
            [[(ny,), (ny+1,)]], ['x_data'])
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

    def __setattr__(self, name, value):
        _check_unit_conversion(self, name, value,
                               ['x_data_unit', 'y_data_unit'])
        super(Spectrum1D, self).__setattr__(name, value)

    def broaden(self, x_width, shape='gauss'):
        """
        Broaden y_data and return a new broadened Spectrum1D object

        Parameters
        ----------
        x_width : float Quantity
            The broadening FWHM
        shape : {'gauss', 'lorentz'}, optional
            The broadening shape

        Returns
        -------
        broadened_spectrum : Spectrum1D
            A new Spectrum1D object with broadened y_data
        """
        broadening = _distribution_1d(self._get_bin_centres().magnitude,
                                      x_width.to(self.x_data_unit).magnitude,
                                      shape=shape)
        y_broadened = signal.fftconvolve(
            self.y_data.magnitude, broadening, mode='same')*ureg(
                self.y_data_unit)
        return Spectrum1D(
            np.copy(self.x_data.magnitude)*ureg(self.x_data_unit),
            y_broadened, deepcopy((self.x_tick_labels)))

    def _set_data(self, data, attr_name):
        setattr(self, f'_{attr_name}_data',
                np.array(data.magnitude, dtype=np.float64))
        setattr(self, f'_internal_{attr_name}_data_unit', str(data.units))
        setattr(self, f'{attr_name}_data_unit', str(data.units))

    def _is_bin_edge(self, bin_ax, data_ax='y'):
        enum = {'x': 0, 'y': 1}
        bins = getattr(self, f'{bin_ax}_data')
        data = getattr(self, f'{data_ax}_data')
        if len(bins) == data.shape[enum[bin_ax]] + 1:
            is_bin_edge = True
        else:
            is_bin_edge = False
        return is_bin_edge, bins.magnitude, bins.units

    def _get_bin_edges(self, bin_ax='x'):
        is_bin_edge, bins, bins_units = self._is_bin_edge(bin_ax)
        if is_bin_edge:
            return bins*bins_units
        else:
            bin_edges = np.concatenate((
                [bins[0]], (bins[1:] + bins[:-1])/2, [bins[-1]]))
            return bin_edges*bins_units

    def _get_bin_centres(self, bin_ax='x'):
        is_bin_edge, bins, bins_units = self._is_bin_edge(bin_ax)
        if is_bin_edge:
             bin_centres = bins[:-1] + 0.5*np.diff(bins)
             return bin_centres*bins_units
        else:
            return bins*bins_units

    def to_dict(self):
        """
        Convert to a dictionary. See Spectrum1D.from_dict for details on
        keys/values

        Returns
        -------
        dict
        """
        d = _obj_to_dict(self, ['x_data', 'y_data', 'x_tick_labels'])
        return d

    def to_json_file(self, filename):
        """
        Write to a JSON file. JSON fields are equivalent to
        from_dict keys

        Parameters
        ----------
        filename : str
            Name of the JSON file to write to
        """
        _obj_to_json_file(self, filename)

    @classmethod
    def from_dict(cls, d):
        """
        Convert a dictionary to a Spectrum1D object

        Parameters
        ----------
        d : dict
            A dictionary with the following keys/values:

            - 'x_data': (n_x_data,) or (n_x_data + 1,) float ndarray
            - 'x_data_unit': str
            - 'y_data': (n_x_data,) float ndarray
            - 'y_data_unit': str

            There are also the following optional keys:

            - 'x_tick_labels': list of (int, string) tuples

        Returns
        -------
        Spectrum1D
        """
        d = _process_dict(d, quantities=['x_data', 'y_data'],
                          optional=['x_tick_labels'])
        return Spectrum1D(d['x_data'], d['y_data'], d['x_tick_labels'])

    @classmethod
    def from_json_file(cls, filename):
        """
        Read from a JSON file. See from_dict for required fields

        Parameters
        ----------
        filename : str
            The file to read from
        """
        type_dict = {'x_tick_labels': tuple}
        return _obj_from_json_file(cls, filename, type_dict)


class Spectrum2D(Spectrum1D):
    """
    For storing generic 2D spectra e.g. S(Q,w)

    Attributes
    ----------
    x_data : (n_x_data,) or (n_x_data + 1,) float Quantity
        The x_data points (if size (n_x_data,)) or x_data bin edges (if
        size (n_x_data + 1,))
    y_data : (n_y_data,) or (n_y_data + 1,) float Quantity
        The y_data bin points (if size (n_y_data,)) or y_data bin edges
        (if size (n_y_data + 1,))
    z_data : (n_x_data, n_y_data) float Quantity
        The plot data in z
    x_tick_labels : list (int, string) tuples or None
        Special tick labels e.g. for high-symmetry points. The int
        refers to the index in x_data the label should be applied to
    """
    def __init__(self, x_data, y_data, z_data, x_tick_labels=None):
        """
        Attributes
        ----------
        x_data : (n_x_data,) or (n_x_data + 1,) float Quantity
            The x_data points (if size (n_x_data,)) or x_data bin edges
            (if size (n_x_data + 1,))
        y_data : (n_y_data,) or (n_y_data + 1,) float Quantity
            The y_data bin points (if size (n_y_data,)) or y_data bin
            edges (if size (n_y_data + 1,))
        z_data : (n_x_data, n_y_data) float Quantity
            The plot data in z
        x_tick_labels : list (int, string) tuples or None
            Special tick labels e.g. for high-symmetry points. The int
            refers to the index in x_data the label should be applied to
        """
        _check_constructor_inputs(
            [z_data, x_tick_labels],
            [Quantity, [list, type(None)]],
            [(-1, -1), ()],
            ['z_data', 'x_tick_labels'])
        nx = z_data.shape[0]
        ny = z_data.shape[1]
        _check_constructor_inputs(
            [x_data, y_data],
            [Quantity, Quantity],
            [[(nx,), (nx + 1,)], [(ny,), (ny + 1,)]],
            ['x_data', 'y_data'])
        self._set_data(x_data, 'x')
        self._set_data(y_data, 'y')
        self.x_tick_labels = x_tick_labels
        self._set_data(z_data, 'z')

    @property
    def z_data(self):
        return self._z_data*ureg(self._internal_z_data_unit).to(
            self.z_data_unit)

    def __setattr__(self, name, value):
        _check_unit_conversion(self, name, value,
                               ['z_data_unit'])
        super(Spectrum2D, self).__setattr__(name, value)

    def broaden(self, x_width=None, y_width=None, shape='gauss'):
        """
        Broaden z_data and return a new broadened Spectrum2D object

        Parameters
        ----------
        x_width : float Quantity, optional
            The broadening FWHM in x
        y_width : float Quantity, optional
            The broadening FWHM in y
        shape : {'gauss', 'lorentz'}, optional
            The broadening shape

        Returns
        -------
        broadened_spectrum : Spectrum2D
            A new Spectrum2D object with broadened z_data
        """
        # If no width has been set, make widths small enough to have
        # effectively no broadening
        if x_width is None:
            x_width = 0.1*(self.x_data[1] - self.x_data[0])
        if y_width is None:
            y_width = 0.1*(self.y_data[1] - self.y_data[0])
        broadening = _distribution_2d(self._get_bin_centres('x').magnitude,
                                      self._get_bin_centres('y').magnitude,
                                      x_width.to(self.x_data_unit).magnitude,
                                      y_width.to(self.y_data_unit).magnitude,
                                      shape=shape)
        z_broadened = signal.fftconvolve(
            self.z_data.magnitude, np.transpose(broadening), mode='same')*ureg(
                self.z_data_unit)
        return Spectrum2D(
            np.copy(self.x_data.magnitude)*ureg(self.x_data_unit),
            np.copy(self.y_data.magnitude)*ureg(self.y_data_unit),
            z_broadened, deepcopy((self.x_tick_labels)))

    def _is_bin_edge(self, bin_ax, data_ax='z'):
        return super()._is_bin_edge(bin_ax, data_ax)

    def to_dict(self):
        """
        Convert to a dictionary. See Spectrum2D.from_dict for details on
        keys/values

        Returns
        -------
        dict
        """
        d1 = super().to_dict()
        d2 = _obj_to_dict(self, ['z_data'])
        d2.update(d1)
        return d2

    @classmethod
    def from_dict(cls, d):
        """
        Convert a dictionary to a Spectrum2D object

        Parameters
        ----------
        d : dict
            A dictionary with the following keys/values:

            - 'x_data': (n_x_data,) or (n_x_data + 1,) float ndarray
            - 'x_data_unit': str
            - 'y_data': (n_y_data,) or (n_y_data + 1,) float ndarray
            - 'y_data_unit': str
            - 'z_data': (n_x_data, n_y_data) float Quantity
            - 'z_data_unit': str

            There are also the following optional keys:

            - 'x_tick_labels': list of (int, string) tuples

        Returns
        -------
        Spectrum2D
        """
        d = _process_dict(d, quantities=['x_data', 'y_data', 'z_data'],
                          optional=['x_tick_labels'])
        return Spectrum2D(d['x_data'], d['y_data'], d['z_data'],
                          d['x_tick_labels'])


def _gaussian(x, sigma):
    return np.exp(-np.square(x)/(2*sigma**2))/(math.sqrt(2*math.pi)*sigma)


def _lorentzian(x, gamma):
    return gamma/(2*math.pi*(np.square(x) + (gamma/2)**2))


def _get_dist_bins(bins, fwhm, extent):
    # Ensure nbins is always odd, and each bin has the same approx width
    # as original x/ybins
    bin_width = np.mean(np.diff(bins))
    nbins = int(np.ceil(2*extent*fwhm/bin_width)/2)*2 + 1
    width = extent*fwhm
    # Prevent xbins from being too large. If user accidentally selects a
    # very large broadening, xwidth and therefore xbins could be
    # extremely large. But for most cases the original nxbins should be
    # smaller
    if nbins > len(bins):
        nbins = int(len(bins)/2)*2 + 1
        width = (bins[-1] - bins[0])/2
    return np.linspace(-width, width, nbins)


def _distribution_1d(xbins, xwidth, shape='gauss', extent=3.0):
    x = _get_dist_bins(xbins, xwidth, extent)
    if shape == 'gauss':
        # Gauss FWHM = 2*sigma*sqrt(2*ln2)
        xsigma = xwidth/(2*math.sqrt(2*math.log(2)))
        dist = _gaussian(x, xsigma)
    elif shape == 'lorentz':
        dist = _lorentzian(x, xwidth)
    else:
        raise ValueError(
            f'Distribution shape \'{shape}\' not recognised')
    dist = dist/np.sum(dist)  # Naively normalise
    return dist


def _distribution_2d(xbins, ybins, xwidth, ywidth, shape='gauss', extent=3.0):
    x = _get_dist_bins(xbins, xwidth, extent)
    y = _get_dist_bins(ybins, ywidth, extent)

    if shape == 'gauss':
        # Gauss FWHM = 2*sigma*sqrt(2*ln2)
        xsigma = xwidth/(2*math.sqrt(2*math.log(2)))
        ysigma = ywidth/(2*math.sqrt(2*math.log(2)))
        xdist = _gaussian(x, xsigma)
        ydist = _gaussian(y, ysigma)
    elif shape == 'lorentz':
        xdist = _lorentzian(x, xwidth)
        ydist = _lorentzian(y, ywidth)
    else:
        raise ValueError(
            f'Distribution shape \'{shape}\' not recognised')
    xgrid = np.tile(xdist, (len(ydist), 1))
    ygrid = np.transpose(np.tile(ydist, (len(xdist), 1)))
    dist = xgrid*ygrid
    dist = dist/np.sum(dist)  # Naively normalise

    return dist
