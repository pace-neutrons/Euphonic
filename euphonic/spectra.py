from abc import ABC, abstractmethod
import collections
from copy import deepcopy
import math
from typing import (Any, Dict, List, Optional, overload,
                    Sequence, Tuple, TypeVar, Union)

import numpy as np
from scipy.ndimage import gaussian_filter1d, correlate1d, gaussian_filter

from euphonic.validate import _check_constructor_inputs, _check_unit_conversion
from euphonic.io import (_obj_to_json_file, _obj_from_json_file,
                         _obj_to_dict, _process_dict)
from euphonic import ureg, Quantity


S = TypeVar('S', bound='Spectrum')
SC = TypeVar('SC', bound='Spectrum1DCollection')


class Spectrum(ABC):
    def __setattr__(self, name, value):
        _check_unit_conversion(self, name, value,
                               ['x_data_unit', 'y_data_unit'])
        super().__setattr__(name, value)

    def _set_data(self, data, attr_name):
        setattr(self, f'_{attr_name}_data',
                np.array(data.magnitude, dtype=np.float64))
        setattr(self, f'_internal_{attr_name}_data_unit', str(data.units))
        setattr(self, f'{attr_name}_data_unit', str(data.units))

    def _gfwhm_to_sigma(self, fwhm, ax_bin_centres):
        """
        Convert a Gaussian FWHM to sigma in units of the mean ax bin size

        Parameters
        ----------
        fwhm : float Quantity
            The Gaussian broadening FWHM
        ax_bin_centres : Quantity ndarray
            The bin centres along the axis the broadening is applied to

        Returns
        -------
        sigma : float
            Sigma in units of mean ax bin size
        """
        ax_units = ax_bin_centres.units
        sigma = fwhm/(2*math.sqrt(2*math.log(2)))
        mean_bin_size = np.mean(np.diff(ax_bin_centres.magnitude))
        sigma_bin = sigma.to(ax_units).magnitude/mean_bin_size
        return sigma_bin

    @abstractmethod
    def to_dict(self) -> Dict[str, Union[float, np.ndarray]]:
        """Write to dict using euphonic.io._obj_to_dict"""
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls: S, d: Dict[str, Union[float, np.ndarray]]) -> S:
        """Initialise a Spectrum object from dictionary"""
        ...

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

    @abstractmethod
    def _split_by_indices(self: S,
                        indices: Union[Sequence[int], np.ndarray]) -> List[S]:
        """Split data along x axis at given indices"""
        ...

    def _split_by_tol(self: S, btol: float = 10.0) -> List[S]:
        """Split data along x-axis at detected breakpoints"""
        diff = np.diff(self.x_data.magnitude)
        median = np.median(diff)
        breakpoints = np.where((diff / median) > btol)[0] + 1
        return self._split_by_indices(breakpoints)

    @staticmethod
    def _ranges_from_indices(indices: Union[Sequence[int], np.ndarray]
                             ) -> List[Tuple[int, int]]:
        """Convert a series of breakpoints to a series of slice ranges"""
        if len(indices) == 0:
            ranges = [(0, None)]
        else:
            ranges = [(0, indices[0])]
            if len(indices) > 1:
                ranges = ranges + [(indices[i], indices[i+1])
                                   for i in range(len(indices) - 1)]
            ranges = ranges + [(indices[-1], None)]
        return ranges

    @staticmethod
    def _cut_x_ticks(x_tick_labels: Union[List[Tuple[int, str]], None],
                     x0: int,
                     x1: Union[int, None]) -> List[Tuple[int, str]]:
        """Crop and shift x labels to new x range"""
        if x_tick_labels is None:
            return None
        else:
            return [(int(x - x0), label) for (x, label) in x_tick_labels
                    if (x >= x0) and ((x1 is None) or (x < x1))]

    def split(self: S, indices: Union[Sequence[int], np.ndarray] = None,
              btol: float = None) -> List[S]:
        """Split to multiple spectra

        Data may be split by index. Alternatively, x-axis data may be
        split automatically by searching for unusually large gaps in
        energy values.  These would usually correspond to disconnected
        special-points in a band-structure diagram.

        Parameters
        ----------
        indices:
            positions in data of breakpoints
        btol:
            parameter used to identify breakpoints. This is a ratio
            between the gap in values and the median gap between
            neighbouring x-values. If neither indices nor btol is
            specified, this is set to 10.0.

        Returns
        -------
            Separated spectrum regions. If passed to the appropriate
            functions in euphonic.plot this would be interpreted as a
            series of subplots.

        """

        if indices is None:
            if btol is None:
                btol = 10.0
            return self._split_by_tol(btol=btol)
        else:
            if btol is not None:
                raise ValueError("Cannot set both indices and btol")
            return self._split_by_indices(indices)

    @staticmethod
    def _bin_edges_to_centres(bin_edges: np.ndarray) -> np.ndarray:
        return bin_edges[:-1] + 0.5*np.diff(bin_edges)

    @staticmethod
    def _bin_centres_to_edges(bin_centres: np.ndarray) -> np.ndarray:
        return np.concatenate((
            [bin_centres[0]],
            (bin_centres[1:] + bin_centres[:-1])/2,
            [bin_centres[-1]]))

    @staticmethod
    def _is_bin_edge(data_length, bin_length) -> bool:
        """Determine if axis data are bin edges or centres"""
        if bin_length == data_length + 1:
            return True
        elif bin_length == data_length:
            return False
        else:
            raise ValueError((
                f'Unexpected data axis length {data_length} '
                f'for bin axis length {bin_length}'))

    def get_bin_edges(self) -> Quantity:
        """
        Get x-axis bin edges. If the size of x_data is one element larger
        than y_data, x_data is assumed to contain bin edges, but if x_data
        is the same size, x_data is assumed to contain bin centres and
        a conversion is made. In the conversion, the bin edges will
        not go outside the existing data bounds so the first and last
        bins may be half-size. In addition, each bin edge is assumed
        to be halfway between each bin centre, which may not be an
        accurate assumption in the case of differently sized bins.
        """
        # Need to use -1 index for y_data so it also works for
        # Spectrum1DCollection which has y_data shape (n_spectra, bins)
        if self._is_bin_edge(self.y_data.shape[-1], self.x_data.shape[0]):
            return self.x_data
        else:
            return self._bin_centres_to_edges(
                self.x_data.magnitude)*self.x_data.units


    def get_bin_centres(self) -> Quantity:
        """
        Get x-axis bin centres. If the size of x_data is the same size
        as y_data, x_data is assumed to contain bin centres, but if x_data
        is one element larger, x_data is assumed to contain bin edges and
        a conversion is made. In this conversion, the bin centres are
        assumed to be in the middle of each bin, which may not be an
        accurate assumption in the case of differently sized bins.
        """
        # Need to use -1 index for y_data so it also works for
        # Spectrum1DCollection which has y_data shape (n_spectra, bins)
        if self._is_bin_edge(self.y_data.shape[-1], self.x_data.shape[0]):
            return self._bin_edges_to_centres(
                self.x_data.magnitude)*self.x_data.units
        else:
            return self.x_data


class Spectrum1D(Spectrum):
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

    def _split_by_indices(self: S,
                        indices: Union[Sequence[int], np.ndarray]) -> List[S]:
        """Split data along x-axis at given indices"""
        ranges = self._ranges_from_indices(indices)

        return [type(self)(self.x_data[x0:x1], self.y_data[x0:x1],
                           x_tick_labels=self._cut_x_ticks(self.x_tick_labels,
                                                           x0, x1))
                           for x0, x1 in ranges]

    def to_dict(self):
        """
        Convert to a dictionary. See Spectrum1D.from_dict for details on
        keys/values

        Returns
        -------
        dict
        """
        return _obj_to_dict(self, ['x_data', 'y_data', 'x_tick_labels'])

    @classmethod
    def from_dict(cls: S, d) -> S:
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
        return cls(d['x_data'], d['y_data'], x_tick_labels=d['x_tick_labels'])

    def broaden(self, x_width, shape='gauss'):
        """
        Broaden y_data and return a new broadened spectrum object

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

        Raises
        ------
        ValueError
            If shape is not one of the allowed strings
        """
        if shape == 'gauss':
            xsigma = self._gfwhm_to_sigma(x_width, self.get_bin_centres())
            y_broadened = gaussian_filter1d(
                    self.y_data.magnitude, xsigma,
                    mode='constant')*ureg(self.y_data_unit)
        elif shape == 'lorentz':
            broadening = _distribution_1d(
                    self.get_bin_centres().magnitude,
                    x_width.to(self.x_data_unit).magnitude,
                    shape=shape)
            y_broadened = correlate1d(
                    self.y_data.magnitude, broadening,
                    mode='constant')*ureg(self.y_data_unit)
        else:
            raise ValueError(
                f"Distribution shape '{shape}' not recognised")

        return Spectrum1D(
            np.copy(self.x_data.magnitude)*ureg(self.x_data_unit),
            y_broadened, deepcopy((self.x_tick_labels)))


class Spectrum1DCollection(collections.abc.Sequence, Spectrum):
    """A collection of Spectrum1D with common x_data and x_tick_labels

    Intended for convenient storage of band structures, projected DOS
    etc.  This object can be indexed or iterated to obtain individual
    Spectrum1D.

    x_data : (n_x_data,) or (n_x_data + 1,) float Quantity
        The x_data points (if size (n_x_data,)) or x_data bin edges (if
        size (n_x_data + 1,))
    y_data : (n_entries, n_x_data) float Quantity
        The plot data in y, in rows corresponding to separate 1D spectra
    x_tick_labels : list (int, string) tuples or None
        Special tick labels e.g. for high-symmetry points. The int
        refers to the index in x_data the label should be applied to

    """
    def __init__(self, x_data: Quantity, y_data: Quantity,
                 x_tick_labels: Optional[Sequence[Tuple[int, str]]] = None,
                 metadata: Optional[Dict[Any, Any]] = None) -> None:

        _check_constructor_inputs(
            [y_data, x_tick_labels, metadata],
            [Quantity, [list, type(None)], [dict, type(None)]],
            [(-1,-1), (), ()],
            ['y_data', 'x_tick_labels', 'metadata'])
        ny = len(y_data[0])
        _check_constructor_inputs(
            [x_data], [Quantity],
            [[(ny,), (ny+1,)]], ['x_data'])

        self._set_data(x_data, 'x')
        self._set_data(y_data, 'y')
        self.x_tick_labels = x_tick_labels
        self.metadata = metadata

    @property
    def x_data(self):
        return self._x_data*ureg(self._internal_x_data_unit).to(
            self.x_data_unit)

    @property
    def y_data(self):
        return self._y_data*ureg(self._internal_y_data_unit).to(
            self.y_data_unit)

    def _split_by_indices(self: S,
                        indices: Union[Sequence[int], np.ndarray]) -> List[S]:
        """Split data along x-axis at given indices"""

        ranges = self._ranges_from_indices(indices)

        return [type(self)(self.x_data[x0:x1], self.y_data[:, x0:x1],
                           x_tick_labels=self._cut_x_ticks(self.x_tick_labels,
                                                           x0, x1))
                           for x0, x1 in ranges]

    def __len__(self):
        return self.y_data.magnitude.shape[0]

    @overload
    def __getitem__(self, item: int) -> Spectrum1D:
        ...

    @overload  # noqa: F811
    def __getitem__(self, item: slice) -> 'Spectrum1DCollection':
        ...

    def __getitem__(self, item):  # noqa: F811
        if isinstance(item, int):
            return Spectrum1D(self.x_data,
                              self.y_data[item, :],
                              x_tick_labels=self.x_tick_labels)
        elif isinstance(item, slice):
            if (item.stop is not None) and (item.stop >= len(self)):
                raise IndexError(f'index "{item.stop}" out of range')

            return type(self)(self.x_data,
                              self.y_data[item, :],
                              x_tick_labels=self.x_tick_labels)
        else:
            raise TypeError(f'Index "{item}" should be an integer or a slice')

    @classmethod
    def from_spectra(cls: SC, spectra: Sequence[Spectrum1D]) -> SC:
        if len(spectra) < 1:
            raise IndexError("At least one spectrum is needed for collection")

        def _type_check(spectrum):
            if not isinstance(spectrum, Spectrum1D):
                raise TypeError(
                    "from_spectra() requires a sequence of Spectrum1D")

        _type_check(spectra[0])
        x_data = spectra[0].x_data
        x_tick_labels = spectra[0].x_tick_labels
        y_data_length = len(spectra[0].y_data.magnitude)
        y_data_magnitude = np.empty((len(spectra), y_data_length))
        y_data_magnitude[0, :] = spectra[0].y_data.magnitude
        y_data_units = spectra[0].y_data.units

        for i, spectrum in enumerate(spectra[1:]):
            _type_check(spectrum)
            assert spectrum.y_data.units == y_data_units
            assert np.allclose(spectrum.x_data.magnitude, x_data.magnitude)
            assert spectrum.x_data.units == x_data.units
            assert spectrum.x_tick_labels == x_tick_labels
            y_data_magnitude[i + 1, :] = spectrum.y_data.magnitude

        y_data = Quantity(y_data_magnitude, y_data_units)

        return cls(x_data, y_data, x_tick_labels=x_tick_labels)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a dictionary consistent with from_dict()

        Returns
        -------
        dict
        """
        return _obj_to_dict(self, ['x_data', 'y_data', 'x_tick_labels',
                                   'metadata'])

    @classmethod
    def from_dict(cls: S, d) -> S:
        """
        Convert a dictionary to a Spectrum1DCollection object

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
        Spectrum1DCollection
        """
        d = _process_dict(d, quantities=['x_data', 'y_data'],
                          optional=['x_tick_labels', 'metadata'])
        return cls(d['x_data'], d['y_data'], x_tick_labels=d['x_tick_labels'],
                   metadata=d['metadata'])


class Spectrum2D(Spectrum):
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

    def __setattr__(self, name, value):
        _check_unit_conversion(self, name, value,
                               ['z_data_unit'])
        super(Spectrum2D, self).__setattr__(name, value)

    def _split_by_indices(self: S,
                        indices: Union[Sequence[int], np.ndarray]) -> List[S]:
        """Split data along x-axis at given indices"""
        ranges = self._ranges_from_indices(indices)
        return [type(self)(self.x_data[x0:x1], self.y_data,
                           self.z_data[x0:x1, :],
                           x_tick_labels=self._cut_x_ticks(self.x_tick_labels,
                                                           x0, x1))
                           for x0, x1 in ranges]

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

        Raises
        ------
        ValueError
            If shape is not one of the allowed strings
        """
        if shape == 'gauss':
            if x_width is None:
                xsigma = 0.0
            else:
                xsigma = self._gfwhm_to_sigma(
                    x_width, self.get_bin_centres('x'))
            if y_width is None:
                ysigma = 0.0
            else:
                ysigma = self._gfwhm_to_sigma(
                    y_width, self.get_bin_centres('y'))
            z_broadened = gaussian_filter(
                self.z_data.magnitude, [xsigma, ysigma],
                mode='constant')
        elif shape == 'lorentz':
            z_broadened = self.z_data.magnitude
            if x_width is not None:
                x_broadening = _distribution_1d(
                    self.get_bin_centres('x').magnitude,
                    x_width.to(self.x_data_unit).magnitude,
                    shape=shape)
                z_broadened = correlate1d(
                    z_broadened, x_broadening, mode='constant', axis=0)
            if y_width is not None:
                y_broadening = _distribution_1d(
                    self.get_bin_centres('y').magnitude,
                    y_width.to(self.y_data_unit).magnitude,
                    shape=shape)
                z_broadened = correlate1d(
                    z_broadened, y_broadening, mode='constant', axis=1)
        else:
            raise ValueError(
                f"Distribution shape '{shape}' not recognised")
        return Spectrum2D(
            np.copy(self.x_data.magnitude)*ureg(self.x_data_unit),
            np.copy(self.y_data.magnitude)*ureg(self.y_data_unit),
            z_broadened*ureg(self.z_data_unit), deepcopy((self.x_tick_labels)))

    def get_bin_edges(self, bin_ax: str = 'x') -> Quantity:
        """
        Get bin edges for the axis specified by bin_ax. If the size of
        bin_ax is one element larger than z_data along that axis, bin_ax
        is assumed to contain bin edges, but if bin_ax is the same size,
        bin_ax is assumed to contain bin centres and a conversion is made.
        In the conversion, the bin edges will not go outside the existing
        data bounds so the first and last bins may be half-size. In addition,
        each bin edge is assumed to be halfway between each bin centre,
        which may not be an accurate assumption in the case of differently
        sized bins.

        Parameters
        ----------
        bin_ax
            The axis to get the bin edges for, 'x' or 'y'
        """
        enum = {'x': 0, 'y': 1}
        bin_data = getattr(self, f'{bin_ax}_data')
        data_ax_len = self.z_data.shape[enum[bin_ax]]
        if self._is_bin_edge(data_ax_len, bin_data.shape[0]):
            return bin_data
        else:
            return self._bin_centres_to_edges(bin_data.magnitude)*bin_data.units

    def get_bin_centres(self, bin_ax: str = 'x') -> Quantity:
        """
        Get bin centres for the axis specified by bin_ax. If the size of
        bin_ax is the same size as z_data along that axis, bin_ax is
        assumed to contain bin centres, but if bin_ax is one element
        larger, bin_ax is assumed to contain bin centres and a conversion
        is made. In this conversion, the bin centres are assumed to be in
        the middle of each bin, which may not be an accurate assumption in
        the case of differently sized bins.

        Parameters
        ----------
        bin_ax
            The axis to get the bin centres for, 'x' or 'y'
        """
        enum = {'x': 0, 'y': 1}
        bin_data = getattr(self, f'{bin_ax}_data')
        data_ax_len = self.z_data.shape[enum[bin_ax]]
        if self._is_bin_edge(data_ax_len, bin_data.shape[0]):
            return self._bin_edges_to_centres(
                bin_data.magnitude)*bin_data.units
        else:
            return bin_data

    def to_dict(self):
        """
        Convert to a dictionary. See Spectrum2D.from_dict for details on
        keys/values

        Returns
        -------
        dict
        """
        return _obj_to_dict(self, ['x_data', 'y_data', 'z_data',
                                   'x_tick_labels'])

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
        return cls(d['x_data'], d['y_data'], d['z_data'],
                   x_tick_labels=d['x_tick_labels'])


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


def _distribution_1d(xbins, xwidth, shape='lorentz', extent=3.0):
    x = _get_dist_bins(xbins, xwidth, extent)
    if shape == 'lorentz':
        dist = _lorentzian(x, xwidth)
    dist = dist/np.sum(dist)  # Naively normalise
    return dist
