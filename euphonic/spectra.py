from abc import ABC, abstractmethod
import collections
import copy
import itertools
import math
import json
from numbers import Integral, Real
from typing import (Any, Callable, Dict, List, Literal, Optional, overload,
                    Sequence, Tuple, TypeVar, Union, Type)
import warnings

from pint import DimensionalityError, Quantity
import numpy as np
from scipy.ndimage import correlate1d, gaussian_filter

from euphonic import ureg, __version__
from euphonic.broadening import (ErrorFit, KernelShape,
                                 variable_width_broadening)
from euphonic.io import (_obj_to_json_file, _obj_from_json_file,
                         _obj_to_dict, _process_dict)
from euphonic.readers.castep import read_phonon_dos_data
from euphonic.util import _get_unique_elems_and_idx
from euphonic.validate import _check_constructor_inputs, _check_unit_conversion


CallableQuantity = Callable[[Quantity], Quantity]


class Spectrum(ABC):
    T = TypeVar('T', bound='Spectrum')

    def __setattr__(self, name: str, value: Any) -> None:
        _check_unit_conversion(self, name, value,
                               ['x_data_unit', 'y_data_unit'])
        super().__setattr__(name, value)

    def _set_data(self, data: Quantity, attr_name: str) -> None:
        setattr(self, f'_{attr_name}_data',
                np.array(data.magnitude, dtype=np.float64))
        setattr(self, f'_internal_{attr_name}_data_unit', str(data.units))
        setattr(self, f'{attr_name}_data_unit', str(data.units))

    @property
    def x_data(self) -> Quantity:
        return self._x_data*ureg(self._internal_x_data_unit).to(
            self.x_data_unit)

    @x_data.setter
    def x_data(self, value: Quantity) -> None:
        self.x_data_unit = str(value.units)
        self._x_data = value.to(self._internal_x_data_unit).magnitude

    @property
    def y_data(self) -> Quantity:
        return self._y_data*ureg(self._internal_y_data_unit).to(
            self.y_data_unit)

    @y_data.setter
    def y_data(self, value: Quantity) -> None:
        self.y_data_unit = str(value.units)
        self._y_data = value.to(self._internal_y_data_unit).magnitude

    def __imul__(self: T, other: Real) -> T:
        """Scale spectral data in-place"""
        self._y_data *= other
        return self

    def __mul__(self: T, other: Real) -> T:
        """Get a new spectrum with scaled data"""
        new_spec = self.copy()
        new_spec *= other
        return new_spec

    @abstractmethod
    def copy(self: T) -> T:
        """Get an independent copy of spectrum"""
        ...

    @property
    def x_tick_labels(self) -> List[Tuple[int, str]]:
        return self._x_tick_labels

    @x_tick_labels.setter
    def x_tick_labels(self, value: Sequence[Tuple[int, str]]) -> None:
        err_msg = ('x_tick_labels should be of type '
                   'Sequence[Tuple[int, str]] e.g. '
                   '[(0, "label1"), (5, "label2")]')
        if value is not None:
            if isinstance(value, Sequence):
                for elem in value:
                    if not (isinstance(elem, tuple)
                            and len(elem) == 2
                            and isinstance(elem[0], Integral)
                            and isinstance(elem[1], str)):
                        raise TypeError(err_msg)
                # Ensure indices in x_tick_labels are plain ints as
                # np.int64/32 etc. are not JSON serializable
                value = [(int(idx), label) for idx, label in value]
            else:
                raise TypeError(err_msg)
        self._x_tick_labels = value

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Write to dict using euphonic.io._obj_to_dict"""
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls: Type[T], d: Dict[str, Any]) -> T:
        """Initialise a Spectrum object from dictionary"""
        ...

    def to_json_file(self, filename: str) -> None:
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
    def from_json_file(cls: Type[T], filename: str) -> T:
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
    def _split_by_indices(self: T, indices: Union[Sequence[int], np.ndarray]
                          ) -> List[T]:
        """Split data along x axis at given indices"""
        ...

    def _split_by_tol(self: T, btol: float = 10.0) -> List[T]:
        """Split data along x-axis at detected breakpoints"""
        diff = np.diff(self.x_data)
        median = np.median(diff)
        breakpoints = np.where((diff / median) > btol)[0] + 1
        return self._split_by_indices(breakpoints)

    @staticmethod
    def _ranges_from_indices(indices: Union[Sequence[int], np.ndarray]
                             ) -> List[Tuple[int, Optional[int]]]:
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
    def _cut_x_ticks(x_tick_labels: Union[Sequence[Tuple[int, str]], None],
                     x0: int,
                     x1: Union[int, None]) -> Union[List[Tuple[int, str]],
                                                    None]:
        """Crop and shift x labels to new x range"""
        if x_tick_labels is None:
            return None
        else:
            return [(int(x - x0), label) for (x, label) in x_tick_labels
                    if (x >= x0) and ((x1 is None) or (x < x1))]

    def split(self: T, indices: Union[Sequence[int], np.ndarray] = None,
              btol: float = None) -> List[T]:
        """Split to multiple spectra

        Data may be split by index. Alternatively, x-axis data may be
        split automatically by searching for unusually large gaps in
        energy values.  These would usually correspond to disconnected
        special-points in a band-structure diagram.

        Parameters
        ----------
        indices
            positions in data of breakpoints
        btol
            parameter used to identify breakpoints. This is a ratio
            between the gap in values and the median gap between
            neighbouring x-values. If neither indices nor btol is
            specified, this is set to 10.0.

        Returns
        -------
        split_spectra
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
    def _broaden_data(data: np.ndarray, bin_centres: Sequence[np.ndarray],
                      widths: Sequence[float],
                      shape: KernelShape = 'gauss',
                      method: Optional[Literal['convolve']] = None,
                      ) -> np.ndarray:
        """
        Broaden data along each axis by widths, returning the broadened
        data. Data can be 1D or 2D, and  the length of bin_centres and
        widths should match the number of dimensions in data.
        bin_centres and widths are assumed to be in the same units
        """
        shape_opts = ('gauss', 'lorentz')
        if shape not in shape_opts:
            raise ValueError(f'Invalid value for shape, got {shape}, '
                             f'should be one of {shape_opts}')
        method_opts = ('convolve', None)
        if method not in method_opts:
            raise ValueError(f'Invalid value for method, got {method}, '
                             f'should be one of {method_opts}')

        # We only want to check for unequal bins if using a method that
        # is not correct for unequal bins (currently this is the only
        # option but leave the 'if' anyway for future implementations)
        if method in ('convolve', None):
            axes = ['x_data', 'y_data']
            unequal_bin_axes = []
            for ax, (width, bin_data) in enumerate(zip(widths, bin_centres)):
                if width is not None:
                    bin_widths = np.diff(bin_data)
                    if not np.all(np.isclose(bin_widths, bin_widths[0])):
                        unequal_bin_axes += [axes[ax]]
            if len(unequal_bin_axes) > 0:
                msg = (f'{" and ".join(unequal_bin_axes)} bin widths are '
                       f'not equal, so broadening by convolution may give '
                       f'incorrect results.')
                if method is None:
                    raise ValueError(
                        msg + ' If you still want to broaden by convolution '
                        'please explicitly use the method="convolve" option.')
                else:
                    warnings.warn(msg, stacklevel=3)

        if shape == 'gauss':
            sigmas = []
            for width, bin_data in zip(widths, bin_centres):
                if width is None:
                    sigmas.append(0.0)
                else:
                    sigmas.append(Spectrum._gfwhm_to_sigma(width, bin_data))
            data_broadened = gaussian_filter(data, sigmas, mode='constant')
        elif shape == 'lorentz':
            data_broadened = data
            for ax, (width, bin_data) in enumerate(zip(widths, bin_centres)):
                if width is not None:
                    broadening = _distribution_1d(bin_data, width)
                    data_broadened = correlate1d(data_broadened, broadening,
                                                 mode='constant',
                                                 axis=ax)

        return data_broadened

    @staticmethod
    def _gfwhm_to_sigma(fwhm: float, ax_bin_centres: np.ndarray) -> float:
        """
        Convert a Gaussian FWHM to sigma in units of the mean ax bin size

        Parameters
        ----------
        fwhm
            The Gaussian broadening FWHM
        ax_bin_centres
            Shape (n_bins,) float np.ndarray.
            The bin centres along the axis the broadening is applied to

        Returns
        -------
        sigma
            Sigma in units of mean ax bin size
        """
        sigma = fwhm/(2*math.sqrt(2*math.log(2)))
        mean_bin_size = np.mean(np.diff(ax_bin_centres))
        sigma_bin = sigma/mean_bin_size
        return sigma_bin

    @staticmethod
    def _bin_edges_to_centres(bin_edges: Quantity) -> Quantity:
        return bin_edges[:-1] + 0.5*np.diff(bin_edges)

    @staticmethod
    def _bin_centres_to_edges(bin_centres: Quantity) -> Quantity:
        return np.concatenate((
            [bin_centres[0]],
            (bin_centres[1:] + bin_centres[:-1])/2,
            [bin_centres[-1]]))

    @staticmethod
    def _is_bin_edge(data_length: int, bin_length: int) -> bool:
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
            return self._bin_centres_to_edges(self.x_data)

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
            return self._bin_edges_to_centres(self.x_data)
        else:
            return self.x_data

    def get_bin_widths(self) -> Quantity:
        """
        Get x-axis bin widths
        """
        return np.diff(self.get_bin_edges())

    def assert_regular_bins(self, message: str = '',
                            rtol: float = 1e-5,
                            atol: float = 0.) -> None:
        """Raise AssertionError if x-axis bins are not evenly spaced.

        Parameters
        ----------
        message
            Text appended to ValueError for more informative output.

        rtol
            Relative tolerance for 'close enough' values

        atol
            Absolute tolerance for 'close enough' values. Note this is a bare
            float and follows the stored units of the bins.

        """
        bin_widths = self.get_bin_widths()
        # Need to cast to magnitude to use isclose() with atol before Pint 0.21
        if not np.all(np.isclose(bin_widths.magnitude, bin_widths.magnitude[0],
                                 rtol=rtol, atol=atol)):
            raise AssertionError("Not all x-axis bins are the same width. "
                                 + message)


class Spectrum1D(Spectrum):
    """
    For storing generic 1D spectra e.g. density of states

    Attributes
    ----------
    x_data
        Shape (n_x_data,) or (n_x_data + 1,) float Quantity. The x_data
        points (if size == (n_x_data,)) or x_data bin edges (if size
        == (n_x_data + 1,))
    y_data
        Shape (n_x_data,) float Quantity. The plot data in y
    x_tick_labels
        Sequence[Tuple[int, str]] or None. Special tick labels e.g. for
        high-symmetry points. The int refers to the index in x_data the
        label should be applied to
    metadata
        Dict[str, Union[int, str]]. Contains metadata about the
        spectrum. Keys should be strings and values should be strings
        or integers
        There are some functional keys:

          - 'label' : str. This is used label lines on a 1D plot
    """
    T = TypeVar('T', bound='Spectrum1D')

    def __init__(self, x_data: Quantity, y_data: Quantity,
                 x_tick_labels: Optional[Sequence[Tuple[int, str]]] = None,
                 metadata: Optional[Dict[str, Union[int, str]]] = None
                 ) -> None:
        """
        Parameters
        ----------
        x_data
            Shape (n_x_data,) or (n_x_data + 1,) float Quantity. The
            x_data points (if size == (n_x_data,)) or x_data bin edges
            (if size == (n_x_data + 1,))
        y_data
            Shape (n_x_data,) float Quantity. The plot data in y
        x_tick_labels
            Special tick labels e.g. for high-symmetry points. The int
            refers to the index in x_data the label should be applied to
        metadata
            Contains metadata about the spectrum. Keys should be
            strings and values should be strings or integers
            There are some functional keys:

              - 'label' : str. This is used label lines on a 1D plot
        """
        _check_constructor_inputs(
            [y_data, x_tick_labels, metadata],
            [Quantity, [list, type(None)], [dict, type(None)]],
            [(-1,), (), ()],
            ['y_data', 'x_tick_labels', 'metadata'])
        ny = len(y_data)
        _check_constructor_inputs(
            [x_data], [Quantity],
            [[(ny,), (ny+1,)]], ['x_data'])
        self._set_data(x_data, 'x')
        self._set_data(y_data, 'y')
        self.x_tick_labels = x_tick_labels
        self.metadata = {} if metadata is None else metadata

    def __add__(self, other: 'Spectrum1D') -> 'Spectrum1D':
        """
        Sums the y_data of two Spectrum1D objects together,
        their x_data axes must be equal, and their y_data must
        have compatible units and the same number of y_data
        entries

        Any metadata key/value pairs that are common to both
        spectra are retained, any others are discarded
        """
        spec_col = Spectrum1DCollection.from_spectra([self, other])
        return spec_col.sum()

    def _split_by_indices(self: T,
                          indices: Union[Sequence[int], np.ndarray]
                          ) -> List[T]:
        """Split data along x-axis at given indices"""
        ranges = self._ranges_from_indices(indices)

        return [type(self)(self.x_data[x0:x1], self.y_data[x0:x1],
                           x_tick_labels=self._cut_x_ticks(self.x_tick_labels,
                                                           x0, x1),
                           metadata=self.metadata)
                for x0, x1 in ranges]

    def copy(self: T) -> T:
        """Get an independent copy of spectrum"""
        return type(self)(np.copy(self.x_data),
                          np.copy(self.y_data),
                          x_tick_labels=copy.copy(self.x_tick_labels),
                          metadata=copy.deepcopy(self.metadata))

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a dictionary. See Spectrum1D.from_dict for details on
        keys/values

        Returns
        -------
        dict
        """
        return _obj_to_dict(self, ['x_data', 'y_data', 'x_tick_labels',
                                   'metadata'])

    def to_text_file(self, filename: str,
                     fmt: Optional[Union[str, Sequence[str]]] = None) -> None:
        """
        Write to a text file. The header contains metadata and unit
        information, the first column contains x_data and the second
        column contains y_data. Note that text files written in this
        format cannot be read back in by Euphonic.

        Parameters
        ----------
        filename
            Name of the text file to write to
        fmt
            A format specifier or sequence of specifiers (one for each
            column), to be passed to numpy.savetxt
        """
        spec = Spectrum1DCollection.from_spectra([self])
        spec.to_text_file(filename, fmt)

    @classmethod
    def from_dict(cls: Type[T], d: Dict[str, Any]) -> T:
        """
        Convert a dictionary to a Spectrum1D object

        Parameters
        ----------
        d
            A dictionary with the following keys/values:

            - 'x_data': (n_x_data,) or (n_x_data + 1,) float ndarray
            - 'x_data_unit': str
            - 'y_data': (n_x_data,) float ndarray
            - 'y_data_unit': str

            There are also the following optional keys:

            - 'x_tick_labels': list of (int, string) tuples
            - 'metadata': dict

        Returns
        -------
        spectrum
        """
        d = _process_dict(d, quantities=['x_data', 'y_data'],
                          optional=['x_tick_labels', 'metadata'])
        return cls(d['x_data'], d['y_data'], x_tick_labels=d['x_tick_labels'],
                   metadata=d['metadata'])

    @classmethod
    def from_castep_phonon_dos(cls: Type[T], filename: str,
                               element: Optional[str] = None) -> T:
        """
        Reads DOS from a CASTEP .phonon_dos file

        Parameters
        ----------
        filename
            The path and name of the .phonon_dos file to read
        element
            Which element's PDOS to read. If not supplied reads
            the total DOS.
        """
        data = read_phonon_dos_data(filename)
        metadata = {}
        if element is None:
            element = 'Total'
        else:
            metadata['species'] = element
        metadata['label'] = element
        return cls(data['dos_bins']*ureg(data['dos_bins_unit']),
                   data['dos'][element]*ureg(data['dos_unit']),
                   metadata=metadata)

    @overload
    def broaden(self: T, x_width: Quantity,
                shape: KernelShape = 'gauss',
                method: Optional[Literal['convolve']] = None
                ) -> T:
        ...

    @overload
    def broaden(self: T, x_width: CallableQuantity,
                shape: KernelShape = 'gauss',
                method: Optional[Literal['convolve']] = None,
                width_lower_limit: Optional[Quantity] = None,
                width_convention: Literal['fwhm', 'std'] = 'fwhm',
                width_interpolation_error: float = 0.01,
                width_fit: ErrorFit = 'cheby-log'
                ) -> T:  # noqa: F811
        ...

    def broaden(self: T, x_width,
                shape='gauss',
                method=None,
                width_lower_limit=None,
                width_convention='fwhm',
                width_interpolation_error=0.01,
                width_fit='cheby-log'
                ) -> T:  # noqa: F811
        """
        Broaden y_data and return a new broadened spectrum object

        Parameters
        ----------
        x_width
            Scalar float Quantity, giving broadening FWHM for all y data, or
            a function handle accepting Quantity consistent with x-axis as
            input and returning FWHM corresponding to input array. This would
            typically be an energy-dependent resolution function.
        shape
            One of {'gauss', 'lorentz'}. The broadening shape
        method
            Can be None or 'convolve'. Currently the only broadening
            method available is convolution with a broadening kernel
            but this may not produce correct results for unequal bin
            widths. To use convolution anyway, explicitly set
            method='convolve'
        width_lower_limit
            Set a lower bound to width obtained calling x_width function. By
            default, this is equal to bin width. To disable, set to -Inf.
        width_convention
            By default ('fwhm'), x_width is interpreted as full-width
            half-maximum. Set to 'std' to instead define standard deviation.
        width_interpolation_error
            When x_width is a callable function, variable-width broadening is
            implemented by an approximate kernel-interpolation scheme. This
            parameter determines the target error of the kernel approximations.
        width_fit
            Select parametrisation of kernel width spacing to
            width_interpolation_error.  'cheby-log' is recommended: for shape
            'gauss', 'cubic' is also available.

        Returns
        -------
        broadened_spectrum
            A new Spectrum1D object with broadened y_data

        Raises
        ------
        ValueError
            If shape is not one of the allowed strings
        ValueError
            If method is None and bins are not of equal size
        """
        if isinstance(x_width, Quantity):
            y_broadened = self._broaden_data(
                self.y_data.magnitude,
                [self.get_bin_centres().magnitude],
                [x_width.to(self.x_data_unit).magnitude],
                shape=shape, method=method) * ureg(self.y_data_unit)

        elif isinstance(x_width, Callable):
            self.assert_regular_bins(message=(
                'Broadening with convolution requires a regular sampling grid.'
            ))
            y_broadened = variable_width_broadening(
                self.get_bin_edges(),
                self.get_bin_centres(),
                x_width,
                (self.y_data * self.get_bin_widths()[0]),
                width_lower_limit=width_lower_limit,
                width_convention=width_convention,
                adaptive_error=width_interpolation_error,
                shape=shape,
                fit=width_fit
            )
        else:
            raise TypeError("x_width must be a Quantity or Callable")

        new_spectrum = self.copy()
        new_spectrum.y_data = y_broadened
        return new_spectrum


LineData = Sequence[Dict[str, Union[str, int]]]


class Spectrum1DCollection(collections.abc.Sequence, Spectrum):
    """A collection of Spectrum1D with common x_data and x_tick_labels

    Intended for convenient storage of band structures, projected DOS
    etc. This object can be indexed or iterated to obtain individual
    Spectrum1D.

    Attributes
    ----------
    x_data
        Shape (n_x_data,) or (n_x_data + 1,) float Quantity. The x_data
        points (if size == (n_x_data,)) or x_data bin edges (if size
        == (n_x_data + 1,))
    y_data
        Shape (n_entries, n_x_data) float Quantity. The plot data in y,
        in rows corresponding to separate 1D spectra
    x_tick_labels
        Sequence[Tuple[int, str]] or None. Special tick labels e.g. for
        high-symmetry points. The int refers to the index in x_data the
        label should be applied to
    metadata
        Dict[str, Union[int, str, LineData]] or None. Contains metadata
        about the spectra. Keys should be strings and values should be
        strings or integers.
        There are some functional keys:

          - 'line_data' : LineData
                          This is a Sequence[Dict[str, Union[int, str]],
                          it contains metadata for each spectrum in
                          the collection, and must be of length
                          n_entries
    """
    T = TypeVar('T', bound='Spectrum1DCollection')

    def __init__(
            self, x_data: Quantity, y_data: Quantity,
            x_tick_labels: Optional[Sequence[Tuple[int, str]]] = None,
            metadata: Optional[Dict[str, Union[str, int, LineData]]] = None
    ) -> None:
        """
        Parameters
        ----------
        x_data
            Shape (n_x_data,) or (n_x_data + 1,) float Quantity. The
            x_data points (if size == (n_x_data,)) or x_data bin edges
            (if size == (n_x_data + 1,))
        y_data
            Shape (n_entries, n_x_data) float Quantity. The plot data
            in y, in rows corresponding to separate 1D spectra
        x_tick_labels
            Special tick labels e.g. for high-symmetry points. The int
            refers to the index in x_data the label should be applied to
        metadata
            Contains metadata about the spectra. Keys should be
            strings and values should be strings or integers.
            There are some functional keys:

              - 'line_data' : LineData
                              This is a Sequence[Dict[str, Union[int, str]],
                              it contains metadata for each spectrum
                              in the collection, and must be of length
                              n_entries
        """

        _check_constructor_inputs(
            [y_data, x_tick_labels, metadata],
            [Quantity, [list, type(None)], [dict, type(None)]],
            [(-1, -1), (), ()],
            ['y_data', 'x_tick_labels', 'metadata'])
        ny = len(y_data[0])
        _check_constructor_inputs(
            [x_data], [Quantity],
            [[(ny,), (ny+1,)]], ['x_data'])

        self._set_data(x_data, 'x')
        self._set_data(y_data, 'y')
        self.x_tick_labels = x_tick_labels
        if metadata and 'line_data' in metadata.keys():
            if len(metadata['line_data']) != len(y_data):
                raise ValueError(
                    f'y_data contains {len(y_data)} spectra, but '
                    f'metadata["line_data"] contains '
                    f'{len(metadata["line_data"])} entries')
        self.metadata = {} if metadata is None else metadata

    def __add__(self: T, other: T) -> T:
        """
        Appends the y_data of 2 Spectrum1DCollection objects,
        creating a single Spectrum1DCollection that contains
        the spectra from both objects. The two objects must
        have equal x_data axes, and their y_data must
        have compatible units and the same number of y_data
        entries

        Any metadata key/value pairs that are common to both
        spectra are retained in the top level dictionary, any
        others are put in the individual 'line_data' entries
        """
        return type(self).from_spectra([*self, *other])

    def _split_by_indices(self,
                          indices: Union[Sequence[int], np.ndarray]
                          ) -> List[T]:
        """Split data along x-axis at given indices"""

        ranges = self._ranges_from_indices(indices)

        return [type(self)(self.x_data[x0:x1], self.y_data[:, x0:x1],
                           x_tick_labels=self._cut_x_ticks(self.x_tick_labels,
                                                           x0, x1),
                           metadata=self.metadata)
                for x0, x1 in ranges]

    def __len__(self):
        return self.y_data.shape[0]

    @overload
    def __getitem__(self, item: int) -> Spectrum1D:
        ...

    @overload  # noqa: F811
    def __getitem__(self, item: slice) -> T:
        ...

    @overload  # noqa: F811
    def __getitem__(self, item: Union[Sequence[int], np.ndarray]) -> T:
        ...

    def __getitem__(self, item: Union[int, slice, Sequence[int], np.ndarray]
                    ):  # noqa: F811
        new_metadata = copy.deepcopy(self.metadata)
        line_metadata = new_metadata.pop('line_data',
                                         [{} for _ in self._y_data])
        if isinstance(item, Integral):
            new_metadata.update(line_metadata[item])
            return Spectrum1D(self.x_data,
                              self.y_data[item, :],
                              x_tick_labels=self.x_tick_labels,
                              metadata=new_metadata)

        if isinstance(item, slice):
            if (item.stop is not None) and (item.stop >= len(self)):
                raise IndexError(f'index "{item.stop}" out of range')
            new_metadata.update(self._combine_metadata(line_metadata[item]))
        else:
            try:
                item = list(item)
                if not all([isinstance(i, Integral) for i in item]):
                    raise TypeError
            except TypeError:
                raise TypeError(f'Index "{item}" should be an integer, slice '
                                f'or sequence of ints')
            new_metadata.update(self._combine_metadata(
                [line_metadata[i] for i in item]))
        return type(self)(self.x_data,
                          self.y_data[item, :],
                          x_tick_labels=self.x_tick_labels,
                          metadata=new_metadata)

    @classmethod
    def from_spectra(cls: Type[T], spectra: Sequence[Spectrum1D]) -> T:
        if len(spectra) < 1:
            raise IndexError("At least one spectrum is needed for collection")

        def _type_check(spectrum):
            if not isinstance(spectrum, Spectrum1D):
                raise TypeError(
                    "from_spectra() requires a sequence of Spectrum1D")

        _type_check(spectra[0])
        x_data = spectra[0].x_data
        x_tick_labels = spectra[0].x_tick_labels
        y_data_length = len(spectra[0].y_data)
        y_data_magnitude = np.empty((len(spectra), y_data_length))
        y_data_magnitude[0, :] = spectra[0].y_data.magnitude
        y_data_units = spectra[0].y_data.units

        for i, spectrum in enumerate(spectra[1:]):
            _type_check(spectrum)
            assert spectrum.y_data.units == y_data_units
            assert np.allclose(spectrum.x_data, x_data)
            assert spectrum.x_data.units == x_data.units
            assert spectrum.x_tick_labels == x_tick_labels
            y_data_magnitude[i + 1, :] = spectrum.y_data.magnitude

        metadata = cls._combine_metadata([spec.metadata for spec in spectra])
        y_data = Quantity(y_data_magnitude, y_data_units)
        return cls(x_data, y_data, x_tick_labels=x_tick_labels,
                   metadata=metadata)

    @staticmethod
    def _combine_metadata(all_metadata: Sequence[Dict[str, Union[int, str]]]
                          ) -> Dict[str, Union[int, str, LineData]]:
        """
        From a sequence of metadata dictionaries, combines all common
        key/value pairs into the top level of a metadata dictionary,
        all unmatching key/value pairs are put into the 'line_data'
        key, which is a list of metadata dicts for each element in
        all_metadata
        """
        # This is for combining multiple separate spectrum metadata,
        # they shouldn't have line_data
        for metadata in all_metadata:
            assert 'line_data' not in metadata.keys()
        # Combine all common key/value pairs
        combined_metadata = dict(
            set(all_metadata[0].items()).intersection(
                *[metadata.items() for metadata in all_metadata[1:]]))
        # Put all other per-spectrum metadata in line_data
        line_data = []
        for i, metadata in enumerate(all_metadata):
            sdata = copy.deepcopy(metadata)
            for key in combined_metadata.keys():
                sdata.pop(key)
            line_data.append(sdata)
        if any(line_data):
            combined_metadata['line_data'] = line_data
        return combined_metadata

    def _combine_line_metadata(self, indices: Optional[Sequence[int]] = None
                               ) -> Dict[str, Any]:
        """
        For a metadata dictionary, combines all common key/value
        pairs in 'line_data' and puts them in a top-level dictionary.
        If indices is supplied, only those indices in 'line_data' are
        combined. Unmatching key/value pairs are discarded
        """
        line_data = self.metadata.get('line_data', [{}]*len(self))
        if indices is not None:
            line_data = [line_data[idx] for idx in indices]
        combined_line_data = self._combine_metadata(line_data)
        combined_line_data.pop('line_data', None)
        return combined_line_data

    def _get_line_data_vals(self, *line_data_keys: str) -> np.ndarray:
        """
        Get value of the key(s) for each element in
        metadata['line_data']. Returns a 1D array of tuples, where each
        tuple contains the value(s) for each key in line_data_keys, for
        a single element in metadata['line_data']. This allows easy
        grouping/selecting by specific keys

        For example, if we have a Spectrum1DCollection with the following
        metadata:
            {'desc': 'Quartz', 'line_data': [
                {'inst': 'LET', 'sample': 0, 'index': 1},
                {'inst': 'MAPS', 'sample': 1, 'index': 2},
                {'inst': 'MARI', 'sample': 1, 'index': 1},
            ]}
        Then:
            _get_line_data_vals('inst', 'sample') = [('LET', 0),
                                                     ('MAPS', 1),
                                                     ('MARI', 1)]

        Raises a KeyError if 'line_data' or the key doesn't exist
        """
        line_data = self.metadata['line_data']
        line_data_vals = np.empty(len(line_data), dtype=object)
        for i, data in enumerate(line_data):
            line_data_vals[i] = tuple([data[key] for key in line_data_keys])
        return line_data_vals

    def copy(self: T) -> T:
        """Get an independent copy of spectrum"""
        return Spectrum1D.copy(self)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a dictionary consistent with from_dict()

        Returns
        -------
        dict
        """
        return _obj_to_dict(self, ['x_data', 'y_data', 'x_tick_labels',
                                   'metadata'])

    def to_text_file(self, filename: str,
                     fmt: Optional[Union[str, Sequence[str]]] = None) -> None:
        """
        Write to a text file. The header contains metadata and unit
        information, the first column is x_data and each subsequent
        column is a y_data spectrum. Note that text files written in
        this format cannot be read back in by Euphonic.

        Parameters
        ----------
        filename
            Name of the text file to write to
        fmt
            A format specifier or sequence of specifiers (one for each
            column), to be passed to numpy.savetxt
        """
        common_metadata = copy.deepcopy(self.metadata)
        line_data = common_metadata.pop('line_data',
                                        [{} for i in self.y_data])
        header = [f'Generated by Euphonic {__version__}',
                  f'x_data in ({self.x_data.units})',
                  f'y_data in ({self.y_data.units})',
                  f'Common metadata: {json.dumps(common_metadata)}',
                  r'Column 1: x_data']
        for i, line in enumerate(line_data):
            header += [f'Column {i + 2}: y_data[{i}] {json.dumps(line)}']
        out_data = np.hstack((self.get_bin_centres().magnitude[:, np.newaxis],
                              self.y_data.transpose().magnitude))
        kwargs = {'header': '\n'.join(header)}
        if fmt is not None:
            kwargs['fmt'] = fmt
        np.savetxt(filename, out_data, **kwargs)

    @classmethod
    def from_dict(cls: Type[T], d) -> T:
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
            - 'metadata': dict

        Returns
        -------
        spectrum_collection
        """
        d = _process_dict(d, quantities=['x_data', 'y_data'],
                          optional=['x_tick_labels', 'metadata'])
        return cls(d['x_data'], d['y_data'], x_tick_labels=d['x_tick_labels'],
                   metadata=d['metadata'])

    @classmethod
    def from_castep_phonon_dos(cls: Type[T], filename: str) -> T:
        """
        Reads total DOS and per-element PDOS from a CASTEP
        .phonon_dos file

        Parameters
        ----------
        filename
            The path and name of the .phonon_dos file to read
        """
        data = read_phonon_dos_data(filename)
        n_spectra = len(data['dos'].keys())
        metadata = {'line_data': [{} for x in range(n_spectra)]}
        for i, (species, dos_data) in enumerate(data['dos'].items()):
            if i == 0:
                y_data = np.zeros((n_spectra, len(dos_data)))
            y_data[i] = dos_data
            if species != 'Total':
                metadata['line_data'][i]['species'] = species
            metadata['line_data'][i]['label'] = species
        return Spectrum1DCollection(
            data['dos_bins']*ureg(data['dos_bins_unit']),
            y_data*ureg(data['dos_unit']),
            metadata=metadata)

    @overload
    def broaden(self: T, x_width: Quantity,
                shape: KernelShape = 'gauss',
                method: Optional[Literal['convolve']] = None
                ) -> T:
        ...

    @overload
    def broaden(self: T, x_width: CallableQuantity,
                shape: KernelShape = 'gauss',
                method: Optional[Literal['convolve']] = None,
                width_lower_limit: Optional[Quantity] = None,
                width_convention: Literal['fwhm', 'std'] = 'fwhm',
                width_interpolation_error: float = 0.01,
                width_fit: ErrorFit = 'cheby-log'
                ) -> T:  # noqa: F811
        ...

    def broaden(self: T,
                x_width,
                shape='gauss',
                method=None,
                width_lower_limit=None,
                width_convention='fwhm',
                width_interpolation_error=0.01,
                width_fit='cheby-log'
                ) -> T:  # noqa: F811
        """
        Individually broaden each line in y_data, returning a new
        Spectrum1DCollection

        Parameters
        ----------
        x_width
            Scalar float Quantity, giving broadening FWHM for all y data, or
            a function handle accepting Quantity consistent with x-axis as
            input and returning FWHM corresponding to input array. This would
            typically be an energy-dependent resolution function.
        shape
            One of {'gauss', 'lorentz'}. The broadening shape
        method
            Can be None or 'convolve'. Currently the only broadening
            method available is convolution with a broadening kernel
            but this may not produce correct results for unequal bin
            widths. To use convolution anyway, explicitly set
            method='convolve'
        width_lower_limit
            Set a lower bound to width obtained calling x_width function. By
            default, this is equal to bin width. To disable, set to -Inf.
        width_convention
            By default ('fwhm'), x_width is interpreted as full-width
            half-maximum. Set to 'std' to instead define standard deviation.
        width_interpolation_error
            When x_width is a callable function, variable-width broadening is
            implemented by an approximate kernel-interpolation scheme. This
            parameter determines the target error of the kernel approximations.
        width_fit
            Select parametrisation of kernel width spacing to
            width_interpolation_error. 'cheby-log' is recommended: for shape
            'gauss', 'cubic' is also available.

        Returns
        -------
        broadened_spectrum
            A new Spectrum1DCollection object with broadened y_data

        Raises
        ------
        ValueError
            If shape is not one of the allowed strings
        ValueError
            If method is None and bins are not of equal size
        """
        if isinstance(x_width, Quantity):
            y_broadened = np.zeros_like(self.y_data)
            x_centres = [self.get_bin_centres().magnitude]
            x_width_calc = [x_width.to(self.x_data_unit).magnitude]
            for i, yi in enumerate(self.y_data.magnitude):
                y_broadened[i] = self._broaden_data(
                    yi, x_centres, x_width_calc, shape=shape,
                    method=method)

            new_spectrum = self.copy()
            new_spectrum.y_data = y_broadened * ureg(self.y_data_unit)
            return new_spectrum

        elif isinstance(x_width, Callable):
            return type(self).from_spectra([
                spectrum.broaden(
                    x_width=x_width,
                    shape=shape,
                    method=method,
                    width_lower_limit=width_lower_limit,
                    width_convention=width_convention,
                    width_interpolation_error=width_interpolation_error,
                    width_fit=width_fit)
                for spectrum in self])

        else:
            raise TypeError("x_width must be a Quantity or Callable")

    def group_by(self, *line_data_keys: str) -> T:
        """
        Group and sum y_data for each spectrum according to the values
        mapped to the specified keys in metadata['line_data']

        Parameters
        ----------
        line_data_keys
            The key(s) to group by. If only one line_data_key is
            supplied, if the value mapped to a key is the same for
            multiple spectra, they are placed in the same group and
            summed. If multiple line_data_keys are supplied, the values
            must be the same for all specified keys for them to be
            placed in the same group

        Returns
        -------
        grouped_spectrum
            A new Spectrum1DCollection with one line for each group. Any
            metadata in 'line_data' not common across all spectra in a
            group will be discarded
        """
        grouping_dict = _get_unique_elems_and_idx(
            self._get_line_data_vals(*line_data_keys))

        new_y_data = np.zeros((len(grouping_dict), self._y_data.shape[-1]))
        group_metadata = copy.deepcopy(self.metadata)
        group_metadata['line_data'] = [{}]*len(grouping_dict)
        for i, idxs in enumerate(grouping_dict.values()):
            # Look for any common key/values in grouped metadata
            group_i_metadata = self._combine_line_metadata(idxs)
            group_metadata['line_data'][i] = group_i_metadata
            new_y_data[i] = np.sum(self._y_data[idxs], axis=0)
        new_y_data = new_y_data*ureg(self._internal_y_data_unit).to(
            self.y_data_unit)

        new_data = self.copy()
        new_data.y_data = new_y_data
        new_data.metadata = group_metadata

        return new_data

    def sum(self) -> Spectrum1D:
        """
        Sum y_data over all spectra

        Returns
        -------
        summed_spectrum
            A Spectrum1D created from the summed y_data. Any metadata
            in 'line_data' not common across all spectra will be
            discarded
        """
        metadata = copy.deepcopy(self.metadata)
        metadata.pop('line_data', None)
        metadata.update(self._combine_line_metadata())
        summed_y_data = np.sum(self._y_data, axis=0)*ureg(
            self._internal_y_data_unit).to(self.y_data_unit)
        return Spectrum1D(np.copy(self.x_data),
                          summed_y_data,
                          x_tick_labels=copy.copy(self.x_tick_labels),
                          metadata=copy.deepcopy(metadata))

    def select(self, **select_key_values: Union[
            str, int, Sequence[str], Sequence[int]]) -> T:
        """
        Select spectra by their keys and values in metadata['line_data']

        Parameters
        ----------
        **select_key_values
            Key-value/values pairs in metadata['line_data'] describing
            which spectra to extract. For example, to select all spectra
            where metadata['line_data']['species'] = 'Na' or 'Cl' use
            spectrum.select(species=['Na', 'Cl']). To select 'Na' and
            'Cl' spectra where weighting is also coherent, use
            spectrum.select(species=['Na', 'Cl'], weighting='coherent')

        Returns
        -------
        selected_spectra
           A Spectrum1DCollection containing the selected spectra

        Raises
        ------
        ValueError
            If no matching spectra are found
        """
        select_val_dict = _get_unique_elems_and_idx(
            self._get_line_data_vals(*select_key_values.keys()))
        for key, value in select_key_values.items():
            if isinstance(value, (int, str)):
                select_key_values[key] = [value]
        value_combinations = itertools.product(*select_key_values.values())
        select_idx = np.array([], dtype=np.int32)
        for value_combo in value_combinations:
            try:
                idx = select_val_dict[value_combo]
            # Don't require every combination to match e.g.
            # spec.select(sample=[0, 2], inst=['MAPS', 'MARI'])
            # we don't want to error simply because there are no
            # inst='MAPS' and sample=2 combinations
            except KeyError:
                continue
            select_idx = np.append(select_idx, idx)
        if len(select_idx) == 0:
            raise ValueError(f'No spectra found with matching metadata '
                             f'for {select_key_values}')
        return self[select_idx]


class Spectrum2D(Spectrum):
    """
    For storing generic 2D spectra e.g. S(Q,w)

    Attributes
    ----------
    x_data
        Shape (n_x_data,) or (n_x_data + 1,) float Quantity. The x_data
        points (if size == (n_x_data,)) or x_data bin edges (if size
        == (n_x_data + 1,))
    y_data
        Shape (n_y_data,) or (n_y_data + 1,) float Quantity. The y_data
        bin points (if size == (n_y_data,)) or y_data bin edges (if size
        == (n_y_data + 1,))
    z_data
        Shape (n_x_data, n_y_data) float Quantity. The plot data in z
    x_tick_labels
        Sequence[Tuple[int, str]] or None. Special tick labels e.g. for
        high-symmetry points. The int refers to the index in x_data the
        label should be applied to
    metadata
        Dict[str, Union[int, str]]. Contains metadata about the
        spectrum. Keys should be strings and values should be strings
        or integers
    """
    T = TypeVar('T', bound='Spectrum2D')

    def __init__(self, x_data: Quantity, y_data: Quantity,
                 z_data: Quantity,
                 x_tick_labels: Optional[Sequence[Tuple[int, str]]] = None,
                 metadata: Optional[Dict[str, Union[int, str]]] = None
                 ) -> None:
        """
        Parameters
        ----------
        x_data
            Shape (n_x_data,) or (n_x_data + 1,) float Quantity. The
            x_data points (if size == (n_x_data,)) or x_data bin edges
            (if size == (n_x_data + 1,))
        y_data
            Shape (n_y_data,) or (n_y_data + 1,) float Quantity. The
            y_data bin points (if size == (n_y_data,)) or y_data bin
            edges (if size == (n_y_data + 1,))
        z_data
            Shape (n_x_data, n_y_data) float Quantity. The plot data in z
        x_tick_labels
            Special tick labels e.g. for high-symmetry points. The int
            refers to the index in x_data the label should be applied to
        metadata
            Contains metadata about the spectrum. Keys should be
            strings and values should be strings or integers.
        """
        _check_constructor_inputs(
            [z_data, x_tick_labels, metadata],
            [Quantity, [list, type(None)], [dict, type(None)]],
            [(-1, -1), (), ()],
            ['z_data', 'x_tick_labels', 'metadata'])
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
        self.metadata = {} if metadata is None else metadata

    @property
    def z_data(self) -> Quantity:
        return self._z_data*ureg(self._internal_z_data_unit).to(
            self.z_data_unit)

    @z_data.setter
    def z_data(self, value: Quantity) -> None:
        self.z_data_unit = str(value.units)
        self._z_data = value.to(self._internal_z_data_unit).magnitude

    def __imul__(self: T, other: Real) -> T:
        """Scale spectral data in-place"""
        self.z_data = self.z_data * other
        return self

    def __setattr__(self, name: str, value: Any) -> None:
        _check_unit_conversion(self, name, value,
                               ['z_data_unit'])
        super(Spectrum2D, self).__setattr__(name, value)

    def _split_by_indices(self,
                          indices: Union[Sequence[int], np.ndarray]
                          ) -> List[T]:
        """Split data along x-axis at given indices"""
        ranges = self._ranges_from_indices(indices)
        return [type(self)(self.x_data[x0:x1], self.y_data,
                           self.z_data[x0:x1, :],
                           x_tick_labels=self._cut_x_ticks(self.x_tick_labels,
                                                           x0, x1),
                           metadata=self.metadata)
                for x0, x1 in ranges]

    def broaden(self: T,
                x_width: Optional[Union[Quantity, CallableQuantity]] = None,
                y_width: Optional[Union[Quantity, CallableQuantity]] = None,
                shape: KernelShape = 'gauss',
                method: Optional[Literal['convolve']] = None,
                x_width_lower_limit: Quantity = None,
                y_width_lower_limit: Quantity = None,
                width_convention: Literal['fwhm', 'std'] = 'fwhm',
                width_interpolation_error: float = 0.01,
                width_fit: ErrorFit = 'cheby-log'
                ) -> T:
        """
        Broaden z_data and return a new broadened Spectrum2D object

        Callable functions can be used to access variable-width broadening.
        In this case, broadening is implemented with a kernel-interpolating
        approximate scheme.

        Parameters
        ----------
        x_width
            Either a scalar float Quantity representing the broadening width,
            or a callable function that accepts and returns Quantity consistent
            with x-axis units.
        y_width
            Either a scalar float Quantity representing the broadening width,
            or a callable function that accepts and returns Quantity consistent
            with y-axis units.
        shape
            One of {'gauss', 'lorentz'}. The broadening shape
        method
            Can be None or 'convolve'. Currently the only broadening
            method available is convolution with a broadening kernel,
            but this may not produce correct results for unequal bin
            widths. To use convolution anyway, explicitly set
            method='convolve'
        x_width_lower_limit
            Set a lower bound to width obtained calling x_width function. By
            default, this is equal to x bin width. To disable, set to -Inf.
        y_width_lower_limit
            Set a lower bound to width obtained calling y_width function. By
            default, this is equal to y bin width. To disable, set to -Inf.
        width_convention
            By default ('fwhm'), widths are interpreted as full-width
            half-maximum. Set to 'std' to instead define standard deviation.
        width_interpolation_error
            When x_width is a callable function, variable-width broadening is
            implemented by an approximate kernel-interpolation scheme. This
            parameter determines the target error of the kernel approximations.
        width_fit
            Select parametrisation of kernel width spacing to
            width_interpolation_error. 'cheby-log' is recommended: for shape
            'gauss', 'cubic' is also available.

        Returns
        -------
        broadened_spectrum
            A new Spectrum2D object with broadened z_data

        Raises
        ------
        ValueError
            If shape is not one of the allowed strings
        ValueError
            If method is None and bins are not of equal size
        """
        # First apply fixed-width broadening; this can be applied to both axes
        # in one function call if both are specified.
        widths_in_bin_units = [None]*2

        if isinstance(x_width, Quantity):
            try:
                self.assert_regular_bins('x', message=(
                    'Broadening by convolution may give incorrect results.'))
            except AssertionError as e:
                warnings.warn(str(e), UserWarning)
            widths_in_bin_units[0] = x_width.to(self.x_data_unit).magnitude
        if isinstance(y_width, Quantity):
            widths_in_bin_units[1] = y_width.to(self.y_data_unit).magnitude

        if any(widths_in_bin_units):
            bin_centres = [self.get_bin_centres(ax).magnitude
                           for ax in ['x', 'y']]
            z_broadened = self._broaden_data(self.z_data.magnitude,
                                             bin_centres,
                                             widths_in_bin_units,
                                             shape=shape,
                                             method=method)
            spectrum = Spectrum2D(np.copy(self.x_data),
                                  np.copy(self.y_data),
                                  z_broadened*ureg(self.z_data_unit),
                                  copy.copy(self.x_tick_labels),
                                  copy.deepcopy(self.metadata))
        else:
            spectrum = self

        # Then apply variable-width broadening to any axes with Callable
        for axis, width, lower_limit in (('x', x_width, x_width_lower_limit),
                                         ('y', y_width, y_width_lower_limit)):
            if isinstance(width, Callable):
                spectrum = self._broaden_spectrum2d_with_function(
                    spectrum, width, axis=axis, width_lower_limit=lower_limit,
                    width_convention=width_convention,
                    width_interpolation_error=width_interpolation_error,
                    shape=shape,
                    width_fit=width_fit)

        return spectrum

    @staticmethod
    def _broaden_spectrum2d_with_function(
            spectrum: 'Spectrum2D',
            width_function: Callable[[Quantity], Quantity],
            axis: Literal['x', 'y'] = 'y',
            width_lower_limit: Quantity = None,
            width_convention: Literal['fwhm', 'std'] = 'fwhm',
            width_interpolation_error: float = 1e-2,
            shape: KernelShape = 'gauss',
            width_fit: ErrorFit = 'cheby-log'
        ) -> 'Spectrum2D':
        """
        Apply value-dependent Gaussian broadening to one axis of Spectrum2D
        """
        assert axis in ('x', 'y')

        bins = spectrum.get_bin_edges(bin_ax=axis)
        bin_widths = np.diff(bins)

        if not np.all(np.isclose(bin_widths, bin_widths[0])):
            bin_width = bin_widths.mean()
        else:
            bin_width = bin_widths[0]

        # Input data: rescale to sparse-like data values
        z_data = spectrum.z_data * bin_width

        if axis == 'x':
            z_data = z_data.T

        # Output data: matches input units
        z_broadened = np.empty_like(z_data) * spectrum.z_data.units

        for i, row in enumerate(z_data):
            z_broadened[i] = variable_width_broadening(
                bins,
                spectrum.get_bin_centres(bin_ax=axis),
                width_function,
                row,
                width_lower_limit=width_lower_limit,
                width_convention=width_convention,
                adaptive_error=width_interpolation_error,
                shape=shape,
                fit=width_fit)

        if axis == 'x':
            z_broadened = z_broadened.T

        return Spectrum2D(np.copy(spectrum.x_data),
                          np.copy(spectrum.y_data),
                          z_broadened,
                          copy.copy(spectrum.x_tick_labels),
                          copy.copy(spectrum.metadata))

    def copy(self: T) -> T:
        """Get an independent copy of spectrum"""
        return type(self)(np.copy(self.x_data),
                          np.copy(self.y_data),
                          np.copy(self.z_data),
                          copy.copy(self.x_tick_labels),
                          copy.deepcopy(self.metadata))

    def get_bin_edges(self, bin_ax: Literal['x', 'y'] = 'x') -> Quantity:
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
            return self._bin_centres_to_edges(bin_data)

    def get_bin_centres(self, bin_ax: Literal['x', 'y'] = 'x') -> Quantity:
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
            return self._bin_edges_to_centres(bin_data)
        else:
            return bin_data

    def get_bin_widths(self, bin_ax: Literal['x', 'y'] = 'x') -> Quantity:
        """
        Get x-bin widths along specified axis

        Parameters
        ----------
        bin_ax
            Axis of interest, 'x' or 'y'
        """
        bins = self.get_bin_edges(bin_ax)
        return np.diff(bins)

    def assert_regular_bins(self, bin_ax: Literal['x', 'y'],
                            message: str = '',
                            rtol: float = 1e-5,
                            atol: float = 0.) -> None:
        """Raise AssertionError if x-axis bins are not evenly spaced.

        Parameters
        ----------
        bin_ax
            Axis of interest, 'x' or 'y'

        message
            Text appended to ValueError for more informative output.

        rtol
            Relative tolerance for 'close enough' values

        atol
            Absolute tolerance for 'close enough' values. Note this is a bare
            float and follows the stored units of the bins.

        """
        bin_widths = self.get_bin_widths(bin_ax)
        if not np.all(np.isclose(bin_widths, bin_widths[0],
                                 rtol=rtol, atol=atol)):
            raise AssertionError(
                f"Not all {bin_ax}-axis bins are the same width. " + message)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a dictionary. See Spectrum2D.from_dict for details on
        keys/values

        Returns
        -------
        dict
        """
        return _obj_to_dict(self, ['x_data', 'y_data', 'z_data',
                                   'x_tick_labels', 'metadata'])

    @classmethod
    def from_dict(cls: Type[T], d: Dict[str, Any]) -> T:
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
            - 'metadata': dict

        Returns
        -------
        spectrum
        """
        d = _process_dict(d, quantities=['x_data', 'y_data', 'z_data'],
                          optional=['x_tick_labels', 'metadata'])
        return cls(d['x_data'], d['y_data'], d['z_data'],
                   x_tick_labels=d['x_tick_labels'],
                   metadata=d['metadata'])


def apply_kinematic_constraints(spectrum: Spectrum2D,
                                e_i: Quantity = None,
                                e_f: Quantity = None,
                                angle_range: Tuple[float] = (0, 180.)
                                ) -> Spectrum2D:
    """
    Set events to NaN which violate energy/momentum limits:

      - Energy transfer greater than e_i
      - q outside region accessible for given e_i and angle range

    This requires x_data to be in wavevector units and y_data to be energy.

    Either e_i or e_f should be set, according to direct/instrument
    geometry. The other values will be inferred, interpreting y_data as
    energy transfer.

    Parameters
    ----------
    spectrum
        input 2-D spectrum, with |q| x_data and energy y_data
    e_i
        incident energy of direct-geometry spectrometer
    e_f
        final energy of indirect-geometry spectrometer
    angle_range
        min and max scattering angles (2θ) of detector bank in degrees

    Returns
    -------
    Masked spectrum with inaccessible bins set to NaN in z_data.
    """
    try:
        (1 * spectrum.x_data.units).to('1/angstrom')
    except DimensionalityError as error:
        raise ValueError(
            "x_data needs to have wavevector units (i.e. 1/length)"
            ) from error
    try:
        (1 * spectrum.y_data.units).to('eV', 'spectroscopy')
    except DimensionalityError as error:
        raise ValueError(
            "y_data needs to have energy (or wavenumber) units"
            ) from error

    momentum2_to_energy = 0.5 * (ureg('hbar^2 / neutron_mass')
                                 .to('meV angstrom^2'))

    if (e_i is None) == (e_f is None):
        raise ValueError("Exactly one of e_i and e_f should be set. "
                         "(The other value will be derived from energy "
                         "transfer).")

    if e_i is None:
        # Indirect geometry: final energy is fixed, incident energy unlimited
        e_f = e_f.to('meV')
        e_i = (spectrum.get_bin_centres(bin_ax='y').to('meV') + e_f)
    elif e_f is None:
        # Direct geometry: incident energy is fixed, max energy transfer = e_i
        e_i = e_i.to('meV')
        e_f = e_i - spectrum.get_bin_centres(bin_ax='y').to('meV')

    k2_i = (e_i / momentum2_to_energy)
    k2_f = (e_f / momentum2_to_energy)

    cos_values = np.asarray(
        _get_cos_range(np.asarray(angle_range) * np.pi / 180.))

    # Momentum goes negative where final energy greater than incident
    # energy; detect this as complex component and set extreme q-bounds to
    # enforce conservation of energy

    # (Complex number sqrt separated from units sqrt for compatibility with
    # older library versions; in newer versions this is not necessary.)
    q_bounds = np.sqrt(k2_i + k2_f
                       - 2 * cos_values[:, np.newaxis]
                           * np.sqrt(k2_i.magnitude * k2_f.magnitude,
                                     dtype=complex)
                           * (k2_i.units * k2_f.units)**0.5
                       )
    q_bounds.magnitude.T[np.any(q_bounds.imag, axis=0)] = [float('Inf'),
                                                           float('-Inf')]
    q_bounds = q_bounds.real

    mask = np.logical_or((spectrum.get_bin_edges(bin_ax='x')[1:, np.newaxis]
                          < q_bounds[0][np.newaxis, :]),
                         (spectrum.get_bin_edges(bin_ax='x')[:-1, np.newaxis]
                          > q_bounds[-1][np.newaxis, :]))

    new_spectrum = spectrum.copy()
    new_spectrum._z_data[mask] = float('nan')

    return new_spectrum


def _get_cos_range(angle_range: Tuple[float]) -> Tuple[float]:
    """
    Get max and min of cosine function over angle range

    These will either be the cosines of the input angles, or, in the case that
    a cosine max/min point lies within the angle range, 1/-1 respectively.

    Method: the angle range is translated such that it starts within 0-2π; then
    we check for the presence of turning points at π and 2π.
    """
    limiting_values = np.cos(angle_range).tolist()

    shift, lower_angle = divmod(min(angle_range), np.pi * 2)
    upper_angle = max(angle_range) - (shift * 2 * np.pi)
    if lower_angle < np.pi < upper_angle:
        limiting_values.append(-1.)
    if lower_angle < 2 * np.pi < upper_angle:
        limiting_values.append(1.)
    return max(limiting_values), min(limiting_values)


def _distribution_1d(xbins: np.ndarray,
                     xwidth: float,
                     shape: Literal['lorentz'] = 'lorentz',
                     ) -> np.ndarray:
    x = _get_dist_bins(xbins)
    if shape == 'lorentz':
        dist = _lorentzian(x, xwidth)
        dist = dist / np.sum(dist)  # Naively normalise
    else:
        raise ValueError("Expected shape: 'lorentz'")

    return dist


def _get_dist_bins(bins: np.ndarray) -> np.ndarray:
    # Get bins with same spacing as original, centered on zero
    # and spanning at least twice as far; this ensures a point in bin can be
    # distributed all the way across bin range.
    bin_range = bins[-1] - bins[0]
    nbins = len(bins) * 2 + 1

    return np.linspace(-bin_range, bin_range, nbins, endpoint=True)


def _lorentzian(x: np.ndarray, gamma: float) -> np.ndarray:
    return gamma/(2*math.pi*(np.square(x) + (gamma/2)**2))
