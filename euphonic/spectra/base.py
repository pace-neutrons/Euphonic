"""Classes for spectral data"""
# pylint: disable=no-member

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
import copy
from functools import partial
import math
from numbers import Integral, Real
from typing import (
    Any,
    Literal,
    Optional,
    TypeVar,
    overload,
)
import warnings

import numpy as np
from pint import DimensionalityError, Quantity
from scipy.ndimage import correlate1d, gaussian_filter

from euphonic.broadening import (
    FWHM_TO_SIGMA,
    ErrorFit,
    KernelShape,
    variable_width_broadening,
)
from euphonic.io import (
    _obj_from_json_file,
    _obj_to_dict,
    _obj_to_json_file,
    _process_dict,
)
from euphonic.readers.castep import read_phonon_dos_data
from euphonic.ureg import ureg
from euphonic.util import dedent_and_fill, zips
from euphonic.validate import _check_constructor_inputs, _check_unit_conversion

CallableQuantity = Callable[[Quantity], Quantity]
XTickLabels = list[tuple[int, str]]

OneSpectrumMetadata = dict[str, str | int]


class Spectrum(ABC):
    """Base class for a spectral data: do not use directly"""
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
        """x-axis data with units"""
        return ureg.Quantity(self._x_data, self._internal_x_data_unit,
                             ).to(self.x_data_unit, 'reciprocal_spectroscopy')

    @x_data.setter
    def x_data(self, value: Quantity) -> None:
        self.x_data_unit = str(value.units)
        self._x_data = value.to(self._internal_x_data_unit).magnitude

    @property
    def y_data(self) -> Quantity:
        """y-axis data with units"""
        return ureg.Quantity(self._y_data, self._internal_y_data_unit).to(
            self.y_data_unit, 'reciprocal_spectroscopy')

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

    @property
    def x_tick_labels(self) -> XTickLabels:
        """x-axis tick labels (e.g. high-symmetry point locations)"""
        return self._x_tick_labels

    @x_tick_labels.setter
    def x_tick_labels(self, value: XTickLabels) -> None:
        err_msg = (
            'x_tick_labels should be of type Sequence[Tuple[int, str]] e.g. '
            '[(0, "label1"), (5, "label2")]'
        )
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
    def to_dict(self) -> dict[str, Any]:
        """Write to dict using euphonic.io._obj_to_dict"""

    @classmethod
    @abstractmethod
    def from_dict(cls: type[T], d: dict[str, Any]) -> T:
        """Initialise a Spectrum object from dictionary"""

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
    def from_json_file(cls: type[T], filename: str) -> T:
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
    def _split_by_indices(self: T, indices: Sequence[int] | np.ndarray,
                          ) -> list[T]:
        """Split data along x axis at given indices"""

    def _split_by_tol(self: T, btol: float = 10.0) -> list[T]:
        """Split data along x-axis at detected breakpoints"""
        diff = np.diff(self.x_data)
        median = np.median(diff)
        breakpoints = np.where((diff / median) > btol)[0] + 1
        return self._split_by_indices(breakpoints)

    @staticmethod
    def _ranges_from_indices(indices: Sequence[int] | np.ndarray,
                             ) -> list[tuple[int, Optional[int]]]:
        """Convert a series of breakpoints to a series of slice ranges"""
        if len(indices) == 0:
            ranges = [(0, None)]
        else:
            ranges = [(0, indices[0])]
            if len(indices) > 1:
                ranges = ranges + [(indices[i], indices[i+1])
                                   for i in range(len(indices) - 1)]
            ranges = [*ranges, (indices[-1], None)]
        return ranges

    @staticmethod
    def _cut_x_ticks(x_tick_labels: XTickLabels | None,
                     x0: int,
                     x1: int | None) -> XTickLabels | None:
        """Crop x labels to new x range, shifting indices accordingly"""
        if x_tick_labels is None:
            return None

        if x1 is None:
            x1 = float('inf')

        return [(int(x - x0), label)
                for (x, label) in x_tick_labels if x0 <= x < x1]

    def split(self: T, indices: Sequence[int] | np.ndarray = None,
              btol: float | None = None) -> list[T]:
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

        if btol is not None:
            msg = 'Cannot set both indices and btol'
            raise ValueError(msg)
        return self._split_by_indices(indices)

    @classmethod
    def _broaden_data(cls,
                      data: np.ndarray,
                      bin_centres: Sequence[np.ndarray],
                      widths: Sequence[float|None],
                      shape: KernelShape = 'gauss',
                      *,
                      method: Optional[Literal['convolve']] = None,
                      width_convention: Literal['fwhm', 'std'] = 'fwhm',
                      ) -> np.ndarray:
        """
        Broaden data along each axis by widths, returning the broadened
        data. Data can be 1D or 2D, and  the length of bin_centres and
        widths should match the number of dimensions in data.
        bin_centres and widths are assumed to be in the same units
        """
        shape_opts = ('gauss', 'lorentz')
        if shape not in shape_opts:
            msg = (
                f'Invalid value for shape, got {shape}, '
                f'should be one of {shape_opts}'
            )
            raise ValueError(msg)
        method_opts = ('convolve', None)
        if method not in method_opts:
            msg = (
                f'Invalid value for method, got {method}, '
                f'should be one of {method_opts}'
            )
            raise ValueError(msg)

        # We only want to check for unequal bins if using a method that
        # is not correct for unequal bins (currently this is the only
        # option but leave the 'if' anyway for future implementations)
        axes = ['x_data', 'y_data']
        unequal_bin_axes = []
        for ax, (width, bin_data) in enumerate(zips(widths, bin_centres)):
            if width is not None:
                bin_widths = np.diff(bin_data)
                if not np.all(np.isclose(bin_widths, bin_widths[0])):
                    unequal_bin_axes += [axes[ax]]
        if len(unequal_bin_axes) > 0:
            msg = dedent_and_fill(f"""
            {" and ".join(unequal_bin_axes)} bin widths are not equal, so
            broadening by convolution may give incorrect results.
            """)
            if method is None:
                raise ValueError(
                    msg + ' If you still want to broaden by convolution '
                    'please explicitly use the method="convolve" option.')

            warnings.warn(msg, stacklevel=3)

        if shape == 'gauss':
            width_to_bin = partial(cls._gaussian_width_to_bin_sigma,
                                   width_convention=width_convention)
            sigmas = list(map(width_to_bin, widths, bin_centres))
            data_broadened = gaussian_filter(data, sigmas, mode='constant')

        elif shape == 'lorentz':
            if width_convention != 'fwhm':
                msg = 'Lorentzian function width must be specified as FWHM'
                raise ValueError(msg)
            data_broadened = data
            for ax, (width, bin_data) in enumerate(zips(widths, bin_centres)):
                if width is not None:
                    broadening = _distribution_1d(bin_data, width)
                    data_broadened = correlate1d(data_broadened, broadening,
                                                 mode='constant',
                                                 axis=ax)
        return data_broadened

    @staticmethod
    def _gaussian_width_to_bin_sigma(
        width: float | None,
        ax_bin_centres: np.ndarray,
        width_convention: Literal['fwhm', 'std'],
    ) -> float:
        """
        Convert a Gaussian FWHM to sigma in units of the mean ax bin size

        Parameters
        ----------
        width
            The Gaussian broadening width parameter FWHM or sigma (STD)
        ax_bin_centres
            Shape (n_bins,) float np.ndarray.
            The bin centres along the axis the broadening is applied to
        width_convention
            Specify width as full-width-half-maximum 'fwhm' or standard
            deviation 'std'

        Returns
        -------
        sigma
            Sigma in units of mean ax bin size
        """
        if width is None:
            return 0.

        match width_convention:
            case 'fwhm':
                sigma = width * FWHM_TO_SIGMA
            case 'std':
                sigma = width
            case _:
                msg = "Width convention must be 'std' or 'fwhm'"
                raise ValueError(msg)

        mean_bin_size = np.mean(np.diff(ax_bin_centres))
        return sigma / mean_bin_size

    @staticmethod
    def _bin_edges_to_centres(bin_edges: Quantity) -> Quantity:
        return bin_edges[:-1] + 0.5*np.diff(bin_edges)

    @staticmethod
    def _bin_centres_to_edges(
            bin_centres: Quantity,
            restrict_range: bool = True,
    ) -> Quantity:
        if restrict_range:
            return np.concatenate((
                [bin_centres[0]],
                (bin_centres[1:] + bin_centres[:-1])/2,
                [bin_centres[-1]],
            ))

        half_diff = np.diff(bin_centres) * 0.5
        return np.concatenate(
            (
                [bin_centres[0] - half_diff[0]],
                bin_centres[:-1] + half_diff,
                [bin_centres[-1] + half_diff[-1]],
            ))

    @staticmethod
    def _is_bin_edge(data_length: int, bin_length: int) -> bool:
        """Determine if axis data are bin edges or centres"""
        if bin_length == data_length + 1:
            return True
        if bin_length == data_length:
            return False
        msg = (
            f'Unexpected data axis length {data_length} '
            f'for bin axis length {bin_length}'
        )
        raise ValueError(msg)

    def get_bin_edges(self, *, restrict_range: bool = True) -> Quantity:
        """
        Get x-axis bin edges. If the size of x_data is one element larger
        than y_data, x_data is assumed to contain bin edges, but if x_data
        is the same size, x_data is assumed to contain bin centres and
        a conversion is made.

        In this case, bin edges are assumed to be halfway between bin centres.

        Parameters
        ----------
        restrict_range
            If True (default), the bin edges will not go outside the existing
            data bounds so the first and last bins may be half-size. This may
            be desirable for plotting.  Otherwise, the outer bin edges will
            extend from the initial data range.
        """
        # Need to use -1 index for y_data so it also works for
        # Spectrum1DCollection which has y_data shape (n_spectra, bins)
        if self._is_bin_edge(self.y_data.shape[-1], self.x_data.shape[0]):
            return self.x_data
        return self._bin_centres_to_edges(
            self.x_data, restrict_range=restrict_range)

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
        return self.x_data

    def get_bin_widths(self, *, restrict_range: bool = True) -> Quantity:
        """
        Get x-axis bin widths

        Parameters
        ----------
        restrict_range
            If True, bin edges will be clamped to the input data range; if
            False, they may be extrapolated beyond the initial range of bin
            centres.

            False is usually preferable, this default behaviour is for backward
            compatibility and will be removed in a future version.

        """
        return np.diff(self.get_bin_edges(restrict_range=restrict_range))

    def assert_regular_bins(self,
                            message: str = '',
                            rtol: float = 1e-5,
                            atol: float = 0.,
                            restrict_range: bool = True,
                            ) -> None:
        """Raise AssertionError if x-axis bins are not evenly spaced.

        Note that the positional arguments are different from
        Spectrum2D.assert_regular_bins: it is strongly recommended to only use
        keyword arguments with this method.

        Parameters
        ----------
        message
            Text appended to ValueError for more informative output.

        rtol
            Relative tolerance for 'close enough' values

        atol
            Absolute tolerance for 'close enough' values. Note this is a bare
            float and follows the stored units of the bins.

        restrict_range
            If True, bin edges will be clamped to the input data range; if
            False, they may be extrapolated beyond the initial range of bin
            centres.

            You should use the value which is consistent with calls to
            get_bin_widths() or get_bin_edges().

        """
        bin_widths = self.get_bin_widths(restrict_range=restrict_range)
        # Need to cast to magnitude to use isclose() with atol before Pint 0.21
        if not np.all(np.isclose(bin_widths.magnitude, bin_widths.magnitude[0],
                                 rtol=rtol, atol=atol)):
            raise AssertionError('Not all x-axis bins are the same width. '
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
                 x_tick_labels: Optional[XTickLabels] = None,
                 metadata: Optional[dict[str, int | str]] = None,
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
        # pylint: disable=import-outside-toplevel
        from .collections import Spectrum1DCollection
        spec_col = Spectrum1DCollection.from_spectra([self, other])
        return spec_col.sum()

    def _split_by_indices(self: T,
                          indices: Sequence[int] | np.ndarray,
                          ) -> list[T]:
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

    def to_dict(self) -> dict[str, Any]:
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
                     fmt: Optional[str | Sequence[str]] = None) -> None:
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
        # pylint: disable=import-outside-toplevel
        from .collections import Spectrum1DCollection
        spec = Spectrum1DCollection.from_spectra([self])
        spec.to_text_file(filename, fmt)

    @classmethod
    def from_dict(cls: type[T], d: dict[str, Any]) -> T:
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
    def from_castep_phonon_dos(cls: type[T], filename: str,
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

        return cls(ureg.Quantity(data['dos_bins'],
                                 units=data['dos_bins_unit']),
                   ureg.Quantity(data['dos'][element], units=data['dos_unit']),
                   metadata=metadata)

    @overload
    def broaden(self: T, x_width: Quantity,
                shape: KernelShape = 'gauss',
                method: Optional[Literal['convolve']] = None,
                width_convention: Literal['fwhm', 'std'] = 'fwhm',
                ) -> T: ...

    @overload
    def broaden(self: T, x_width: CallableQuantity,
                shape: KernelShape = 'gauss',
                method: Optional[Literal['convolve']] = None,
                width_lower_limit: Optional[Quantity] = None,
                width_convention: Literal['fwhm', 'std'] = 'fwhm',
                width_interpolation_error: float = 0.01,
                width_fit: ErrorFit = 'cheby-log',
                ) -> T: ...

    def broaden(self: T, x_width,
                shape='gauss',
                method=None,
                width_lower_limit=None,
                width_convention='fwhm',
                width_interpolation_error=0.01,
                width_fit='cheby-log',
                ) -> T:
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
                shape=shape, method=method, width_convention=width_convention)
            y_broadened = ureg.Quantity(y_broadened, units=self.y_data_unit)

        elif isinstance(x_width, Callable):
            self.assert_regular_bins(
                message=(
                    'Broadening with convolution requires a '
                    'regular sampling grid.'
                ),
                restrict_range=False,
            )
            y_broadened = variable_width_broadening(
                self.get_bin_edges(restrict_range=False),
                self.get_bin_centres(),
                x_width,
                (self.y_data * self.get_bin_widths(restrict_range=False)[0]),
                width_lower_limit=width_lower_limit,
                width_convention=width_convention,
                adaptive_error=width_interpolation_error,
                shape=shape,
                fit=width_fit,
            )
        else:
            msg = 'x_width must be a Quantity or Callable'
            raise TypeError(msg)

        new_spectrum = self.copy()
        new_spectrum.y_data = y_broadened
        return new_spectrum


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
                 x_tick_labels: Optional[XTickLabels] = None,
                 metadata: Optional[OneSpectrumMetadata] = None,
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
        """z-axis data with units"""
        return ureg.Quantity(
            self._z_data, self._internal_z_data_unit,
        ).to(self.z_data_unit, 'reciprocal_spectroscopy')

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
        super().__setattr__(name, value)

    def _split_by_indices(self,
                          indices: Sequence[int] | np.ndarray,
                          ) -> list[T]:
        """Split data along x-axis at given indices"""
        ranges = self._ranges_from_indices(indices)
        return [type(self)(self.x_data[x0:x1], self.y_data,
                           self.z_data[x0:x1, :],
                           x_tick_labels=self._cut_x_ticks(self.x_tick_labels,
                                                           x0, x1),
                           metadata=self.metadata)
                for x0, x1 in ranges]

    def broaden(self: T,
                x_width: Optional[Quantity | CallableQuantity] = None,
                y_width: Optional[Quantity | CallableQuantity] = None,
                shape: KernelShape = 'gauss',
                method: Optional[Literal['convolve']] = None,
                x_width_lower_limit: Quantity = None,
                y_width_lower_limit: Quantity = None,
                width_convention: Literal['fwhm', 'std'] = 'fwhm',
                width_interpolation_error: float = 0.01,
                width_fit: ErrorFit = 'cheby-log',
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
                self.assert_regular_bins(bin_ax='x', message=(
                    'Broadening by convolution may give incorrect results.'))
            except AssertionError as e:
                warnings.warn(str(e), UserWarning, stacklevel=1)
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
                                             method=method,
                                             width_convention=width_convention)
            spectrum = Spectrum2D(
                np.copy(self.x_data),
                np.copy(self.y_data),
                ureg.Quantity(z_broadened, units=self.z_data_unit),
                copy.copy(self.x_tick_labels),
                copy.deepcopy(self.metadata),
            )

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
            width_fit: ErrorFit = 'cheby-log',
    ) -> 'Spectrum2D':
        """
        Apply value-dependent Gaussian broadening to one axis of Spectrum2D
        """
        assert axis in ('x', 'y')

        bins = spectrum.get_bin_edges(bin_ax=axis, restrict_range=False)
        bin_widths = spectrum.get_bin_widths(bin_ax=axis, restrict_range=False)

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

    def get_bin_edges(
            self,
            bin_ax: Literal['x', 'y'] = 'x',
            *,
            restrict_range: bool = True,
    ) -> Quantity:
        """
        Get bin edges for the axis specified by bin_ax. If the size of bin_ax
        is one element larger than z_data along that axis, bin_ax is assumed to
        contain bin edges. If they are the same size, bin_ax is assumed to
        contain bin centres and a conversion is made.

        In this case, bin edges are assumed to be halfway between bin centres.

        Parameters
        ----------
        bin_ax
            The axis to get the bin edges for, 'x' or 'y'

        restrict_range
            If True (default), the bin edges will not go outside the existing
            data bounds so the first and last bins may be half-size. This may
            be desirable for plotting.  Otherwise, the outer bin edges will
            extend from the initial data range.
        """
        enum = {'x': 0, 'y': 1}
        bin_data = getattr(self, f'{bin_ax}_data')
        data_ax_len = self.z_data.shape[enum[bin_ax]]
        if self._is_bin_edge(data_ax_len, bin_data.shape[0]):
            return bin_data
        return self._bin_centres_to_edges(
            bin_data, restrict_range=restrict_range)

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
        return bin_data

    def get_bin_widths(
            self,
            bin_ax: Literal['x', 'y'] = 'x',
            *,
            restrict_range: bool = True,
    ) -> Quantity:
        """
        Get x-bin widths along specified axis

        Parameters
        ----------
        bin_ax
            Axis of interest, 'x' or 'y'

        restrict_range
            If True, bin edges will be clamped to the input data range; if
            False, they may be extrapolated beyond the initial range of bin
            centres.

            False is usually preferable, this default behaviour is for backward
            compatibility and will be removed in a future version.
        """
        bins = self.get_bin_edges(bin_ax, restrict_range=restrict_range)
        return np.diff(bins)

    def assert_regular_bins(  # pylint: disable=arguments-renamed
            self,
            bin_ax: Literal['x', 'y'],
            message: str = '',
            rtol: float = 1e-5,
            atol: float = 0.,
            restrict_range: bool = True,
    ) -> None:
        """Raise AssertionError if x-axis bins are not evenly spaced.

        Note that the positional arguments are different from
        Spectrum1D.assert_regular_bins: it is strongly recommended to only use
        keyword arguments with this method.

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

        restrict_range
            If True, bin edges will be clamped to the input data range; if
            False, they may be extrapolated beyond the initial range of bin
            centres.

            You should use the value which is consistent with calls to
            get_bin_widths() or get_bin_edges().
        """
        bin_widths = self.get_bin_widths(bin_ax, restrict_range=restrict_range)
        if not np.all(np.isclose(bin_widths, bin_widths[0],
                                 rtol=rtol, atol=atol)):
            raise AssertionError(
                f'Not all {bin_ax}-axis bins are the same width. ' + message)

    def to_dict(self) -> dict[str, Any]:
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
    def from_dict(cls: type[T], d: dict[str, Any]) -> T:
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
                                angle_range: tuple[float] = (0, 180.),
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
        msg = 'x_data needs to have wavevector units (i.e. 1/length)'
        raise ValueError(msg) from error
    try:
        (1 * spectrum.y_data.units).to('eV', 'spectroscopy')
    except DimensionalityError as error:
        msg = 'y_data needs to have energy (or wavenumber) units'
        raise ValueError(msg) from error

    momentum2_to_energy = 0.5 * (ureg('hbar^2 / neutron_mass')
                                 .to('meV angstrom^2'))

    if (e_i is None) == (e_f is None):
        msg = dedent_and_fill("""
            Exactly one of e_i and e_f should be set.
            (The other value will be derived from energy transfer).
            """)
        raise ValueError(msg)

    if e_i is None:
        # Indirect geometry: final energy is fixed, incident energy unlimited
        e_f = e_f.to('meV')
        e_i = spectrum.get_bin_centres(bin_ax='y').to('meV') + e_f
    elif e_f is None:
        # Direct geometry: incident energy is fixed, max energy transfer = e_i
        e_i = e_i.to('meV')
        e_f = e_i - spectrum.get_bin_centres(bin_ax='y').to('meV')

    k2_i = e_i / momentum2_to_energy
    k2_f = e_f / momentum2_to_energy

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
                           * (k2_i.units * k2_f.units)**0.5,
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


def _get_cos_range(angle_range: tuple[float]) -> tuple[float]:
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
    if shape != 'lorentz':
        msg = "Expected shape: 'lorentz'"
        raise ValueError(msg)

    dist = _lorentzian(x, xwidth)
    return dist / np.sum(dist)  # Naively normalise


def _get_dist_bins(bins: np.ndarray) -> np.ndarray:
    # Get bins with same spacing as original, centered on zero
    # and spanning at least twice as far; this ensures a point in bin can be
    # distributed all the way across bin range.
    bin_range = bins[-1] - bins[0]
    nbins = len(bins) * 2 + 1

    return np.linspace(-bin_range, bin_range, nbins, endpoint=True)


def _lorentzian(x: np.ndarray, gamma: float) -> np.ndarray:
    return gamma/(2*math.pi*(np.square(x) + (gamma/2)**2))
