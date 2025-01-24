"""Spectrum Collection classes"""
# pylint: disable=no-member

from abc import ABC, abstractmethod
import collections
import copy
from functools import partial, reduce
from itertools import product, repeat

import json
from numbers import Integral
from operator import contains
from typing import (Any, Callable, Dict, Generator, List, Literal, Optional,
                    overload, Sequence, TypeVar, Union, Type)
from typing_extensions import Self

from pint import Quantity
import numpy as np
from toolz.dicttoolz import keyfilter, valmap
from toolz.functoolz import complement
from toolz.itertoolz import groupby, pluck

from euphonic import ureg, __version__
from euphonic.broadening import ErrorFit, KernelShape
from euphonic.io import _obj_to_dict, _process_dict
from euphonic.readers.castep import read_phonon_dos_data
from euphonic.validate import _check_constructor_inputs

from .base import Spectrum, Spectrum1D, Spectrum2D
from .base import CallableQuantity, XTickLabels
from .base import OneSpectrumMetadata as OneLineData


LineData = Sequence[OneLineData]
Metadata = Dict[str, Union[str, int, LineData]]


class SpectrumCollectionMixin(ABC):
    """Help a collection of spectra work with "line_data" metadata file

    This is a Mixin to be inherited by Spectrum collection classes

    To avoid redundancy, spectrum collections store metadata in the form

    {"key1": value1, "key2", value2, "line_data": [{"key3": value3, ...},
                                                   {"key4": value4, ...}...]}

    - It is not guaranteed that all "lines" carry the same keys
    - No key should appear at both top-level and in line-data; any key-value
      pair at top level is assumed to apply to all lines
    - "lines" can actually correspond to N-D spectra, the notation was devised
      for multi-line plots of Spectrum1DCollection and then applied to other
      purposes.

    The _spectrum_axis class attribute determines which axis property contains
    the spectral data, and should be set by subclasses (i.e. to "y" or "z" for
    1D or 2D).
    """

    # Subclasses must define which axis contains the spectral data for
    # purposes of splitting, indexing, etc.
    # Python doesn't support abstract class attributes so we define a default
    # value, ensuring _something_ was set.
    _bin_axes = ("x",)
    _spectrum_axis = "y"
    _item_type = Spectrum1D

    # Define some private methods which wrap this information into useful forms
    @classmethod
    def _spectrum_data_name(cls) -> str:
        return f"{cls._spectrum_axis}_data"

    @classmethod
    def _raw_spectrum_data_name(cls) -> str:
        return f"_{cls._spectrum_axis}_data"

    def _get_spectrum_data(self) -> Quantity:
        return getattr(self, self._spectrum_data_name())

    def _get_raw_spectrum_data(self) -> np.ndarray:
        return getattr(self, self._raw_spectrum_data_name())

    def _set_spectrum_data(self, data: Quantity) -> None:
        setattr(self, self._spectrum_data_name(), data)

    def _set_raw_spectrum_data(self, data: np.ndarray) -> None:
        setattr(self, self._raw_spectrum_data_name(), data)

    def _get_spectrum_data_unit(self) -> str:
        return getattr(self, f"{self._spectrum_data_name()}_unit")

    def _get_internal_spectrum_data_unit(self) -> str:
        return getattr(self, f"_internal_{self._spectrum_data_name()}_unit")

    @property
    def _bin_axis_names(self) -> list[str]:
        return [f"{axis}_data" for axis in self._bin_axes]

    @classmethod
    def _get_item_data(cls, item: Spectrum) -> Quantity:
        return getattr(item, f"{cls._spectrum_axis}_data")

    @classmethod
    def _get_item_raw_data(cls, item: Spectrum) -> np.ndarray:
        return getattr(item, f"_{cls._spectrum_axis}_data")

    @classmethod
    def _get_item_data_unit(cls, item: Spectrum) -> str:
        return getattr(item, f"{cls._spectrum_axis}_data_unit")

    def sum(self) -> Spectrum:
        """
        Sum collection to a single spectrum

        Returns
        -------
        summed_spectrum
            A single combined spectrum from all items in collection. Any
            metadata in 'line_data' not common across all spectra will be
            discarded
        """
        metadata = copy.deepcopy(self.metadata)
        metadata.pop('line_data', None)
        metadata.update(self._tidy_metadata())
        summed_s_data = ureg.Quantity(
            np.sum(self._get_raw_spectrum_data(), axis=0),
            units=self._get_internal_spectrum_data_unit()
        ).to(self._get_spectrum_data_unit())

        bin_kwargs = {axis_name: getattr(self, axis_name)
                      for axis_name in self._bin_axis_names}

        return self._item_type(
            **bin_kwargs,
            **{self._spectrum_data_name(): summed_s_data},
            x_tick_labels=copy.copy(self.x_tick_labels),
            metadata=metadata
        )

    # Required methods
    @classmethod
    @abstractmethod
    def from_spectra(
            cls, spectra: Sequence[Spectrum], *, unsafe: bool = False
    ) -> Self:
        """Construct spectrum collection from a sequence of components

        If 'unsafe', some consistency checks may be skipped to improve
        performance; this should only be used when e.g. combining data iterated
        by another Spectrum collection

        """

    # Mixin methods
    def __len__(self):
        return self._get_raw_spectrum_data().shape[0]

    @overload
    def __getitem__(self, item: int) -> Spectrum: ...

    @overload  # noqa: F811
    def __getitem__(self, item: slice) -> Self: ...

    @overload  # noqa: F811
    def __getitem__(self, item: Union[Sequence[int], np.ndarray]) -> Self: ...

    def __getitem__(
            self, item: Union[Integral, slice, Sequence[Integral], np.ndarray]
    ):  # noqa: F811
        self._validate_item(item)

        if isinstance(item, Integral):
            spectrum = self._item_type.__new__(self._item_type)
        else:
            # Pylint miscounts arguments when we call this staticmethod
            spectrum = self.__new__(type(self))  # pylint: disable=E1120

        self._set_item_data(spectrum, item)

        spectrum.x_tick_labels = self.x_tick_labels
        spectrum.metadata = self._get_item_metadata(item)

        return spectrum

    def _set_item_data(
            self,
            spectrum: Spectrum,
            item: Union[Integral, slice, Sequence[Integral], np.ndarray]
    ) -> None:
        """Write axis and spectrum data from self to Spectrum

        This is intended to set attributes on a 'bare' Spectrum created with
        __new__() from the parent SpectrumCollection.
        """

        for axis in self._bin_axes:
            for prop in ("_{}_data", "_internal_{}_data_unit", "{}_data_unit"):
                name = prop.format(axis)
                setattr(spectrum, name, copy.copy(getattr(self, name)))

        setattr(spectrum, self._raw_spectrum_data_name(),
                self._get_raw_spectrum_data()[item, :].copy())

        setattr(spectrum, f"_internal_{self._spectrum_data_name()}_unit",
                self._get_internal_spectrum_data_unit())
        setattr(spectrum, f"{self._spectrum_data_name()}_unit",
                self._get_spectrum_data_unit())

    def _validate_item(self, item: Integral | slice | Sequence[Integral] | np.ndarray
                       ) -> None:
        """Raise Error if index has inappropriate typing/ranges

        Raises
        ------
        IndexError
            Slice is not compatible with size of collection

        TypeError
            item specification does not have acceptable type; e.g. a sequence
            of float or bool was provided when ints are needed.

        """
        if isinstance(item, Integral):
            return
        if isinstance(item, slice):
            if (item.stop is not None) and (item.stop >= len(self)):
                raise IndexError(f'index "{item.stop}" out of range')
            return

        if not all(isinstance(i, Integral) for i in item):
            raise TypeError(
                f'Index "{item}" should be an integer, slice '
                f'or sequence of ints')

    @overload
    def _get_item_metadata(self, item: Integral) -> OneLineData:
        """Get a single metadata item with no line_data"""

    @overload
    def _get_item_metadata(self, item: slice | Sequence[Integral] | np.ndarray
                           ) -> Metadata:  # noqa: F811
        """Get a metadata collection (may include line_data)"""

    def _get_item_metadata(self, item):  # noqa: F811
        """Produce appropriate metadata for __getitem__"""
        metadata_lines = list(self.iter_metadata())

        if isinstance(item, Integral):
            return metadata_lines[item]
        if isinstance(item, slice):
            return self._combine_metadata(metadata_lines[item])
        # Item must be some kind of integer sequence
        return self._combine_metadata([metadata_lines[i] for i in item])

    def copy(self) -> Self:
        """Get an independent copy of spectrum"""
        return self._item_type.copy(self)

    def __add__(self, other: Self) -> Self:
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

    def iter_metadata(self) -> Generator[OneLineData, None, None]:
        """Iterate over metadata dicts of individual spectra from collection"""
        common_metadata = {key: value for key, value in self.metadata.items()
                           if key != "line_data"}

        line_data = self.metadata.get("line_data")
        if line_data is None:
            line_data = repeat({}, len(self._get_raw_spectrum_data()))

        for one_line_data in line_data:
            yield common_metadata | one_line_data

    def _select_indices(self, **select_key_values) -> list[int]:
        """Get indices of items that match metadata query

        The target key-value pairs are a subset of the matching data, e.g.

        self._select_indices(species="Na", weight="coherent")

        will match metadata rows

        {"species": "Na", "weight": "coherent"}

        and

        {"species": "Na", "weight": "coherent", "mass": "22.9898"}

        but not

        {"species": "Na"}

        or

        {"species": "K", "weight": "coherent"}
        """
        required_metadata = select_key_values.items()
        indices = [i for i, row in enumerate(self.iter_metadata())
                   if required_metadata <= row.items()]
        return indices

    def select(self, **select_key_values: Union[
            str, int, Sequence[str], Sequence[int]]) -> Self:
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
        # Convert all items to sequences of possibilities
        def ensure_sequence(value: int | str | Sequence[int | str]
                            ) -> Sequence[int | str]:
            return (value,) if isinstance(value, (int, str)) else value

        select_key_values = valmap(ensure_sequence, select_key_values)

        # Collect indices that match each combination of values
        selected_indices = []
        for value_combination in product(*select_key_values.values()):
            selection = dict(zip(select_key_values.keys(), value_combination))
            selected_indices.extend(self._select_indices(**selection))

        if not selected_indices:
            raise ValueError(f'No spectra found with matching metadata '
                             f'for {select_key_values}')

        return self[selected_indices]

    @staticmethod
    def _combine_metadata(all_metadata: LineData) -> Metadata:
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
            assert 'line_data' not in metadata

        # Combine key-value pairs common to *all* metadata lines into new dict
        common_metadata = dict(
            reduce(set.intersection,
                   (set(metadata.items()) for metadata in all_metadata)))

        # Put all other per-spectrum metadata in line_data
        is_common = partial(contains, common_metadata)
        line_data = [keyfilter(complement(is_common), one_line_data)
                     for one_line_data in all_metadata]

        if any(line_data):
            return common_metadata | {'line_data': line_data}

        return common_metadata

    def _tidy_metadata(self) -> Metadata:
        """
        For a metadata dictionary, combines all common key/value
        pairs in 'line_data' and puts them in a top-level dictionary.
        """
        line_data = self.metadata.get("line_data", [{}] * len(self))
        combined_line_data = self._combine_metadata(line_data)
        combined_line_data.pop("line_data", None)
        return combined_line_data

    def _check_metadata(self) -> None:
        """Check self.metadata['line_data'] is consistent with collection size

        Raises
        ------
        ValueError
            Metadata contains 'line_data' of incorrect length

        """
        if 'line_data' in self.metadata:
            collection_size = len(self._get_raw_spectrum_data())
            n_lines = len(self.metadata['line_data'])

            if n_lines != collection_size:
                raise ValueError(
                    f'{self._spectrum_data_name()} contains {collection_size} '
                    f'spectra, but metadata["line_data"] contains '
                    f'{n_lines} entries')

    def group_by(self, *line_data_keys: str) -> Self:
        """
        Group and sum elements of spectral data according to the values
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
        def get_key_items(enumerated_metadata: tuple[int, OneLineData]
                          ) -> tuple[str | int, ...]:
            """Get sort keys from an item of enumerated input to groupby

            e.g. with line_data_keys=("a", "b")

              (0, {"a": 4, "d": 5}) --> (4, None)
            """
            return tuple(enumerated_metadata[1].get(item, None)
                         for item in line_data_keys)

        # First element of each tuple is the index
        indices = partial(pluck, 0)

        groups = groupby(get_key_items, enumerate(self.iter_metadata()))

        return self.from_spectra([self[list(indices(group))].sum()
                                  for group in groups.values()])

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a dictionary consistent with from_dict()

        Returns
        -------
        dict
        """
        attrs = [*self._bin_axis_names,
                 self._spectrum_data_name(),
                 'x_tick_labels',
                 'metadata']

        return _obj_to_dict(self, attrs)

    @classmethod
    def from_dict(cls: Self, d: dict) -> Self:
        """Initialise a Spectrum Collection object from dict"""
        data_keys = [f"{dim}_data" for dim in cls._bin_axes]
        data_keys.append(cls._spectrum_data_name())

        d = _process_dict(d,
                          quantities=data_keys,
                          optional=['x_tick_labels', 'metadata'])

        data_args = [d[key] for key in data_keys]
        return cls(*data_args,
                   x_tick_labels=d['x_tick_labels'],
                   metadata=d['metadata'])

    @classmethod
    def _item_type_check(cls, spectrum) -> None:
        if not isinstance(spectrum, cls._item_type):
            raise TypeError(
                f"Item is not of type {cls._item_type.__name__}.")


class Spectrum1DCollection(SpectrumCollectionMixin,
                           Spectrum,
                           collections.abc.Sequence):
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

    # Private attributes used by SpectrumCollectionMixin
    _spectrum_axis = "y"
    _item_type = Spectrum1D

    def __init__(
            self, x_data: Quantity, y_data: Quantity,
            x_tick_labels: Optional[XTickLabels] = None,
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

        self.metadata = metadata if metadata is not None else {}
        self._check_metadata()

    def _split_by_indices(self,
                          indices: Union[Sequence[int], np.ndarray]
                          ) -> List[Self]:
        """Split data along x-axis at given indices"""

        ranges = self._ranges_from_indices(indices)

        return [type(self)(self.x_data[x0:x1], self.y_data[:, x0:x1],
                           x_tick_labels=self._cut_x_ticks(self.x_tick_labels,
                                                           x0, x1),
                           metadata=self.metadata)
                for x0, x1 in ranges]

    @staticmethod
    def _from_spectra_data_check(spectrum, x_data, y_data_units, x_tick_labels
                                 ) -> None:
        if (spectrum.y_data.units != y_data_units
                or spectrum.x_data.units != x_data.units):
            raise ValueError("Spectrum units in sequence are inconsistent")
        if not np.allclose(spectrum.x_data, x_data):
            raise ValueError("x_data in sequence are inconsistent")
        if spectrum.x_tick_labels != x_tick_labels:
            raise ValueError("x_tick_labels in sequence are inconsistent")

    @classmethod
    def from_spectra(
            cls: Self, spectra: Sequence[Spectrum1D], *, unsafe: bool = False
    ) -> Self:
        """Combine Spectrum1D to produce a new collection

        x_bins and x_tick_labels must be consistent (in magnitude and units)
        across the input spectra. If 'unsafe', this will not be checked.

        """
        if len(spectra) < 1:
            raise IndexError("At least one spectrum is needed for collection")

        cls._item_type_check(spectra[0])
        x_data = spectra[0].x_data
        x_tick_labels = spectra[0].x_tick_labels
        y_data_length = len(spectra[0].y_data)
        y_data_magnitude = np.empty((len(spectra), y_data_length))
        y_data_magnitude[0, :] = spectra[0].y_data.magnitude
        y_data_units = spectra[0].y_data.units

        for i, spectrum in enumerate(spectra[1:]):
            if not unsafe:
                cls._item_type_check(spectrum)
                cls._from_spectra_data_check(
                    spectrum, x_data, y_data_units, x_tick_labels)

            y_data_magnitude[i + 1, :] = spectrum.y_data.magnitude

        metadata = cls._combine_metadata([spec.metadata for spec in spectra])
        y_data = ureg.Quantity(y_data_magnitude, y_data_units)
        return cls(x_data, y_data, x_tick_labels=x_tick_labels,
                   metadata=metadata)

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
        dos_size = len(next(iter(data['dos'].values())))
        y_data = np.zeros((n_spectra, dos_size))
        metadata = {'line_data': [{} for x in range(n_spectra)]}

        for i, (species, dos_data) in enumerate(data['dos'].items()):
            y_data[i] = dos_data
            if species != 'Total':
                metadata['line_data'][i]['species'] = species
            metadata['line_data'][i]['label'] = species
        return Spectrum1DCollection(
            ureg.Quantity(data['dos_bins'], units=data['dos_bins_unit']),
            ureg.Quantity(y_data, units=data['dos_unit']),
            metadata=metadata)

    @overload
    def broaden(self: T, x_width: Quantity,
                shape: KernelShape = 'gauss',
                method: Optional[Literal['convolve']] = None
                ) -> T: ...

    @overload
    def broaden(self: T, x_width: CallableQuantity,
                shape: KernelShape = 'gauss',
                method: Optional[Literal['convolve']] = None,
                width_lower_limit: Optional[Quantity] = None,
                width_convention: Literal['fwhm', 'std'] = 'fwhm',
                width_interpolation_error: float = 0.01,
                width_fit: ErrorFit = 'cheby-log'
                ) -> T: ...  # noqa: F811

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
                    method=method, width_convention=width_convention)

            new_spectrum = self.copy()
            new_spectrum.y_data = ureg.Quantity(y_broadened, units=self.y_data_unit)
            return new_spectrum

        if isinstance(x_width, Callable):
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

        raise TypeError("x_width must be a Quantity or Callable")

    @classmethod
    def from_dict(cls: Self, d: dict) -> Self:
        """
        Convert a dictionary to a Spectrum Collection object

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
        return super().from_dict(d)


class Spectrum2DCollection(SpectrumCollectionMixin,
                           Spectrum,
                           collections.abc.Sequence):
    """A collection of Spectrum2D with common x_data, y_data and x_tick_labels

    Intended for convenient storage of contributions to spectral maps such as
    S(Q,w). This object can be indexed or iterated to obtain individual
    Spectrum2D.

    Attributes
    ----------
    x_data
        Shape (n_x_data,) or (n_x_data + 1,) float Quantity. The x_data
        points (if size == (n_x_data,)) or x_data bin edges (if size
        == (n_x_data + 1,))
    y_data
        Shape (n_y_data,) or (n_y_data + 1,) float Quantity. The y_data
        points (if size == (n_y_data,)) or y_data bin edges (if size
        == (n_y_data + 1,))
    z_data
        Shape (n_entries, n_x_data, n_y_data) float Quantity. The spectral data
        in x and y, indexed over components
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

    # Private attributes used by SpectrumCollectionMixin
    _bin_axes = ("x", "y")
    _spectrum_axis = "z"
    _item_type = Spectrum2D

    def __init__(
            self, x_data: Quantity, y_data: Quantity, z_data: Quantity,
            x_tick_labels: Optional[XTickLabels] = None,
            metadata: Optional[Metadata] = None
    ) -> None:
        _check_constructor_inputs(
            [z_data, x_tick_labels, metadata],
            [Quantity, [list, type(None)], [dict, type(None)]],
            [(-1, -1, -1), (), ()],
            ['z_data', 'x_tick_labels', 'metadata'])
        # First axis corresponds to spectra in collection
        _, nx, ny = z_data.shape
        _check_constructor_inputs(
            [x_data, y_data],
            [Quantity, Quantity],
            [[(nx,), (nx + 1,)], [(ny,), (ny + 1,)]],
            ['x_data', 'y_data'])

        self._set_data(x_data, 'x')
        self._set_data(y_data, 'y')
        self.x_tick_labels = x_tick_labels
        self._set_data(z_data, 'z')

        self.metadata = metadata if metadata is not None else {}
        self._check_metadata()

    def _split_by_indices(self, indices: Sequence[int] | np.ndarray
                          ) -> List[Self]:
        """Split data along x axis at given indices"""
        ranges = self._ranges_from_indices(indices)
        return [type(self)(self.x_data[x0:x1],
                           self.y_data,
                           self.z_data[:, x0:x1, :],
                           x_tick_labels=self._cut_x_ticks(
                               self.x_tick_labels, x0, x1),
                           metadata=self.metadata)
                for x0, x1 in ranges]

    @property
    def z_data(self) -> Quantity:
        """intensity data"""
        return ureg.Quantity(
            self._z_data, self._internal_z_data_unit
        ).to(self.z_data_unit, "reciprocal_spectroscopy")

    @z_data.setter
    def z_data(self, value: Quantity) -> None:
        self.z_data_unit = str(value.units)
        self._z_data = value.to(self._internal_z_data_unit).magnitude

    @staticmethod
    def _from_spectra_data_check(
            spectrum_0_data_units, spectrum_i_data_units,
            x_tick_labels, spectrum_i_x_tick_labels
    ) -> None:
        """Check spectrum data units and x_tick_labels are consistent"""
        if spectrum_0_data_units != spectrum_i_data_units:
            raise ValueError("Spectrum units in sequence are inconsistent")
        if x_tick_labels != spectrum_i_x_tick_labels:
            raise ValueError("x_tick_labels in sequence are inconsistent")

    @staticmethod
    def _from_spectra_bins_check(ref_bins_data: dict[str, Quantity],
                                 spectrum: Spectrum2D) -> None:
        """Check bin values and units match between spectra in new collection

        Args:
            ref_bins_data: ref axis names and values in format
                ``{"x_data": Quantity(...), }``
            spectrum: Item from sequence to compare with ref
        """
        for key, ref_bins in ref_bins_data.items():
            item_bins = getattr(spectrum, key)
            if not np.allclose(item_bins.magnitude, ref_bins.magnitude):
                raise ValueError("Bins in sequence are inconsistent")
            if item_bins.units != ref_bins.units:
                raise ValueError("Bin units in sequence are inconsistent")

    @classmethod
    def from_spectra(
            cls, spectra: Sequence[Spectrum2D], *, unsafe: bool = False
    ) -> Self:
        """Combine Spectrum2D to produce a new collection

        bins and x_tick_labels must be consistent (in magnitude and units)
        across the input spectra. If 'unsafe', this will not be checked.

        """

        if len(spectra) < 1:
            raise IndexError("At least one spectrum is needed for collection")

        cls._item_type_check(spectra[0])
        bins_data = {
            f"{ax}_data": getattr(spectra[0], f"{ax}_data")
            for ax in cls._bin_axes
        }
        x_tick_labels = spectra[0].x_tick_labels

        spectrum_0_data = cls._get_item_data(spectra[0])
        spectrum_data_shape = spectrum_0_data.shape
        spectrum_data_magnitude = np.empty(
            (len(spectra), *spectrum_data_shape))
        spectrum_data_magnitude[0, :, :] = spectrum_0_data.magnitude
        spectrum_data_units = spectrum_0_data.units

        for i, spectrum in enumerate(spectra[1:], start=1):
            spectrum_i_raw_data = cls._get_item_raw_data(spectrum)
            spectrum_i_data_units = cls._get_item_data_unit(spectrum)

            if not unsafe:
                cls._item_type_check(spectrum)
                cls._from_spectra_data_check(
                    spectrum_data_units, spectrum_i_data_units,
                    x_tick_labels, spectrum.x_tick_labels)
                cls._from_spectra_bins_check(bins_data, spectrum)

            spectrum_data_magnitude[i, :, :] = spectrum_i_raw_data

        metadata = cls._combine_metadata([spec.metadata for spec in spectra])
        spectrum_data = ureg.Quantity(spectrum_data_magnitude,
                                      spectrum_data_units)
        return cls(**bins_data,
                   **{f"{cls._spectrum_axis}_data": spectrum_data},
                   x_tick_labels=x_tick_labels,
                   metadata=metadata)
