import numpy as np
from numpy.testing import assert_allclose
from pint import Quantity
import pytest

from euphonic import ureg
from euphonic.spectra import Spectrum1D, Spectrum1DCollection

def assert_quantity_allclose(q1, q2):
    assert str(q1.units) == str(q2.units)
    # assert q1.units == q2.units   # why doesn't this work?
    assert_allclose(q1.magnitude, q2.magnitude)


def sample_xy_data(n_spectra=1, n_energies=10, n_labels=2):
    x_data = Quantity(np.linspace(0, 1, n_energies), ureg('1/angstrom'))
    y_data = Quantity(np.random.random((n_spectra, n_energies)),
                      ureg('meV'))
    x_label_positions = np.random.choice(range(n_energies),
                                         size=n_labels, replace=False)
    x_tick_labels = [(i, str(i)) for i in x_label_positions]
    return (x_data, y_data, x_tick_labels)


@pytest.mark.unit
class TestSpectrum1DCollectionCreation:
    @pytest.mark.parametrize('sample_collection_data',
                             [tuple(list(sample_xy_data(n_spectra=2)) + [2]),
                              (Quantity(np.linspace(0, 2, 11),
                                        ureg('1/angstrom')),
                               Quantity(np.random.random((1, 11)),
                                        ureg('meV')),
                               None, 1)])
    def test_init(self, sample_collection_data):
        (x_data, y_data, x_tick_labels, length) = sample_collection_data

        bands = Spectrum1DCollection(x_data, y_data,
                                     x_tick_labels=x_tick_labels)

        assert isinstance(bands, Spectrum1DCollection)
        assert_quantity_allclose(bands.x_data, x_data)
        assert bands.x_tick_labels == x_tick_labels

        assert len(bands) == length
        assert len([spec for spec in bands]) == length

        for i, spec in enumerate(bands):
            assert isinstance(spec, Spectrum1D)
            assert_quantity_allclose(spec.x_data, x_data)
            assert_quantity_allclose(spec.y_data, y_data[i, :])
            assert spec.x_tick_labels == x_tick_labels

    def test_slice(self):
        x_data, y_data, x_tick_labels = sample_xy_data(n_spectra=4)
        bands = Spectrum1DCollection(x_data, y_data,
                                     x_tick_labels=x_tick_labels)

        sliced_collection = bands[1:3]
        assert isinstance(sliced_collection, Spectrum1DCollection)
        assert len(sliced_collection) == 2

        for i in range(2):
            assert_quantity_allclose(sliced_collection[i].x_data, x_data)
            assert sliced_collection[i].x_tick_labels == x_tick_labels
            assert_quantity_allclose(sliced_collection[i].y_data,
                                     bands[i + 1].y_data)
        
        negative_sliced_collection = bands[-3:]
        assert isinstance(negative_sliced_collection, Spectrum1DCollection)
        assert len(negative_sliced_collection) == 3
        
        for i in range(3):
            assert_quantity_allclose(negative_sliced_collection[i].x_data,
                                     x_data)
            assert negative_sliced_collection[i].x_tick_labels == x_tick_labels
            assert_quantity_allclose(negative_sliced_collection[i].y_data,
                                     bands[i + 1].y_data)

        with pytest.raises(TypeError):
            sliced_collection['1']

    def test_from_spectra(self):
        x_data, y_data, x_tick_labels = sample_xy_data(n_spectra=3)
        y_data_rows = [row for row in y_data]

        spectra_1d = [Spectrum1D(x_data, y, x_tick_labels)
                      for y in y_data_rows]

        spectrum_collection = Spectrum1DCollection.from_spectra(spectra_1d)

        for in_spec, out_spec in zip(spectra_1d, spectrum_collection):
            assert_quantity_allclose(in_spec.x_data, out_spec.x_data)
            assert in_spec.x_tick_labels == out_spec.x_tick_labels
            assert_quantity_allclose(in_spec.y_data, out_spec.y_data)

        with pytest.raises(IndexError):
            Spectrum1DCollection.from_spectra([])

    def test_split(self):
        x_data, y_data, x_tick_labels = sample_xy_data(n_spectra=4,
                                                       n_energies=10)
        bands = Spectrum1DCollection(x_data, y_data,
                                     x_tick_labels=x_tick_labels)
        split_bands = bands.split(indices=[3])
        for spectrum in split_bands:
            assert isinstance(spectrum, Spectrum1DCollection)

        assert split_bands[0].x_data_unit == x_data.units
        assert len(split_bands[0].x_data) == 3
        assert len(split_bands[1].x_data) == 7

        for i, spectrum in enumerate(split_bands[1]):
            assert_quantity_allclose(spectrum.y_data, y_data[i, 3:])
