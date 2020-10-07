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

@pytest.mark.unit
class TestSpectrum1DCollectionCreation:

    @pytest.mark.parametrize('sample_collection_data',
                             [(Quantity(np.linspace(0, 1, 10), ureg('1/angstrom')),
                               Quantity(np.random.random((2, 10)), ureg('meV')),
                               [(0, 'first label'), (4, 'second label')],
                               2),
                              (Quantity(np.linspace(0, 2, 11), ureg('1/angstrom')),
                               Quantity(np.random.random((1, 11)), ureg('meV')),
                               None,
                               1)])
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
