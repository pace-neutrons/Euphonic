"""Unit tests for Spectrum2DCollection"""

# Stop the linter from complaining when pytest fixtures are used idiomatically
# pylint: disable=redefined-outer-name

from typing import Optional

import numpy as np
import pytest

from euphonic import Quantity, ureg
from euphonic.spectra import OneLineData, Spectrum2D, Spectrum2DCollection

from tests_and_analysis.test.utils import get_data_path
from .test_spectrum2d import check_spectrum2d, get_spectrum2d


def get_spectrum2dcollection_path(*subpaths):
    """Get Spectrum2DCollection reference data path"""
    return get_data_path('spectrum2dcollection', *subpaths)


def get_spectrum2dcollection(json_filename):
    """Get Spectrum2DCollection reference data object"""
    return Spectrum2DCollection.from_json_file(
        get_spectrum2dcollection_path(json_filename))


@pytest.fixture
def quartz_fuzzy_collection() -> Spectrum2DCollection:
    """Coarsely sampled quartz bands in a few directions"""
    return get_spectrum2dcollection("quartz_fuzzy_map.json")


@pytest.fixture
def quartz_fuzzy_items() -> list[Spectrum2D]:
    """Individual spectra corresponding to quartz_fuzzy_collection"""
    return [get_spectrum2d(f"quartz_fuzzy_map_{i}.json") for i in range(3)]

@pytest.fixture
def inconsistent_x_item() -> Spectrum2D:
    """Spectrum with different x values"""
    item = get_spectrum2d("quartz_fuzzy_map_0.json")
    item._x_data *= 2.
    return item

@pytest.fixture
def inconsistent_x_units_item():
    """Spectrum with different x units"""
    item = get_spectrum2d("quartz_fuzzy_map_0.json")
    item.x_data_unit = "1/bohr"
    return item

@pytest.fixture
def inconsistent_x_length_item():
    """Spectrum with different number of x values"""
    item = get_spectrum2d("quartz_fuzzy_map_0.json")
    item.x_data = item.x_data[:-2]
    item.z_data = item.z_data[:-2, :]
    return item

@pytest.fixture
def inconsistent_y_item():
    """Spectrum with different y values"""
    item = get_spectrum2d("quartz_fuzzy_map_0.json")
    item.y_data = item.y_data * 2.
    return item

def rand_spectrum2d(seed: int = 1,
                    x_bins: Optional[Quantity] = None,
                    y_bins: Optional[Quantity] = None,
                    metadata: Optional[OneLineData] = None) -> Spectrum2D:
    """Generate a Spectrum2D with random axis lengths, ranges, and metadata"""
    rng = np.random.default_rng(seed=seed)

    if x_bins is None:
        x_bins = np.linspace(*sorted([rng.random(), rng.random()]),
                             rng.integers(3, 10),
                             ) * ureg("1 / angstrom")
    if y_bins is None:
        y_bins = np.linspace(*sorted([rng.random(), rng.random()]),
                             rng.integers(3, 10)) * ureg("meV")
    if metadata is None:
        metadata = {"index": rng.integers(10),
                    "value": rng.random(),
                    "tag": "common"}

    spectrum = Spectrum2D(x_data=x_bins,
                          y_data=y_bins,
                          z_data=rng.random([len(x_bins) - 1, len(y_bins) - 1]
                                            ) * ureg("millibarn / meV"),
                          metadata=metadata)
    return spectrum


class TestSpectrum2DCollectionCreation:
    """Unit tests for Spectrum2DCollection constructors"""
    def test_init_from_numbers(self):
        """Construct Spectrum2DCollection with __init__()"""
        n_x = 10
        n_y = 20
        n_z = 5

        x_data = ureg.Quantity(np.linspace(0, 100, n_x), "1 / angstrom")
        y_data = ureg.Quantity(np.linspace(0, 2000, n_y), "meV")
        z_data = ureg.Quantity(np.random.random((n_z, n_x, n_y)), "1 / meV")

        metadata = {"flavour": "chocolate",
                    "line_data": [{"index": i} for i in range(n_z)]}

        x_tick_labels = [(0, "Start"), (n_x - 1, "END")]

        spectrum = Spectrum2DCollection(
            x_data, y_data, z_data,
            x_tick_labels=x_tick_labels, metadata=metadata)

        for attr, data in [("x_data", x_data),
                           ("y_data", y_data),
                           ("z_data", z_data)]:
            assert getattr(spectrum, attr).units == data.units
            np.testing.assert_allclose(getattr(spectrum, attr).magnitude,
                                       data.magnitude)

        assert spectrum.metadata == metadata

    def test_from_spectra(self, quartz_fuzzy_collection, quartz_fuzzy_items):
        """Use alternate constructor Spectrum2DCollection.from_spectra()"""
        collection = Spectrum2DCollection.from_spectra(quartz_fuzzy_items)
        ref_collection = quartz_fuzzy_collection

        for attr in ("x_data", "y_data", "z_data"):
            new, ref = getattr(collection, attr), getattr(ref_collection, attr)
            assert new.units == ref.units
            np.testing.assert_allclose(new.magnitude, ref.magnitude)

        if ref_collection.metadata is None:
            assert collection.metadata is None
        else:
            assert ref_collection.metadata == collection.metadata

    # pylint: disable=R0913  #  These fixtures are "too many arguments"
    def test_from_bad_spectra(
            self,
            quartz_fuzzy_items,
            inconsistent_x_item,
            inconsistent_x_length_item,
            inconsistent_x_units_item,
            inconsistent_y_item):
        """Spectrum2DCollection.from_spectra with inconsistent input"""

        with pytest.raises(ValueError):
            Spectrum2DCollection.from_spectra(
                quartz_fuzzy_items + [inconsistent_x_item]
            )

        with pytest.raises(ValueError):
            Spectrum2DCollection.from_spectra(
                quartz_fuzzy_items + [inconsistent_x_units_item]
            )

        with pytest.raises(ValueError):
            Spectrum2DCollection.from_spectra(
                quartz_fuzzy_items + [inconsistent_x_length_item]
            )

        with pytest.raises(ValueError):
            Spectrum2DCollection.from_spectra(
                quartz_fuzzy_items + [inconsistent_y_item]
            )

class TestSpectrum2DCollectionFunctionality:
    """Unit test indexing and methods of Spectrum2DCollection"""

    def test_indexing(self, quartz_fuzzy_collection, quartz_fuzzy_items):
        """Check indexing an element, slice and iteration

        - Individual index should yield corresponding Spectrum2D
        - A slice should yield a new Spectrum2DCollection
        - Iteration should yield a series of Spectrum2D

        """
        item_1 = quartz_fuzzy_collection[1]
        assert isinstance(item_1, Spectrum2D)
        check_spectrum2d(item_1, quartz_fuzzy_items[1])

        item_1_to_end = quartz_fuzzy_collection[1:]
        assert isinstance(item_1_to_end, Spectrum2DCollection)
        assert item_1_to_end != quartz_fuzzy_collection

        for item, ref in zip(item_1_to_end, quartz_fuzzy_items[1:]):
            assert isinstance(item, Spectrum2D)
            check_spectrum2d(item, ref)

    def test_collection_methods(self, quartz_fuzzy_collection):
        """Check methods from SpectrumCollectionMixin

        These are checked thoroughly for Spectrum1DCollection, but here we
        try to ensure the generic implementation works correctly in 2-D

        """

        total = quartz_fuzzy_collection.sum()
        assert isinstance(total, Spectrum2D)
        assert total.z_data[3, 3] == sum(spec.z_data[3, 3]
                                         for spec in quartz_fuzzy_collection)

        extended = quartz_fuzzy_collection + quartz_fuzzy_collection
        assert len(extended) == 2 * len(quartz_fuzzy_collection)
        np.testing.assert_allclose(extended.sum().z_data.magnitude,
                                   total.z_data.magnitude * 2)

        selection = quartz_fuzzy_collection.select(direction=2, common="yes")
        ref_item_2 = get_spectrum2d("quartz_fuzzy_map_2.json")
        check_spectrum2d(selection.sum(), ref_item_2)
