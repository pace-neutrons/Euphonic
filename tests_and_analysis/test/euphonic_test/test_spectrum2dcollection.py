from typing import Optional

import numpy as np
import pytest

from euphonic import Quantity, ureg
from euphonic.spectra import OneLineData, Spectrum2D, Spectrum2DCollection

from tests_and_analysis.test.utils import get_data_path
from .test_spectrum2d import check_spectrum2d, get_spectrum2d


def get_spectrum2dcollection_path(*subpaths):
    return get_data_path('spectrum2dcollection', *subpaths)


def get_spectrum2dcollection(json_filename):
    return Spectrum2DCollection.from_json_file(
        get_spectrum2dcollection_path(json_filename))


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
    def test_init_from_numbers(self):
        """Construct Spectrum2DCollection with __init__()"""
        N_X = 10
        N_Y = 20
        N_Z = 5

        x_data = ureg.Quantity(np.linspace(0, 100, N_X), "1 / angstrom")
        y_data = ureg.Quantity(np.linspace(0, 2000, N_Y), "meV")
        z_data = ureg.Quantity(np.random.random((N_Z, N_X, N_Y)), "1 / meV")

        metadata = {"flavour": "chocolate",
                    "line_data": [{"index": i} for i in range(N_Z)]}

        x_tick_labels = [(0, "Start"), (N_X - 1, "END")]

        spectrum = Spectrum2DCollection(
            x_data, y_data, z_data,
            x_tick_labels=x_tick_labels, metadata=metadata)

        assert spectrum

    def test_init_from_spectra(self):
        """Construct collection from a series of Spectrum2D"""
        spec_2d = rand_spectrum2d(seed=1)
        spec_2d_consistent = rand_spectrum2d().copy()
        spec_2d_consistent._z_data *= 2
        spec_2d.metadata["index"] = 2

        spectrum = Spectrum2DCollection.from_spectra(
            [spec_2d, spec_2d_consistent])

        spec_2d_inconsistent = rand_spectrum2d(seed=2)
        with pytest.raises(ValueError):
            spectrum = Spectrum2DCollection.from_spectra(
                [spec_2d, spec_2d_inconsistent])
            assert spectrum

    def test_from_spectra(self):
        spectra = [get_spectrum2d(f"quartz_fuzzy_map_{i}.json")
                   for i in range(3)]
        collection = Spectrum2DCollection.from_spectra(spectra)

        ref_collection = get_spectrum2dcollection("quartz_fuzzy_map.json")

        for attr in ("x_data", "y_data", "z_data"):
            new, ref = getattr(collection, attr), getattr(ref_collection, attr)
            assert new.units == ref.units
            np.testing.assert_allclose(new, ref)

        if ref_collection.metadata is None:
            assert collection.metadata is None
        else:
            assert ref_collection.metadata == collection.metadata

    def test_indexing(self):
        """Check indexing an element, slice and iteration

        - Individual index should yield corresponding Spectrum2D
        - A slice should yield a new Spectrum2DCollection
        - Iteration should yield a series of Spectrum2D

        """
        # TODO move spectrum load to a common fixture

        spectra = [get_spectrum2d(f"quartz_fuzzy_map_{i}.json")
                   for i in range(3)]
        collection = get_spectrum2dcollection("quartz_fuzzy_map.json")

        item_1 = collection[1]
        assert isinstance(item_1, Spectrum2D)
        check_spectrum2d(item_1, spectra[1])

        item_1_to_end = collection[1:]
        assert isinstance(item_1_to_end, Spectrum2DCollection)
        assert item_1_to_end != collection

        for item, ref in zip(item_1_to_end, spectra[1:]):
            assert isinstance(item, Spectrum2D)
            check_spectrum2d(item, ref)
