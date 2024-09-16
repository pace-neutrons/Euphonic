from typing import Optional

import numpy as np
import pytest

from euphonic import Quantity, ureg
from euphonic.spectra import OneLineData, Spectrum2D, Spectrum2DCollection

# def check_spectrum2d(actual_spectrum2d, expected_spectrum2d, equal_nan=False,
#                      z_atol=np.finfo(np.float64).eps):


def rand_spectrum2d(seed: int = 1,
                    x_bins: Optional[Quantity] = None,
                    y_bins: Optional[Quantity] = None,
                    metadata: Optional[OneLineData] = None) -> Spectrum2D:
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
