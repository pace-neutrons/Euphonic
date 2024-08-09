import numpy as np

from euphonic import ureg
from euphonic.spectra import Spectrum2DCollection

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

