from importlib.resources import files

from . import data
from .version import __version__

import pint
from pint import UnitRegistry

# Create ureg here so it is only created once
ureg = UnitRegistry()

# Add reciprocal_spectroscopy environment used for tricky conversions
ureg.load_definitions(files(data) / "reciprocal_spectroscopy_definitions.txt")

ureg.enable_contexts('spectroscopy')
Quantity = ureg.Quantity

from .spectra import Spectrum1D, Spectrum1DCollection, Spectrum2D
from .crystal import Crystal
from .debye_waller import DebyeWaller
from .qpoint_frequencies import QpointFrequencies
from .structure_factor import StructureFactor
from .qpoint_phonon_modes import QpointPhononModes
from .force_constants import ForceConstants
