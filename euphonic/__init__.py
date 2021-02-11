from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

import pint
from pint import UnitRegistry
from importlib_resources import files
from distutils.version import LooseVersion

# Create ureg here so it is only created once
if LooseVersion(pint.__version__) < LooseVersion('0.10'):
    # Bohr, unified_atomic_mass_unit not defined in pint 0.9, so load
    # pint 0.16.1 definition file
    ureg = UnitRegistry(str(files('euphonic.data') / 'default_en.txt'))
else:
    ureg = UnitRegistry()
ureg.enable_contexts('spectroscopy')
Quantity = ureg.Quantity

from .spectra import Spectrum1D, Spectrum1DCollection, Spectrum2D
from .crystal import Crystal
from .debye_waller import DebyeWaller
from .qpoint_frequencies import QpointFrequencies
from .structure_factor import StructureFactor
from .qpoint_phonon_modes import QpointPhononModes
from .force_constants import ForceConstants
