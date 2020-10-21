from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

import pint
from pint import UnitRegistry
from importlib_resources import files

# Create ureg here so it is only created once
pint_ver = [int(x) for x in pint.__version__.split('.')]
if pint_ver[0] == 0 and pint_ver[1] < 10:
    # Bohr, unified_atomic_mass_unit not defined in pint 0.9, so load
    # pint 0.16.1 definition file
    ureg = UnitRegistry(str(files('euphonic.data').joinpath('default_en.txt')))
else:
    ureg = UnitRegistry()
ureg.enable_contexts('spectroscopy')
Quantity = ureg.Quantity

from .spectra import Spectrum1D, Spectrum2D
from .crystal import Crystal
from .debye_waller import DebyeWaller
from .structure_factor import StructureFactor
from .qpoint_phonon_modes import QpointPhononModes
from .force_constants import ForceConstants
