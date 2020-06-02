from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

# Create ureg here so it is only created once
from pint import UnitRegistry
ureg = UnitRegistry()
ureg.enable_contexts('spectroscopy')
ureg.define('@alias bohr = INTERNAL_LENGTH_UNIT')
ureg.define('@alias electron_mass = INTERNAL_MASS_UNIT')
ureg.define('@alias hartree = INTERNAL_ENERGY_UNIT')
ureg.define('@alias elementary_charge = INTERNAL_CHARGE_UNIT')
ureg.define('@alias K = INTERNAL_TEMPERATURE_UNIT')

ureg.define('@alias angstrom = DEFAULT_LENGTH_UNIT')
ureg.define('@alias amu = DEFAULT_MASS_UNIT')
ureg.define('@alias eV = DEFAULT_ENERGY_UNIT')
ureg.define('@alias K = DEFAULT_TEMPERATURE_UNIT')

from .spectra import Spectrum1D, Spectrum2D
from .crystal import Crystal
from .debye_waller import DebyeWaller
from .structure_factor import StructureFactor
from .qpoint_phonon_modes import QpointPhononModes
from .force_constants import ForceConstants
