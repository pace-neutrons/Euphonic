__version__ = '0.2.2'

try:
    # Create ureg here so it is only created once. However, this __init__.py
    # needs to be imported by setup.py to find __version__, so allow pint import
    # to fail as in this case pint might not be installed yet
    from pint import UnitRegistry
    ureg = UnitRegistry()
    ureg.define('@alias bohr = INTERNAL_LENGTH_UNIT')
    ureg.define('@alias electron_mass = INTERNAL_MASS_UNIT')
    ureg.define('@alias hartree = INTERNAL_ENERGY_UNIT')
    ureg.define('@alias elementary_charge = INTERNAL_CHARGE_UNIT')
    ureg.define('@alias K = INTERNAL_TEMPERATURE_UNIT')

    ureg.define('@alias angstrom = DEFAULT_LENGTH_UNIT')
    ureg.define('@alias amu = DEFAULT_MASS_UNIT')
    ureg.define('@alias eV = DEFAULT_ENERGY_UNIT')
    ureg.define('@alias K = DEFAULT_TEMPERATURE_UNIT')

    from .crystal import Crystal
    from .debye_waller import DebyeWaller
    from .structure_factor import StructureFactor
    from .qpoint_phonon_modes import QpointPhononModes
    from .force_constants import ForceConstants
except ImportError:
    pass
