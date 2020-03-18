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
except ImportError:
    pass

