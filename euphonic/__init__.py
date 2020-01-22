__version__ = '0.2.0'

try:
    # Create ureg here so it is only created once. However, this __init__.py
    # needs to be imported by setup.py to find __version__, so allow pint import
    # to fail as in this case pint might not be installed yet
    from pint import UnitRegistry

    ureg = UnitRegistry()
    # All values from CODATA 2014
    ureg.define('rydberg = 13.605693009*eV = Ry')
    ureg.define('bohr = 0.52917721067*angstrom = a_0')
    ureg.define('electron_mass = 0.0005485799093*amu = e_mass')
except ImportError:
    pass

