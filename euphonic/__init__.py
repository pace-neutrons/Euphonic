from pint import UnitRegistry

ureg = UnitRegistry()
# All values from CODATA 2014
ureg.define('rydberg = 13.605693009*eV = Ry')
ureg.define('bohr = 0.52917721067*angstrom = a_0')
ureg.define('electron_mass = 0.0005485799093*amu = e_mass')
