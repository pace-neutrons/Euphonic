"""A consistent Pint unit registry for Euphonic Quantity objects.

They can also be imported directly 'from euphonic import Quantity, ureg';
this module exists to avoid import loops between the core modules of Euphonic.

This registry has the 'spectroscopy' context enabled by default to facilitate
conversion between energy and wavenumber. During tricky conversions between
reciprocals of energy and wavenumber, try enabling the
'reciprocal_spectroscopy' ('rs') context.

>>> y = 1. * ureg("1/meV")
>>> y.to("1/ (1/cm)", "rs")
<Quantity(0.123984198, 'centimeter')>

"""

from importlib.resources import files

from pint import UnitRegistry

from . import data

ureg = UnitRegistry()

ureg.load_definitions(files(data) / 'reciprocal_spectroscopy_definitions.txt')
ureg.enable_contexts('spectroscopy')
Quantity = ureg.Quantity

__all__ = ['Quantity', 'ureg']
