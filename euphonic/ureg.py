from importlib.resources import files

from pint import UnitRegistry

from . import data

# Central unit registry for all things Euphonic;
# this is also exported to euphonic.__init__
ureg = UnitRegistry()

# Add reciprocal_spectroscopy environment used for tricky conversions
ureg.load_definitions(files(data) / "reciprocal_spectroscopy_definitions.txt")

ureg.enable_contexts('spectroscopy')
Quantity = ureg.Quantity

__all__ = ["ureg", "Quantity"]
