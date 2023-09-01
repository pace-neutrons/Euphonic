"""Matplotlib stylesheets for plot styling"""
from importlib_resources import files

base_style = files(__package__) / "base.mplstyle"
intensity_widget_style = files(__package__) / "intensity_widget.mplstyle"
