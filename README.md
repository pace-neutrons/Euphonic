# Euphonic
Euphonic is a Python package that can calculate phonon bandstructures and
inelastic neutron scattering intensities from modelling code outputs (e.g.
CASTEP). In addition, it can also do simple plotting, and can also plot
dispersion and density of states from existing CASTEP .bands or .phonon files.

## Installation
The easiest way to install the Euphonic package is using `pip`. First clone
this repository and cd into the top directory containing the `setup.py` script.

This package does all plotting with Matplotlib, to install Euphonic with the
optional Matplotlib dependency do:
```
pip install .[matplotlib]
```
Or to install for just your user:
```
pip install --user .[matplotlib]
```
If you only require the interpolation functionality, and don't need any of the
Matplotlib plotting routines just do:
```
pip install .
```

## Usage
For tutorials and more information, see the [Wiki](https://github.com/pace-neutrons/Euphonic/wiki)
