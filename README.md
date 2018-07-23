# CastepPy
CastepPy is a Python package for plotting dispersion and density of states from
CASTEP .bands or .phonon files

## Installation
The easiest way to install the CastepPy package is using `pip`. First clone this
repository and cd into the top directory containing the `setup.py` script.

Then do:
```
pip install .
```

or to install for just your user:
```
pip install --user .
```

## Usage
The `dispersion.py` and `dos.py` scripts in the scripts directory can be used to easily plot dispersion and density of states respectively.
```
python dispersion.py <CASTEP filename>
```

There are many command line options available, to see these do:
```
python dispersion.py --help
```
Also see the Jupyter notebooks in the tutorials directory for more examples

#### Customising Plots
If you want to customise your plots beyond what is provided by the command line arguments, you can edit the `dispersion.py` and `dos.py` scripts. The `plot_dispersion` and `plot_dos` functions return a Matplotlib figure.Figure instance, which has many functions to alter the axis settings, plot title etc. For more details see the [Matplotlib docs](https://matplotlib.org/api/\_as_gen/matplotlib.figure.Figure.html#matplotlib.figure.Figure).
