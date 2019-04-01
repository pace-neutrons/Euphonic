# SimPhony
SimPhony is a Python package which can plot dispersion and density of states from
existing CASTEP .bands or .phonon files. For phonons, it can also plot dispersion
and density of states at arbitrary q-points via interpolation.

## Installation
The easiest way to install the SimPhony package is using `pip`. First clone this
repository and cd into the top directory containing the `setup.py` script.

This package does all plotting with Matplotlib, to install SimPhony with the
optional Matplotlib dependency do:
```
pip install .[matplotlib]
```
Or to install for just your user:
```
pip install --user .[matplotlib]
```
If you only require the interpolation functionality, and don't need any of the Matplotlib plotting routines just do:
```
pip install .
```

## Usage
### Scripts
The `dispersion.py` and `dos.py` scripts in the scripts directory can be used to
easily plot dispersion and density of states respectively from an existing .phonon
or .bands file.
```
python dispersion.py <CASTEP filename>
```

There are many command line options available, to see these do:
```
python dispersion.py --help
```
Also see the Jupyter notebooks in the tutorials directory for more examples

##### Customising Plots
If you want to customise your plots beyond what is provided by the command line
arguments, you can edit the `dispersion.py` and `dos.py` scripts. The
`plot_dispersion` and `plot_dos` functions return a Matplotlib figure.Figure
instance, which has many functions to alter the axis settings, plot title etc.
For more details see the [Matplotlib docs](https://matplotlib.org/api/\_as_gen/matplotlib.figure.Figure.html#matplotlib.figure.Figure).

### Python API
For custom plots, or calculating dispersion/dos at arbitrary q-points, the Python
API can be used. For more detailed information, see the class/function docstrings.

##### Reading Data
There are 3 different data classes, BandsData, PhononData and InterpolationData for
reading .bands, .phonon and .castep_bin files respectively. This object needs to be
created with a seedname argument, and optionally a path argument. Upon creation, the
file will be read and the object will contain all the required data.
```
#!python

from simphony.data.bands import BandsData
from simphony.data.phonon import PhononData
from simphony.data.interpolation import InterpolationData

seedname = 'graphite'
path = 'data'
bdata = BandsData(seedname, path)
pdata = PhononData(seedname, path)
idata = InterpolationData(seedname, path)
```

##### Phonon Interpolation
If the .castep_bin file contains a force constants matrix calculated with the
supercell method, the phonon frequencies and eigenvectors can be calculated at
any q-point via interpolation.

First, build a (n, 3) array of the q-points that you want to calculate for (where
n = number of q-points), a recommended path can be generated with 
[SeeK-path](https://seekpath.readthedocs.io/en/latest/maindoc.html#), but can also
be done manually. Then pass it as an argument to the
InterpolationData.calculate_fine_phonons function. This function sets the
InterpolationData freqs and eigenvecs attributes, but also returns freqs and
eigenvecs so they may be stored elsewhere
```
#!python

from simphony.data.interpolation import InterpolationData
import seekpath as skp

seedname = 'graphite'
idata = InterpolationData(seedname)

numbers = [1, 1, 1, 1] # A list identifying unique ions in the unit cell
structure = (idata.cell_vec, idata.ion_r, numbers)
qpts = skp.get_explicit_k_path(structure)["explicit_kpoints_rel"]

freqs, eigenvecs = idata.calculate_fine_phonons(qpts)
```

##### Other calculations
You can perform a dynamical structure factor calculation
on a PhononData object or an InterpolationData object that
has the freqs and eigenvecs attributes set.
```
#!python

from simphony.data.interpolation import InterpolationData
from simphony.calculate.scattering import structure_factor

seedname = 'graphite'
scattering_lengths = {'C': 6.646}
n = 100
qpts = np.hstack((np.tile(0, (n, 1)), np.linspace(0, 0.5, n)[:, np.newaxis], np.tile(0, (n, 1))))

idata = InterpolationData(seedname)
idata.calculate_fine_phonons(qpts)
sf = structure_factor(idata, scattering_lengths)
```

##### Plotting
Any object that has the freqs attribute set can produce a dispersion plot, and
any object that has the dos and dos_bins attributes set (by the calculate_dos
function) can produce a DOS plot. The plotting functions return a return a
Matplotlib figure.Figure instance, which has many functions to alter the axis
settings, plot title etc. For more details see the [Matplotlib docs](https://matplotlib.org/api/\_as_gen/matplotlib.figure.Figure.html#matplotlib.figure.Figure).


Dispersion plotting is fairly straightforward. For phonons, the frequencies
can also be reordered to correctly join branches using the reorder_freqs
function:
```
#!python

from simphony.data.phonon import PhononData
from simphony.calculate.dispersion import reorder_freqs
from simphony.plot.dispersion import plot_dispersion

seedname = 'graphite'
pdata = PhononData(seedname)
reorder_freqs(pdata)
fig = plot_dispersion(pdata)
fig.show()
```


To plot DOS, the DOS must first be calculated with the calculate_dos
function, which works on any data object:

```
#!python

from simphony.data.bands import BandsData
from simphony.calculate.dos import calculate_dos
from simphony.plot.dos import plot_dos

seedname = 'graphite'
bdata = BandsData(seedname)

# Widths in the same units as frequencies, in the case of .bands eV
bin_width = 0.05
gauss_width = 10.0
calculate_dos(bdata, bin_width, gauss_width)

fig = plot_dos(bdata)
fig.show()
```
