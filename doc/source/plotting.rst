.. _plotting:

========
Plotting
========

Plotting in Euphonic is contained in a separate ``euphonic.plot`` module,
and most plotting functions return a ``matplotlib.figure.Figure``, which can
be tweaked, then viewed with ``matplotlib.figure.Figure.show()``

.. contents:: :local:

.. _plotting-dispersion:

Plotting Dispersion
===================

Phonon dispersion can be plotted from any Euphonic object that contains
q-points and frequencies. For example, from :ref:`QpointPhononModes <qpoint-phonon-modes>`
using :py:meth:`QpointPhononModes.get_dispersion <euphonic.qpoint_phonon_modes.QpointPhononModes.get_dispersion>`
to convert the frequencies into a :ref:`Spectrum1DCollection` object, which
is plotted with :py:meth:`euphonic.plot.plot_1d`. Converting to
:ref:`Spectrum1DCollection` creates a defined x-axis for the plot. Extra
arguments get passed to :py:meth:`plot_1d <euphonic.plot.plot_1d>`, for
adding axis labels etc.:

.. code-block:: py

  from euphonic import QpointPhononModes
  from euphonic.plot import plot_1d

  phonons = QpointPhononModes.from_castep('quartz.phonon')
  bands = phonons.get_dispersion()  # type: Spectrum1DCollection

  fig = plot_1d(bands, y_label='Energy (meV)')
  fig.show()

Phonon dispersion plots often include large steps between discontinuous
regions in reciprocal space. In order to accommodate this, a single
:ref:`Spectrum1D` or :ref:`Spectrum1DCollection` can be split into
a list of spectra with the :py:meth:`euphonic.spectra.Spectrum1D.split`
method. :py:meth:`plot_1d <euphonic.plot.plot_1d>` will happily accept
this list as input, plotting each region to a series of proportionally-spaced
subplots.

A compact recipe to write a band-structure plot with discontinuities could be

.. code-block:: py

  from euphonic import QpointFrequencies
  from euphonic.plot import plot_1d

  phonon_frequencies = QpointFrequencies.from_castep('quartz.phonon')

  fig = plot_1d(phonon_frequencies.get_dispersion().split())
  fig.savefig('quartz-dispersion.pdf')


1D Plots
========

1D spectra are arranged in a matplotlib Figure with :py:meth:`euphonic.plot.plot_1d`.
For multiple lines on the same axes, use Spectrum1DCollection objects.
A sequence of Spectrum1D or Spectrum1DCollection objects will be interpreted
as a series of axis regions:

.. code-block:: py

  from euphonic import Spectrum1D, Spectrum1DCollection
  from euphonic.plot import plot_1d

  dos = Spectrum1D.from_json_file('dos.json')
  dos_broaden = Spectrum1D.from_json_file('dos_broaden.json')

  dos_collection = Spectrum1DCollection.from_spectra([dos, dos_broaden])

  fig = plot_1d(dos_collection, x_label='Energy (meV)', y_min=0,
                labels=['Density of states', 'Broadened'])
  fig.show()

Plotting to a specific axis
---------------------------

This can be used to have multiple subplots in the same figure, or to plot
multiple :ref:`Spectrum1D` objects on the same axis (for example if they
have different x_data values, so using a :ref:`Spectrum1DCollection` is not
possible). An example of plotting 2 DOS with different energy bins on the
same axis:

.. code-block:: py

  import matplotlib.pyplot as plt
  from euphonic import Spectrum1D
  from euphonic.plot import plot_1d, plot_1d_to_axis

  dos1 = Spectrum1D.from_json_file('dos_ebins1.json')
  dos2 = Spectrum1D.from_json_file('dos_ebins2.json')

  fig = plot_1d(dos1)
  ax = fig.get_axes()[0]
  plot_1d_to_axis(dos2, ax)
  fig.show()

An example of plotting 2 DOS on different axes on the same figure:

.. code-block:: py

  import matplotlib.pyplot as plt
  from euphonic import Spectrum1D
  from euphonic.plot import plot_1d_to_axis

  dos1 = Spectrum1D.from_json_file('dos_ebins1.json')
  dos2 = Spectrum1D.from_json_file('dos_ebins2.json')
  fig, axes = plt.subplots(1, 2)
  plot_1d_to_axis(dos1, axes[0])
  plot_1d_to_axis(dos2, axes[1])
  fig.show()

Docstrings
----------

.. autofunction:: euphonic.plot.plot_1d

.. autofunction:: euphonic.plot.plot_1d_to_axis

2D Plots
========

2D spectra are arranged in a matplotlib Figure with
:py:meth:`euphonic.plot.plot_2d`:

.. code-block:: py

  from euphonic import Spectrum2D
  from euphonic.plot import plot_2d

  sqw = Spectrum2D.from_json_file('sqw.json')
  fig = plot_2d(sqw, cmap='bone')
  fig.show()

Plotting to a specific axis
---------------------------

This can be used to multiple subplots showing :ref:`Spectrum2D` in the same
figure. A `matplotlib.colors.Normalize` object can also be used to ensure both
spectra are on the same colour scale. An example of this for 2 S(Q,w) is below:

.. code-block:: py

  import matplotlib.pyplot as plt
  from matplotlib.colors import Normalize
  from euphonic import Spectrum2D
  from euphonic.plot import plot_2d_to_axis

  sqw1 = Spectrum2D.from_json_file('sqw1.json')
  sqw2 = Spectrum2D.from_json_file('sqw2.json')
  norm = Normalize(vmin=0, vmax=1e-10)
  fig, axes = plt.subplots(1, 2)
  plot_2d_to_axis(sqw1, axes[0], norm=norm)
  plot_2d_to_axis(sqw2, axes[1], norm=norm)
  fig.show()

Docstrings
----------

.. autofunction:: euphonic.plot.plot_2d

.. autofunction:: euphonic.plot.plot_2d_to_axis

Styling
=======

To produce consistent and beautiful plots, it is recommended to use
`Matplotlib style sheets <https://matplotlib.org/stable/tutorials/introductory/customizing.html#temporary-styling>`_.
The cleanest way to apply this is using a context manager.
Within the indented block, a user-provided combination of style sheets
is applied to any new plots.
These can be built-in themes, file paths or parameter dictionaries,
e.g.:

.. code-block:: py

  import matplotlib.pyplot as plt
  from euphonic import Spectrum1D
  from euphonic.plot import plot_1d, plot_1d_to_axis

  dos = Spectrum1D.from_json_file('dos.json')

  with plt.style.context(['dark_background', {'lines.linewidth': 2.0}]):
      fig = plot_1d(dos)
  fig.show()

This approach is used in the Euphonic command-line tools; for more
information see :ref:`styling`. The CLI defaults can be imitated by
using the same style sheet ``euphonic.style.base_style``.
