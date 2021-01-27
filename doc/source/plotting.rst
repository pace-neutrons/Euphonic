.. _plotting:

========
Plotting
========

Plotting in Euphonic is contained in a separate ``euphonic.plot`` module,
and all plotting functions return a ``matplotlib.figure.Figure``, which can
be tweaked, then viewed with ``matplotlib.figure.Figure.show()``

.. contents:: :local:

Plotting Dispersion
===================

Phonon dispersion can be plotted from the q-points/frequencies in a
:ref:`QpointPhononModes <qpoint-phonon-modes>`
via a :ref:`Spectrum1DCollection` object, which is plotted with :py:meth:`euphonic.plot.plot_1d`.
Converting to `Spectrum1DCollection` creates a defined x-axis for the plot.
Extra arguments get passed to :py:meth:`plot_1d <euphonic.plot.plot_1d>`, for adding axis labels
etc.:

.. code-block:: py

  from euphonic import QpointPhononModes
  from euphonic.plot import plot_1d

  phonons = QpointPhononModes.from_castep('quartz.phonon')
  bands = phonons.get_dispersion()  # type: Spectrum1DCollection

  fig = plot_1d(bands, y_label='Energy (meV)')
  fig.show()

Phonon dispersion plots often include large steps between
discontinuous regions in reciprocal space.
In order to accommodate this, a single
Spectrum1D can be split into a list of spectra with the
:py:meth:`euphonic.spectra.Spectrum1D.split` method.
:py:meth:`plot_1d <euphonic.plot.plot_1d>` will happily accept
this list as input, plotting each region to a series of
proportionally-spaced subplots.

A compact recipe to write a band-structure plot with discontinuities could be

.. code-block:: py
  from euphonic import QpointPhononModes
  from euphonic.plot import plot_1d

  phonons = QpointPhononModes.from_castep('quartz.phonon')

  fig = plot_1d(phonons.get_dispersion().split())
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

Docstring
---------

.. autofunction:: euphonic.plot.plot_1d

2D Plots
========

2D spectra are arranged in a matplotlib Figure with
:py:meth:`euphonic.plot.plot_2d`:

.. code-block:: py

  from euphonic import Spectrum2D
  from euphonic.plot import plot_2d

  sqw = Spectrum2D.from_json_file('sqw.json')
  fig, ims = plot_2d(sqw, ratio=1.0)
  fig.show()

Docstring
---------

.. autofunction:: euphonic.plot.plot_2d
