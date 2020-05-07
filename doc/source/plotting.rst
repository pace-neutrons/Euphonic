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

Phonon dispersion can be plotted straight from the q-points/frequencies in a
:ref:`QpointPhononModes` object with
:py:meth:`euphonic.plot.plot_dispersion`. Extra arguments get passed to
:py:meth:`plot_1d <euphonic.plot.plot_dispersion>`, for adding axis labels
etc.:

.. code-block:: py

  from euphonic import QpointPhononModes
  from euphonic.plot import plot_dispersion

  phonons = QpointPhononModes.from_castep('quartz.phonon')
  fig = plot_dispersion(phonons, y_label='Energy (meV)')
  fig.show()

Docstring
---------

.. autofunction:: euphonic.plot.plot_dispersion

1D Plots
========

1D spectra are plotted with :py:meth:`euphonic.plot.plot_1d`. Multiple 1D
spectra can be plotted on the same axis by passing a list:

.. code-block:: py

  from euphonic import Spectrum1D
  from euphonic.plot import plot_1d

  dos = Spectrum1D.from_json_file('dos.json')
  dos_broaden = Spectrum1D.from_json_file('dos_broaden.json')

  fig = plot_1d([dos, dos_broaden], x_label='Energy (meV)', y_min=0,
                labels=['dos', 'broadened dos'])
  fig.show()

Docstring
---------

.. autofunction:: euphonic.plot.plot_1d

2D Plots
========

2D spectra are plotted with :py:meth:`euphonic.plot.plot_2d`:

.. code-block:: py

  from euphonic import Spectrum2D
  from euphonic.plot import plot_2d

  sqw = Spectrum2D.from_json_file('sqw.json')
  fig, ims = plot_2d(sqw, ratio=1.0)
  fig.show()

Docstring
---------

.. autofunction:: euphonic.plot.plot_2d