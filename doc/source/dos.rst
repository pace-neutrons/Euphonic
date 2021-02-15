.. _dos:

Calculating DOS
===============

Density of states can be calculated for any Euphonic object containing
frequencies using ``calculate_dos``. This requires an array of energy bin
edges, with the units specified by wrapping it as a ``pint.Quantity`` (see
:ref:`Units` for details). This function returns a generic
:ref:`Spectrum1D<spectrum1d>` object. For example, using
:py:meth:`QpointFrequencies.calculate_dos <euphonic.qpoint_frequencies.QpointFrequencies.calculate_dos>`.

.. code-block:: py

  from euphonic import ureg, QpointFrequencies
  import numpy as np

  phonons = QpointFrequencies.from_castep('quartz.phonon')

  # Create an array of energy bins 0 - 100 in meV
  energy_bins = np.arange(0, 101, 1)*ureg('meV')

  # Calculate dos
  dos = phonons.calculate_dos(energy_bins)

