.. _dos:

Calculating DOS
===============

Density of states can be calculated for any Euphonic object containing
frequencies using its ``calculate_dos`` method. This requires an array of
energy bin edges, with the units specified by wrapping it as a
``pint.Quantity`` (see :ref:`Units` for details). This function returns a
generic :ref:`Spectrum1D<spectrum1d>` object. For example, using
:py:meth:`QpointFrequencies.calculate_dos <euphonic.qpoint_frequencies.QpointFrequencies.calculate_dos>`.

.. code-block:: py

  from euphonic import ureg, QpointFrequencies
  import numpy as np

  phonons = QpointFrequencies.from_castep('quartz.phonon')

  # Create an array of energy bins 0 - 100 in meV
  energy_bins = np.arange(0, 101, 1)*ureg('meV')

  # Calculate dos
  dos = phonons.calculate_dos(energy_bins)

Adaptive Broadening
-------------------

Adaptive broadening can also be enabled to get a more accurate DOS than with
standard fixed width broadening. In this scheme each mode at each q-point is
broadened individually with a specific width. These mode widths are derived
from the mode gradients, and are calculated at the same time as the phonon
frequencies and eigenvectors, by passing ``return_mode_gradients=True`` to
:py:meth:`ForceConstants.calculate_qpoint_phonon_modes <euphonic.force_constants.ForceConstants.calculate_qpoint_phonon_modes>` or
:py:meth:`ForceConstants.calculate_qpoint_frequencies <euphonic.force_constants.ForceConstants.calculate_qpoint_frequencies>`. These widths can then be passed to ``calculate_dos`` through the
``mode_widths`` keyword argument. An example is shown below.

.. code-block:: py

  from euphonic import ureg, ForceConstants
  from euphonic.util import mp_grid
  import numpy as np

  fc = ForceConstants.from_castep('quartz.castep_bin')
  phonons, mode_widths = fc.calculate_qpoint_frequencies(
      mp_grid([5, 5, 4]),
      return_mode_widths=True)

  energy_bins = np.arange(0, 166, 0.1)*ureg('meV')
  adaptive_dos = phonons.calculate_dos(energy_bins, mode_widths=mode_widths)

