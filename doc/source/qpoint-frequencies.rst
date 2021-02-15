.. _qpoint-frequencies:

QpointFrequencies
=================

The QpointFrequencies object contains  precalculated phonon
frequencies at certain q-points

.. contents:: :local:

Reading From CASTEP
-------------------

Phonon frequencies and eigenvectors can be read from a  ``.phonon`` file using
:py:meth:`QpointFrequencies.from_castep <euphonic.qpoint_frequencies.QpointFrequencies.from_castep>`.

.. code-block:: py

  from euphonic import QpointFrequencies

  filename = 'quartz.phonon'
  phonons = QpointFrequencies.from_castep(filename)

Reading From Phonopy
--------------------

Using :py:meth:`QpointFrequencies.from_phonopy <euphonic.qpoint_frequencies.QpointFrequencies.from_phonopy>`
Euphonic can read frequencies from Phonopy files with the following default names:

- ``mesh.yaml``/``mesh.hdf5``
- ``qpoints.yaml``/``qpoints.hdf5``
- ``bands.yaml``/``bands.hdf5``

The file to be read can be specified with the ``phonon_name`` argument. Some of
these files do not include the crystal information, so it must be read from a
``phonopy.yaml`` file, which can be specified with the ``summary_name``
argument. A path can also be specified.

.. code-block:: py

  from euphonic import QpointFrequencies

  phonons = QpointFrequencies.from_phonopy(path='NaCl', phonon_name='mesh.hdf5')

From Force Constants
--------------------

See :ref:`Force Constants <force-constants>`

Plotting Dispersion
-------------------

See :ref:`Plotting Dispersion <plotting-dispersion>`

.. _dos:

Calculating Density of States
-----------------------------

Density of states can be calculated using 
:py:meth:`QpointFrequencies.calculate_dos <euphonic.qpoint_frequencies.QpointFrequencies.calculate_dos>`.
This requires an array of energy bin edges, with the units specified by
wrapping it as a ``pint.Quantity`` (see :ref:`Units` for details). This
function returns a generic :ref:`Spectrum1D<spectrum1d>` object. For example:

.. code-block:: py

  from euphonic import ureg, QpointFrequencies
  import numpy as np

  phonons = QpointFrequencies.from_castep('quartz.phonon')

  # Create an array of energy bins 0 - 100 in meV
  energy_bins = np.arange(0, 101, 1)*ureg('meV')

  # Calculate dos
  dos = phonons.calculate_dos(energy_bins)

Docstring
---------

.. autoclass:: euphonic.qpoint_frequencies.QpointFrequencies
   :members:
   :inherited-members:
   :exclude-members: frequencies

