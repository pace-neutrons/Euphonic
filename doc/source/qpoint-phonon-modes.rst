.. _qpoint-phonon-modes:

QpointPhononModes
=================

The QpointPhononModes object contains  precalculated phonon
frequencies and eigenvectors at certain q-points.

.. contents:: :local:

Reading From CASTEP
-------------------

Phonon frequencies and eigenvectors can be read from a  ``.phonon`` file using
:py:meth:`QpointPhononModes.from_castep <euphonic.qpoint_phonon_modes.QpointPhononModes.from_castep>`.

.. code-block:: py

  from euphonic import QpointPhononModes

  filename = 'quartz.phonon'
  phonons = QpointPhononModes.from_castep(filename)

Reading From Phonopy
--------------------

Phonopy should be run with the ``--eigvecs`` flag, or ``EIGENVECTORS = .TRUE.``
for use with Euphonic.

Using :py:meth:`QpointPhononModes.from_phonopy <euphonic.qpoint_phonon_modes.QpointPhononModes.from_phonopy>`
Euphonic can read frequencies and eigenvectors from Phonopy files with the following default names:

- ``mesh.yaml``/``mesh.hdf5``
- ``qpoints.yaml``/``qpoints.hdf5``
- ``bands.yaml``/``bands.hdf5``

The file to be read can be specified with the ``phonon_name`` argument. Some of
these files do not include the crystal information, so it must be read from a
``phonopy.yaml`` file, which can be specified with the ``summary_name``
argument. A path can also be specified.

.. code-block:: py

  from euphonic import QpointPhononModes

  phonons = QpointPhononModes.from_phonopy(path='NaCl', phonon_name='mesh.hdf5')

From Force Constants
--------------------

See :ref:`Force Constants <force-constants>`

Reordering frequencies
----------------------

The stored frequencies can be reordered by comparing eigenvectors using 
:py:meth:`QpointPhononModes.reorder_frequencies <euphonic.qpoint_phonon_modes.QpointPhononModes.reorder_frequencies>`.
This reordering can be seen the plotting dispersion (see
:ref:`Plotting <plotting>`)

.. code-block:: py

  from euphonic import QpointPhononModes

  phonons = QpointPhononModes.from_castep('quartz.phonon')
  phonons.reorder_frequencies()

Plotting Dispersion
-------------------

See :ref:`Plotting Dispersion <plotting-dispersion>`


Calculating The Coherent Neutron Structure Factor
-------------------------------------------------

The neutron structure factor can be calculated for each branch and q-point
using the
:py:meth:`QpointPhononModes.calculate_structure_factor <euphonic.qpoint_phonon_modes.QpointPhononModes.calculate_structure_factor>`
method, which returns a :ref:`StructureFactor<structure-factor>` object.
(See the docstring for algorithm details.)


Scattering lengths
^^^^^^^^^^^^^^^^^^
The coherent scattering length is a physical property of the nucleus.
To provide this data explicitly, the ``scattering_lengths`` argument can
be set as a dictionary mapping each atom identity to a ``pint.Quantity``
(see :ref:`Units` for details).

Alternatively, this argument may be a string referring to a data file
with the *coherent_scattering_length* property. (See :ref:`Reference Data <ref_data>` for details.)
By default, :py:meth:`QpointPhononModes.calculate_structure_factor <euphonic.qpoint_phonon_modes.QpointPhononModes.calculate_structure_factor>` will use the ``"Sears1992"`` data set included in Euphonic.
If you have a custom data file, this can be used instead.

Debye-Waller factor
^^^^^^^^^^^^^^^^^^^
Inclusion of the
Debye-Waller factor is optional, and can be provided in the ``dw`` keyword
argument, see `Calculating The Debye-Waller Exponent`_.

Example
^^^^^^^
The following example shows a full calculation from the force constants to the
structure factor with Debye-Waller:

.. code-block:: py

  import seekpath
  import numpy as np
  from euphonic import ureg, QpointPhononModes, ForceConstants
  from euphonic.util import mp_grid

  # Read the force constants
  fc = ForceConstants.from_castep('quartz.castep_bin')

  # Generate a recommended q-point path to calculate the structure factor on
  # using seekpath
  cell = crystal.to_spglib_cell()
  qpts = seekpath.get_explicit_k_path(cell)["explicit_kpoints_rel"]
  # Calculate frequencies/eigenvectors for the q-point path
  phonons = fc.calculate_qpoint_phonon_modes(qpts, asr='reciprocal')

  # For the Debye-Waller calculation, generate and calculate
  # frequencies/eigenvectors on a grid (generate a Monkhorst-Pack grid of
  # q-points using the mp-grid helper function)
  q_grid = mp_grid([5,5,5])
  phonons_grid = fc.calculate_qpoint_phonon_modes(q_grid, asr='reciprocal')
  # Now calculate the Debye-Waller exponent
  temperature = 5*ureg('K')
  dw = phonons_grid.calculate_debye_waller(temperature)

  # Calculate the structure factor for each q-point in phonons. A
  # StructureFactor object is returned
  fm = ureg('fm')
  scattering_lengths = {'Si': 4.1491*fm, 'O': 5.803*fm}
  sf = phonons.calculate_structure_factor(scattering_lengths, dw=dw)

Calculating The Debye-Waller Exponent
-------------------------------------

The Debye-Waller factor is an optional part of the structure factor
calculation. The exponent part of the Debye-Waller factor is independent of Q
and should be precalculated using
:py:meth:`QpointPhononModes.calculate_debye_waller <euphonic.qpoint_phonon_modes.QpointPhononModes.calculate_debye_waller>`
(see the docstring for algorithm details). This requires a QpointPhononModes
object calculated on a grid of q-points and a temperature, and returns a
:ref:`DebyeWaller<debye-waller>` object. The Debye-Waller exponent can be
calculated by:

.. code-block:: py

  from euphonic import ureg, QpointPhononModes

  phonons = QpointPhononModes.from_castep('quartz-grid.phonon')
  temperature = 5*ureg('K')
  dw = phonons.calculate_debye_waller(temperature)

.. _dos:

Calculating Density of States
-----------------------------

Density of states can be calculated using 
:py:meth:`QpointPhononModes.calculate_dos <euphonic.qpoint_phonon_modes.QpointPhononModes.calculate_dos>`.
This requires an array of energy bin edges, with the units specified by
wrapping it as a ``pint.Quantity`` (see :ref:`Units` for details). This
function returns a generic :ref:`Spectrum1D<spectrum1d>` object. For example:

.. code-block:: py

  from euphonic import ureg, QpointPhononModes
  import numpy as np

  phonons = QpointPhononModes.from_castep('quartz.phonon')

  # Create an array of energy bins 0 - 100 in meV
  energy_bins = np.arange(0, 101, 1)*ureg('meV')

  # Calculate dos
  dos = phonons.calculate_dos(energy_bins)

Docstring
---------

.. autoclass:: euphonic.qpoint_phonon_modes.QpointPhononModes
   :members:
   :exclude-members: frequencies


