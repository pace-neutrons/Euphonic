.. _force-constants:

ForceConstants
==============

The ``ForceConstants`` object contains the force constants, supercell, and
crystal structure information required to calculate phonon frequencies and
eigenvectors at any arbitrary q via Fourier interpolation. 

.. contents:: :local:

Reading From CASTEP
-------------------

The force constants matrix and other required information can be read from a
``.castep_bin`` or ``.check`` file with
:py:meth:`ForceConstants.from_castep <euphonic.force_constants.ForceConstants.from_castep>`:

.. code-block:: py

  from euphonic import ForceConstants

  filename = 'quartz/quartz.castep_bin'
  fc = ForceConstants.from_castep(filename)

By default CASTEP may not write the force constants, if you receive an error
saying the force constants could not be read, in the ``.param`` file ensure a
``PHONON_FINE_METHOD`` has been chosen e.g. ``PHONON_FINE_METHOD: interpolate``,
and set ``PHONON_WRITE_FORCE_CONSTANTS: true``, then rerun CASTEP to trigger the
force constants to be written.

Reading From Phonopy
------------

When using Phonopy with Euphonic, it is recommended that all the required data
(force constants, crystal structure, born charges if applicable) be collected
in a single ``phonopy.yaml`` file. This can be done by running Phonopy with the
``--include-all`` flag or with ``INCLUDE_ALL = .TRUE.``
(``phonopy >= 2.5.0 only``).

Required information is read from Phonopy output files using
:py:meth:`ForceConstants.from_phonopy <euphonic.force_constants.ForceConstants.from_phonopy>`.
A path keyword argument can be supplied (if the files are in another
directory), and by default ``phonopy.yaml`` is read, but the filename can be
changed with the ``summary_name`` keyword argument:

.. code-block:: py

  from euphonic import ForceConstants

  fc = ForceConstants.from_phonopy(path='NaCl',
                                   summary_name='phonopy_fc.yaml')

If you are using an older version of Phonopy, the force constants and born
charges can also be read from Phonopy plaintext or hdf5 files by specifying the
``fc_name`` and ``born_name`` keyword arguments:

.. code-block:: py

  from euphonic import ForceConstants

  fc = ForceConstants.from_phonopy(path='NaCl',
                                   fc_name='force_constants.hdf5',
                                   born_name='BORN')

Calculating phonon frequencies/eigenvectors
-------------------------------------------

Phonon frequencies and eigenvectors are calculated using
:py:meth:`ForceConstants.calculate_qpoint_phonon_modes <euphonic.force_constants.ForceConstants.calculate_qpoint_phonon_modes>`
(see the docstring for algorithm details). A Numpy array of q-points of shape
(n_qpts, 3) must be provided, and a
:ref:`QpointPhononModes<qpoint-phonon-modes>` object is returned. A
recommended q-point path for plotting bandstructures can be generated using
`seekpath <https://seekpath.readthedocs.io/en/latest/module_guide/index.html#seekpath.getpaths.get_explicit_k_path>`_:

.. code-block:: py

  import seekpath
  import numpy as np
  from euphonic import ForceConstants

  # Read quartz data from quartz.castep_bin
  filename = 'quartz/quartz.castep_bin'
  fc = ForceConstants.from_castep(filename)

  # Generate a recommended q-point path using seekpath
  _, unique_atoms = np.unique(fc.crystal.atom_type, return_inverse=True)
  structure = (fc.crystal.cell_vectors.magnitude,
               fc.crystal.atom_r, unique_atoms)
  qpts = seekpath.get_explicit_k_path(structure)["explicit_kpoints_rel"]

  # Calculate frequencies/eigenvectors
  phonons = fc.calculate_qpoint_phonon_modes(qpts, asr='reciprocal')

Docstring
---------

.. autoclass:: euphonic.force_constants.ForceConstants
   :members:
   :exclude-members: force_constants, born, dielectric
