.. _force-constants:

ForceConstants
==============

The ``ForceConstants`` object contains the force constants, supercell, and
crystal structure information required to calculate phonon frequencies and
eigenvectors at any arbitrary q via Fourier interpolation. 

.. contents:: :local:

Force Constants Shape and Phase Conventions
-------------------------------------------


The :py:attr:`ForceConstants.force_constants <euphonic.force_constants.ForceConstants.force_constants>`
attribute contains the force constants matrix

.. math::

  \phi_{\alpha, {\alpha}'}^{\kappa, {\kappa}'} =
  \frac{\delta^{2}E}{{\delta}u_{\kappa,\alpha}{\delta}u_{{\kappa}',{\alpha}'}}

This describes the change in total energy when atom :math:`\kappa` is displaced in direction
:math:`\alpha` and atom :math:`\kappa\prime` is displaced in direction
:math:`\alpha\prime`. In Euphonic the force constants are stored
in 'compact' form, which means that the minimum required information is
stored. The change in energy in displacing atom :math:`\kappa` and atom
:math:`\kappa\prime`, is the same as displacing atom :math:`\kappa\prime` and
atom :math:`\kappa`, so only :math:`N(3n)^2` values need to be stored rather than :math:`(3Nn)^2`,
where :math:`n` is the number of atoms in the unit cell and :math:`N` is the
number of unit cells in the supercell.

**Phase Convention**

To calculate phonon frequencies and eigenvectors, the dynamical matrix at
:math:`q` must be calculated. This is calculated by taking a Fourier transform
of the force constants matrix

.. math::

  D_{\alpha, {\alpha}'}^{\kappa, {\kappa}'}(q) =
  \frac{1}{\sqrt{M_\kappa M_{\kappa '}}}
  \sum_{a}\phi_{\alpha, \alpha '}^{\kappa, \kappa '}e^{-iq\cdot r_a}

**Indexing**

The force constants matrix has shape ``(n_cells, 3*n_atoms, 3*n_atoms)``, where ``n_cells`` is
the number of cells in the supercell, and ``n_atoms`` is the number of atoms
in the unit cell. The force constants index ``[n, l, m]``, where
``l = 3*i + a`` and ``m = 3*j + b``, describes the change in energy when atom
``i`` in unit cell ``0`` is displaced in direction ``a`` and atom ``j``
in unit cell ``n`` is displaced in direction ``b``. For example,
``ForceConstants.force_constants[5, 8, 1]`` is the change in energy when atom
``2`` in unit cell ``0`` is displaced in the ``z`` direction and atom ``0`` in
unit cell ``5`` is displaced in the ``y`` direction.

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
--------------------

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

Calculating phonon frequencies and eigenvectors
-----------------------------------------------

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
  cell = fc.crystal.to_spglib_cell()
  qpts = seekpath.get_explicit_k_path(cell)["explicit_kpoints_rel"]

  # Calculate frequencies/eigenvectors
  phonons = fc.calculate_qpoint_phonon_modes(qpts, asr='reciprocal')

Calculating phonon frequencies only
-----------------------------------

This uses the same algorithm as for calculating both the frequencies and
eigenvectors, only with lower memory requirements as the eigenvectors
are not stored. This is done using
:py:meth:`ForceConstants.calculate_qpoint_frequencies <euphonic.force_constants.ForceConstants.calculate_qpoint_frequencies>`
which returns a :ref:`QpointFrequencies<qpoint-frequencies>` object.


Docstring
---------

.. autoclass:: euphonic.force_constants.ForceConstants
   :members:
   :exclude-members: force_constants, born, dielectric
