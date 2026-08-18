.. _readers:

============
File Readers
============

.. note::

   The ``euphonic.readers`` subpackage is part of Euphonic's public API and provides low-level functions that extract raw Python dictionaries from calculation files.

   In most standard workflows, end-users are not expected to call these functions directly. Instead, you should construct high-level Euphonic objects (:py:class:`~euphonic.force_constants.ForceConstants`, :py:class:`~euphonic.qpoint_phonon_modes.QpointPhononModes`, or :py:class:`~euphonic.qpoint_frequencies.QpointFrequencies`) via their respective ``.from_vasp()``, ``.from_castep()``, or ``.from_phonopy()`` classmethod constructors.

.. contents:: :local:

CASTEP Reader
-------------

Euphonic supports reading CASTEP calculation files (``.castep_bin``, ``.check``, ``.phonon``, ``.phonon_dos``).

File Structure & Records
^^^^^^^^^^^^^^^^^^^^^^^^

* **Binary Files (``.castep_bin``, ``.check``)**: Binary records containing unit cell vectors, ionic positions, species, force constants matrix, Born effective charges, and high-frequency dielectric tensors.
* **Phonon Dispersion Files (``.phonon``)**: Text blocks containing q-points, phonon frequencies, and eigenvectors.
* **Density of States Files (``.phonon_dos``)**: Text tables containing total density of states and per-element partial DOS.

Module API Reference
^^^^^^^^^^^^^^^^^^^^

.. automodule:: euphonic.readers.castep
   :members:
   :undoc-members:
   :show-inheritance:


Phonopy Reader
--------------

Euphonic supports reading Phonopy calculation files in YAML and HDF5 formats.

File Structure & Records
^^^^^^^^^^^^^^^^^^^^^^^^

* **Summary Files (``phonopy.yaml``)**: Unified metadata file containing crystal structure, supercell transformation matrix, Born effective charges, and calculation parameters.
* **Force Constant Files (``FORCE_CONSTANTS`` / ``force_constants.hdf5``)**: Standalone supercell force constant matrices.
* **Dielectric Files (``BORN``)**: Standalone Born effective charges and dielectric tensors.
* **Phonon Dispersion / Mesh Files (``mesh.*``, ``qpoints.*``, ``bands.*``)**: Precalculated q-points, frequencies, and eigenvectors.

Separate Files & Fallback Behavior
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When using modern Phonopy versions with ``--include-all``, a single ``phonopy.yaml`` contains all necessary data. However, for older Phonopy versions or split calculations where force constants and Born charges are stored in separate files (e.g. ``FORCE_CONSTANTS``, ``force_constants.hdf5``, or ``BORN``), Euphonic will look inside ``phonopy.yaml`` first and fall back to external files if needed:

* **Python API**: Specify explicit filenames using keyword arguments such as ``fc_name='force_constants.hdf5'`` or ``born_name='BORN'`` in :py:meth:`ForceConstants.from_phonopy <euphonic.force_constants.ForceConstants.from_phonopy>`.
* **CLI Tools**: Euphonic's command-line tools automatically detect and load accompanying ``FORCE_CONSTANTS`` or ``force_constants.hdf5`` files if they reside in the same directory as ``phonopy.yaml``.

.. note::

   Force constants from Phonopy use the atomic-coordinate phase convention (:math:`\mathbf{q}\cdot\mathbf{r}_\kappa`). Euphonic automatically converts these into cell-origin convention (:math:`\mathbf{q}\cdot\mathbf{R}_l`) during ingestion using :py:func:`euphonic.util.convert_fc_phases`. For more details, see :ref:`Phase Convention <fc_format>`.

Module API Reference
^^^^^^^^^^^^^^^^^^^^

.. automodule:: euphonic.readers.phonopy
   :members:
   :undoc-members:
   :show-inheritance:


VASP HDF5 Reader
----------------

Euphonic supports direct import of VASP 6 HDF5 calculation files (e.g. ``vaspout.h5``) containing supercell force constants, Born effective charges, dielectric tensors, and precalculated phonon data.

HDF5 Group Structure Mapping
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``euphonic.readers.vasp`` module extracts data from standard VASP 6 HDF5 hierarchies to construct native Euphonic objects:

* **Crystal Structure & Masses** (used by :py:class:`~euphonic.crystal.Crystal`):

  * Equilibrium lattice and ion positions: ``input/poscar`` (or fallback ``results/positions``)
  * Primitive cell structure (when available): ``results/phonons/primitive``
  * Atomic masses (POMASS): Extracted from ``input/incar/POMASS``, ``original/incar/content``, or ``input/potcar/content``.

* **Force Constants & Dielectric Properties** (used by :py:meth:`ForceConstants.from_vasp <euphonic.force_constants.ForceConstants.from_vasp>`):

  * Force constants / Hessian matrix: ``results/linear_response/force_constants`` or ``results/linear_response/hessian``
  * Born effective charges: ``results/linear_response/born_charges``
  * High-frequency electronic dielectric tensor (:math:`\epsilon^\infty`): ``results/linear_response/electron_dielectric_tensor``

* **Precalculated Phonons** (used by :py:meth:`QpointPhononModes.from_vasp <euphonic.qpoint_phonon_modes.QpointPhononModes.from_vasp>` and :py:meth:`QpointFrequencies.from_vasp <euphonic.qpoint_frequencies.QpointFrequencies.from_vasp>`):

  * Dispersion / mesh q-points, frequencies, and eigenvectors: ``results/phonons``


Module API Reference
^^^^^^^^^^^^^^^^^^^^

.. automodule:: euphonic.readers.vasp
   :members:
   :undoc-members:
   :show-inheritance:
