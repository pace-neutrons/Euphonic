.. _readers:

============
File Readers
============

.. note::

   The ``euphonic.readers`` subpackage is part of Euphonic's public API and provides low-level functions that extract raw Python dictionaries from calculation files.

   In most standard workflows, end-users are not expected to call these functions directly. Instead, you should construct high-level Euphonic objects (``ForceConstants``, ``QpointPhononModes``, ``QpointFrequencies``, or ``Crystal``) via their respective ``.from_vasp()``, ``.from_castep()``, or ``.from_phonopy()`` classmethod constructors.

.. contents:: :local:

VASP HDF5 Reader
----------------

Euphonic supports direct import of VASP 6+ HDF5 calculation files (e.g. ``vaspout.h5``) containing supercell force constants, Born effective charges, dielectric tensors, and precalculated phonon data.

### HDF5 Group Structure Mapping

The `euphonic.readers.vasp` module extracts data from standard VASP 6+ HDF5 hierarchies:

* **Crystal Structure & Masses**:
  - Equilibrium lattice and ion positions: `input/poscar` (or fallback `results/positions`)
  - Primitive cell structure (when available): `results/phonons/primitive`
  - Atomic masses (POMASS): Extracted from `input/incar/POMASS`, `original/incar/content`, or `input/potcar/content`.
* **Force Constants & Dielectric Properties**:
  - Force constants / Hessian matrix: `results/linear_response/force_constants` or `results/linear_response/hessian`
  - Born effective charges: `results/linear_response/born_charges`
  - High-frequency electronic dielectric tensor ($\epsilon^\infty$): `results/linear_response/electron_dielectric_tensor`
* **Precalculated Phonons**:
  - Dispersion / mesh q-points, frequencies, and eigenvectors: `results/phonons`

### Module API Reference

.. automodule:: euphonic.readers.vasp
   :members:
   :undoc-members:
   :show-inheritance:
