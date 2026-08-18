.. _isotopes:

============
Isotope Data
============

.. contents:: :local:

Overview
========

Euphonic uses isotope and reference data when calculating neutron scattering quantities, such as structure factors (:py:meth:`QpointPhononModes.calculate_structure_factor <euphonic.qpoint_phonon_modes.QpointPhononModes.calculate_structure_factor>`) or neutron-weighted partial density of states (:py:meth:`QpointPhononModes.calculate_pdos <euphonic.qpoint_phonon_modes.QpointPhononModes.calculate_pdos>`). 

Depending on your use case, isotope and scattering length properties can be provided in two main ways:

1. **Problem-Specific Dictionaries**: A simple, convenient way to define scattering lengths or atomic weights tied directly to element or atom labels in a specific calculation, without needing external files.
2. **General-Purpose Tabular Datasets (``CsvData``)**: A robust framework for loading comprehensive property tables across the periodic table (such as the built-in ``sears_1992`` dataset).

Using Custom Dictionaries for Problem-Specific Data
===================================================

For quick calculations or when testing custom scattering lengths for specific atomic labels in your crystal structure, you can pass a plain Python dictionary (or ``AtomTypeDictData``) directly to calculation methods. The dictionary maps element or atom symbols to `pint.Quantity` objects:

.. code-block:: python

  from euphonic import ureg, ForceConstants

  fc = ForceConstants.from_castep('quartz.castep_bin')
  phonons = fc.calculate_qpoint_phonon_modes(qpts, asr='reciprocal')

  fm = ureg('fm')
  # Simple dictionary mapping element symbols to scattering lengths
  scattering_lengths = {'Si': 4.1491 * fm, 'O': 5.803 * fm}
  sf = phonons.calculate_structure_factor(scattering_lengths=scattering_lengths)

This approach is lightweight and keeps problem-specific overrides self-contained within your script.

Using General-Purpose Datasets (``sears_1992`` & ``CsvData``)
=============================================================

For robust, general-purpose isotopic tables supporting automatic mass and scattering length lookups across all elements and isotopes, Euphonic provides :py:class:`~euphonic.isotopes._csv.CsvData`. 

By default, Euphonic preloads the Sears (1992) neutron scattering length dataset as ``sears_1992``:

.. code-block:: python

  from euphonic.isotopes import sears_1992

  # Access scattering length for a specific isotope or element
  si_b = sears_1992.get_array('Si', 'bound_coherent_scattering_length')

You can also load custom CSV data tables containing physical properties and unit definitions using :py:class:`~euphonic.isotopes._csv.CsvData`.


Module API Reference
====================

.. automodule:: euphonic.isotopes
   :members:
   :undoc-members:
   :show-inheritance:
