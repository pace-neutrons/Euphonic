.. _structure-factor:

StructureFactor
===============

The StructureFactor object contains the structure factor at every q-point and
mode, and optionally a temperature if temperature-dependent effects such as
the Debye-Waller factor were used in the calculation.

.. contents:: :local:

.. _scattering-intensities:

From Phonon Modes
-----------------

See :ref:`QpointPhononModes <qpoint-phonon-modes>`

Calculating Scattering Intensities
----------------------------------

The structure factors can be used to create a :math:`S(Q, \omega)` map
with Q on the x-axis and energy on the y-axis using
:py:meth:`StructureFactor.calculate_sqw_map <euphonic.structure_factor.StructureFactor.calculate_sqw_map>`
(see docstring for algorithm details). This requires an array of energy bin
edges as a ``pint.Quantity``. Calculating the Bose population factor
is optional, but if ``calc_bose=True`` the temperature stored in
StructureFactor is used. If there is no temperature in StructureFactor,
then it must be provided in the function arguments. This function returns a
generic :ref:`Spectrum2D` object. 

.. code-block:: py

  from euphonic import ureg, StructureFactor

  sf = StructureFactor.from_json_file('sf_100K.json')
  energy_bins = np.arange(-100, 101, 1)*ureg('meV')
  sqw_map = sf.calculate_sqw_map(energy_bins calc_bose=True)

Plotting Dispersion
-------------------

See :ref:`Plotting Dispersion <plotting-dispersion>`

Calculating Density of States
-----------------------------

See :ref:`Calculating DOS <dos>`

Docstring
---------

.. autoclass:: euphonic.structure_factor.StructureFactor
   :members:
   :inherited-members:
   :exclude-members: frequencies, temperature
