.. _units:

=====
Units
=====

In Euphonic, units are handled by `Pint <https://pint.readthedocs.io/>`_. Pint
wraps Numpy arrays as a ``Quantity`` object, so you can easily see which units
values are in.

.. contents:: :local:

Euphonic's ureg
---------------

Units in euphonic are accessed through ``euphonic.ureg``, which is a
``Pint.UnitRegistry`` object that contains all required units. For example, to
create temperature as a ``Quantity`` with units in Kelvin rather than a plain
float:

.. code-block:: py

  from euphonic import ureg
  temperature = 5.0*ureg('K')

This can also be done with Numpy arrays, to create an array with units. For
example, to create an array of energy bins in meV:

.. code-block:: py

  from euphonic import ureg
  import numpy as np
  arr = np.arange(0, 100, 1)
  energy_bins = arr*ureg('meV')

**Quantities as Function Arguments**

Many Euphonic functions require ``Quantity`` wrapped values as arguments.
Simply import the unit registry and create a ``Quantity``, then pass it to
a function:

.. code-block:: py

  from euphonic import ureg, QpointPhononModes

  phonons = QpointPhononModes.from_castep('quartz.phonon')

  fm = ureg('fm')
  scattering_lengths = {'Si': 4.1491*fm, 'O': 5.803*fm}
  sf = phonons.calculate_structure_factor(scattering_lengths)

Changing Units
--------------

When creating a Euphonic object, any attributes with units will be
wrapped as a ``Quantity`` object in default units (e.g. meV for frequencies,
angstrom for cell vectors). Each ``Quantity`` attribute has an associated
string attribute that can be used to change the units. See the following
example to change the units of frequency in ``QpointPhononModes``:

.. code-block:: py

  >>> from euphonic import QpointPhononModes
  >>> phonons = QpointPhononModes.from_castep('quartz.phonon')
  >>> phonons.frequencies[0]
  <Quantity([  0.38193331   0.61106305   0.69303768  15.86537413  15.88218266
    27.64203018  31.98753254  32.21911407  42.20189648  42.90662972
    46.91907755  48.26610037  54.57118794  56.89834674  60.53664449
    61.56378648  86.09430023  86.53104436  95.78403943  98.8705196
   100.38739904 132.68487558 133.97647412 134.71659207 142.55121327
   142.97971084 152.80832126], 'millielectron_volt')>
  >>> phonons.frequencies_unit = '1/cm'
  >>> phonons.frequencies[0]
  <Quantity([   3.0805      4.928556    5.589726  127.962875  128.098445  222.948014
    257.996855  259.864686  340.381258  346.065315  378.42789   389.292362
    440.146324  458.916126  488.260977  496.545436  694.397377  697.919956
    772.550396  797.444538  809.678996 1070.175718 1080.593163 1086.562617
   1149.7531   1153.209166 1232.482257], '1 / centimeter')>

The pattern is the same for any ``Quantity`` attribute e.g.
``ForceConstants.force_constants`` has ``ForceConstants.force_constants_unit``,
``Crystal.cell_vectors`` has ``Crystal.cell_vectors_unit``