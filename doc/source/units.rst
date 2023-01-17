.. _units:

Units
*****

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

.. testcode::

  from euphonic import ureg
  temperature = 5.0*ureg('K')

This can also be done with Numpy arrays, to create an array with units. For
example, to create an array of energy bins in meV:

.. testcode::

  from euphonic import ureg
  import numpy as np
  arr = np.arange(0, 100, 1)
  energy_bins = arr*ureg('meV')

**Quantities as Function Arguments**

Many Euphonic functions require ``Quantity`` wrapped values as arguments.
Simply import the unit registry and create a ``Quantity``, then pass it to
a function:

.. testsetup:: quartz_phonon

    fnames = 'quartz.phonon'
    shutil.copyfile(
        get_castep_path('quartz', 'quartz_nosplit.phonon'), fnames)

    import numpy
    numpy.set_printoptions(precision=3, linewidth=52, suppress=True)

.. testcode:: quartz_phonon

  from euphonic import QpointPhononModes, ureg

  phonons = QpointPhononModes.from_castep('quartz.phonon')

  fm = ureg('fm')
  scattering_lengths = {'Si': 4.1491*fm, 'O': 5.803*fm}
  sf = phonons.calculate_structure_factor(scattering_lengths)

Object Attributes
-----------------

Any dimensioned attributes (attributes with units) on a Euphonic object,
for example ``cell_vectors`` or ``frequencies``, are actually properties.
They are stored internally in atomic units, and only wrapped as a
``Quantity`` in user-friendly units once they are accessed.

**Changing attribute units**

When a Euphonic object is created, its dimensioned attributes will be
in default units (e.g. meV for frequencies, angstrom for cell vectors).
Each ``Quantity`` attribute has an associated string attribute that can
be used to change the units. See the following example to change the units
of frequency in ``QpointPhononModes``:

.. doctest:: quartz_phonon

   >>> from euphonic import QpointPhononModes
   >>> phonons = QpointPhononModes.from_castep('quartz.phonon')
   >>> phonons.frequencies[5]
   <Quantity([  7.597  14.964  15.853  17.     21.604  27.9
     33.892  36.467  37.509  41.034  46.834  50.083
     52.757  53.321  58.286  62.749  80.168  88.254
     98.021 100.962 101.436 132.376 134.143 134.503
    142.526 145.486 149.365], 'millielectron_volt')>
   >>> phonons.frequencies_unit = '1/cm'
   >>> phonons.frequencies[5]
   <Quantity([  61.271  120.695  127.866  137.113  174.25
     225.028  273.359  294.127  302.533  330.961
     377.744  403.949  425.516  430.063  470.105
     506.108  646.598  711.815  790.593  814.312
     818.136 1067.682 1081.933 1084.837 1149.553
    1173.42  1204.709], '1 / centimeter')>

The pattern is the same for any ``Quantity`` attribute e.g.
``ForceConstants.force_constants`` has ``ForceConstants.force_constants_unit``,
``Crystal.cell_vectors`` has ``Crystal.cell_vectors_unit``

**Changing attribute values**

Each dimensioned property also has a setter which allows it to be set. For
example, to set new ``Crystal.cell_vectors``:

.. testsetup:: quartz_fc

    fnames = 'quartz.castep_bin'
    shutil.copyfile(
        get_castep_path('quartz', 'quartz.castep_bin'), fnames)

    import numpy
    numpy.set_printoptions(precision=3, linewidth=52)

.. doctest:: quartz_fc

   >>> import numpy as np
   >>> from euphonic import ForceConstants, ureg
   >>> fc = ForceConstants.from_castep('quartz.castep_bin')
   >>> fc.crystal.cell_vectors
   <Quantity([[ 2.426 -4.202  0.   ]
    [ 2.426  4.202  0.   ]
    [ 0.     0.     5.35 ]], 'angstrom')>
   >>> fc.crystal.cell_vectors = np.ones((3, 3))*ureg('angstrom')
   >>> fc.crystal.cell_vectors
   <Quantity([[1. 1. 1.]
    [1. 1. 1.]
    [1. 1. 1.]], 'angstrom')>

However as dimensioned attributes are properties, individual elements can't be
set by indexing, for example the following to set a single element of
``Crystal.atom_mass`` does not work:

.. doctest:: quartz_fc

   >>> import numpy as np
   >>> from euphonic import ForceConstants, ureg
   >>> fc = ForceConstants.from_castep('quartz.castep_bin')
   >>> fc.crystal.atom_mass
   <Quantity([15.999 15.999 15.999 15.999 15.999 15.999 28.085
    28.085 28.085], 'unified_atomic_mass_unit')>
   >>> fc.crystal.atom_mass[0] = 17.999*ureg('amu')
   >>> fc.crystal.atom_mass
   <Quantity([15.999 15.999 15.999 15.999 15.999 15.999 28.085
    28.085 28.085], 'unified_atomic_mass_unit')>

Nothing has changed! Instead, get the entire array, change any desired entries and
then set the whole attribute as follows:

.. doctest:: quartz_fc

   >>> from euphonic import ForceConstants, ureg
   >>> fc = ForceConstants.from_castep('quartz.castep_bin')
   >>> atom_mass = fc.crystal.atom_mass
   >>> atom_mass[0] = 17.999*ureg('amu')
   >>> fc.crystal.atom_mass = atom_mass
   >>> fc.crystal.atom_mass
   <Quantity([17.999 15.999 15.999 15.999 15.999 15.999 28.085
    28.085 28.085], 'unified_atomic_mass_unit')>