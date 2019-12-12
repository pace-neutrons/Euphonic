========
Euphonic
========

Euphonic is a Python package that can efficiently calculate phonon
bandstructures and inelastic neutron scattering intensities from a force
constants matrix (e.g. from a .castep_bin file). Euphonic can also do
simple plotting, and can plot dispersion and density of states from
precalculated phonon frequencies (e.g. CASTEP .bands or .phonon).

For more information, see the :ref:`tutorials <tutorials>`

Installation
============
Pip
---
To do plotting, you will also have to install the optional Matplotlib
dependency alongside Euphonic:

.. code-block:: bash

    pip install euphonic[matplotlib]

If you only need the core calculations (no plotting) just use:

.. code-block:: bash

    pip install euphonic

Github
------
To get the latest unreleased version, clone the Git repository at
``https://github.com/pace-neutrons/Euphonic`` and cd into the top directory
containing the ``setup.py`` script.
To install with the optional Matplotlib dependency use:

.. code-block:: bash

    pip install .[matplotlib]

If you only require the interpolation functionality, and don't need any of the
Matplotlib plotting routines just use:

.. code-block:: bash

    pip install .



.. toctree::
   :hidden:
   :maxdepth: 2

   tutorials