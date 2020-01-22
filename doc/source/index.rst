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

Installing the C extension (optional)
=====================================
Euphonic has an optional C extension, which can lead to increased performance
and enable use of multiple cores when interpolating phonons. By default
Euphonic will attempt to install this extension, but will print a warning and
fall back to the pure Python version if this fails. To determine if the C
extension is installing properly and investigate any problems, it is best to
increase pip's verbosity:

.. code-block:: bash

    pip install -vvv euphonic

**Windows**

On Windows, the C extension can be compiled with the Microsoft Visual Studio
Compiler, which can be downloaded with
`Visual Studio <https://visualstudio.microsoft.com/downloads/>`_. If downloaded
to a standard location your command line software may pick it up automatically,
or you may need to manually add the compiler executable (``cl.exe``) to your
path. The Euphonic extension should then be installed automatically when using
the same pip commands as above.

**Linux**

You should have a version of ``gcc`` on your path (currently tested with
``4.8.5``). If ``gcc`` can be found the Euphonic extension will be automatically
installed when using the same pip commands as above.


.. toctree::
   :hidden:
   :maxdepth: 2

   tutorials