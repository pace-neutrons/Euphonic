========
Euphonic
========

Euphonic is a Python package that can efficiently calculate phonon
bandstructures and inelastic neutron scattering intensities from a force
constants matrix (e.g. from a .castep_bin file). Euphonic can also do
simple plotting, and can plot dispersion and density of states from
precalculated phonon frequencies (e.g. CASTEP .phonon files).

For more information on how to use Euphonic, see the
:ref:`tutorials <tutorials>`

.. contents:: :local:

Installation
============

Euphonic requires Python ``3.6``, ``3.7`` or ``3.8``

Pip
---

If you want to do plotting, or read Phonopy files, you will need to install the
optional ``matplotlib`` and ``phonopy_reader`` extensions:

.. code-block:: bash

  pip install euphonic[matplotlib,phonopy_reader]

The dependencies can also be installed individually:

.. code-block:: bash

  pip install euphonic[matplotlib]

If you don't require plotting or reading of Phonopy files, just use:

.. code-block:: bash

  pip install euphonic

Conda
-----

For Conda users there is a ``euphonic`` package on conda-forge.
This may be preferable to Pip installation as it should grab existing
binary-compatible builds of Euphonic and its dependencies.
The package is quite minimal, it is usually recommended to also
install plotting, import and progress-bar dependencies.
To create a "complete" installation in a new environment:

.. code-block:: bash

  conda create -n euphonic-forge -c conda-forge python=3.8 euphonic matplotlib-base pyyaml tqdm h5py

This creates an environment named "euphonic-forge", which can be
entered with ``activate euphonic-forge`` and exited with
``deactivate``. Once could also install the packages into an existing environment:

.. code-block:: bash

  conda install -c conda-forge euphonic matplotlib-base pyyaml tqdm h5py

Github
------
To get the latest unreleased version, clone the Git repository at
``https://github.com/pace-neutrons/Euphonic`` and cd into the top directory
containing the ``setup.py`` script.
To install with the optional Matplotlib dependency for plotting and pyyaml/h5py
dependencies for reading Phonopy files use:

.. code-block:: bash

  pip install .[matplotlib,phonopy_reader]

If you don't require plotting or reading of Phonopy files, just use:

.. code-block:: bash

  pip install .

Installing the C extension
--------------------------

By default, Euphonic will attempt to build and install the C extension,
which can lead to increased performance and enable use of multiple cores for
interpolating phonons. If installed, the C extension will be used automatically,
with the number of threads automatically set by ``multiprocessing.cpu_count()``.
A specific number of threads can be used by setting the environment variable
``EUPHONIC_NUM_THREADS``. The number of threads and whether to use the C
extension at all can also be controlled on each function call with the
``n_threads`` and ``use_c`` arguments to
:py:meth:`ForceConstants.calculate_qpoint_phonon_modes <euphonic.force_constants.ForceConstants.calculate_qpoint_phonon_modes>`.

See below for information on installing the extension for different platforms.
If you are having trouble installing the C extension and don't require it, see
`Installing Euphonic without the C extension`_

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
``4.8.5``). If ``gcc`` can be found the Euphonic extension will be
automatically installed when using the same pip commands as above.

**Mac OSX**

Requires a brew install of the llvm-clang compiler,
before running pip install run:

.. code-block:: bash

  brew install llvm

Installing Euphonic without the C extension
-------------------------------------------

If you don't need the extra performance the C extension provides, you can
install the Python parts only with:

.. code-block:: bash

  pip install --install-option="--python-only" euphonic

Note that using this option disables the use of wheels which, if they haven't
been installed already, actually makes installing other packages such as Numpy
more difficult. The easiest way around this is running the usual install
command first (which will install all the dependencies), then running again
with the ``--install-option="--python-only"`` option.

Citing Euphonic
===============

Euphonic has a `CITATION.cff <https://citation-file-format.github.io/>`_ that
contains all the metadata required to cite Euphonic. The correct citation file
for your version of Euphonic can be found in Euphonic's installation directory,
or it can be read programatically as follows:

.. code-block:: py

  import yaml
  import euphonic
  from importlib_resources import open_text

  with open_text(euphonic, 'CITATION.cff') as fp:
      citation_data = yaml.safe_load(fp)

The latest version of the citation file can also be found in Euphonic's code
repository


.. toctree::
   :hidden:
   :maxdepth: 2

   tutorials
