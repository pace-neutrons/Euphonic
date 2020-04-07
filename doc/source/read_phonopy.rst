.. _read_phonopy:

=====================
Reading Phonopy Output
=====================

To use this functionality, ensure you have installed the optional ``pyyaml`` and
``h5py`` dependencies. This can be done with:

.. code-block:: bash

    pip install euphonic[phonopy_reader]

InterpolationData
-----------------

When using Phonopy with Euphonic, it is recommended that all the required data
(force constants, crystal structure, born charges if applicable) be collected
in a single ``phonopy.yaml`` file. This can be done by running Phonopy with the
``--include-all`` flag or with ``INCLUDE_ALL = .TRUE.``
(``phonopy >= 2.5.0 only``).

Required information is read from Phonopy output files using
``InterpolationData.from_phonopy``. A path keyword argument can be supplied (if
the files are in another directory), and by default ``phonopy.yaml`` is read, but
the filename can be changed with the ``summary_name`` keyword argument:

.. code-block:: py

    from euphonic.data.interpolation import InterpolationData

    idata = InterpolationData.from_phonopy(path='NaCl',
                                           summary_name='phonopy_fc.yaml')

If you are using an older version of Phonopy, the force constants and born
charges can also be read from Phonopy plaintext or hdf5 files by specifying the
``fc_name`` and ``born_name`` keyword arguments:

.. code-block:: py

    from euphonic.data.interpolation import InterpolationData

    idata = InterpolationData.from_phonopy(path='NaCl',
                                           fc_name='force_constants.hdf5',
                                           born_name='BORN')

.. autofunction:: euphonic.data.interpolation.InterpolationData.from_phonopy

PhononData
----------

Note that Phonopy should be run with the ``--eigenvecs`` flag, or
``EIGENVECTORS = .TRUE.`` for use with Euphonic.

Using ``PhononData.from_phonopy`` Euphonic can read frequencies and eigenvectors
from Phonopy files with the following default names:

- ``mesh.yaml``/``mesh.hdf5``
- ``qpoints.yaml``/``qpoints.hdf5``
- ``bands.yaml``/``bands.hdf5``

The file to be read can be specified with the ``phonon_name`` argument. Some of
these files do not include the crystal information, so it must be read from a
``phonopy.yaml`` file, which can be specified with the ``summary_name``
argument. A path can also be specified.

.. code-block:: py

    from euphonic.data.phonon import PhononData

    pdata = PhononData.from_phonopy(path='NaCl', phonon_file='mesh.hdf5')


.. autofunction:: euphonic.data.phonon.PhononData.from_phonopy
