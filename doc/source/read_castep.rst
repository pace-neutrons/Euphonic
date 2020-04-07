.. _read_castep:

=====================
Reading CASTEP Output
=====================

InterpolationData
-----------------

The force constants matrix and other required information is read from a
``.castep_bin`` file by:

.. code-block:: py

    from euphonic.data.interpolation import InterpolationData

    seedname = 'quartz'
    idata = InterpolationData.from_castep(seedname)

By default CASTEP may not write the force constants, if you receive an error
saying the force constants could not be read, set
``PHONON_WRITE_FORCE_CONSTANTS: true`` in the ``.param`` file, and rerun CASTEP
to trigger the force constants to be written.

PhononData
----------

Frequencies and eigenvectors for PhononData are read from a ``.phonon`` file.

.. code-block:: py

    from euphonic.data.phonon import PhononData

    seedname = 'quartz'
    pdata = PhononData.from_castep(seedname)

BandsData
---------
Electronic fequencies can be read from a ``.bands`` file.

.. code-block:: py

    from euphonic.data.bands import BandsData

    seedname = 'quartz'
    bdata = BandsData.from_castep(seedname)
