.. _qpoint-frequencies:

QpointFrequencies
=================

The QpointFrequencies object contains precalculated phonon 
frequencies at certain q-points. This object does not contain
eigenvectors, so allows some quantities such as a basic DOS
or bandstructures to be calculated with lower memory requirements.

.. contents:: :local:

Reading From CASTEP
-------------------

Phonon frequencies and eigenvectors can be read from a  ``.phonon`` file using
:py:meth:`QpointFrequencies.from_castep <euphonic.qpoint_frequencies.QpointFrequencies.from_castep>`.

.. testsetup:: quartz_phonon

   fnames = 'quartz.phonon'
   shutil.copyfile(
       get_castep_path('quartz', 'quartz_nosplit.phonon'), fnames)

.. testcode:: quartz_phonon

  from euphonic import QpointFrequencies

  filename = 'quartz.phonon'
  phonons = QpointFrequencies.from_castep(filename)

Reading From Phonopy
--------------------

Using :py:meth:`QpointFrequencies.from_phonopy <euphonic.qpoint_frequencies.QpointFrequencies.from_phonopy>`
Euphonic can read frequencies from Phonopy files with the following default names:

- ``mesh.yaml``/``mesh.hdf5``
- ``qpoints.yaml``/``qpoints.hdf5``
- ``bands.yaml``/``bands.hdf5``

The file to be read can be specified with the ``phonon_name`` argument. Some of
these files do not include the crystal information, so it must be read from a
``phonopy.yaml`` file, which can be specified with the ``summary_name``
argument. A path can also be specified.

.. testsetup:: nacl_mesh

   fnames = 'NaCl'
   shutil.copytree(get_phonopy_path('NaCl', 'mesh'), fnames)

.. testcode:: nacl_mesh

  from euphonic import QpointFrequencies

  phonons = QpointFrequencies.from_phonopy(path='NaCl', phonon_name='mesh.hdf5')

From Force Constants
--------------------

See :ref:`Force Constants <force-constants>`

Plotting Dispersion
-------------------

See :ref:`Plotting Dispersion <plotting-dispersion>`

Calculating Density of States
-----------------------------

See :ref:`Calculating DOS <dos>`

Docstring
---------

.. autoclass:: euphonic.qpoint_frequencies.QpointFrequencies
   :members:
   :inherited-members:
   :exclude-members: frequencies

