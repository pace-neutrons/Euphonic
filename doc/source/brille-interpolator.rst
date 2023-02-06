.. _brille-interpolator:

BrilleInterpolator
==================

The ``BrilleInterpolator`` object provides an easy interface to the `Brille <https://brille.github.io/>`_ library for phonon frequencies and eigenvectors.
It can be created from a ``ForceConstants`` object, then used to perform linear (rather than Fourier) interpolation to calculate phonon frequencies and eigenvectors at specific q-points.
Linear interpolation may be less accurate than the Fourier interpolation performed by ``ForceConstants``,
but should be faster for large unit cells, particularly those tha require the computationally expensive dipole correction calculation.
You should test this on your particular machine and material first to see if it provides a performance benefit.

When creating a ``BrilleInterpolator`` from ``ForceConstants``, Brille calculates the first Brillouin Zone (BZ) for the provided ``ForceConstants.crystal``,
then creates a grid of points distributed across it.
Euphonic calculates frequencies and eigenvectors at each of these grid vertices (q-points) by Fourier interpolation, which are then used to fill the grid with data.
Both the grid and the crystal are stored in the created ``BrilleInterpolator`` object.
To calculate the frequencies and eigenvectors at a particular q-point, Brille folds the q-point into the 1st BZ, then searches for the nearest grid points and linearly interpolates between them.

.. contents:: :local:

Creating From Force Constants
-----------------------------

Once a :py:attr:`ForceConstants <euphonic.force_constants.ForceConstants>` object has been created, it can be used to create a :py:attr:`BrilleInterpolator <euphonic.brille.BrilleInterpolator>` object with :py:meth:`BrilleInterpolator.from_force_constants <euphonic.brille.BrilleInterpolator.from_force_constants>`.

The Brille grid type can be chosen with the ``grid_type`` argument (although the default ``trellis`` is recommended),
and the number or density of q-points in the grid can be chosen with the ``grid_npts`` and ``grid_density`` arguments.
A grid with more points will take longer to initialise and require more memory, but can give more accurate results.
For more information on what these mean, see the Brille documentation at https://brille.github.io/.

There is also a command-line tool, :ref:`euphonic-brille-convergence <brille-convergence-script>` which can help with choosing the number of grid points by comparing Euphonic and Brille frequencies and eigenvectors.

An example of creating a ``BrilleInterpolator`` object from ``ForceConstants`` is shown below:

.. testsetup:: lzo_fc

  fnames = 'La2Zr2O7.castep_bin'
  shutil.copyfile(get_castep_path('LZO', fnames), fnames)

.. testcode:: lzo_fc

  from euphonic import ForceConstants
  from euphonic.brille import BrilleInterpolator

  fc = ForceConstants.from_castep('La2Zr2O7.castep_bin')

  bri = BrilleInterpolator.from_force_constants(fc, grid_npts=1000)

.. testoutput:: lzo_fc
   :hide:

   Generating grid...
   Grid generated with 2094 q-points. Calculating frequencies/eigenvectors...
   Filling grid...

Calculating Phonon Frequencies and Eigenvectors
-----------------------------------------------

Phonon frequencies and eigenvectors are calculated using
:py:meth:`BrilleInterpolator.calculate_qpoint_phonon_modes <euphonic.brille.BrilleInterpolator.calculate_qpoint_phonon_modes>`,
and is very similar to 
:py:meth:`ForceConstants.calculate_qpoint_phonon_modes <euphonic.force_constants.ForceConstants.calculate_qpoint_phonon_modes>`.
A Numpy array of q-points of shape (n_qpts, 3) must be provided, and a :ref:`QpointPhononModes<qpoint-phonon-modes>` object is returned.
Kwargs will also be passed to Brille's ``ir_interpolate_at`` function, which can be used to set the number of threads with ``threads`` for example, but by default the number of threads will be the same as used in the Euphonic Fourier interpolation.
An example is below:

.. testcode:: lzo_fc

  import seekpath
  import numpy as np
  from euphonic import ForceConstants
  from euphonic.brille import BrilleInterpolator

  fc = ForceConstants.from_castep('La2Zr2O7.castep_bin')
  bri = BrilleInterpolator.from_force_constants(fc, grid_npts=1000)

  # Generate a recommended q-point path using seekpath
  cell = bri.crystal.to_spglib_cell()
  qpts = seekpath.get_explicit_k_path(cell)["explicit_kpoints_rel"]

  # Calculate frequencies/eigenvectors
  phonons = bri.calculate_qpoint_phonon_modes(qpts, threads=4)

.. testoutput:: lzo_fc
   :hide:

   Generating grid...
   Grid generated with 2094 q-points. Calculating frequencies/eigenvectors...
   Filling grid...

Docstring
---------

.. autoclass:: euphonic.brille.BrilleInterpolator
   :members:
