.. _dos:

=================
Density of States
=================

.. contents:: :local:

DOS representation in Euphonic
------------------------------

Density of states in Euphonic is represented by a generic
:ref:`Spectrum1D<spectrum1d>` object. If there are multiple
DOS with the same energy bins (e.g. per-element partial DOS
and total DOS) they can be contained in a
:ref:`Spectrum1DCollection<spectrum1dcollection>` object. PDOS
can be labelled with the ``metadata`` attributes, e.g. for a
``Spectrum1D`` object:

.. testsetup:: si_pdos

   from euphonic import Spectrum1D
   si_pdos = Spectrum1D.from_json_file(get_data_path(
       'spectrum1d', 'quartz_554_full_castep_si_adaptive_dos.json'))

.. doctest:: si_pdos

   >>> si_pdos.metadata
   {'label': 'Si', 'species': 'Si'}

Or, for a ``Spectrum1DCollection`` object:

.. testsetup:: all_dos

   from euphonic import Spectrum1DCollection
   all_dos = Spectrum1DCollection.from_json_file(get_data_path(
       'spectrum1dcollection', 'quartz_554_full_castep_adaptive_dos.json'))

.. doctest:: all_dos

   >>> all_dos.metadata
   {'line_data': [{'label': 'Total'}, {'label': 'O', 'species': 'O'}, {'label': 'Si', 'species': 'Si'}]}

See the :ref:`Spectrum1D<spectrum1d>` or
:ref:`Spectrum1DCollection<spectrum1dcollection>` documentation for further
information on processing and plotting.

.. contents:: :local:

Reading DOS
===========

DOS can be read from a CASTEP ``.phonon_dos`` file using
:py:meth:`Spectrum1D.from_castep_phonon_dos <euphonic.spectra.Spectrum1D.from_castep_phonon_dos>` or
:py:meth:`Spectrum1DCollection.from_castep_phonon_dos <euphonic.spectra.Spectrum1DCollection.from_castep_phonon_dos>`.
The ``Spectrum1D`` version will return either the total DOS or a specific
PDOS from the file, which can be specified with the ``element`` argument.
The ``Spectrum1DCollection`` version will read both the total DOS and
per-element PDOS. Each DOS is labelled by the
``Spectrum1DCollection.metadata`` attribute. An example is shown below.

.. testsetup:: castep_dos

   fnames = 'quartz-151512.phonon_dos'
   shutil.copyfile(
       get_castep_path('quartz', 'quartz-554-full.phonon_dos'), fnames)

.. testcode:: castep_dos

  from euphonic import Spectrum1D, Spectrum1DCollection

  # Read total DOS
  dos_total = Spectrum1D.from_castep_phonon_dos('quartz-151512.phonon_dos')

  # Read Silicon PDOS
  dos_si = Spectrum1D.from_castep_phonon_dos('quartz-151512.phonon_dos', element='Si')

  # Read all DOS and PDOS
  dos_all = Spectrum1DCollection.from_castep_phonon_dos('quartz-151512.phonon_dos')
  # View DOS labels
  print(dos_all.metadata)

.. testoutput:: castep_dos

   {'line_data': [{'label': 'Total'}, {'species': 'O', 'label': 'O'}, {'species': 'Si', 'label': 'Si'}]}

Calculating DOS
===============

Density of states can be calculated for any Euphonic object containing
frequencies using its ``calculate_dos`` method. This requires an array of
energy bin edges, with the units specified by wrapping it as a
``pint.Quantity`` (see :ref:`Units` for details). This function returns a
generic :ref:`Spectrum1D<spectrum1d>` object. For example, using
:py:meth:`QpointFrequencies.calculate_dos <euphonic.qpoint_frequencies.QpointFrequencies.calculate_dos>`.

.. testsetup:: quartz_phonon

   fnames = 'quartz.phonon'
   shutil.copyfile(
       get_castep_path('quartz', 'quartz_nosplit.phonon'), fnames)

.. testcode:: quartz_phonon

  from euphonic import ureg, QpointFrequencies
  import numpy as np

  phonons = QpointFrequencies.from_castep('quartz.phonon')

  # Create an array of energy bins 0 - 100 in meV
  energy_bins = np.arange(0, 101, 1)*ureg('meV')

  # Calculate dos
  dos = phonons.calculate_dos(energy_bins)

.. _adaptive_broadening:

Adaptive Broadening
-------------------

Adaptive broadening can also be enabled to get a more accurate DOS than with
standard fixed width broadening.  For adaptive broadening each
mode at each q-point is broadened individually with a specific width.
There are two adaptive broadening methods available, the 'reference' and 'fast' methods.
The 'reference' scheme explicitly calculates a gaussian for each mode width. 
These mode widths are derived from the mode gradients, and the mode gradients
can be  calculated at the same time as the phonon frequencies and eigenvectors,
by passing ``return_mode_gradients=True`` to
:py:meth:`ForceConstants.calculate_qpoint_phonon_modes <euphonic.force_constants.ForceConstants.calculate_qpoint_phonon_modes>` or
:py:meth:`ForceConstants.calculate_qpoint_frequencies <euphonic.force_constants.ForceConstants.calculate_qpoint_frequencies>`.
The mode widths can be estimated from the mode gradients using
:py:meth:`euphonic.util.mode_gradients_to_widths <euphonic.util.mode_gradients_to_widths>`.
These widths can then be passed to ``calculate_dos`` through the
``mode_widths`` keyword argument. An example is shown below.

.. testsetup:: quartz_fc

   fnames = 'quartz.castep_bin'
   shutil.copyfile(
       get_castep_path('quartz', fnames), fnames)

.. testcode:: quartz_fc

  from euphonic import ureg, ForceConstants
  from euphonic.util import mp_grid, mode_gradients_to_widths
  import numpy as np

  fc = ForceConstants.from_castep('quartz.castep_bin')
  phonons, mode_grads = fc.calculate_qpoint_frequencies(
      mp_grid([5, 5, 4]),
      return_mode_gradients=True)
  mode_widths = mode_gradients_to_widths(mode_grads, fc.crystal.cell_vectors)

  energy_bins = np.arange(0, 166, 0.1)*ureg('meV')
  adaptive_dos = phonons.calculate_dos(energy_bins, mode_widths=mode_widths)

The 'fast' approximate adaptive brodening method reduces computation time by
reducing the number of Gaussian functions that have to be evaluated. Rather than
individually broadening each mode at each q-point with a Gaussian of specific width,
broadening kernels need only be computed for regularly spaced values across range of
mode widths. The kernels at intermediate mode width values can then be approximated
using interpolation. Interpolation weights can then be used to scale the input spectrum, 
before scaled spectra are then convolved by each of the broadening functions associated
with the mode width sample values and then summed.

For fast adaptive broadening the ``adaptive_method`` keyword argument must be set to 'fast'
when passed to ``calculate_dos``. Optionally, an acceptable error level for the interpolated kernels
can be specified, using the ``adaptive_error`` keyword argument. The error is defined as the absolute
difference between the areas of the true and approximate Gaussians. Changing the ``adaptive_error`` value
will change the number of mode width samples; more samples will make the gaussian approximations more
accurate but will also increase computation time. Following on from the above example,
fast adaptive broadening can be performed as follows:

.. testcode:: quartz_fc

  fast_adaptive_dos = phonons.calculate_dos(energy_bins, 
                                            mode_widths=mode_widths,
                                            adaptive_method='fast')

Calculating Partial and Neutron-weighted DOS
--------------------------------------------

See :ref:`Calculating PDOS <calculating_pdos>`

