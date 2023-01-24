.. _spectra:

=======
Spectra
=======

These are generic objects for storing 1D/2D spectra e.g. density of states or
S(Q,w) maps.

**Metadata**

All spectra objects have a ``metadata`` attribute. By default it is an empty
dictionary, but it can contain keys/values to help describe the contained
spectra. The keys should be strings and values should be only strings or
integers. Note there are some special 'functional' keys, see the metadata
docstring below for each specific spectrum class for details.


.. contents:: :local:

.. _spectrum1d:

Spectrum1D
==========

Adding Spectrum1D
-----------------

Two ``Spectrum1D`` objects can be added together (provided they have the
same ``x_data`` axes) with the ``+`` operator. This will add their ``y_data``
and return a single ``Spectrum1D``. Note that any metadata key/value pairs
that aren't common to both spectra will be omitted from the new object. For
example:

.. testsetup:: si_o_pdos

   fnames = ['si_pdos.json', 'o_pdos.json']
   for fname in fnames:
       shutil.copyfile(
           get_data_path('spectrum1d', 'toy_band.json'), fname)

.. testcode:: si_o_pdos

  from euphonic import Spectrum1D

  pdos_si = Spectrum1D.from_json_file('si_pdos.json')
  pdos_o = Spectrum1D.from_json_file('o_pdos.json')
  total_dos = pdos_si + pdos_o

Broadening
----------

A 1D spectrum can be broadened using 
:py:meth:`Spectrum1D.broaden <euphonic.spectra.Spectrum1D.broaden>`,
which broadens along the x-axis and returns a new :ref:`Spectrum1D`
object. It can broaden with either a Gaussian or Lorentzian and requires
a broadening FWHM in the same type of units as ``x_data``. For example:

.. testsetup:: dos

   fnames = 'dos.json'
   shutil.copyfile(
       get_data_path('spectrum1d', 'toy_quartz_dos.json'), fnames)

.. testcode:: dos

  from euphonic import ureg, Spectrum1D

  dos = Spectrum1D.from_json_file('dos.json')
  fwhm = 1.5*ureg('meV')
  dos_broaden = dos.broaden(fwhm, shape='lorentz')

Plotting
--------

See :ref:`Plotting <plotting>`

Docstring
---------

.. autoclass:: euphonic.spectra.Spectrum1D
   :inherited-members:
   :members:

.. _spectrum1dcollection:

Spectrum1DCollection
====================

This is an object for storing multiple 1D spectra which share the same
x-axis, e.g. bands in a dispersion plot.

From Spectrum1D
---------------

If you have multiple ``Spectrum1D`` objects with the same ``x_data``,
they can be grouped together into a ``Spectrum1DCollection`` object.
For example:

.. testcode:: si_o_pdos

  from euphonic import Spectrum1D, Spectrum1DCollection

  pdos_si = Spectrum1D.from_json_file('si_pdos.json')
  pdos_o = Spectrum1D.from_json_file('o_pdos.json')

  pdos_collection = Spectrum1DCollection.from_spectra([pdos_si, pdos_o])

Adding Spectrum1DCollection
---------------------------

Two ``Spectrum1DCollection`` objects can be added together (provided they
have the same ``x_data`` axes) with the ``+`` operator. This will concatenate
their ``y_data``, returning a single ``Spectrum1DCollection`` that contains
all the ``y_data`` of both objects. For example:

.. testsetup:: coh_incoh_pdos

   fnames = ['coherent_pdos.json', 'incoherent_pdos.json']
   for fname in fnames:
       shutil.copyfile(
           get_data_path('spectrum1dcollection', 'quartz_666_coh_pdos.json'), fname)

.. testcode:: coh_incoh_pdos

  from euphonic import Spectrum1DCollection

  coh_pdos = Spectrum1DCollection.from_json_file('coherent_pdos.json')
  incoh_pdos = Spectrum1DCollection.from_json_file('incoherent_pdos.json')
  all_pdos = coh_pdos + incoh_pdos

Indexing
--------

A ``Spectrum1DCollection`` can be indexed just like a list to obtain
a specific spectrum as a ``Spectrum1D``, or a subset of spectra as
another ``Spectrum1DCollection``, for example, to plot only specific
spectra:

.. testcode:: coh_incoh_pdos

  from euphonic import Spectrum1DCollection
  from euphonic.plot import plot_1d

  spec1d_col = Spectrum1DCollection.from_json_file('coherent_pdos.json')
  # Plot the 1st spectrum
  spec1d_0 = spec1d_col[0]
  fig1 = plot_1d(spec1d_0)

  # Plot the 2nd - 5th spectra
  spec1d_col_1_5 = spec1d_col[1:5]
  fig2 = plot_1d(spec1d_col_1_5)

Broadening
----------

A collection of 1D spectra can also be broadened using
:py:meth:`Spectrum1DCollection.broaden <euphonic.spectra.Spectrum1DCollection.broaden>`,
which broadens each spectrum individually, giving the same result
as using :py:meth:`Spectrum1D.broaden <euphonic.spectra.Spectrum1D.broaden>`
on each contained spectrum.

Grouping By Metadata
---------------------

You can group and sum specific spectra from a ``Spectrum1DCollection``
based on their metadata using
:py:meth:`Spectrum1DCollection.group_by <euphonic.spectra.Spectrum1DCollection.group_by>`.
For example, if you have a collection ``spec1d_col`` containing
8 spectra with the following metadata:

.. testsetup:: metadata_spec

   import numpy
   from euphonic import Spectrum1DCollection, ureg
   fake_metadata =  {'line_data': [
       {'index': 1, 'species': 'Si', 'weighting': 'coherent'},
       {'index': 2, 'species': 'Si', 'weighting': 'coherent'},
       {'index': 3, 'species': 'O', 'weighting': 'coherent'},
       {'index': 4, 'species': 'O', 'weighting': 'coherent'},
       {'index': 1, 'species': 'Si', 'weighting': 'incoherent'},
       {'index': 2, 'species': 'Si', 'weighting': 'incoherent'},
       {'index': 3, 'species': 'O', 'weighting': 'incoherent'},
       {'index': 4, 'species': 'O', 'weighting': 'incoherent'}]}
   spec1d_col = Spectrum1DCollection(
       numpy.arange(5)*ureg('meV'),
       numpy.ones((len(fake_metadata['line_data']), 5))*ureg('1/meV'),
       metadata=fake_metadata)

.. doctest:: metadata_spec

   >>> spec1d_col.metadata
   {'line_data': [{'index': 1, 'species': 'Si', 'weighting': 'coherent'}, {'index': 2, 'species': 'Si', 'weighting': 'coherent'}, {'index': 3, 'species': 'O', 'weighting': 'coherent'}, {'index': 4, 'species': 'O', 'weighting': 'coherent'}, {'index': 1, 'species': 'Si', 'weighting': 'incoherent'}, {'index': 2, 'species': 'Si', 'weighting': 'incoherent'}, {'index': 3, 'species': 'O', 'weighting': 'incoherent'}, {'index': 4, 'species': 'O', 'weighting': 'incoherent'}]}

If you want to group and sum spectra that have the same ``weighting``,
pass ``'weighting'`` to the ``group_by`` method. This would produce a
collection containing 2 spectra with the following metadata (the ``species``
and ``index`` metadata are not common across all the grouped spectra, so have
been discarded):

.. doctest:: metadata_spec

  >>> weighting_pdos = spec1d_col.group_by('weighting')
  >>> weighting_pdos.metadata
  {'line_data': [{'weighting': 'coherent'}, {'weighting': 'incoherent'}]}

You can also group by multiple keys, for example you can group and sum
spectra that have both the same ``weighting`` and ``species``, producing
the following metadata:

.. doctest:: metadata_spec
   :skipif: True # Skip due to non-deterministic metadata

   >>> weighting_species_pdos = spec1d_col.group_by('weighting', 'species')
   >>> weighting_species_pdos.metadata
   {'line_data': [{'weighting': 'coherent', 'species': 'Si'}, {'species': 'O', 'weighting': 'coherent'}, {'weighting': 'incoherent', 'species': 'Si'}, {'weighting': 'incoherent', 'species': 'O'}]}

Selecting By Metadata
---------------------

You can select specific spectra from a ``Spectrum1DCollection`` based
on their metadata using
:py:meth:`Spectrum1DCollection.select <euphonic.spectra.Spectrum1DCollection.select>`.
For example, if you have a collection ``spec1d_col`` containing
8 spectra with the following metadata:

.. doctest:: metadata_spec

   >>> spec1d_col.metadata
   {'line_data': [{'index': 1, 'species': 'Si', 'weighting': 'coherent'}, {'index': 2, 'species': 'Si', 'weighting': 'coherent'}, {'index': 3, 'species': 'O', 'weighting': 'coherent'}, {'index': 4, 'species': 'O', 'weighting': 'coherent'}, {'index': 1, 'species': 'Si', 'weighting': 'incoherent'}, {'index': 2, 'species': 'Si', 'weighting': 'incoherent'}, {'index': 3, 'species': 'O', 'weighting': 'incoherent'}, {'index': 4, 'species': 'O', 'weighting': 'incoherent'}]}

If you want to select only the spectra where ``weighting='coherent'``,
use the ``select`` method, which would create a collection containing
4 spectra, with the following metadata:

.. doctest:: metadata_spec

   >>> coh_pdos = spec1d_col.select(weighting='coherent')
   >>> coh_pdos.metadata
   {'weighting': 'coherent', 'line_data': [{'index': 1, 'species': 'Si'}, {'index': 2, 'species': 'Si'}, {'index': 3, 'species': 'O'}, {'index': 4, 'species': 'O'}]}

You can also select multiple values for a specific key. For example, to
select spectra where ``index=1`` or ``index=2``:

.. doctest:: metadata_spec

   >>> coh_or_incoh_pdos = spec1d_col.select(index=[1, 2])
   >>> coh_or_incoh_pdos.metadata
   {'species': 'Si', 'line_data': [{'index': 1, 'weighting': 'coherent'}, {'index': 1, 'weighting': 'incoherent'}, {'index': 2, 'weighting': 'coherent'}, {'index': 2, 'weighting': 'incoherent'}]}

You can also select by multiple key/values. To select only the spectra with
``weighting='coherent'`` and ``species='Si'``:

.. doctest:: metadata_spec
   :skipif: True # Skip due to non-deterministic metadata

   >>> coh_si_pdos = spec1d_col.select(weighting='coherent', species='Si')
   >>> coh_si_pdos.metadata
   {'weighting': 'coherent', 'species': 'Si', 'line_data': [{'index': 1}, {'index': 2}]}

Summing Spectra
---------------

All spectra in a ``Spectrum1DCollection`` can be summed with
:py:meth:`Spectrum1DCollection.sum <euphonic.spectra.Spectrum1DCollection.sum>`.
This produces a single ``Spectrum1D`` object.


Plotting
--------

See :ref:`Plotting <plotting>`

Docstring
---------

.. autoclass:: euphonic.spectra.Spectrum1DCollection
   :inherited-members:
   :members:

.. _spectrum2d:

Spectrum2D
==========

Broadening
----------

A 2D spectrum can be broadened using 
:py:meth:`Spectrum2D.broaden <euphonic.spectra.Spectrum2D.broaden>`, which
broadens along either or both of the x/y-axes and returns a new
:ref:`Spectrum2D` object. It can broaden with either a Gaussian or Lorentzian
and requires a broadening FWHM in the same type of units as
``x_data``/``y_data`` for broadening along the x/y-axis respectively.
For example:

.. testsetup:: sqw

   fnames = 'sqw.json'
   shutil.copyfile(
       get_data_path('spectrum2d', 'lzo_57L_bragg_sqw.json'), fnames)

.. testcode:: sqw

  from euphonic import ureg, Spectrum2D

  sqw = Spectrum2D.from_json_file('sqw.json')
  x_fwhm = 0.05*ureg('1/angstrom')
  y_fwhm = 1.5*ureg('meV')
  sqw_broaden = sqw.broaden(x_width=x_fwhm, y_width=y_fwhm, shape='lorentz')

Plotting
--------

See :ref:`Plotting <plotting>`

Kinematic constraints
---------------------

Inelastic neutron-scattering (INS) experiments are often performed on
time-of-flight (ToF) spectrometers where a wide ToF range yields a wide
range of energy transfers which are measured simultaneously.
In "direct geometry" the incident energy is fixed (e.g. by a Fermi
chopper) while in "indirect geometry" the scattered energy is fixed
(e.g. by scattering from an "analyser" crystal).
Conservation laws allow the overall energy and momentum
transfer to be determined for a given scattering angle and crystal
orientation. In powder measurements, the crystal orientation is not
needed so the kinematic limits --- the accessible :math:`(q, \omega)`
range --- determined by the conservation laws are given solely
by these instrument parameters.

The function :py:func:`euphonic.spectra.apply_kinematic_constraints
<euphonic.spectra.apply_kinematic_constraints>` applies these limits
to a powder-averaged :ref:`Spectrum2D` object with appropriate dimensions
(i.e. the x- and y-axes represent :math:`|q|` and :math:`\omega` respectively).
Inaccessible data values are set to ``NaN``; in Matplotlib colour maps
this will leave them unset.

.. list-table:: Sample values for INS spectrometers
   :header-rows: 1

   * - Facility
     - Instrument
     - :math:`E_i` / meV
     - :math:`E_f` / meV
     - :math:`2\theta` / :math:`{}^\circ`
   * - ILL
     - LAGRANGE
     -
     - 4.5
     - 10--90
   * - ISIS
     - LET
     - 1--25
     -
     - 5--140
   * - ISIS
     - MAPS
     - 15--2000
     -
     - 3--60
   * - ISIS
     - MARI
     - 7--1000
     -
     - 3--135
   * - ISIS
     - MERLIN
     - 7--2000
     -
     - 3--135
   * - ILL
     - PANTHER
     - 76, 112, 150
     -
     - 5--136
   * - ISIS
     - TOSCA
     -
     - 3.97
     - 45, 135
.. autofunction:: euphonic.spectra.apply_kinematic_constraints

Docstring
---------

.. autoclass:: euphonic.spectra.Spectrum2D
   :members:
   :inherited-members:

