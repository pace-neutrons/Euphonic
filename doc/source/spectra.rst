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

.. code-block:: py

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

.. code-block:: py

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

.. code-block:: py

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

.. code-block:: py

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

.. code-block:: py

  from euphonic import Spectrum1DCollection
  from euphonic.plot import plot_1d

  spec1d_col = Spectrum1DCollection.from_json_file('dos.json')
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

.. code-block:: py

  {'line_data': [
     {'index': 1, 'species': 'Si', 'weighting': 'coherent'},
     {'index': 2, 'species': 'Si', 'weighting': 'coherent'},
     {'index': 3, 'species': 'O', 'weighting': 'coherent'},
     {'index': 4, 'species': 'O', 'weighting': 'coherent'},
     {'index': 1, 'species': 'Si', 'weighting': 'incoherent'},
     {'index': 2, 'species': 'Si', 'weighting': 'incoherent'},
     {'index': 3, 'species': 'O', 'weighting': 'incoherent'},
     {'index': 4, 'species': 'O', 'weighting': 'incoherent'}]
  }

If you want to group and sum spectra that have the same ``weighting``:

.. code-block:: py

  weighting_pdos = spec1d_col.group_by('weighting')

This would produce a collection containing 2 spectra with the following metadata
(the ``index`` and ``species`` metadata are not common across all the grouped
spectra, so have been discarded):

.. code-block:: py

  {'line_data': [
     {'weighting': 'coherent'},
     {'weighting': 'incoherent'}]
  }

You can also group by multiple keys, for example to group and sum spectra that
have both the same ``weighting`` and ``species``:

.. code-block:: py

  weighting_species_pdos = spec1d_col.group_by('weighting', 'species')

This would produce a collection containing 4 spectra with the following metadata:

.. code-block:: py

  {'line_data': [
     {'species': 'Si', 'weighting': 'coherent'},
     {'species': 'O', 'weighting': 'coherent'},
     {'species': 'Si', 'weighting': 'incoherent'},
     {'species': 'O', 'weighting': 'incoherent'}]
  }


Selecting By Metadata
---------------------

You can select specific spectra from a ``Spectrum1DCollection`` based
on their metadata using
:py:meth:`Spectrum1DCollection.select <euphonic.spectra.Spectrum1DCollection.select>`.
For example, if you have a collection ``spec1d_col`` containing
6 spectra with the following metadata:

.. code-block:: py

  {'line_data': [
     {'species': 'Si', 'weighting': 'coherent'},
     {'species': 'O', 'weighting': 'coherent'},
     {'species': 'Si', 'weighting': 'incoherent'},
     {'species': 'O', 'weighting': 'incoherent'},
     {'species': 'Si', 'weighting': 'coherent-plus-incoherent'},
     {'species': 'O', 'weighting': 'coherent-plus-incoherent'}]
  }

If you want to select only the spectra where ``weighting='coherent'``:

.. code-block:: py

  coh_pdos = spec1d_col.select(weighting='coherent')

This would create a collection containing 2 spectra, with the following
metadata:

.. code-block:: py

  {'line_data': [
     {'species': 'Si', 'weighting': 'coherent'},
     {'species': 'O', 'weighting': 'coherent'}]
  }

You can also select multiple values for a specific key. To select spectra
where ``weighting='coherent'`` or ``weighting='incoherent'``:

.. code-block:: py

  coh_or_incoh_pdos = spec1d_col.select(weighting=['coherent', 'incoherent'])

This would create a collection containing 4 spectra, with the following
metadata:

.. code-block:: py

  {'line_data': [
     {'species': 'Si', 'weighting': 'coherent'},
     {'species': 'O', 'weighting': 'coherent'},
     {'species': 'Si', 'weighting': 'incoherent'},
     {'species': 'O', 'weighting': 'incoherent'}]
  }

You can also select by multiple key/values. To select only the spectrum with
``weighting='coherent'`` and ``species='Si'``:

.. code-block:: py

  coh_si_pdos = spec1d_col.select(weighting='coherent', species='Si')

This would create a collection containing only one spectrum, with the
following metadata:

.. code-block:: py

  {'species': 'Si', 'weighting': 'coherent'}

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

.. code-block:: py

  from euphonic import ureg, Spectrum2D

  sqw = Spectrum2D.from_json_file('sqw.json')
  x_fwhm = 0.05*ureg('1/angstrom')
  y_fwhm = 1.5*ureg('meV')
  sqw_broaden = sqw.broaden(x_width=x_fwhm, y_width=y_fwhm, shape='lorentz')

Plotting
--------

See :ref:`Plotting <plotting>`

Docstring
---------

.. autoclass:: euphonic.spectra.Spectrum2D
   :members:
   :inherited-members:

