=======
Spectra
=======

These are generic objects for storing 1D/2D spectra e.g. density of states or
S(Q,w) maps.

.. contents:: :local:

.. _spectrum1d:

Spectrum1D
==========

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
x-axis, e.g. modes in a dispersion plot.

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
===============

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

