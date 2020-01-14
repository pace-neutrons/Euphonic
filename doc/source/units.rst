.. _units:

=====
Units
=====

In Euphonic, units are handled by `Pint <https://pint.readthedocs.io/>`_. Pint
wraps Numpy arrays as a Quantity object, so you can easily see which units
values are in.

By default, length units (e.g cell vectors) are in ``angstrom``, and energy
units (e.g. frequencies) are in ``meV`` for vibrational data objects
(``PhononData`` and ``InterpolationData``) and ``eV`` for electronic data
objects (``BandsData``). The units used will show up automatically on plot axis
titles.

To change either the length or energy units used, use the ``convert_e_units``
and ``convert_l_units`` methods of any of the ``Data`` objects.

.. code-block:: py

   >>> from euphonic.data.phonon import PhononData
   >>> pdata = PhononData.from_castep('quartz')
   >>> pdata.freqs[0]
   <Quantity([  0.38193331   0.61106305   0.69303768  15.86537413  15.88218266
     27.64203018  31.98753254  32.21911407  42.20189648  42.90662972
     46.91907755  48.26610037  54.57118794  56.89834674  60.53664449
     61.56378648  86.09430023  86.53104436  95.78403943  98.8705196
    100.38739904 132.68487558 133.97647412 134.71659207 142.55121327
    142.97971084 152.80832126], 'millielectron_volt')>
   >>> pdata.convert_e_units('1/cm')
   >>> pdata.freqs[0]
   <Quantity([   3.0805      4.928556    5.589726  127.962875  128.098445  222.948014
     257.996855  259.864686  340.381258  346.065315  378.42789   389.292362
     440.146324  458.916126  488.260977  496.545436  694.397377  697.919956
     772.550396  797.444538  809.678996 1070.175718 1080.593163 1086.562617
    1149.7531   1153.209166 1232.482257], '1 / centimeter')>


Docstrings
==========

.. autofunction:: euphonic.data.data.Data.convert_e_units
.. autofunction:: euphonic.data.data.Data.convert_l_units
