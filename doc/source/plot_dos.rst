.. _plot_dos:

============
Plotting DOS
============

Usage
-----

Any ``Data`` object that contains the ``freqs`` attribute can be used to
calculate a DOS with the ``calculate_dos`` method, and then plot it with the
``plot_dos`` function. There is also a useful helper function, ``mp_grid`` which
can be used to generate a Monkhorst-Pack grid of q-points to interpolate onto.

.. code-block:: py

    import numpy as np
    from euphonic.data.interpolation import InterpolationData
    from euphonic.plot.dos import plot_dos
    from euphonic.util import mp_grid

    # Read quartz data from quartz.castep_bin
    idata = InterpolationData('quartz')

    # Generate a grid of q-points
    qpts = mp_grid([10, 10, 10])

    # Calculate frequencies/eigenvectors
    idata.calculate_fine_phonons(qpts, asr='reciprocal')

    # Create an array of energy bin edges
    ebins = np.arange(0, 160, 0.25)
    # Now calculate the DOS
    # This sets the 'dos' and 'dos_bins' attributes of the Data object, and also
    # returns the dos for each bin
    dos = idata.calculate_dos(ebins, gwidth=0.75)

   # plot_dos takes the Data object as an argument and returns a
   # Matplotlib figure
   fig = plot_dos(idata)

   # Example removing y axis labels
   ax = fig.get_axes()[0]
   ax.set_yticklabels([])

   fig.show()

.. image:: figures/quartz_dos.png
   :align: center

As with dispersion plots, some settings can be changed via optional arguments
(see docstrings below). The returned ``matplotlib.figure.Figure`` instance can
also be tweaked directly, for more information see the
`Matplotlib docs <https://matplotlib.org/api/_as_gen/matplotlib.figure.Figure.html#matplotlib.figure.Figure>`_

Docstrings
----------

.. autofunction:: euphonic.data.data.Data.calculate_dos

.. autofunction:: euphonic.plot.dos.plot_dos
