.. _powder-map-script:
.. highlight:: bash

======================
euphonic-powder-map
======================

The ``euphonic-powder-map`` program can be used to sample
spherically-averaged properties from force constants data over a range
of :math:`|q|`. The results are plotted as a 2-dimensional map in :math:`(|q|, \omega)`.

For example, to plot a coherent neutron-weighted powder spectrum from CASTEP
force constants over a recommended :math:`|q|` range, one could run::

   euphonic-powder-map --weighting coherent --energy-broadening 1.5 quartz.castep_bin

Or, to plot a DOS-weighted intensity from Phonopy phonon frequencies over a specified q range with denser sampling::

   euphonic-powder-map --weighting dos --energy-unit THz --energy-broadening 0.15 --q-min 0.01 --q-max 4. --q-spacing 0.1 phonopy.yaml

To see all the command line options, run::

   euphonic-powder-map -h

You can also see the available command line options at the bottom of this page.

Spherical averaging options
---------------------------

Spherical averaging is performed in a series of constant-q shells. The
``--npts``, ``--npts-density``, ``--npts-min`` and ``--npts-max``
options control the number of samples in each shell, while the
``--sampling`` and ``--jitter`` options control the sampling scheme.
The :ref:`euphonic-show-sampling <sampling-script>` tool can be used
to visualise different sampling schemes.

While the default scheme is recommended for all production
calculations, it is generally necessary to tune the NPTS parameters.
While ``--npts`` sets a constant number of samples for each shell,
``--npts-density`` sets the number of samples at a
1/LENGTH_UNIT-radius sphere, and applies quadratic scaling for other
distances. This may lead to inappropriately small or large numbers of
samples at low or high :math:`|q|`, so the range is limited by
``--npts-min`` and ``--npts-max``. The program will print "Final
npts:" with the number of samples used at the largest sampling
sphere. If this is equal to ``--npts-max`` then the upper limit is in
use; you may wish to experiment with reducing ``--npts-density`` or
increasing ``--npts-max`` in such cases.

Progress bars
-------------

Sampling many q-points can be computationally expensive, so a progress
bar will automatically be displayed if `tqdm <https://tqdm.github.io/>`_
is installed

Command Line Options
--------------------

.. argparse::
   :module: euphonic.cli.powder_map
   :func: get_parser
   :prog: euphonic-powder-map
