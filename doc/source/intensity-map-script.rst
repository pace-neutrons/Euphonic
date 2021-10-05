.. _intensity-map-script:

======================
euphonic-intensity-map
======================

.. highlight:: bash

The ``euphonic-intensity-map`` program can be used to plot a 2D intensity
map either along a specific trajectory from precalculated phonon frequencies
and eigenvectors, or along a recommended reciprocal space path from force
constants.

For example, to plot a coherent neutron-weighted intensity map from CASTEP
force constants along a recommended q-point path, one could run::

   euphonic-intensity-map --weighting coherent --energy-broadening 1.5 quartz.castep_bin

Or, to plot a DOS-weighted intensity from Phonopy phonon frequencies::

   euphonic-intensity-map --weighting dos --energy-unit THz --energy-broadening 0.15 band.yaml

To see all the command line options, run::

   euphonic-intensity-map -h

You can also see the available command line options below.
For information on advanced plot styling, see :ref:`styling`.

Command Line Options
--------------------

.. argparse::
   :module: euphonic.cli.intensity_map
   :func: get_parser
   :prog: euphonic-intensity-map
