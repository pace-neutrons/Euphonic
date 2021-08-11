.. _dos-script:
.. highlight:: bash

============
euphonic-dos
============

The ``euphonic-dos`` program can be used to plot density of states,
partial density of states, and/or neutron-weighted density of states.
It can use pre-calculated frequencies, or use force constants to
generate frequencies and DOS over a Monkhorst-Pack grid. For example,
to plot DOS from a CASTEP ``.phonon`` file in units of 1/cm, run::

   euphonic-dos --energy-unit 1/cm quartz-554-grid.phonon

Or, to plot coherent neutron-weighted DOS from CASTEP force constants in a
``.castep_bin`` file on a 15x15x12 grid with broadening, run::

   euphonic-dos --weighting coherent-dos --grid 15 15 12 --energy-broadening 1.5 quartz.castep_bin

To see all the command line options, run::

   euphonic-dos -h

You can also see the available command line options below

Command Line Options
--------------------

.. argparse::
   :module: euphonic.cli.dos
   :func: get_parser
   :prog: euphonic-dos
