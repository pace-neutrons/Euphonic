.. _dos-script:

============
euphonic-dos
============

The ``euphonic-dos`` program can be used to plot a density of states
from precalculated frequencies, or on a specific Monkhorst-Pack grid
from force constants. For example, to plot DOS from a CASTEP ``.phonon``
file in units of 1/cm, run::

   euphonic-dos --energy-unit 1/cm quartz-554-grid.phonon

Or, to plot DOS from CASTEP force constants in a ``.castep_bin`` file
on a 5x5x4 grid with broadening, run::

   euphonic-dos --grid 5 5 4 --energy-broadening 1.5 quartz.castep_bin

To see all the command line options, run::

   euphonic-dos -h

You can also see the available command line options below

Command Line Options
--------------------

.. argparse::
   :module: euphonic.cli.dos
   :func: get_parser
   :prog: euphonic-dos
