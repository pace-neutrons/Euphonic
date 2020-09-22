.. _dos-script:

============
euphonic-dos
============

The ``euphonic-dos`` program can be used to plot a density of states
from precalculated frequencies in a ``.phonon`` or ``.bands``
file. For example, to plot DOS from a ``quartz.phonon`` file, run::

   euphonic-dos quartz.phonon

To see all the command line options, run::

   euphonic-dos -h

You can also see the available command line options below

Command Line Options
--------------------

.. argparse::
   :module: euphonic.cli.dos
   :func: get_parser
   :prog: euphonic-dos
