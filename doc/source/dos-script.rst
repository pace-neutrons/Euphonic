.. _dos-script:

=============
dos.py Script
=============

``dos.py`` can be used to plot a density of states from precalculated
frequencies in a ``.phonon`` or ``.bands`` file. For example, to plot DOS from a
``quartz.phonon`` file, run::

   dos.py quartz.phonon

To see all the command line options, run::

   dos.py -h

You can also see the available command line options below

Command Line Options
--------------------

.. argparse::
   :module: euphonic.cli.dos
   :func: get_parser
   :prog: euphonic-dos
