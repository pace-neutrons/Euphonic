.. _disp-script:

===================
euphonic-dispersion
===================

The ``euphonic-dispersion`` program can be used to plot precalculated
frequencies from a CASTEP ``.phonon`` file. For example, to plot from
a ``quartz.phonon`` file, run::

   euphonic-dispersion quartz.phonon

To see all the command line options, run::

   euphonic-dispersion -h

You can also see the available command line options below

Command Line Options
--------------------

.. argparse::
   :module: euphonic.cli.dispersion
   :func: get_parser
   :prog: euphonic-dispersion
