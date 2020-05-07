.. _disp-script:

====================
dispersion.py Script
====================

``dispersion.py`` can be used to plot precalculated frequencies from a CASTEP
``.phonon`` file. For example, to plot from a ``quartz.phonon``
file, run::

   dispersion.py quartz.phonon

To see all the command line options, run::

   dispersion.py -h

You can also see the available command line options below

Command Line Options
--------------------

.. argparse::
   :filename: ../../scripts/dispersion.py
   :func: get_parser
   :prog: dispersion.py

