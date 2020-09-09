.. _sampling-script:

======================
euphonic-show-sampling
======================

``euphonic-show-sampling`` can be used to visualise the spherical
sampling schemes implemented in :mod:`euphonic.sampling`.  For
example, to see how the 'golden' sphere sampling approach works for
100 points, run::

  euphonic-show-sampling 100 golden-sphere

To see all the command line options, run::

   euphonic-show-sampling -h

You can also see the available command line options below

Command Line Options
--------------------

.. argparse::
   :module: euphonic.cli.show_sampling
   :func: get_parser
   :prog: euphonic-show-sampling
