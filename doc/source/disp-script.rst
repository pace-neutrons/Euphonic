.. _disp-script:
.. highlight:: bash

===================
euphonic-dispersion
===================

The ``euphonic-dispersion`` program can be used to plot dispersion
either along a specific trajectory from precalculated phonon frequencies,
or along a recommended reciprocal space path from force constants. For
example, to plot from a Euphonic ``.json`` file containing
`QpointPhononModes`, with frequencies reordered to follow equivalent modes
across Q, run::

   euphonic-dispersion --reorder si_qpoint_phonon_modes.json

Or, to plot along a recommended q-point path from Phonopy force constants
with an acoustic sum rule, run::

   euphonic-dispersion --asr reciprocal phonopy.yaml

To see all the command line options, run::

   euphonic-dispersion -h

You can also see the available command line options below

Command Line Options
--------------------

.. argparse::
   :module: euphonic.cli.dispersion
   :func: get_parser
   :prog: euphonic-dispersion
