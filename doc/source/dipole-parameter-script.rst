.. _dipole-parameter-script:

==================================
euphonic-optimise-dipole-parameter
==================================

.. highlight:: bash

This program is useful for users wanting to efficiently calculate phonon
frequencies on many q-points for polar materials.

Polar materials have a long range force constants matrix, so an analytic
correction must be applied, which is computationally expensive. The correction
is calculated using an Ewald sum, and the balance between the real and
reciprocal sums can be tuned to reduce the computational cost without affecting
the result. This is done with the ``dipole_parameter`` argument to
:py:meth:`ForceConstants.calculate_qpoint_phonon_modes <euphonic.force_constants.ForceConstants.calculate_qpoint_phonon_modes>`

The program runs a calculation for a small test number of q-points (100 by
default) repeatedly for different values of ``dipole_parameter``, and times both the
initialisation time, and time per q-point. Euphonic precalculates as much of the
correction as it can to minimise the calculation time per q-point, so although
some dipole_parameter values will have higher initialisation time, the time per
q-point may be lower, which is what's important when calculating for many q-points.

Usage
-----

Simply run the program on a ``.castep_bin`` file to get an output suggesting the
optimum ``dipole_parameter``. For example, for ``quartz.castep_bin``:

.. code-block:: none

   euphonic-optimise-dipole-parameter quartz.castep_bin

   Results for dipole_parameter  0.25
   Ewald init time     :  0.39 s
   Ewald time/qpt      :  27.69 ms

   Results for dipole_parameter  0.50
   Ewald init time     :  0.08 s
   Ewald time/qpt      :  5.57 ms

   Results for dipole_parameter  0.75
   Ewald init time     :  0.03 s
   Ewald time/qpt      :  4.17 ms

   Results for dipole_parameter  1.00
   Ewald init time     :  0.03 s
   Ewald time/qpt      :  6.79 ms

   Results for dipole_parameter  1.25
   Ewald init time     :  0.06 s
   Ewald time/qpt      :  10.30 ms

   Results for dipole_parameter  1.50
   Ewald init time     :  0.06 s
   Ewald time/qpt      :  15.04 ms

   ******************************
   Suggested optimum dipole_parameter is  0.75
   init time           :  0.03 s
   time/qpt            :  4.17 ms

As you can see above, the time per q-point has been reduced from ``6.79ms`` per
q-point to ``4.17ms`` per q-point, just by correctly tuning the real and
reciprocal sums.

You can change the number of q-points to test, the minimum dipole_parameter,
maximum dipole_parameter, and step size between dipole_parameter values. To
see the command line options, run::

   euphonic-optimise-dipole-parameter -h

Or see the command line options in more detail below

Command Line Options
--------------------

.. argparse::
   :module: euphonic.cli.optimise_dipole_parameter
   :func: get_parser
   :prog: euphonic-optimise-dipole-parameter
