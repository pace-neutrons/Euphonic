.. _calc_ins:

====================================================
Calculating Inelastic Neutron Scattering Intensities
====================================================

Theory
------

The one-phonon inelastic neutron scattering intensity can be calculated as:

.. math::

   S(Q, \omega) = |F(Q, \nu)|^2
   (n_\nu+\frac{1}{2}\pm\frac{1}{2})
   \delta(\omega\mp\omega_{q\nu})

   F(Q, \nu) = \frac{b_\kappa}{M_{\kappa}^{1/2}\omega_{q\nu}^{1/2}}
   [Q\cdot\epsilon_{q\nu\kappa\alpha}]e^{Q{\cdot}r_\kappa}e^{-W}

:math:`n_\nu` is the Bose-Einstein distribution:

.. math::

   n_\nu = \frac{1}{e^{\frac{\hbar\omega_\nu}{k_{B}T}} - 1}

:math:`e^{-W}` is the Debye-Waller factor:

.. math::

   e^{-\sum_{\alpha\beta}\frac{W^{\kappa}_{\alpha\beta}Q_{\alpha}Q_{\beta}}{2}}

.. math::

   W^{\kappa}_{\alpha\beta} =
   \frac{1}{2N_{q}M_{\kappa}}
   \sum_{BZ}\frac{\epsilon_{q\nu\kappa\alpha}\epsilon^{*}_{q\nu\kappa\beta}}
   {\omega_{q\nu}}
   coth(\frac{\omega_{q\nu}}{2k_BT})

Where :math:`\nu` runs over phonon modes, :math:`\kappa` runs over atoms, 
:math:`\alpha,\beta` run over the Cartesian directions, :math:`b_\kappa` is the
coherent neutron scattering length, :math:`M_{\kappa}` is the atom mass,
:math:`r_{\kappa}` is the vector to atom :math:`\kappa` in the unit cell, 
:math:`\epsilon_{q\nu\kappa\alpha}` are the eigevectors, :math:`\omega_{q\nu}`
are the frequencies, :math:`\sum_{BZ}` is a sum over the 1st Brillouin Zone, and
:math:`N_q` is the number of q-point samples in the BZ.

Usage
-----

Docstring
---------
.. autofunction:: euphonic.data.interpolation.InterpolationData.calculate_structure_factor