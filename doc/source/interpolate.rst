.. _interpolate:

===========================================
Calculating Phonon Frequencies/Eigenvectors
===========================================

Phonon frequencies/eigenvectors can be calculated at any q-point from a force
constants matrix via Fourier interpolation. The force constants matrix is
defined as:

.. math::

   \phi_{\alpha, {\alpha}'}^{\kappa, {\kappa}'} =
   \frac{\delta^{2}E}{{\delta}u_{\kappa,\alpha}{\delta}u_{{\kappa}',{\alpha}'}}

Which gives the Dynamical matrix at q:

.. math::

   D_{\alpha, {\alpha}'}^{\kappa, {\kappa}'}(q) =
   \frac{1}{\sqrt{M_\kappa M_{\kappa '}}}
   \sum_{a}\phi_{\alpha, \alpha '}^{\kappa, \kappa '}e^{-iq\cdot r_a}

The eigenvalue equation for the dynamical matrix is:

.. math::

   D_{\alpha, {\alpha}'}^{\kappa, {\kappa}'}(q) \epsilon_{\nu\kappa\alpha q} =
   \omega_{\nu q}^{2} \epsilon_{\nu\kappa\alpha q}

Where :math:`\kappa` runs over atoms, :math:`\alpha` runs over the Cartesian
directions, :math:`\nu` runs over phonon modes, :math:`a` runs over unit cells
in the supercell, :math:`u_{\kappa, \alpha}` is the displacement of atom
:math:`\kappa` in direction :math:`\alpha`, :math:`M_{\kappa}` is the mass of
atom :math:`\kappa`, :math:`r_{a}` is the vector to the origin of cell :math:`a`
in the supercell, :math:`\epsilon_{\nu\kappa\alpha q}` are the eigevectors, and
:math:`\omega_{\nu q}^{2}` are the frequencies squared.
