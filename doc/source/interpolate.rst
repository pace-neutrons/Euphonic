.. _interpolate:

===========================================
Calculating Phonon Frequencies/Eigenvectors
===========================================

Theory
------

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

The eigenvalue equation for the dynamical matrix is then:

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

Usage
-----

Phonon frequencies and eigenvectors are calculated from an ``InterpolationData``
object using the ``calculate_fine_phonons`` method. A Numpy array of q-points
of shape ``(n_qpts, 3)`` must be provided. For example, a recommended q-point
path for plotting bandstructures can be generated using `seekpath \
<https://seekpath.readthedocs.io/en/latest/module_guide/index.html#seekpath.get\
paths.get_explicit_k_path>`_

.. code-block:: py

    import seekpath
    import numpy as np
    from euphonic.data.interpolation import InterpolationData

    # Read quartz data from quartz.castep_bin
    seedname = 'quartz'
    idata = InterpolationData(seedname)

    # Generate a recommended q-point path using seekpath
    _, unique_ions = np.unique(idata.ion_type, return_inverse=True)
    structure = (idata.cell_vec.magnitude, idata.ion_r, unique_ions)
    qpts = seekpath.get_explicit_k_path(structure)["explicit_kpoints_rel"]

    # Calculate frequencies/eigenvectors. They are stored in the freqs and
    # eigenvecs attributes of the InterpolationData object, but also returned
    freqs, eigenvecs = idata.calculate_fine_phonons(qpts, asr='reciprocal')

Optional Keyword Arguments
--------------------------

**asr**

It is well-known that any crystal has three acoustic modes at q=0 with zero
frequency. This is due to translational invariance and is defined as the
*Acoustic Sum Rule*. However, this rule is never exactly satisfied as the atoms
translate with respect to a fixed FFT grid, so the acoustic frequencies at q=0
can deviate significantly from zero. This rule can be enforced on the force
constants matrix using ``asr='realspace'``, or on the dynamical matrix at each
q-point using ``asr='reciprocal'``. For polar materials, the long range tail of
the force constants matrix means that ``asr='reciprocal'`` should be used. For
other materials either can be used, but generally for large numbers of cells in
the supercell ``asr='reciprocal'`` will be faster, although ``asr='realspace'``
might be faster for large numbers of q-points. By default no ASR is applied.

**dipole**

In polar materials, the displacement of an ion induces a dipole, and the
resulting dipole-dipole interaction contributes to the force constants as
:math:`\frac{1}{r^3}`, compared to the other interatomic interactions which
decay as :math:`\frac{1}{r^5}` or faster. Fourier interpolation is only
performed on the short ranged part of the force constants matrix, and a
non-analytical correction must be applied to account for the longer ranged
dipole-dipole interactions. This is computationally quite expensive, and is
performed by default if the ``born`` and ``dielectric`` attributes are present in
the ``InterpolationData`` object. It can be turned off with ``dipole=False``.

**eta_scale**

The analytical dipole correction is applied to the force constants matrix using
an Ewald sum. The ``eta_scale`` argument changes the cutoff in real/reciprocal
space and does not change the result, but if tuned correctly can result in
significant performance improvements. The value will usually be somewhere
between 0.5 and 2.0, higher values use more reciprocal terms. The
:ref:`optimise_eta.py <eta_script>` script can be used to suggest a good value
for ``eta_scale``.

**splitting**

This argument defines whether to calculate LO-TO splitting, and is calculated by
default if ``dipole=True``. The indices of the q-points at which splitting has
been calculated will be stored in ``InterpolationData.split_i``. As there are
now 2 sets of freqs/eigenvecs for some q-points, one set will be stored in
``InterpolationData.freqs`` at the index specified in ``split_i``, as expected,
and the extra set of frequencies will be stored in
``InterpolationData.split_freqs``, in the same order as ``split_i``.

**reduce_qpts**

If the q-points span over more than 1 Brillouin Zone (perhaps for a neutron
scattering intensity calculation), computation time can be reduced by using
periodicity to only calculate frequencies/eigenvectors over the 1st BZ. Euphonic
keeps track of which q-points are equivalent, and frequencies and eigenvectors
can still be accessed as normal with the same indices as the input q-point list.
This is turned on by default, but can be turned off using ``reduce_qpts=False``

Docstring
---------
.. autofunction:: euphonic.data.interpolation.InterpolationData.calculate_fine_phonons
