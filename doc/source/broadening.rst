.. _broadening:

======================
Broadening Utilities
======================

.. note::

   If you are performing standard spectrum broadening (e.g. broadening a density of states or :class:`Spectrum1D <euphonic.spectra.base.Spectrum1D>` / :class:`Spectrum2D <euphonic.spectra.base.Spectrum2D>` object with a fixed or variable FWHM), you should use the high-level :py:meth:`Spectrum1D.broaden <euphonic.spectra.base.Spectrum1D.broaden>` or :py:meth:`Spectrum2D.broaden <euphonic.spectra.base.Spectrum2D.broaden>` methods (see :ref:`Spectra <spectra>`).

.. contents:: :local:

Overview
========

The ``euphonic.broadening`` module provides the low-level standalone convolution and interpolation algorithms used internally by Euphonic to apply Gaussian or Lorentzian broadening to 1-D and 2-D spectral data series.

Key Capabilities
================

- **Variable-Width Broadening**: :py:func:`~euphonic.broadening.variable_width_broadening` handles x-dependent broadening functions (such as instrumental resolution functions varying with energy transfer).
- **Kernel Interpolation**: :py:func:`~euphonic.broadening.width_interpolated_broadening` and :py:func:`~euphonic.broadening.find_coeffs` accelerate calculations by evaluating broadening kernels on regularly spaced intervals and interpolating across the spectrum.

Module API Reference
====================

.. automodule:: euphonic.broadening
   :members:
   :undoc-members:
   :show-inheritance:
