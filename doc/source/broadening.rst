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
:py:func:`~euphonic.broadening.variable_width_broadening` evaluates broadening width functions (such as instrumental resolution functions varying with energy transfer)
and passes them to :py:func:`~euphonic.broadening.width_interpolated_broadening`, which implements a fast approximate broadening scheme.
The data is divided and convolved with multiple kernels to construct an interpolated spectrum without broadening every bin individually [Farmer2023]_.

References
==========

.. [Farmer2023] J. Farmer and A. J. Jackson. *A fast approximate method for variable-width broadening of spectra*. `arXiv:2309.12135 <https://arxiv.org/abs/2309.12135>`_ (2023).

Module API Reference
====================

.. automodule:: euphonic.broadening
   :members:
   :undoc-members:
   :show-inheritance:
