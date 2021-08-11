Euphonic
********

Euphonic is a Python package that can efficiently calculate phonon
bandstructures and inelastic neutron scattering intensities from a force
constants matrix (e.g. from a .castep_bin file). Euphonic can also do
simple plotting, and can plot dispersion and density of states from
precalculated phonon frequencies (e.g. CASTEP .phonon files).

The main way to interact with Euphonic is via the Python API, but there are
also some useful command line tools. See the links below for more
information.

Getting Started
===============
.. toctree::

  installation

Python API
==========
.. toctree::
  :maxdepth: 1

  Force Constants <force-constants>
  Phonon Frequencies and Eigenvectors <qpoint-phonon-modes>
  Phonon Frequencies Only <qpoint-frequencies>
  Density of States <dos>
  Structure Factors <structure-factor>
  Scattering Intensities <scattering-intensities>
  Debye-Waller <debye-waller>
  Spectra <spectra>
  powder
  plotting
  sampling
  utils
  Units in Euphonic <units>

Command-line Tools
==================
.. toctree::
  :maxdepth: 1

  euphonic-dispersion <disp-script>
  euphonic-dos <dos-script>
  euphonic-intensity-map <intensity-map-script>
  euphonic-powder-map <powder-map-script>
  euphonic-optimise-dipole-parameter <dipole-parameter-script>
  euphonic-show-sampling <sampling-script>

:ref:`Changelog <changelog>`
============================

:ref:`Citing Euphonic <cite>`
=============================

.. toctree::
  :hidden:

  changelog
  cite
