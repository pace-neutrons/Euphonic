========
Euphonic
========

|PyPI Version| |Conda Version| |Documentation Status| |Tests| |License| |DOI|

.. |PyPI Version| image:: https://img.shields.io/pypi/v/euphonic
   :target: https://pypi.org/project/euphonic/
   :alt: PyPI Version

.. |Conda Version| image:: https://img.shields.io/conda/vn/conda-forge/euphonic
   :target: https://anaconda.org/conda-forge/euphonic
   :alt: Conda Version

.. |Documentation Status| image:: https://readthedocs.org/projects/euphonic/badge/?version=stable
   :target: https://euphonic.readthedocs.io/en/stable/
   :alt: Documentation Status

.. |Tests| image:: https://github.com/pace-neutrons/Euphonic/actions/workflows/run_tests.yml/badge.svg
   :target: https://github.com/pace-neutrons/Euphonic/actions/workflows/run_tests.yml
   :alt: Tests

.. |License| image:: https://img.shields.io/pypi/l/euphonic
   :target: https://github.com/pace-neutrons/Euphonic/blob/master/LICENSE
   :alt: License

.. |DOI| image:: https://img.shields.io/badge/DOI-10.1107%2FS1600576722009256-blue
   :target: https://doi.org/10.1107/S1600576722009256
   :alt: Paper DOI

Euphonic is a Python package that can efficiently calculate phonon
bandstructures, density of states and inelastic neutron scattering
intensities from force constants. The force constants can be read
from various sources, including CASTEP ``.castep_bin`` files, or Phonopy
``phonopy.yaml`` files. While Euphonic is primarily a library, multiple
command-line tools are also provided for convenient plotting of the above
quantities.

For more information, see the `docs <http://euphonic.readthedocs.io/en/latest/>`_.
