.. _cite:

Citing Euphonic
***************

Euphonic has a `CITATION.cff <https://citation-file-format.github.io/>`_ that
contains all the metadata required to cite Euphonic. The correct citation file
for your version of Euphonic can be found in Euphonic's installation directory,
or it can be read programatically as follows:

.. testcode::

  import yaml
  import euphonic
  from importlib.resources import files

  with open(files(euphonic) / 'CITATION.cff') as fp:
      citation_data = yaml.safe_load(fp)

The latest version of the citation file can also be found in Euphonic's code
repository `here <https://github.com/pace-neutrons/Euphonic/blob/master/CITATION.cff>`_.

Please cite both the Euphonic software itself and the following publication:

* Fair R.L., Jackson A.J., Voneshen D.J., Jochym D.B., Le M.D., Refson K., Perring T.G.
  *Euphonic: inelastic neutron scattering simulations from force constants and visualization tools for phonon properties*.
  Journal of Applied Crystallography **55** 1689-1703 (2022).
  DOI: https://doi.org/10.1107/S1600576722009256.
