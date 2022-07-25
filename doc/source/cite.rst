.. _cite:

Citing Euphonic
***************

Euphonic has a `CITATION.cff <https://citation-file-format.github.io/>`_ that
contains all the metadata required to cite Euphonic. The correct citation file
for your version of Euphonic can be found in Euphonic's installation directory,
or it can be read programatically as follows:

.. code-block:: py

  import yaml
  import euphonic
  from importlib_resources import files

  with open(files(euphonic)/'CITATION.cff') as fp:
      citation_data = yaml.safe_load(fp)

The latest version of the citation file can also be found in Euphonic's code
repository
