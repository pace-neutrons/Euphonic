Contributing
------------

Code contributions may be made to this project via Github ``Pull Request
<https://github.com/pace-neutrons/Euphonic/pulls>``_ (PR).  For
non-trivial changes, it may be helpful to discuss the idea first by
creating an ``Issue
<https://github.com/pace-neutrons/Euphonic/issues>``_.

Contributions are also welcome in the form of Issues for bug reports
and feature requests, and by reviewing PRs.  If you have a great idea
to improve the docs but are not confident in the RST syntax (see
below), write it down in an Issue and we should be able to help turn
it into a PR.

Those who make substantial contributions to the project will be added
to the CITATION.cff. We do not generally add authors for formatting
changes or "one-liner" bug fixes, but if you think you have been
overlooked please get in touch.


Pull requests
~~~~~~~~~~~~~

Accepted pull requests will usually be squashed to the master branch,
and included in the next release of Euphonic.

Please ensure that:

- Tests are passing, including ``ruff check``.
  - Ruff checks can be ignored with ``#noqa: R123`` if necessary
    (where R123 is the rule code) but please use this capability
    responsibly.
- Only relevant files are changed; it can be difficult to review code
  if the diff includes many irrelevant formatting changes.
- An appropriate note is included in the *CHANGELOG.rst* file.
- Any new features are explained in the documentation.

Releases
~~~~~~~~

Releases are created by Euphonic maintainers using a Github Actions
pipeline. Version numbers are incremented automatically; you do not
need to edit these as part of a PR. If you need something on
``master`` to be released to PyPI/conda-forge urgently, let us know.


Documentation
~~~~~~~~~~~~~

The documentation uses ``Sphinx <https://www.sphinx-doc.org>``_ and is
published with ``readthedocs.com <https://about.readthedocs.com>``_.
The source files are under *doc/source*; if a new Python module is
added, it is necessary to create a new docs page and add it to the
*python-api.rst* index.
