.. _writers:

===========================
Writing to external formats
===========================

Phonon website visualisation
----------------------------

The `phonon visualisation website <https://henriquemiranda.github.io/phononwebsite/phonon.html>`_ is a useful tool for the interpretation of phonon band structures with related eigenvectors.
A :class:`QpointPhononModes <euphonic.qpoint_phonon_modes.QpointPhononModes>` object can be combined with a corresponding set of :math:`\mathbf{q}`-point labels to write an appropriate JSON file, which is importable as a "custom file".

.. autofunction:: euphonic.writers.phonon_website.write_phonon_website_json


