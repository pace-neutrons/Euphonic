.. _refactor:

Refactor details
----------------

There has been a major refactor from ``v0.2.2`` to ``v0.3``. The main changes
are:

- Crystal parameters (cell vectors, atomic positions etc.) are now contained
  within a ``Crystal`` object
- Crystal parameters referencing ions have been renamed to atoms e.g.
  ``n_atoms``, ``atom_r``, ``atom_type``
- Shortened names have been extended ``cell_vecs`` -> ``cell_vectors``,
  ``eigenvecs``-> ``eigenvectors``, ``freqs``->``frequencies`` etc.
- ``InterpolationData`` has been renamed to ``ForceConstants`` and is now a
  top-level import e.g. ``from euphonic import ForceConstants``
- ``PhononData`` has been renamed to ``QpointPhononModes`` and is now a
  top-level import e.g. ``from euphonic import QpointPhononModes``
- LO-TO split frequencies are no longer contained in their own arrays
  (``split_freqs``, ``split_eigenvecs``, ``split_i``), instead gamma-points are
  duplicated in the main ``frequencies`` array
- ``ForceConstants`` is no longer a subclass of ``QpointPhononModes``. When
  ``ForceConstants.calculate_fine_phonons`` is called, a new
  ``QpointPhononModes`` object is returned, rather than writing the frequencies
  into the ``ForceConstants`` object.
- Some attributes are now pint ``Quantity`` objects to ensure they have the
  correct units. e.g. ``ForceConstants.dielectric``
- All objects have ``from_dict`` methods
- ``BandsData`` and the ability to read CASTEP .bands files has been removed.
  This will be implemented in another project. Get in contact for details.
