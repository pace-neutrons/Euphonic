`Unreleased <https://github.com/pace-neutrons/Euphonic/compare/v1.4.0.post1...HEAD>`_
----------------------------------------------------------------------------------

- Bug fixes

  - CASTEP 25.1 allows Born effective charges to be calculated by
    Berry Phase methods without a corresponding dielectric tensor. In
    such cases, no long-range term can be subtracted from the Force
    Constants (or reconstructed).  Euphonic uses the presence of Born
    effective charges to indicate such a subtraction; to prevent
    ill-defined cases, ForceConstants now sets both Born charges and
    dielectric tensor to None if only one was provided.

  - An optional parameter is provided to change how bin edges are
    obtained from bin centres: previously the bin edges were constrained
    to the initial data range, but this can lead to incorrect scaling
    when performing broadening. Variable-width broadening schemes are
    now allowed to extrapolate the bin edges in order to get the correct
    width scaling.

    Outside of broadening, the default behaviour is unchanged in order
    to maintain backward compatibility. This is likely to be changed
    in the next "major version" (i.e. API-breaking release) of
    Euphonic.

- Maintenance

  - The euphonic.spectra module has been broken up into a subpackage
    with the single-spectrum classes defined in euphonic.spectra.base
    and the collections in euphonic.spectra.collections. This is not a
    breaking change: the public classes, functions and type annotations
    remain importable from euphonic.spectra.

  - CASTEP 25.1 includes an extra field in .castep_bin files,
    indicating whether Born effective charges were read from an
    external file. For clarity and safety, this field is now
    explicitly read by the Euphonic .castep_bin parser, but remains unused.

  - The release process has been reworked to reduce manual steps: a
    top-level "release" action will now sequence most of the
    steps. (Post-release testing is still separate.)

`v1.4.0.post1 <https://github.com/pace-neutrons/Euphonic/compare/v1.4.0...v1.4.0.post1>`_
-----------------------------------------------------------------------------------------

This post-release makes some changes to the source-distribution build process:
- A bug is fixed in the version-numbering script; this particularly affected Windows
- A copies of the unit tests and documentation were mistakenly included
  in tarballs, making them excessively large. This is no longer present.


- Bug fixes

  - Fixed a bug in the version numbering mechanism affecting builds from sdist
    on Windows and environments where Git is unavailable

  - Reduce size of sdist, removing docs and tests from
    archive. (Restoring similar size to pre-v1.4.0 releases.)

- Maintenance

  - Source builds automatically tested on Windows as part of
    build/deployment process

`v1.4.0 <https://github.com/pace-neutrons/Euphonic/compare/v1.3.2...v1.4.0>`_
-----------------------------------------------------------------------------

This release includes some significant modernisation and maintenance,
as well as new features and performance enhancements.

- Requirements

  - Python 3.8, 3.9 is no longer supported

  - Python 3.12 is supported

  - importlib_resources backport is no longer required

  - `toolz <https://toolz.readthedocs.io/en/latest/index.html>`_ is
    a new requirement

  - Some other dependency requirements have been increased in order
    to simplify maintenance and testing:

    - Minimum version of numpy increased from 1.19.3 to 1.24.0

    - Minimum version of matplotlib increased from 3.2 to 3.8

    - Minimum version of Pint increased from 0.19 to 0.22

    - Minimum version of PyYAML increased from 3.13 to 6.0

    - Minimum version of h5py increaased from 2.10 to 3.6

    - Minimum version of threadpoolctl increased from 1.0 to 3.0.


- Improvements

  - A "reciprocal_spectroscopy" Pint context is made available in the
    unit registry for tricky conversions between reciprocal
    frequency/energy units. It is not active by default but can be
    enabled with e.g.

      (10 * ureg("1 / meV")).to("cm", "reciprocal_spectroscopy")

    This can also help to avoid divide-by-zero issues when performing
    energy <-> wavenumber conversions.

  - A Spectrum2DCollection class has been added to euphonic.spectra,
    which shares many features with Spectrum1DCollection

    - In particular, the ``iter_metadata`` method is recommended when
      one needs to iterate over the collection metadata without
      copying the spectral data to new objects.

  - Both Spectrum1DCollection and Spectrum2DCollection have a
    ``.from_spectra()`` constructor with an "unsafe" option which
    bypasses some consistency checks on the component data. This
    should only be used when confident that these will be consistent,
    such as when iterating over an existing collection.

  - Performance optimisations have been made to the "item getter" for
    Spectrum1DCollection (and Spectrum2DCollection); it should now be
    significantly faster to access and iterate over the contained
    spectra.

  - A ``euphonic.writers.phonon_website`` module has been added with a
    function to export QpointPhononModes to appropriate JSON for use
    with the phonon visualisation website
    http://henriquemiranda.github.io/phononwebsite/

    From the command-line, this can be accessed with a
    ``--save-web-json`` argument to the ``euphonic-dispersion`` tool.

- Bug fixes

  - Metadata strings from Castep-imported PDOS data are now converted
    from numpy strings to native Python strings.

  - Spectra from CASTEP .phonon_dos files are now imported with units
    of reciprocal energy (e.g. 1/meV)

- Maintenance

  - Cleared up unit-conversion-related warnings, de-cluttering the
    expected test suite output.

  - The Spectrum1DCollection class was significantly refactored to
    support addition of Spectrum2DCollection and improve
    maintainability.

  - Entire build system rework, migrating to ``pyproject.toml`` form
    with ``meson-python``, ``cibuildwheel`` and removing
    ``versioneer`` to simplify future development and maintenance.

`v1.3.2 <https://github.com/pace-neutrons/Euphonic/compare/v1.3.1...v1.3.2>`_
-----------------------------------------------------------------------------

- Requirements

  - ``packaging`` library added to dependencies.

- Bug fixes

  - Fixed an error loading QpointPhononModes from JSON when there is a
    single q-point in the data

- Improvements

  - When loading ``.castep_bin`` files, explicitly check the CASTEP
    version number and give a useful error message if this is < 17.1.
    (These files are missing information about the unit cell origins,
    and would previously cause an error with an unhelpful message.)

- Maintenance

  - Compatibility fix for spglib 2.4 update: a new sanity-check in
    spglib raises TypeError when using empty unit cell and this needs
    handling when looking for high-symmetry labels

  - Compatibility fix for Numpy 2.0 update: avoid some
    broadcasting issues with array shape returned by ``np.unique``

  - Update reference to scipy.integrate.simpson (scipy.integrate.simps
    is deprecated)

  - Filter out spglib deprecation warnings caused by SeeK-path.

`v1.3.1 <https://github.com/pace-neutrons/Euphonic/compare/v1.3.0...v1.3.1>`_
-----------------------------------------------------------------------------

- Maintenance

  - Updated versioneer for compatibility with Python 3.12
  - In tests, avoid checking an attribute of 3D plots which is unreliable in
    recent matplotlib versions
  - Update readthedocs configuration to fix documentation builds

`v1.3.0 <https://github.com/pace-neutrons/Euphonic/compare/v1.2.1...v1.3.0>`_
-----------------------------------------------------------------------------

- Requirements

  - Python 3.7 is no longer supported

  - Minimum version of scipy increased from 1.1 to 1.10

    - This requires numpy >= 1.19.5

  - Minimum version of matplotlib increased from 2.2.2 to 3.2.0

  - Minimum version of pint increased from 0.10.1 to 0.19

  - Minimum version of h5py increaased form 2.8 to 2.10

- Improvements

  - Added variable-width broadening for 1-D and 2-D spectra. An
    arbitrary Callable can be provided relating the axis position to
    Gaussian or Lorentzian width parameter. ``euphonic-dos`` and
    ``euphonic-powder-map`` CLI tools accept polynomial coefficients
    as input. The broadening is implemented with the fast approximate
    interpolation method already available for adaptive broadening of
    DOS.

  - Added features to Spectrum classes

    - Added ``copy()`` methods returning an independent duplicate of data

    - Added ``__mul__`` and ``__imul__`` methods to Spectrum
      classes. This allows results to be conveniently scaled with
      infix notation ``*`` or ``*=``

  - Added `--scale` parameter to ``euphonic-dos``,
    ``euphonic-intensity-map``, ``euphonic-powder-map`` to allow
    arbitrary scaling of results from command-line. (e.g. for
    comparison with experiment, or changing DOS normalisation from 1
    to 3N.)

- Bug Fixes:

  - Changed the masking logic for kinematic constraints: instead of
    requiring energy bin to _entirely_ fall within accessible range at
    Q-bin mid-point, unmask bins if _any_ part of energy range is
    accessible at this Q value. This gives much more intuitive
    behaviour, especially for narrow angle ranges.

`v1.2.1 <https://github.com/pace-neutrons/Euphonic/compare/v1.2.0...v1.2.1>`_
-----------------------------------------------------------------------------

- Improvements

  - Added "prefer_non_loto" option to Castep *.phonon* file
    importers. When this is enabled, a block of q-points are
    encountered with splitting directions, and one q-point does not
    have a splitting direction, the data at this "exact" q-point is
    preferred and the other weights in the group are set to zero.

    This provides the *intended* behaviour of the Abins Castep parser
    and should give a reasonable result for Gamma-point only Castep
    calculations.

    The option is disabled by default, so existing scripts will not be
    affected.

- Bug Fixes:

  - Allow ``color`` to be passed as an extra kwarg to ``plot_1d`` and
    ``plot_1d_to_axis``. Previously this caused a ``TypeError``.
  - Fix bug where ``Py_None`` was not incremented before returning from
    ``calculate_phonons()`` in the C-extension causing a deallocation crash
  - Support phonopy.yaml files from Phonopy versions >= 1.18, which
    have moved the data relating to dipole-dipole
    corrections. (i.e. Born effective charges, static dielectric
    tensor and a related unit conversion factor.)

- Maintenance:

  - A deprecation in Numpy 1.25, which indirectly caused a test failure, has been addressed.

`v1.2.0 <https://github.com/pace-neutrons/Euphonic/compare/v1.1.0...v1.2.0>`_
-----------------------------------------------------------------------------

- Improvements:

  - Euphonic now tests on Python 3.11
  - Euphonic now provides PyPI wheels for Python 3.11

- New features:

  - You can now perform linear interpolation of phonon frequencies and
    eigenvectors with the `Brille <https://brille.github.io/stable/index.html>`_
    library using the new
    ``euphonic.brille.BrilleInterpolator`` object. This should provide
    performance improvements for large unit cells which require the
    dipole correction.
  - There is a new command-line tool ``euphonic-brille-convergence`` to
    assist with choosing the ``BrilleInterpolator.from_force_constants``
    arguments to achieve the desired accuracy.
  - Brille interpolation can be accessed from the ``euphonic-powder-map`` tool
    using the new ``--use-brille``, ``--brille-grid-type``, ``--brille-npts``
    and ``--brille-npts-density`` arguments.

`v1.1.0 <https://github.com/pace-neutrons/Euphonic/compare/v1.0.0...v1.1.0>`_
-----------------------------------------------------------------------------

- New features:

  - There is a new function ``ForceConstants.from_total_fc_with_dipole`` to allow
    reading force constants from other programs which contain long-ranged
    dipole-dipole interactions.

- Bug fixes:

  - Avoid occasional segmentation faults when using OpenBLAS, workaround for
    `#191 <https://github.com/pace-neutrons/Euphonic/issues/191>`_
  - Correctly read force constants from Phonopy with dipole-dipole
    interactions, see `#239 <https://github.com/pace-neutrons/Euphonic/issues/239>`_.

`v1.0.0 <https://github.com/pace-neutrons/Euphonic/compare/v0.6.5...v1.0.0>`_
-----------------------------------------------------------------------------

- Changes:

  - Support for Python 3.6 has been dropped. This has also resulted in
    changes to the following dependencies:

    - numpy requirement increased from ``1.12.1`` to ``1.14.5``
    - scipy requirement increased from ``1.0.0`` to ``1.1.0``
    - pint requirement increased from ``0.9`` to ``0.10.1``
    - matplotlib requirement increased from ``2.0.0`` to ``2.2.2``
    - h5py requirement increased from ``2.7.0`` to ``2.8.0``

  - The following deprecated features have been removed:

    - The ``return_mode_widths`` argument in ``ForceConstants.calculate_qpoint_phonon_modes``
      and ``ForceConstants.calculate_qpoint_frequencies`` has been removed
    - The ``eta_scale`` argument in ``calculate_qpoint_phonon_modes/frequencies``
      has been removed
    - The alias command-line tool argument ``--weights`` has been removed
    - The alias arguments ``x_label``, ``y_label``, ``y_min`` and ``y_max`` to
      ``plot_1d/2d`` have been removed
    - The ``modes_from_file`` and ``force_constants_from_file`` functions from
      ``euphonic.cli.utils`` have been removed
    - Calling ``broaden`` on a ``Spectrum`` with uneven bin widths without
      specifying the ``method='convolve'`` argument will now raise a ``ValueError``

  - DOS and PDOS calculated by the ``calculate_dos`` and
    ``calculate_dos_map`` methods of ``QpointPhononModes`` and
    ``QpointFrequencies``, and ``QpointPhononModes.calculate_pdos`` are
    now calculated per atom rather than per unit cell (integrated area
    is ``3`` rather than ``3*N_atom``). This is to keep consistency with
    the structure factors calculated by
    ``QpointPhononModes.calculate_structure_factor`` which are calculated
    per atom.

  - The option ``average_repeat_points`` when importing q-point modes or
    frequencies from a CASTEP .phonon file with
    ``QpointFrequencies/QpointPhononModes.from_castep`` is now ``True``
    by default. To recover previous behaviour set this to ``False``.

`v0.6.5 <https://github.com/pace-neutrons/Euphonic/compare/v0.6.4...v0.6.5>`_
-----------------------------------------------------------------------------

- New Features:

  - Kinematic constraints have been implemented for 2-D S(q,w)-like data.

    - A function ``euphonic.spectra.apply_kinematic_constraints(Spectrum2d, **kwargs) -> Spectrum2D``
      is implemented which masks out inaccessible data, replacing it with NaN.
    - Both direct-geometry and indirect-geometry are supported, by
      using the appropriate argument to set incident or final neutron energy.
    - This function is exposed to the ``euphonic-powder-map`` tool, so these
      plots can be produced directly from the CLI.
    - Some parameters from real-world instruments are collected in the
      documentation for convenience.

  - There is a new function ``euphonic.util.convert_fc_phases``, which converts
    a force constants matrix which uses the atom coordinates in the phase
    during interpolation (Phonopy-like), to one which uses the cell origin
    coordinates (Euphonic, CASTEP-like).

  - When importing q-point modes or frequencies from a CASTEP .phonon
    file, a new option (``average_repeat_points=True``) allows
    repeated entries (with the same q-point index) to be identified
    and their weights divided down by the number of entries. This
    option should give better statistics for sampling meshes that
    include the Gamma-point with LO-TO splitting.

- Improvements:

  - Documentation on the shape and format of the force constants, and how to
    read them from other programs has been improved.

  - The ``euphonic.util.get_qpoint_labels`` function, which is called when
    importing band-structure data to identify and label significant points,
    primarily identifies these points by searching for turning-points
    in the band path. The function will now also pick up any q-point
    that appears twice in succession. This is a common convention in
    band-structure calculations and helps with edge-cases such as when
    the path passes through a high-symmetry point without changing
    direction. This may pick up some previously-missing points in
    band-structure plots generated with ``euphonic-dispersion`` and
    ``euphonic-intensity-map``

- Bug fixes:

  - Allow read of ``phonopy.yaml`` quantities in ``'au'`` (bohr) units.
    Previously this was interpreted as an astronomical unit by Pint.

`v0.6.4 <https://github.com/pace-neutrons/Euphonic/compare/v0.6.3...v0.6.4>`_
-----------------------------------------------------------------------------

- Improvements:

  - The ``euphonic-dos``, ``euphonic-dispersion`` and
    ``euphonic-intensity-map`` command-line tools can now read
    files that don't contain eigenvectors, if eigenvectors are
    not required for the chosen options.
  - A new ``--save-json`` option is available for command-line tools
    which produce plots, this will output the produced spectrum to
    a Euphonic .json file.
  - There is now the option to use a fast, approximate variable-width broadening method when
    adaptively broadening dos:

    - Added new ``adaptive_method`` and ``adaptive_error`` arguments for ``calculate_dos``
      which specify which adaptive broadening method to use (``reference`` or ``fast``) and an
      acceptable error level when using the ``fast`` method.
    - Fast adaptive broadening can be used in the ``euphonic-dos`` tool with the
      ``--adaptive-method`` and ``--adaptive-error`` arguments.

- Changes:

  - ``euphonic.cli.force_constants_from_file`` and ``modes_from_file``
    have been deprecated in favour of ``euphonic.cli.load_data_from_file``.
  - Using ``Spectrum1D/1DCollection/2D.broaden`` on an axis with unequal
    bin widths is now deprecated, as broadening is performed via convolution,
    which is incorrect in this case. In the future, this will raise a
    ``ValueError``. To broaden anyway, ``method='convolve'`` can be supplied,
    which will just emit a warning.

`v0.6.3 <https://github.com/pace-neutrons/Euphonic/compare/v0.6.2...v0.6.3>`_
-----------------------------------------------------------------------------

- New Features:

  - New ``Spectrum1D.to_text_file`` and ``Spectrum1DCollection.to_text_file``
    methods to write to column text files

  - An expanded and consistent set of styling options is made
    available for command-line tools that produce plots.

  - Consistent styling and advanced changes can be made using
    Matplotlib stylesheet files, either as a CLI argument or
    using ``matplotlib.style.context()`` in a Python script.

- Improvements:

  - Internally, plot theming has been adjusted to rely on Matplotlib
    style contexts. This means user changes and style context are more
    likely to be respected.
  - Additional aliases for plot arguments in the command-line tools have
    been added, for example either ``--x-label`` or ``--xlabel`` can be used.

- Changes:

  - ``x_label``, ``y_label``, ``y_min`` and ``y_max`` in ``euphonic.plot``
    functions have been deprecated in favour of ``xlabel``, ``ylabel``,
    ``ymin`` and ``ymax`` respectively, to match the Matplotlib arguments
    they refer to, and to match other arguments like ``vmin``, ``vmax``.

`v0.6.2 <https://github.com/pace-neutrons/Euphonic/compare/v0.6.1...v0.6.2>`_
-----------------------------------------------------------------------------

- Improvements:

  - Wheels are now provided with PyPI releases
  - Type hinting is now handled more consistently across different Euphonic
    classes and functions

- Bug Fixes:

  - Will no longer raise a KeyError reading from ``phonopy.yaml`` if
    ``physical_unit`` key is not present, instead will assume default units
  - Can now read Phonopy BORN files where the (optional) NAC conversion
    factor is not present

`v0.6.1 <https://github.com/pace-neutrons/Euphonic/compare/v0.6.0...v0.6.1>`_
-----------------------------------------------------------------------------

- Bug fixes:

  - The scaling of S(Q,w) as produced by ``StructureFactor.calculate_sqw_map``
    was incorrect, and did not correctly scale with energy bin size (given its
    units are now ``length**2/energy``). This has been fixed, and S(Q,w) scale
    has changed by a factor of (hartee to energy bin unit conversion)/(energy
    bin width magnitude). e.g. if using an energy bin width of 0.1 meV, the new
    S(Q,w) will be scaled by 2.72e4/0.1 = 2.72e5. The original structure factors
    can now be correctly recovered by multiplying S(Q,w) by the energy bin width.

`v0.6.0 <https://github.com/pace-neutrons/Euphonic/compare/v0.5.2...v0.6.0>`_
-----------------------------------------------------------------------------

- Euphonic can now calculate neutron-weighted partial density of states, and
  has new ``Spectra`` features to handle PDOS data:

  - Added ``QpointPhononModes.calculate_pdos`` method
  - Added ``QpointFrequencies.calculate_dos_map`` method
  - New ``Spectrum1D.__add__`` method, which adds 2 spectra together
  - New ``Spectrum1DCollection.__add__`` method, which concatenates 2 collections
  - Enabled indexing of ``Spectrum1DCollection`` by a sequence
  - Added ``Spectrum1DCollection.group_by`` method, which allows grouping and
    summing spectra by metadata keys e.g. ``group_by('species')``
  - Added ``Spectrum1DCollection.select`` method, which allows selection
    of spectra by metadata keys e.g. ``select(species='Si')``
  - Added ``Spectrum1DCollection.sum`` method, which sums all spectra in a
    collection
  - Added ``-w={'coherent-dos','incoherent-dos','coherent-plus-incoherent-dos'}``
    neutron-weighted PDOS options to ``euphonic-dos`` and ``euphonic-powder-map``
  - Added ``--pdos`` options for plotting specific species PDOS to
    ``euphonic-dos`` and ``euphonic-powder-map``
  - Deprecated ``--weights`` command-line argument in favour of ``--weighting``
    for consistency with ``calculate_pdos``

- Improvements:

  - LICENSE and `CITATION.cff <https://citation-file-format.github.io/>`_
    files are now included in Euphonic's installation
  - Add ability to interactively change the colormap intensity limits
    in ``euphonic-powder-map``
  - ``euphonic-optimise-dipole-parameter`` can now read from Phonopy sources
  - ``euphonic-optimise-dipole-parameter`` can now also be used for non-polar
    materials to get general per-qpoint timings
  - Dimensioned Euphonic properties (e.g. ``frequencies``, ``cell_vectors``)
    now have setters so can be set, previously this would raise an
    ``AttributeError``

- Changes:

  - The units of density of states as produced by ``calculate_dos`` have
    changed from dimensionless to ``1/energy``
  - The scaling of density of states has also changed. Previously the
    integration would sum to 1 (if the ``x_data`` were converted to Hartree
    units), now the integration will sum to 3N in the same units as ``x_data``
  - ``StructureFactor.structure_factors`` have been changed to be in absolute
    units per atom (rather than per unit cell) so will have changed by a
    factor of `1/2*n_atoms`, this formulation change has been reflected in the
    ``calculate_structure_factor`` docstring
  - The default unit of ``StructureFactor.structure_factors`` has been changed
    from ``angstrom**2`` to ``millibarn``
  - The unit of S(Q,w) as produced by ``StructureFactor.calculate_sqw_map``
    has changed dimension from ``length**2`` to ``length**2/energy``. Also,
    as its unit is derived from the input ``StructureFactor`` object, its
    default units are now ``millibarn/meV``
  - The ``eta_scale`` argument in ``calculate_qpoint_phonon_modes`` has been
    deprecated, ``dipole_parameter`` should be used instead.
  - This means the ``euphonic-optimise-eta`` script has been renamed to
    ``euphonic-optimise-dipole-parameter``.

`v0.5.2 <https://github.com/pace-neutrons/Euphonic/compare/v0.5.1...v0.5.2>`_
-----------------------------------------------------------------------------

- Improvements:

  - Added ``broaden`` method to ``Spectrum1DCollection``

- Changes:

  - The ``return_mode_widths`` argument in ``calculate_qpoint_phonon_modes``
    has been deprecated in favour of ``return_mode_gradients``. The mode
    widths can still be obtained from the mode gradients with
    ``util.mode_gradients_to_widths``

- Bug fixes:

  - Fixed memory leak when using the C extension and making multiple calls to
    ``calculate_qpoint_phonon_modes/frequencies``
  - Fixed bug which resulted in incorrect energy bins being generated
    in ``euphonic-powder-map`` if units other than meV are used and
    ``--e-max`` and ``--e-min`` aren't specified
  - Use correct number of energy bins in ``euphonic-intensity-map``,
    ``euphonic-powder-map`` and ``euphonic-dos``. Previously only
    ``ebins - 1`` bins were generated

`v0.5.1 <https://github.com/pace-neutrons/Euphonic/compare/v0.5.0...v0.5.1>`_
-----------------------------------------------------------------------------

- New Features:

  - New ``Crystal.get_symmetry_equivalent_atoms`` method which uses spglib
    to get the symmetry operations and equivalent atoms under each operation

- Improvements:

  - Added ``symmetrise`` argument to ``QpointPhononModes.calculate_debye_waller``
    which will symmetrise it under the crystal symmetry operations. This
    means that there will no longer be a discrepancy between ``DebyeWaller``
    calculated on a symmetry-reduced or full Monkhorst-Pack grid. By default,
    ``symmetrise=True``
  - Added ``frequencies_min`` argument to ``calculate_debye_waller`` to
    exclude very small frequencies. This will also exclude negative
    frequencies. This improves on the previous behaviour which only excluded
    gamma-point acoustic modes, so would miss small/negative frequencies
    elsewhere
  - Loading the LAPACK libraries for the C extension now uses the
    `interface <https://docs.scipy.org/doc/scipy/reference/linalg.cython_lapack.html>`_
    provided by `scipy` for `cython` instead of loading directly from a DLL.
    The new method means we don't have to guess the DLL filename anymore!

- Changes:

  - New dependency on ``spglib>=1.9.4``
  - Fixed formula in ``calculate_debye_waller`` docstring to match actual
    implementation: moved ``1/2`` factor and added explicit q-point weights

`v0.5.0 <https://github.com/pace-neutrons/Euphonic/compare/v0.4.0...v0.5.0>`_
-----------------------------------------------------------------------------

- New Features:

  - New command-line tool ``euphonic-powder-map`` allows generation
    and plotting of powder-averaged S(|q|,w) and DOS maps.
  - New ``QpointFrequencies`` object which allows storage of frequencies
    without eigenvectors, meaning that memory usage can be reduced if
    eigenvectors are not required.
  - ``StructureFactor`` now has a ``weights`` attribute and can be used
    to calculate DOS with ``calculate_dos`` and get dispersion with
    ``get_dispersion``
  - ``Spectrum1D``, ``Spectrum1DCollection`` and ``Spectrum2D`` objects
    have a new ``metadata`` attribute, see their docstrings for details
  - Euphonic can now read DOS/PDOS from CASTEP .phonon_dos files with
    ``Spectrum1D.from_castep_phonon_dos`` and
    ``Spectrum1DCollection.from_castep_phonon_dos``
  - **Adaptive broadening** is now available for DOS, which can obtain a
    more representative DOS than standard fixed-width broadening. See
    `the docs <https://euphonic.readthedocs.io/en/latest/dos.html#adaptive-broadening>`__
    for details
  - Adaptive broadening can be used in the ``euphonic-dos`` tool with the
    ``--adaptive`` argument

- Improvements:

  - Improved default behaviour for C extension use and number of threads:

    - By default the C extension will be used if it is installed
    - By default the number of threads will be set by
      ``multiprocessing.cpu_count()``
    - The environment variable ``EUPHONIC_NUM_THREADS`` can be used to set
      a specific number of threads, which takes priority over
      ``multiprocessing.cpu_count()``
    - ``fall_back_on_python`` argument has been removed and superseded by the
      default ``use_c=None`` behaviour
    - ``threadpoolctl.threadpool_limits`` is used to limit the number of threads
      used by numerical libraries in Euphonic C function calls, resulting in
      better overall performance

  - Command-line interfaces have been refactored, giving a more
    uniform set of options and clearer sections of related arguments
    on the interactive help pages.

    - It is now possible where appropriate to specify Monkhorst-Pack
      sampling with a single-parameter ``--q-spacing`` as an
      alternative to setting Monkhorst-Pack divisions. This approach
      will account for the size and shape of reciprocal-lattice cells.

  - Build process tweaks

    - On Linux, the build process will now respect a user-defined
      C-compiler variable ``CC``.

    - On Mac OSX, the build process will now respect a user-defined
      C-compiler variable ``CC``. Homebrew library paths will only be
      set if ``CC`` is empty and the ``brew`` command is available.

    These tweaks are intended to facilitate Conda packaging.

- Breaking changes:

  - The ``--q-distance`` argument to ``euphonic-intensity-map`` has
    been renamed to ``--q-spacing`` for consistency with other tools.

  - Debye-Waller calculation in ``euphonic-intensity-map`` is now
    enabled by setting ``--temperature``, which no longer has a
    default value.

  - Default Monkhorst-Pack meshes (i.e. [6, 6, 6] in ``euphonic-dos``
    and [20, 20, 20] in ``sample_sphere_structure_factor()``) have
    been replaced by default grid-spacing values.

  - The scaling of density of states has changed, due to a change
    in implementation

`v0.4.0 <https://github.com/pace-neutrons/Euphonic/compare/v0.3.2...v0.4.0>`_
-----------------------------------------------------------------------------

- There have been some major changes and improvements to spectra, plotting
  and command line tools, including:

  - New command line tool ``euphonic-intensity-map`` for plotting weighted
    2D Spectra e.g. Coherent neutron S(Q,w)
  - Existing command line tools ``euphonic-dispersion`` and ``euphonic-dos``
    have been updated to also read force constants and Phonopy files.
    Arguments are also more consistent across tools so some may have changed,
    check the command line tool help for details.
  - New ``Spectrum1DCollection`` object for containing 1D spectra with a
    shared x-axis (e.g. phonon dispersion modes)
  - New ``plot_1d_to_axis`` and ``plot_2d_to_axis`` functions to allow
    plotting on specific axes
  - ``get_bin_centres`` and ``get_bin_edges`` utility functions on spectra
  - The ``ratio`` argument to ``plot_2d`` has been removed, it should no longer
    be required due to better management of relative axis sizes.
  - The ``btol`` argument to ``plot_1d`` has been removed, it is recommended
    to use ``Spectrum1D.split()`` or ``Spectrum1DCollection.split()`` instead.
  - The ``plot_dispersion`` function has been removed. It is now recommended
    to plot dispersion using ``plot_1d(QpointPhononModes.get_dispersion())``.
    See docs for details.

- Other changes:

  - Some of Euphonic's dependency version requirements have been changed, but
    can now be relied on with more certainty due to better CI testing. This
    includes:

    - numpy requirement increased from ``1.9.1`` to ``1.12.1``
    - matplotlib requirement increased from ``1.4.2`` to ``2.0.0``
    - pint requirement decreased from ``0.10.1`` to ``0.9``
    - h5py requirement decreased from ``2.9.0`` to ``2.7.0``
    - pyyaml requirement decreased from ``5.1.2`` to ``3.13``

- Improvements:

  - ``yaml.CSafeLoader`` is now used instead of ``yaml.SafeLoader`` by
    default, so Phonopy ``.yaml`` files should load faster
  - Metadata ``__euphonic_version__`` and ``__euphonic_class__`` have been
    added to .json file output for better provenance

- Bug fixes:

  - Fix read of Phonopy 'full' force constants from phonopy.yaml and
    FORCE_CONSTANTS files
  - Fix structure factor calculation at gamma points with splitting, see
    `#107 <https://github.com/pace-neutrons/Euphonic/issues/107>`_
  - Change broadening implementation from ``scipy.signal.fftconvolve``
    to use ``scipy.ndimage`` functions for better handling of bright
    Bragg peaks, see
    `#108 <https://github.com/pace-neutrons/Euphonic/issues/108>`_

`v0.3.2 <https://github.com/pace-neutrons/Euphonic/compare/v0.3.1...v0.3.2>`_
-----------------------------------------------------------------------------

- New Features:

  - Added `weights` as an argument to
    `ForceConstants.calculate_qpoint_phonon_modes`, this will allow easier
    use of symmetry reduction for calculating density of states, for example.
  - Modules have been added to support spherical averaging from 3D
    q-points to mod(q)

    - euphonic.sampling provides pure functions for the generation of
      points on (2D) unit square and (3D) unit sphere surfaces.
    - A script is provided for visualisation of the different schemes
      implemented in euphonic.sampling. This is primarily intended for
      education and debugging.
    - euphonic.powder provides functions which, given force constants
      data, can use these sampling methods to obtain
      spherically-averaged phonon DOS and coherent structure factor
      data as 1D spectrum objects. (It is anticipated that this module
      will grow to include schemes beyond this average over a single
      sphere.)
  - Added ``Crystal.to_spglib_cell`` convenience function

- Changes:

  - The Scripts folder has been removed. Command-line tools are now
    located in the euphonic.cli module. The entry-points are managed
    in setup.py, and each tool has the prefix "euphonic-" to avoid
    namespace clashes with other tools on the user's
    computer. (e.g. euphonic-dos)
  - From an interactive shell with tab-completion, one can find all
    the euphonic tools by typing "euphonic-<TAB>".
  - Changed arguments for ``util.get_qpoint_labels(Crystal, qpts)``
    to ``util.get_qpoint_labels(qpts, cell=None)`` where
    ``cell = Crystal.to_spglib_cell()``

- Bug fixes:

  - Correctly convert from Phonopy's q-point weight convention to Euphonic's
    when reading from mesh.yaml (see
    `7509043 <https://github.com/pace-neutrons/Euphonic/commit/7509043>`_)
  - Avoid IndexError in ``ForceConstants.calculate_qpoint_phonon_modes`` when
    there is only one q-point (which is gamma) and ``splitting=True``

`v0.3.1 <https://github.com/pace-neutrons/Euphonic/compare/v0.3.0...v0.3.1>`_
-----------------------------------------------------------------------------

- New Features:

  - A system has been added for reference data in JSON files. These
    are accessed via ``euphonic.utils.get_reference_data`` and some
    data has been added for coherent scattering lengths and cross-sections.
    This system has been made available to the
    ``calculate_structure_factor()`` method; it is no longer necessary to
    craft a data dict every time a program uses this function.

- Changes:

  - Fixed structure factor formula in docs (``|F(Q, nu)|`` -> ``|F(Q, \\nu)|^2``
    and ``e^(Q.r)`` -> ``e^(iQ.r)``)

- Bug fixes:

  - Fix ``'born':null`` in ``ForceConstants`` .json files when Born is not
    present in the calculation (see
    `c20679c <https://github.com/pace-neutrons/Euphonic/commit/c20679c>`_)
  - Fix incorrect calculation of LO-TO splitting when ``reduce_qpts=True``,
    as the 'reduced' q rather than the actual q was used as the q-direction
    (see `3958072 <https://github.com/pace-neutrons/Euphonic/commit/3958072>`_)
  - Fix interpolation for materials with non-symmetric supcercell matrices,
    see `#81 <https://github.com/pace-neutrons/Euphonic/issues/81>`_
  - Fix interpolation for force constants read from Phonopy for materials that
    have a primitive matrix and more than 1 species, see
    `#77 <https://github.com/pace-neutrons/Euphonic/issues/77>`_

`v0.3.0 <https://github.com/pace-neutrons/Euphonic/compare/v0.2.2...v0.3.0>`_
-----------------------------------------------------------------------------

- Breaking Changes:

  - There has been a major refactor, for see the v0.3.0
    `docs <https://euphonic.readthedocs.io/en/v0.3.0>`_ for how to use, or
    `here <https://euphonic.readthedocs.io/en/v0.3.0/refactor.html>`_ for
    refactor details
  - Python 2 is no longer supported. Supported Python versions are ``3.6``,
    ``3.7`` and ``3.8``

- New Features:

  - Euphonic can now read Phonopy input! See
    `the docs <https://euphonic.readthedocs.io/en/v0.3.0>`_
    for details.

- Improvements:

  - Added ``fall_back_on_python`` boolean keyword argument to
    ``ForceConstants.calculate_qpoint_phonon_modes`` to control
    whether the Python implementation is used as a fallback to the C
    extension or not, see
    `#35 <https://github.com/pace-neutrons/Euphonic/issues/35>`_
  - Added ``--python-only`` option to ``setup.py`` to enable install
    without the C extension

- Bug fixes:

  - On reading CASTEP phonon file header information, switch from a fixed
    number of lines skipped to a search for a specific line, fixing issue
    `#23 <https://github.com/pace-neutrons/Euphonic/issues/23>`_
  - Fix NaN frequencies/eigenvectors for consecutive gamma points, see
    `#25 <https://github.com/pace-neutrons/Euphonic/issues/25>`_
  - Fix issue saving plots to file with dispersion.py, see
    `#27 <https://github.com/pace-neutrons/Euphonic/issues/27>`_
  - Fix incorrect frequencies at gamma point when using dipole correction
    in C, `#45 <https://github.com/pace-neutrons/Euphonic/issues/45>`_

`v0.2.2 <https://github.com/pace-neutrons/Euphonic/compare/v0.2.1...v0.2.2>`_
-----------------------------------------------------------------------------

- Bug fixes:

  - Add MANIFEST.in for PyPI distribution

`v0.2.1 <https://github.com/pace-neutrons/Euphonic/compare/v0.2.0...v0.2.1>`_
-----------------------------------------------------------------------------

- Bug fixes:

  - Cannot easily upload C header files to PyPI without an accompanying source
    file, so refactor C files to avoid this

`v0.2.0 <https://github.com/pace-neutrons/Euphonic/compare/v0.1-dev3...v0.2.0>`_
--------------------------------------------------------------------------------

- There are several breaking changes:

  - Changes to the object instantiation API. The former interface
    ``InterpolationData(seedname)`` has been changed to
    ``InterpolationData.from_castep(seedname)`` in anticipation of more codes
    being added which require more varied arguments.
  - Changes to the Debye-Waller calculation API when calculating the structure
    factor. The previous ``dw_arg`` kwarg accepted either a seedname or length
    3 list describing a grid. The new kwarg is now ``dw_data`` and accepts a
    ``PhononData`` or ``InterpolationData`` object with the frequencies
    calculated on a grid. This is to make it clearer to the user exactly what
    arguments are being used when calculating phonons on the grid.
  - Changes to parallel functionality. The previous parallel implementation
    based on Python's multiprocessing has been removed and replaced by a
    C/OpenMP version. This has both better performance and is more robust. As
    a result the ``n_procs`` kwarg to ``calculate_fine_phonons`` has been
    replaced by ``use_c`` and ``n_threads`` kwargs.

- Improvements:

  - The parallel implementation based on Python's multiprocessing has been
    removed and now uses C/OpenMP which both has better performance and is more
    robust
  - Documentation has been moved to readthedocs and is more detailed
  - Clearer interface for calculating the Debye-Waller factor
  - Better error handling (e.g. empty ``InterpolationData`` objects, Matplotlib
    is not installed...)

- Bug fixes:

  - Fix gwidth for DOS not being converted to correct units
  - Fix qwidth for S(Q,w) broadening being incorrectly calculated
