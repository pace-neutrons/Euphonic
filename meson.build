project('euphonic', 'c', meson_version: '>=1.1')

build = get_option('python_only') ? disabler() : []

src = ['c/_euphonic.c', 'c/dyn_mat.c', 'c/util.c', 'c/py_util.c', 'c/load_libs.c']

openmp = dependency('openmp', required: false, language: 'c')

py = import('python').find_installation(pure: false)
py_dep = py.dependency()

py.install_sources(
    'euphonic/__init__.py',
    'euphonic/brille.py',
    'euphonic/broadening.py',
    'euphonic/crystal.py',
    'euphonic/debye_waller.py',
    'euphonic/force_constants.py',
    'euphonic/io.py',
    'euphonic/plot.py',
    'euphonic/powder.py',
    'euphonic/qpoint_frequencies.py',
    'euphonic/qpoint_phonon_modes.py',
    'euphonic/sampling.py',
    'euphonic/spectra.py',
    'euphonic/structure_factor.py',
    'euphonic/util.py',
    'euphonic/validate.py',
    subdir : 'euphonic',
)

py.install_sources(
    'euphonic/cli/__init__.py',
    'euphonic/cli/brille_convergence.py',
    'euphonic/cli/dispersion.py',
    'euphonic/cli/dos.py',
    'euphonic/cli/intensity_map.py',
    'euphonic/cli/optimise_dipole_parameter.py',
    'euphonic/cli/powder_map.py',
    'euphonic/cli/show_sampling.py',
    'euphonic/cli/utils.py',    
    subdir : 'euphonic/cli',
)

py.install_sources(
    'euphonic/data/__init__.py',
    'euphonic/data/bluebook.json',
    'euphonic/data/reciprocal_spectroscopy_definitions.txt',
    'euphonic/data/sears-1992.json',
    subdir : 'euphonic/data',
)

py.install_sources(
    'euphonic/readers/__init__.py',
    'euphonic/readers/castep.py',
    'euphonic/readers/phonopy.py',
    subdir : 'euphonic/readers',
)

py.install_sources(
    'euphonic/styles/__init__.py',
    'euphonic/styles/base.mplstyle',
    'euphonic/styles/intensity_widget.mplstyle',
    subdir : 'euphonic/styles',
)

py.install_sources(
    'euphonic/writers/__init__.py',
    'euphonic/writers/phonon_website.py',
    subdir : 'euphonic/writers',
)

np = dependency('numpy', required: false)

if not np.found() # Try default
  np_inc = include_directories(py.get_path('platlib') / 'numpy/core/include')
  np = declare_dependency(include_directories: np_inc)
endif

py.extension_module(
  '_euphonic',
  src,
  dependencies: [build, py_dep, np, openmp],
  install: true,
)
