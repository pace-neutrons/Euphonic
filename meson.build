project('euphonic',
        'c',
        version: run_command('python', 'build_utils/version.py', check: true).stdout().strip(),
        meson_version: '>=1.6',
       )

build = get_option('python_only') ? disabler() : []

fs = import('fs')
py = import('python').find_installation(pure: false)

py_dep = py.dependency()

py_src = {
  'euphonic': ['__init__.py', 'brille.py', 'broadening.py',
               'crystal.py', 'debye_waller.py', 'force_constants.py',
               'io.py', 'plot.py', 'powder.py', 'qpoint_frequencies.py',
               'qpoint_phonon_modes.py', 'sampling.py',
               'structure_factor.py', 'util.py', 'validate.py'],
  'euphonic/cli': ['__init__.py', 'brille_convergence.py',
                   'dispersion.py', 'dos.py', 'intensity_map.py',
                   'optimise_dipole_parameter.py', 'powder_map.py',
                   'show_sampling.py', 'utils.py'],
  'euphonic/data': ['__init__.py', 'bluebook.json', 'sears-1992.json',
                    'reciprocal_spectroscopy_definitions.txt'],
  'euphonic/readers': ['__init__.py', 'castep.py', 'phonopy.py'],
  'euphonic/spectra': ['__init__.py', 'base.py', 'collections.py'],
  'euphonic/styles': ['__init__.py', 'base.mplstyle', 'intensity_widget.mplstyle'],
  'euphonic/writers': ['__init__.py', 'phonon_website.py'],
}

foreach folder, sources : py_src
  inc = []
  foreach source : sources
    inc += folder / source
  endforeach
  py.install_sources(inc, subdir: folder)
endforeach

version_script = files('./build_utils/version.py')
meson.add_dist_script('python', version_script, '--dist')

# If building from sdist not under git so will be read from last version
version = configure_file(input: 'build_utils/version.py.in',
                         output: 'version.py',
                         configuration: {'VERSION': meson.project_version()})
py.install_sources(version, 'CITATION.cff', 'LICENSE', subdir: 'euphonic')

src = ['c/_euphonic.c', 'c/dyn_mat.c', 'c/util.c', 'c/py_util.c', 'c/load_libs.c']

np = dependency('numpy', required: false)
if not np.found() # Try default
  np_inc = include_directories(py.get_path('platlib') / 'numpy/core/include')
  np = declare_dependency(include_directories: np_inc)
endif

openmp = dependency('openmp', required: true, language: 'c')

py.extension_module(
  '_euphonic',
  src,
  dependencies: [build, py_dep, np, openmp],
  install: true,
  subdir: 'euphonic',
)
