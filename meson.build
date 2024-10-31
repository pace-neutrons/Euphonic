project('euphonic', 'c', meson_version: '>=1.1')

build = get_option('python_only') ? disabler() : []

src = ['c/_euphonic.c', 'c/dyn_mat.c', 'c/util.c', 'c/py_util.c', 'c/load_libs.c']

openmp = dependency('openmp', required: false, language: 'c')

py = import('python').find_installation(pure: false)
py_dep = py.dependency()

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
