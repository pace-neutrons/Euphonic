from euphonic import __version__

try:
    from setuptools import setup, find_packages, Extension
except ImportError:
    from distutils.core import setup, find_packages, Extension

def run_setup(build_c=True):

    if build_c:
        import os
        import numpy as np
        from sys import platform
        import subprocess
        include_dirs = [np.get_include(), 'c']
        sources = ['c/_euphonic.c', 'c/dyn_mat.c', 'c/util.c', 'c/py_util.c',
                   'c/load_libs.c']
        if platform == 'win32':
            # Windows - assume MSVC compiler
            compile_args = ['/openmp']
            link_args = None
        elif platform == 'darwin':
            # OSX - assume brew install llvm
            brew_prefix_cmd_return = subprocess.run(["brew", "--prefix"], stdout=subprocess.PIPE)
            brew_prefix = brew_prefix_cmd_return.stdout.decode("utf-8").strip()
            os.environ['CC'] = '{}/opt/llvm/bin/clang'.format(brew_prefix)
            compile_args = ['-fopenmp']
            link_args = ['-L{}/opt/llvm/lib'.format(brew_prefix), '-fopenmp']
        else:
            # Linux - assume gcc
            os.environ['CC'] = 'gcc'
            compile_args = ['-fopenmp']
            link_args = ['-fopenmp']

        euphonic_extension = Extension(
            'euphonic._euphonic',
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            include_dirs=include_dirs,
            sources=sources
        )
        ext_modules = [euphonic_extension]
    else:
        ext_modules = None


    with open('README.rst', 'r') as f:
        long_description = f.read()

    packages = ['euphonic',
                'euphonic.data',
                'euphonic.plot',
                'euphonic._readers']

    scripts = ['scripts/dispersion.py',
               'scripts/dos.py',
               'scripts/optimise_eta.py']

    setup(
        name='euphonic',
        version=__version__,
        author='Rebecca Fair',
        author_email='rebecca.fair@stfc.ac.uk',
        description=(
            'Euphonic calculates phonon bandstructures and inelastic neutron '
            'scattering intensities from modelling code output (e.g. CASTEP)'),
        long_description=long_description,
        long_description_content_type='text/x-rst',
        url='https://github.com/pace-neutrons/Euphonic',
        packages=packages,
        install_requires=[
            'numpy>=1.9.1',
            'scipy>=1.0.0',
            'seekpath>=1.1.0',
            'pint>=0.8.0'
        ],
        extras_require={
            'matplotlib': ['matplotlib>=1.4.2'],
            'phonopy_reader': ['h5py>=2.9.0', 'PyYAML>=5.1.2']
        },
        scripts=scripts,
        ext_modules=ext_modules
    )

try:
    run_setup()
except:
    print('*'*79)
    print(('Failed to build Euphonic C extension, installing pure Python '
           'version instead'))
    print('*'*79)
    print("Unexpected error: {}".format(sys.exc_info()[0]))
    run_setup(build_c=False)
