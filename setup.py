import versioneer
try:
    from setuptools import setup, Extension
    from setuptools.command.install import install
except ImportError:
    from distutils.core import setup, Extension
    from distutils.command.install import install


class InstallCommand(install):
    user_options = install.user_options + [('python-only', None, 'Install only Python')]

    def initialize_options(self):
        install.initialize_options(self)
        self.python_only = 0

    def finalize_options(self):
        install.finalize_options(self)
        if not bool(self.python_only):
            self.distribution.ext_modules = [get_c_extension()]


def get_c_extension():
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
        # OSX - if CC not set, assume brew install llvm
        try:
            brew_prefix_cmd_return = subprocess.run(["brew", "--prefix"],
                                                    stdout=subprocess.PIPE)
            brew_prefix = brew_prefix_cmd_return.stdout.decode("utf-8").strip()
        except FileNotFoundError:
            brew_prefix = None

        if brew_prefix and not os.environ.get('CC'):
            os.environ['CC'] = '{}/opt/llvm/bin/clang'.format(brew_prefix)
            link_args = ['-L{}/opt/llvm/lib'.format(brew_prefix), '-fopenmp']
        else:
            link_args = ['-fopenmp']

        compile_args = ['-fopenmp']            

    else:
        # Linux - assume gcc if CC not set
        if not os.environ.get('CC'):
            os.environ['CC'] = 'gcc'

        compile_args = ['-fopenmp']
        link_args = ['-fopenmp']

    euphonic_c_extension = Extension(
        'euphonic._euphonic',
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        include_dirs=include_dirs,
        sources=sources
    )
    return euphonic_c_extension


def run_setup(build_c=True):

    with open('README.rst', 'r') as f:
        long_description = f.read()

    packages = ['euphonic',
                'euphonic.cli',
                'euphonic.readers',
                'euphonic.data']

    package_data = {'euphonic' : ['data/*.json', 'data/constants_en.txt',
                                  'data/default_en.txt']}

    cmdclass = versioneer.get_cmdclass()
    cmdclass['install'] = InstallCommand

    setup(
        name='euphonic',
        version=versioneer.get_version(),
        cmdclass=cmdclass,
        author='Rebecca Fair',
        author_email='rebecca.fair@stfc.ac.uk',
        classifiers=[
            'Development Status :: 4 - Beta',
            'Environment :: Console',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
            'Programming Language :: C',
            'Programming Language :: Python :: 3',
            'Topic :: Scientific/Engineering :: Physics',
            ],
        description=(
            'Euphonic calculates phonon bandstructures and inelastic '
            'neutron scattering intensities from modelling code output '
            '(e.g. CASTEP)'),
        license='GPLv3',
        license_files=('LICENSE',),
        long_description=long_description,
        long_description_content_type='text/x-rst',
        url='https://github.com/pace-neutrons/Euphonic',
        packages=packages,
        package_data=package_data,
        install_requires=[
            'numpy>=1.12.1',
            'scipy>=1.0.0',
            'seekpath>=1.1.0',
            'pint>=0.9',
            'importlib_resources>=1.3.0',
            'threadpoolctl>=1.0.0'
        ],
        extras_require={
            'matplotlib': ['matplotlib>=2.0.0'],
            'phonopy_reader': ['h5py>=2.7.0', 'PyYAML>=3.13']
        },
        entry_points={'console_scripts': [
            'euphonic-dispersion = euphonic.cli.dispersion:main',
            'euphonic-dos = euphonic.cli.dos:main',
            'euphonic-optimise-eta = euphonic.cli.optimise_eta:main',
            'euphonic-show-sampling = euphonic.cli.show_sampling:main',
            'euphonic-intensity-map = euphonic.cli.intensity_map:main',
            'euphonic-powder-map = euphonic.cli.powder_map:main']}
    )

run_setup()
