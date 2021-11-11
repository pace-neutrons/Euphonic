import os
import shutil
import warnings

import versioneer
from setuptools import setup, Extension, Distribution
from setuptools.command.install import install


# As the C extension is optional, this is not detected when building
# wheels and they are incorrectly marked as universal. Explicitly
# specify the distribution is not pure Python
try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False
except ImportError:
    bdist_wheel = None


class InstallCommand(install):
    user_options = install.user_options + [('python-only', None, 'Install only Python')]

    def initialize_options(self):
        install.initialize_options(self)
        self.python_only = 0

    def finalize_options(self):
        install.finalize_options(self)
        if not bool(self.python_only):
            self.distribution.ext_modules = [get_c_extension()]
        if self.distribution.has_ext_modules():
            self.install_lib = self.install_platlib


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


def run_setup():

    license_file = 'LICENSE'
    # A list of files outside the Euphonic package directory that should
    # be included in the installation in site-packages. They must
    # also be added to MANIFEST.in
    ex_install_files = [license_file,
                        'CITATION.cff']
    # MANIFEST.in will add any included files to the site-packages
    # installation, but only if they are inside the package directory,
    # so temporarily copy them there if needed
    for ex_install_file in ex_install_files:
        try:
            shutil.copy(ex_install_file, 'euphonic')
        except (PermissionError, OSError) as err:
            warnings.warn(f'{err}', stacklevel=2)

    packages = ['euphonic',
                'euphonic.cli',
                'euphonic.readers',
                'euphonic.data',
                'euphonic.styles']

    with open('README.rst', 'r') as f:
        long_description = f.read()


    cmdclass = versioneer.get_cmdclass()
    cmdclass['install'] = InstallCommand
    cmdclass['bdist_wheel'] = bdist_wheel

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
        license_files=(license_file,),
        long_description=long_description,
        long_description_content_type='text/x-rst',
        url='https://github.com/pace-neutrons/Euphonic',
        packages=packages,
        include_package_data=True,
        install_requires=[
            'numpy>=1.12.1',
            'scipy>=1.0.0',
            'seekpath>=1.1.0',
            'spglib>=1.9.4',
            'pint>=0.9',
            'importlib_resources>=1.3.0',
            'threadpoolctl>=1.0.0'
        ],
        extras_require={
            'matplotlib': ['matplotlib>=2.0.0'],
            'phonopy_reader': ['h5py>=2.7.0', 'PyYAML>=3.13'],
            'brille': ['brille>=0.5.4']
        },
        entry_points={'console_scripts': [
            'euphonic-dispersion = euphonic.cli.dispersion:main',
            'euphonic-dos = euphonic.cli.dos:main',
            'euphonic-optimise-dipole-parameter = euphonic.cli.optimise_dipole_parameter:main',
            'euphonic-show-sampling = euphonic.cli.show_sampling:main',
            'euphonic-intensity-map = euphonic.cli.intensity_map:main',
            'euphonic-powder-map = euphonic.cli.powder_map:main']}
    )

    for ex_install_file in ex_install_files:
        # Only delete file if there is another copy in the current dir
        # (e.g. from a git clone) in a PyPI dist the only copy will be
        # in the euphonic subdir, we don't want to delete that!
        if os.path.isfile(ex_install_file):
            try:
                os.remove(os.path.join('euphonic', ex_install_file))
            except (PermissionError, OSError, FileNotFoundError) as err:
                warnings.warn(f'{err}', stacklevel=2)

run_setup()
