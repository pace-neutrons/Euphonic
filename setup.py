import os

from setuptools import setup, Extension
from setuptools.command.install import install

import versioneer

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


# cmdclass = versioneer.get_cmdclass()
# cmdclass['install'] = InstallCommand
# cmdclass['bdist_wheel'] = bdist_wheel

# setup(
#     cmdclass=cmdclass,
# )
