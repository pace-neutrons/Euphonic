try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='CastepPy',
    version='0.1dev',
    description="""Module to read CASTEP electronic/vibrational frequency data
                   and output a dispersion/dos plot"""
    packages=['casteppy'],
    install_requires=[
        'numpy>=1.9.1',
        'matplotlib>=1.4.2',
        'seekpath>=1.1.0',
        'pint>=0.8.0'
    ]
)
