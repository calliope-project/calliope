#!/usr/bin/env python

# Sets the __version__ variable
exec(open('calliope/_version.py').read())

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='calliope',
    version=__version__,
    author='Stefan Pfenninger',
    author_email='stefan@pfenninger.org',
    description='A multi-scale energy systems (MUSES) modeling framework',
    packages=['calliope'],
    install_requires=[
        "coopr >= 3.5.8669",
        "numpy >= 1.9.0",
        "numexpr >= 2.4",
        "pandas >= 0.15.0",
        "pyyaml >= 3.10",
        "tables >= 3.1.0"  # Requires cython to build
    ],
    entry_points={
        'console_scripts': [
            'calliope_run = calliope.parallel:main',
        ]
    }
)