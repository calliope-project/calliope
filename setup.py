#!/usr/bin/env python

from setuptools import setup, find_packages


# Sets the __version__ variable
exec(open('calliope/_version.py').read())

setup(
    name='calliope',
    version=__version__,
    author='Stefan Pfenninger',
    author_email='stefan@pfenninger.org',
    description='A multi-scale energy systems (MUSES) modeling framework',
    packages=find_packages(),
    package_data={'calliope': ['config/*.yaml',
                               'example_model/*.yaml',
                               'example_model/model_config/*.yaml',
                               'example_model/model_config/data/*.csv',
                               'test/common/*.yaml',
                               'test/common/t_1h/*.csv',
                               'test/common/t_6h/*.csv',
                               'test/common/t_erroneous/*.csv']},
    install_requires=[
        "coopr >= 3.5.8669",
        "numpy >= 1.9.0",
        "numexpr >= 2.4",
        "pandas >= 0.15.0",
        "pyyaml >= 3.10",
        "tables >= 3.1.0",  # Requires cython to build
        "click >= 3.3"
    ],
    entry_points={
        'console_scripts': [
            'calliope = calliope.cli:cli'
        ]
    }
)
