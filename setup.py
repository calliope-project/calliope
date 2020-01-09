#!/usr/bin/env python

from pathlib import Path
from setuptools import setup, find_packages


# Sets the __version__ variable
exec(open('calliope/_version.py').read())

with open('README.md') as f:
    long_description = f.read()


def get_subdirs(path, glob_string):
    return [
        i.relative_to(str(path) + '/')
        for i in path.glob(glob_string) if i.is_dir()
    ]


def find_calliope_package_data():
    """Returns a list of found directories with package data files"""
    path = Path('./calliope')
    package_data = ['config/*.yaml', 'config/*.html', 'test/common/*.yaml']
    for subdir_level in ['*', '*/*', '*/*/*']:
        for example_dir in get_subdirs(path, 'example_models/' + subdir_level):
            package_data.append(str(example_dir) + '/*.csv')
            package_data.append(str(example_dir) + '/*.yaml')
            package_data.append(str(example_dir) + '/*.rst')
    for test_case_dir in get_subdirs(path, 'test/common/*'):
        package_data.append(str(test_case_dir) + '/*.csv')
    print(package_data)
    return package_data


setup(
    name='calliope',
    version=__version__,
    author='Calliope contributors listed in AUTHORS',
    author_email='stefan@pfenninger.org',
    description='A multi-scale energy systems modelling framework',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='Apache 2.0',
    url='https://www.callio.pe/',
    download_url='https://github.com/calliope-project/calliope/releases',
    packages=find_packages(),
    package_data={'calliope': find_calliope_package_data()},
    install_requires=[
        "bottleneck >= 1.2",
        "click >= 7, < 8",
        "ipython >= 7",
        "ipdb >= 0.11",
        "jinja2 >= 2.10",
        "natsort >= 5",
        "netcdf4 >= 1.2.2",
        "numexpr >= 2.3.1",
        "numpy >= 1.15",
        "pandas >= 0.25, < 0.26",
        "plotly >= 3.3, < 4.0",
        "pyomo >= 5.6, < 5.7",
        "ruamel.yaml >= 0.16",
        "scikit-learn >= 0.22",
        "xarray >= 0.14, < 0.15",
    ],
    entry_points={
        'console_scripts': [
            'calliope = calliope.cli:cli'
        ]
    },
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only'
    ],
    keywords=['energy systems', 'optimisation', 'mathematical programming']
)
