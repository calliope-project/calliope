#!/usr/bin/env python

from setuptools import setup, find_packages


# Sets the __version__ variable
exec(open('calliope/_version.py').read())

setup(
    name='calliope',
    version=__version__,
    author='Stefan Pfenninger and Calliope contributors listed in AUTHORS',
    author_email='stefan@pfenninger.org',
    description='A multi-scale energy systems (MUSES) modeling framework',
    license='Apache 2.0',
    url='http://www.callio.pe/',
    download_url='https://github.com/calliope-project/calliope/releases',
    packages=find_packages(),
    package_data={'calliope': ['config/*.yaml',
                               'example_models/national_scale/*.yaml',
                               'example_models/national_scale/model_config/*.yaml',
                               'example_models/national_scale/model_config/data/*.csv',
                               'test/common/*.yaml',
                               'test/common/t_1h/*.csv',
                               'test/common/t_6h/*.csv',
                               'test/common/t_erroneous/*.csv']},
    install_requires=[
        "click >= 3.3",
        "netcdf4 >= 1.2.2",
        "numexpr >= 2.3.1",
        "numpy >= 1.12",
        "pandas >= 0.19, < 0.20",
        "pyomo >= 5.0, < 5.1",
        "pyyaml >= 3.11",
        "xarray >= 0.9.5, < 0.10",
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
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3 :: Only'
    ],
    keywords=['energy systems', 'optimization', 'mathematical programming']
)
