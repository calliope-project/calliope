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
    license='Apache 2.0',
    url='http://www.callio.pe/',
    download_url='https://github.com/calliope-project/calliope/releases',
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
        "pyomo >= 4.2, < 4.3",
        "numpy >= 1.10.1",
        "numexpr >= 2.3.1",
        "pandas >= 0.18, < 0.19",
        "xarray >= 0.7.2, < 0.8",
        "netcdf4 >= 1.2.2",
        "pyyaml >= 3.11",
        "click >= 3.3"
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
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3 :: Only'
    ],
    keywords=['energy systems', 'optimization', 'mathematical programming']
)
