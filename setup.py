#!/usr/bin/env python

from pathlib import Path
from setuptools import setup, find_packages

exec(Path("calliope/_version.py").read_text())  # Sets the __version__ variable

requirements = Path("requirements.txt").read_text().strip().split("\n")
long_description = Path("README.md").read_text()


def get_subdirs(path, glob_string):
    return [
        i.relative_to(str(path) + "/") for i in path.glob(glob_string) if i.is_dir()
    ]


def find_calliope_package_data():
    """Returns a list of found directories with package data files"""
    path = Path("./calliope")
    package_data = ["config/*.yaml", "config/*.html", "test/common/*.yaml"]
    for subdir_level in ["*", "*/*", "*/*/*"]:
        for example_dir in get_subdirs(path, "example_models/" + subdir_level):
            package_data.append(str(example_dir) + "/*.csv")
            package_data.append(str(example_dir) + "/*.yaml")
            package_data.append(str(example_dir) + "/*.rst")
    for test_case_dir in get_subdirs(path, "test/common/*"):
        package_data.append(str(test_case_dir) + "/*.csv")
    print(package_data)
    return package_data


setup(
    name="calliope",
    version=__version__,
    author="Calliope contributors listed in AUTHORS",
    author_email="stefan@pfenninger.org",
    description="A multi-scale energy systems modelling framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache 2.0",
    url="https://www.callio.pe/",
    download_url="https://github.com/calliope-project/calliope/releases",
    packages=find_packages(),
    package_data={"calliope": find_calliope_package_data()},
    install_requires=requirements,
    entry_points={"console_scripts": ["calliope = calliope.cli:cli"]},
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords=["energy systems", "optimisation", "mathematical programming"],
)
