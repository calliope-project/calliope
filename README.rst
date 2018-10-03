|badge_gitter| |badge_travis| |badge_appveyor| |badge_rtd| |badge_coveralls| |badge_pypi| |badge_conda| |badge_joss|

-----

.. image:: https://raw.githubusercontent.com/calliope-project/calliope/master/doc/_static/logo.png

*A multi-scale energy systems modelling framework* | `www.callio.pe <http://www.callio.pe/>`_

-----

.. contents::

.. section-numbering::

-----

About
-----

Calliope is a framework to develop energy system models, with a focus on flexibility, high spatial and temporal resolution, the ability to execute many runs based on the same base model, and a clear separation of framework (code) and model (data). Its primary focus is on planning energy systems at scales ranging from urban districts to entire continents. In an optional operational it can also test a pre-defined system under different operational conditions.

A Calliope model consists of a collection of text files (in YAML and CSV formats) that fully define a model, with details on technologies, locations, resource potentials, etc. Calliope takes these files, constructs an optimization problem, solves it, and reports back results. Results can be saved to CSV or NetCDF files for further processing, or analysed directly in Python through Python's extensive scientific data processing capabilities provided by libraries like `Pandas <http://pandas.pydata.org/>`_ and `xarray <http://xarray.pydata.org/>`_.

Calliope comes with several built-in analysis and visualisation tools. Having some knowledge of the Python programming language helps when running Calliope and using these tools, but is not a prerequisite.

Quick start
-----------

Calliope can run on Windows, macOS and Linux. Installing it is quickest with the ``conda`` package manager by running a single command: ``conda create -c conda-forge -n calliope python=3.6 calliope``. See the documentation for more `information on installing <https://calliope.readthedocs.io/en/stable/user/installation.html>`_.

Several easy to understand example models are `included with Calliope <calliope/example_models>`_ and accessible through the ``calliope.examples`` submodule.

The `tutorials in the documentation run through these examples <https://calliope.readthedocs.io/en/stable/user/tutorials.html>`_. A good place to start is to look at these tutorials to get a feel for how Calliope works, and then to read the "Introduction", "Building a model", "Running a model", and "Analysing a model" sections in the online documentation.

A fully-featured example model is `UK-Calliope <https://github.com/sjpfenninger/uk-calliope>`_, which models the power system of Great Britain (England+Scotland+Wales), and has been used in several peer-reviewed scientific publications.

Documentation
-------------

Documentation is available on Read the Docs:

* `Read the documentation online (recommended) <https://calliope.readthedocs.io/en/stable/>`_
* `Download all documentation in a single PDF file <https://readthedocs.org/projects/calliope/downloads/pdf/stable/>`_

Contributing
------------

To contribute changes:

1. Fork the project on GitHub
2. Create a feature branch to work on in your fork (``git checkout -b new-feature``)
3. Add your name to the AUTHORS file
4. Commit your changes to the feature branch
5. Push the branch to GitHub (``git push origin my-new-feature``)
6. On GitHub, create a new pull request from the feature branch

See our `contribution guidelines <https://github.com/calliope-project/calliope/blob/master/CONTRIBUTING.md>`_ for more information -- and `join us on Gitter <https://gitter.im/calliope-project/calliope>`_ to ask questions or discuss code.

What's new
----------

See changes made in recent versions in the `changelog <https://github.com/calliope-project/calliope/blob/master/changelog.rst>`_.

Citing Calliope
---------------

If you use Calliope for academic work please cite:

Stefan Pfenninger and Bryn Pickering (2018). Calliope: a multi-scale energy systems modelling framework. *Journal of Open Source Software*, 3(29), 825. `doi: 10.21105/joss.00825 <https://doi.org/10.21105/joss.00825>`_

License
-------

Copyright 2013-2018 Calliope contributors listed in AUTHORS

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

.. |link-latest-doi| image:: https://zenodo.org/badge/9581/calliope-project/calliope.svg
.. _link-latest-doi: https://zenodo.org/badge/latestdoi/9581/calliope-project/calliope

.. |badge_pypi| image:: https://img.shields.io/pypi/v/calliope.svg?style=flat-square
    :target: https://pypi.python.org/pypi/calliope
    :alt: PyPI version

.. |badge_conda| image:: https://img.shields.io/conda/vn/conda-forge/calliope.svg?style=flat-square&label=conda
    :target: https://anaconda.org/conda-forge/calliope
    :alt: Anaconda.org/conda-forge version

.. |badge_license| image:: https://img.shields.io/pypi/l/calliope.svg?style=flat-square
    :target: #license

.. |badge_coveralls| image:: https://img.shields.io/coveralls/calliope-project/calliope.svg?style=flat-square
    :target: https://coveralls.io/r/calliope-project/calliope
    :alt: Test coverage

.. |badge_travis| image:: https://img.shields.io/travis/calliope-project/calliope/master.svg?style=flat-square
    :target: https://travis-ci.org/calliope-project/calliope
    :alt: Build status on Linux

.. |badge_appveyor|  image:: https://img.shields.io/appveyor/ci/sjpfenninger/calliope/master.svg?style=flat-square&label=windows%20build
    :target: https://ci.appveyor.com/project/sjpfenninger/calliope
    :alt: Build status on Windows

.. |badge_rtd| image:: https://img.shields.io/readthedocs/calliope.svg?style=flat-square
    :target: https://readthedocs.org/projects/calliope/builds/
    :alt: Documentation build status

.. |badge_gitter|  image:: https://img.shields.io/gitter/room/calliope-project/calliope.svg?style=flat-square
    :target: https://gitter.im/calliope-project/calliope
    :alt: Chat on Gitter

.. |badge_joss|  image:: https://img.shields.io/badge/JOSS-10.21105/joss.00825-green.svg?style=flat-square
    :target: https://doi.org/10.21105/joss.00825
    :alt: JOSS DOI
