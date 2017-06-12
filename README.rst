|badge_gitter| |badge_travis| |badge_appveyor| |badge_coveralls| |badge_pypi| |badge_conda| |badge_license|

-----

.. image:: https://raw.githubusercontent.com/calliope-project/calliope/master/doc/_static/logo.png

*A multi-scale energy systems (MUSES) modeling framework* | `www.callio.pe <http://www.callio.pe/>`_

-----

.. contents::

.. section-numbering::

-----

About
-----

Calliope is a framework to develop energy system models, with a focus on flexibility, high spatial and temporal resolution, the ability to execute many runs based on the same base model, and a clear separation of framework (code) and model (data).

A model based on Calliope consists of a collection of text files (in YAML and CSV formats) that define the technologies, locations and resource potentials. Calliope takes these files, constructs an optimization problem, solves it, and reports results in the form of `Pandas <http://pandas.pydata.org/>`_ and `xarray <http://xarray.pydata.org/>`_ data structures for easy analysis with Calliope's built-in tools or the standard Python data analysis stack.

Two simple example models are `included with Calliope <calliope/example_models>`_ and accessible through the ``calliope.examples`` submodule.

A more elaborate example is `UK-Calliope <https://github.com/sjpfenninger/uk-calliope>`_, which models the power system of Great Britain (England+Scotland+Wales).

Quick start
-----------

Install Calliope and all dependencies with conda:

.. code-block:: bash

    $ conda create -c conda-forge -n calliope python=3.5 calliope

Calliope can be run from the command line:

.. code-block:: bash

    $ calliope new example  # Create a copy of the national-scale example model, in the `example` dir

    $ calliope run example/run.yaml  # Run the model by pointing to its run configuration file

It can also be run interactively from a Python session:

.. code-block:: python

    import calliope
    model = calliope.Model('path/to/run.yaml')
    model.run()
    solution = model.solution  # An xarray.Dataset

Documentation
-------------

Documentation is available on Read the Docs:

* `Stable version <https://calliope.readthedocs.io/en/stable/>`_
* `Development version <https://calliope.readthedocs.io/en/latest/>`_

Changelog
---------

See `changelog.rst <https://github.com/calliope-project/calliope/blob/master/changelog.rst>`_.

Citing Calliope
---------------

If you use Calliope, please cite the following paper:

Stefan Pfenninger (2017). Dealing with multiple decades of hourly wind and PV time series in energy models: a comparison of methods to reduce time resolution and the planning implications of inter-annual variability. *Applied Energy*. `doi: 10.1016/j.apenergy.2017.03.051 <https://dx.doi.org/10.1016/j.apenergy.2017.03.051>`_

All Calliope releases are archived on Zenodo and you can also refer to specific versions of Calliope with their Zenodo DOI. The most recent archived version is: |link-latest-doi|_

License
-------

Copyright 2013-2017 Calliope contributors listed in AUTHORS

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

.. |badge_conda| image:: https://anaconda.org/conda-forge/calliope/badges/version.svg
    :target: https://anaconda.org/conda-forge/calliope
    :alt: Anaconda.org version

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

.. |badge_gitter|  image:: https://img.shields.io/gitter/room/calliope-project/calliope.svg?style=flat-square
    :target: https://gitter.im/calliope-project/calliope
    :alt: Chat on Gitter
