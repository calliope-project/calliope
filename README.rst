|badge_travis| |badge_appveyor| |badge_coveralls| |badge_pypi| |badge_license|

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

A simple example model is `included with Calliope <calliope/example_model>`_.

A more elaborate example is `UK-Calliope <https://github.com/sjpfenninger/uk-calliope>`_, which models the power system of Great Britain (England+Scotland+Wales).

Quick start
-----------

Calliope can be run from the command line:

.. code-block:: bash

    $ calliope new example  # Create a copy of the example model, in the `example` dir

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

Stefan Pfenninger and James Keirstead (2015). Renewables, nuclear, or fossil fuels? Scenarios for Great Britain's power system considering costs, emissions and energy security. *Applied Energy*, 152, pp. 83–93. `doi: 10.1016/j.apenergy.2015.04.102 <http://dx.doi.org/10.1016/j.apenergy.2015.04.102>`_

All Calliope releases are archived on Zenodo and you can also refer to specific versions of Calliope with their Zenodo DOI. The most recent archived version is: |link-latest-doi|_

License
-------

Copyright 2013-2016 Stefan Pfenninger

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

.. |badge_pypi| image:: https://img.shields.io/pypi/v/calliope.svg
    :target: https://pypi.python.org/pypi/calliope
    :alt: PyPI version

.. |badge_license| image:: https://img.shields.io/pypi/l/calliope.svg
    :target: #license

.. |badge_coveralls| image:: https://img.shields.io/coveralls/calliope-project/calliope.svg
    :target: https://coveralls.io/r/calliope-project/calliope
    :alt: Test coverage

.. |badge_travis| image:: https://travis-ci.org/calliope-project/calliope.svg
    :target: https://travis-ci.org/calliope-project/calliope
    :alt: Build status on Mac/Linux

.. |badge_appveyor|  image:: https://ci.appveyor.com/api/projects/status/16aic413nfm35u4b/branch/master?svg=true
    :target: https://ci.appveyor.com/project/sjpfenninger/calliope
    :alt: Build status on Windows
