.. image:: https://travis-ci.org/calliope-project/calliope.svg
    :target: https://travis-ci.org/calliope-project/calliope
.. image:: https://img.shields.io/coveralls/calliope-project/calliope.svg
    :target: https://coveralls.io/r/calliope-project/calliope
.. image:: https://img.shields.io/pypi/l/calliope.svg
    :target: #license
.. image:: https://img.shields.io/pypi/v/calliope.svg
    :target: https://pypi.python.org/pypi/calliope

::

       _____     _____
      / ___/__ _/ / (_)__  ___  ___
     / /__/ _ `/ / / / _ \/ _ \/ -_)
     \___/\_,_/_/_/_/\___/ .__/\__/
                        /_/

Calliope
========

*A multi-scale energy systems (MUSES) modeling framework* | `www.callio.pe <http://www.callio.pe/>`_

About
-----

Calliope is a framework to develop energy system models, with a focus on flexibility, high spatial and temporal resolution, the ability to execute many runs based on the same base model, and a clear separation of framework (code) and model (data). It is under active development (see the `roadmap <roadmap.rst>`_).

A model based on Calliope consists of a collection of text files (in YAML and CSV formats) that define the technologies, locations and resource potentials. Calliope takes these files, constructs an optimization problem, solves it, and reports results in the form of `Pandas <http://pandas.pydata.org/>`_ data structures for easy analysis with Calliope's built-in tools or the standard Python data analysis stack.

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
    solution = model.solution  # A dict of DataFrames

Documentation
-------------

Documentation is available at docs.callio.pe:

* `Stable version <http://docs.callio.pe/en/stable/>`_
* `Development version <http://docs.callio.pe/en/latest/>`_

DOI for the most recent stable version on Zenodo: |badge_doi|

.. |badge_doi| image:: https://zenodo.org/badge/9581/calliope-project/calliope.svg
    :target: https://zenodo.org/search?ln=en&p=Calliope%3A+a+multi-scale+energy+systems+%28MUSES%29+modeling+framework&action_search=

License
-------

Copyright 2013-2015 Stefan Pfenninger

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
