.. automodule:: calliope

.. raw:: html
   :file: _static/github-ribbon.html

==================================================================
Calliope: a multi-scale energy systems (MUSES) modelling framework
==================================================================

v\ |version| (:doc:`Release history <history>`)

------------

This is the documentation for version |version|. See the `main project website <http://www.callio.pe/>`_ for contact details and other useful information.

------------

Calliope focuses on flexibility, high spatial and temporal resolution, the ability to execute many runs based on the same base model, and a clear separation of framework (code) and model (data).

A model based on Calliope consists of a collection of text files (in YAML and CSV formats) that define the technologies, locations and resource potentials. Calliope takes these files, constructs an optimisation problem, solves it, and reports results in the form of `xarray Datasets <http://xarray.pydata.org/en/stable/>`_ which in turn can easily be converted into `Pandas <http://pandas.pydata.org/>`_ data structures, for easy analysis with Calliope's built-in tools or the standard Python data analysis stack.

Calliope's built-in tools allow interactive exploration of results, as shown in the following example of a model that includes three energy carriers (electricity, heat, and gas):

.. raw:: html
   :file: user/images/plot_timeseries.html

Calliope is developed in the open `on GitHub <https://github.com/calliope-project/calliope>`_ and contributions are very welcome (see the :doc:`user/develop`). See the list of `open issues <https://github.com/calliope-project/calliope/issues>`_ and planned `milestones <https://github.com/calliope-project/calliope/milestones>`_ for an overview of where development is heading, and `join us on Gitter <https://gitter.im/calliope-project/calliope>`_ to ask questions or discuss code.

Key features of Calliope include:

* Model specification in an easy-to-read and machine-processable YAML format
* Generic technology definition allows modelling any mix of production, storage and consumption
* Resolved in space: define locations with individual resource potentials
* Resolved in time: read time series with arbitrary resolution
* Able to run on high-performance computing (HPC) clusters
* Uses a state-of-the-art Python toolchain based on `Pyomo <https://software.sandia.gov/trac/coopr/wiki/Pyomo>`_, `xarray <http://xarray.pydata.org/>`_, and `Pandas <http://pandas.pydata.org/>`_
* Freely available under the Apache 2.0 license

----------
User guide
----------

.. Use :numbered: to get section numbering

.. toctree::
   :maxdepth: 2

   user/introduction
   user/installation
   user/whatsnew
   user/building
   user/running
   user/analysing
   user/tutorials
   user/more_info
   user/develop

-----------------
API documentation
-----------------

Documents functions, classes and methods:

.. toctree::
   :maxdepth: 1

   api/api
   genindex

---------------
Release history
---------------

.. toctree::
   :hidden:
   :maxdepth: 1

   history

:doc:`Release history <history>`

-------
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
