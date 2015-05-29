.. automodule:: calliope

=================================================================
Calliope: a multi-scale energy systems (MUSES) modeling framework
=================================================================

v\ |version| (:doc:`Release history <history>`)

Calliope is a framework to develop energy system models using a modern and open source Python-based toolchain.

------------

This is the documentation for version |version|. See the `main project website <http://www.callio.pe/>`_ for contact details and other useful information.

------------

Main features:

* Generic technology definition allows modeling any mix of production, storage and consumption
* Resolved in space: define locations with individual resource potentials
* Resolved in time: read time series with arbitrary resolution
* Model specification in an easy-to-read and machine-processable YAML format
* Able to run on computing clusters
* Easily extensible in a modular way: custom constraint generator functions and custom time mask functions
* Uses a state-of-the-art Python toolchain based on `Pyomo <https://software.sandia.gov/trac/coopr/wiki/Pyomo>`_ and `Pandas <http://pandas.pydata.org/>`_
* Freely available under the Apache 2.0 license

----------
User guide
----------

.. Use :numbered: to get section numbering

.. toctree::
   :maxdepth: 2

   user/introduction
   user/installation
   user/components
   user/tutorial
   user/formulation
   user/configuration
   user/run_configuration
   user/running
   user/analysis
   user/configuration_reference
   user/example_model
   user/develop

-----------------
API documentation
-----------------

Documents functions, classes and methods:

.. toctree::
   :maxdepth: 1

   api/api

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

Copyright 2013--2015 Stefan Pfenninger

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
