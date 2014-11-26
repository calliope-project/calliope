.. automodule:: calliope

=================================================================
Calliope: a multi-scale energy systems (MUSES) modeling framework
=================================================================

v\ |version| (:doc:`Release history <history>`)

Calliope is a framework to develop energy systems models using a modern and open source Python-based toolchain.

.. Note:: Calliope is under active development. Functionality and/or data formats may change in future versions. See the :doc:`contact page <contact>` to keep up-to-date with changes.

Main features:

* Generic technology definition allows modeling any mix of production, storage and consumption
* Resolved in space: define locations with individual resource potentials
* Resolved in time: can read time series with arbitrary resolution
* Model specification in an easy-to-read YAML format
* Able to run on computing clusters
* Easily extensible
* Uses a state-of-the-art Python toolchain based on `IPython <http://ipython.org/>`_, `Pandas <http://pandas.pydata.org/>`_ and `Pyomo <https://software.sandia.gov/trac/coopr/wiki/Pyomo>`_
* `Freely available <https://github.com/sjpfenninger/calliope>`_ under the Apache 2.0 license

Development is funded by the `Grantham Institute <http://www.imperial.ac.uk/grantham>`_ at Imperial College London, and the European Institute of Innovation & Technology's `Climate-KIC program <http://www.climate-kic.org>`_.

-----------
Quick start
-----------

Models are defined with a mixture of YAML files and CSV files. See the included example model for details, and read the :doc:`configuration section <user/configuration>` and the :doc:`data section <user/data>`.

To run a model once::

   import calliope
   model = calliope.Model(config_run='/path/to/run_configuration.yaml')
   model.run()

On successfully finding a solution, the ``Model`` instance makes available its results for further analysis::

   # Get a pandas DataFrame
   system_vars = model.get_system_variables()
   # Plot system-wide energy production
   system_vars.plot()

.. TODO To set up parallel runs, see :ref:`parallel_runs`.

----------
User guide
----------

.. toctree::
   :maxdepth: 1

   user/introduction
   user/installation
   user/components
   user/constraints
   user/configuration
   user/data
   user/running
   user/develop

-----------------
API documentation
-----------------

Documents functions, methods and classes:

.. toctree::
   :maxdepth: 1

   api/api

---------------
Release history
---------------

.. toctree::
   :maxdepth: 1

   history

-------
License
-------

Copyright 2013 Stefan Pfenninger

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
