.. automodule:: calliope

=================================================================
Calliope: a multi-scale energy systems (MUSES) modeling framework
=================================================================

v\ |version| (:doc:`Release history <history>`)

Calliope is a framework to develop energy systems models using a modern and open source Python-based toolchain.

.. warning:: Calliope and its documentation are still undergoing active development and functionality may change without prior warning. If you are interested in using Calliope at this stage you are encouraged to `get in touch <mailto:stefan.pfenninger@imperial.ac.uk>`_.

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

Models are defined with a mixture of YAML files and CSV files. See the included example model for details, and read the :doc:`configuration section <model/configuration>` and the :doc:`data section <model/data>`.

To run a model once::

   import calliope
   model = calliope.Model(config_run='/path/to/run.yaml')
   model.run()

On successfully finding a solution, the ``Model`` instance makes available its results for further analysis::

   # Returns a pandas DataFrame
   system_vars = model.get_system_variables()
   # Plot system-level variables with matplotlib
   system_vars.plot()

To set up parallel runs, see :ref:`parallel_runs`.

-------------------
Model documentation
-------------------

.. toctree::
   :maxdepth: 2

   model/introduction
   model/installation
   model/components
   model/constraints
   model/configuration
   model/data
   model/running

-----------------
API Documentation
-----------------

.. toctree::

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
