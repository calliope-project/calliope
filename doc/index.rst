.. automodule:: calliope

=================================================================
Calliope: a multi-scale energy systems (MUSES) modeling framework
=================================================================

v\ |version| (:doc:`Release history <history>`)

Calliope is a framework to develop energy systems models using a modern Python-based toolchain.

Main features:

* Generic technology definition allows modeling any mix of production, storage and consumption
* Resolved in space: define locations with individual resource potentials
* Resolved in time: can read time series with arbitrary resolution
* Model specification in an easy-to-read YAML format
* Able to run on computing clusters
* Easily extensible
* Uses a state-of-the-art Python toolchain based on `IPython <http://ipython.org/>`_, `Pandas <http://pandas.pydata.org/>`_ and `Pyomo <https://software.sandia.gov/trac/coopr/wiki/Pyomo>`_
* `Freely available <https://github.com/sjpfenninger/calliope>`_ under the GNU GPLv3 license

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

Copyright © 2013  Stefan Pfenninger

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see [http://www.gnu.org/licenses/].

.. Indices and tables
.. ==================
..
.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`

