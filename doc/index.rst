.. automodule:: calliope

=================================================================
Calliope: a multi-scale energy systems (MUSES) modeling framework
=================================================================

v\ |version| (:doc:`Release history <history>`)

Calliope is a framework to develop energy systems models using a modern Python-based toolchain.

Main features:

* Abstract nodes allow modeling any mix of production, storage and consumption
* Resolved in space: define nodes with individual resource potentials
* Resolved in time: can read time series with arbitrary resolution
* Model specification in an easy-to-read YAML format
* Able to run on computing clusters
* Easily extensible
* Uses a state-of-the-art Python toolchain based on `IPython <http://ipython.org/>`_, `Pandas <http://pandas.pydata.org/>`_ and `Pyomo <https://software.sandia.gov/trac/coopr/wiki/Pyomo>`_

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

   model/installation
   model/components
   model/constraints
   model/configuration
   model/data
   model/running
   api/api

.. toctree::
   :maxdepth: 1

   history

.. Indices and tables
.. ==================
..
.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`

