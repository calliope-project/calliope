
=================
Analyzing results
=================

-------------------
The solution object
-------------------

On successfully solving a model, Calliope creates a ``solution``, which is simply a key-value store with a number of result tables (implemented as an :class:`~calliope.utils.AttrDict`).

Most of the results are returned as `pandas <http://pandas.pydata.org/>`_ Series, DataFrames, or Panels indexed by cost class (``k``), carrier (``c``), location (``x``), technology (``y``), or time (``t``).

The analysis tools included with Calliope require a solution object (see below).

The solution contains the following data keys:

* ``capacity_factor``: a Panel with the axes ``c``, ``y``, ``x``. The ``x`` axis includes a ``total`` row, giving total capacity factors per technology.
* ``time_res``: a Series indexed by date-times containing the time resolution of each timestep
* ``node``: a 4-dimensional Panel with node-level variables. The axes ``variable``, ``y``, ``t``, ``x``. The variables contained are ``rs``, ``s``, ``rbs``, and an ``e`` variable per carrier.
* ``summary``: a DataFrame containing summary information on each technology (index is ``y``)
* ``parameters``: a Panel with the axes ``variable``, ``x``, ``y``. The variables contained are ``e_cap``, ``e_cap_net``, ``r_area``, ``r_cap``, ``s_cap``, and ``rb_cap``.
* ``shares``: a DataFrame containing technology and group based shares of production, consumption and installed capacity (index is ``y``).
* ``metadata``: a DataFrame containing metadata for each technology (such as its ``stack_weight`` or ``color``), used for analysis and plotting.
* ``config_model``: the complete model configuration used as an AttrDict
* ``costs``: a Panel with costs by cost classes, with the axes ``k``, ``x``, ``y``
* ``config_run``: the complete run configuration used as an AttrDict
* ``totals``: a 4-dimensional Panel indexed by ``c``, ``variable``, ``x``, ``y``. The variables contained are ``ec_con``, ``ec_prod``, ``es_prod``, ``es_con``.
* ``levelized_cost``: a 4-dimensional Panel indexed by ``k``, ``c``, ``x``, ``y``, giving the computed levelized costs. The ``x`` axis includes a ``total`` row, giving the total levelized cost per technology.

-----------------
Reading solutions
-----------------

If the solution was written to an HDF file. Calliope provides functionality to read the HDF file and re-construct a ``solution`` object for further analysis in a Python session:

.. code-block:: python

   solution = calliope.read.read_hdf('my_solution.hdf')

If the solution was written to CSV files, they can be processed by any data analysis tool. Similarly, the HDF files written contain only standard numerical and text data and can be read by any other software that can handle the HDF5 format.

----------------------------------
Reading results from parallel runs
----------------------------------

A successfully completed parallel run will contain multiple solutions inside its "Output" directory. To read all solutions, including information about the iterations they correspond to,

.. code-block:: python

   results = calliope.read.read_dir('path/to/Output')

The ``results`` variable is an :class:`~calliope.utils.AttrDict` with two keys:

* ``iterations``: a DataFrame containing the iterations from this parallel run
* ``solutions``: an AttrDict containing each individual ``solution`` object indexed by their iteration IDs

This allows easy access to and analysis of solutions.

-------------------
Analyzing solutions
-------------------

Refer to the :ref:`API documentation for the analysis module<api_analysis>` for an overview of available analysis functionality.

Refer to the :doc:`tutorial <tutorial>` for some basic analysis techniques.

.. Note:: More information about the built-in analysis and plotting functionality to be added.

.. TODO describe the use of the calliope.analysis module inside an interactive IPython session (maybe using an IPython notebook?)
