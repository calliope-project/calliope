
=================
Analyzing results
=================

-------------------
The solution object
-------------------

On successfully solving a model, Calliope creates a ``solution``, which is a multi-dimensional `xarray.Dataset <http://xarray.pydata.org/en/stable/data-structures.html#dataset>`_, with the model and run configuration stored as :class:`~calliope.utils.AttrDict` attributes of the dataset (``config_model`` and ``config_run``).

The analysis tools included with Calliope expect to operate on a dataset.

The solution contains model variables such as ``rs``, ``s``, ``e_cap``, ``r_area``, etc, as well as variables derived from them such as ``capacity_factor`` and ``levelized_cost``. It also contains several two-dimensional summary and metadata tables:

* ``metadata``: metadata for each technology (such as its ``stack_weight`` or ``color``), used for analysis and plotting.
* ``groups``: definition of technology groups and their members.
* ``shares``: technology and group based shares of production, consumption and installed capacity (index is ``y``).
* ``summary``: summary information on each technology.

-----------------
Reading solutions
-----------------

Calliope provides functionality to read a solution from a single NetCDF file or a collection of CSV files and re-construct a ``solution`` object for further analysis in a Python session:

.. code-block:: python

   solution_from_netcdf = calliope.read.read_netcdf('my_solution.nc')

   solution_from_csv = calliope.read.read_csv('path/to/output_directory')

----------------------------------
Reading results from parallel runs
----------------------------------

A successfully completed parallel run will contain multiple solutions inside its "Output" directory. To read all solutions, including information about the iterations they correspond to, use:

.. code-block:: python

   results = calliope.read.read_dir('path/to/Output')

The ``results`` variable is an :class:`~calliope.utils.AttrDict` with two keys:

* ``iterations``: a DataFrame containing the iterations from this parallel run
* ``solutions``: an AttrDict with iteration IDs as keys and the individual ``solution`` objects as values

This allows easy access to and analysis of solutions.

-------------------
Analyzing solutions
-------------------

Refer to the :ref:`API documentation for the analysis module<api_analysis>` for an overview of available analysis functionality.

Refer to the :doc:`tutorials <tutorials>` for some basic analysis techniques.

.. Note:: The built-in analysis and plotting functionality is still experimental. More documentation on it will be added in a future release.

.. TODO describe the use of the calliope.analysis module inside an interactive IPython session (maybe using an IPython notebook?)
