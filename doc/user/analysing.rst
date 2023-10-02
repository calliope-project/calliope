=================
Analysing a model
=================

Calliope inputs and results are designed for easy handling. Whatever software you prefer to use for data processing, either the NetCDF or CSV output options should provide a path to importing your Calliope results.

--------------------------------
Accessing model data and results
--------------------------------

A model which solved successfully has two primary Datasets with data of interest:

* :python:`model.inputs`: contains all input data, such as renewable resource capacity factors
* :python:`model.results`: contains all results data, such as dispatch decisions and installed capacities

Both of these are `xarray.Datasets <https://docs.xarray.dev/en/v2022.03.0/user-guide/data-structures.html#dataset>`_ and can be further processed with Python.

Data is indexed over a subset of the core Calliope sets, e.g. `techs` (technologies), `nodes`, `timesteps`.
Not all combinations of the set items will contain data; if you did not define your `PV` technology at node `X1` then there will be no data for :python:`model.inputs.flow_cap.sel(techs="PV", nodes="X1")`.
In fact, there is likely to be more empty (`NaN`) data points than filled ones.
In Python you can quickly "densify" your data to look at only filled data points: :python:`model.inputs.flow_cap.to_series().dropna()`

.. note:: On saving to CSV (see the :ref:`command-line interface documentation <running_cli>`), each data variable is saved to its own file with all empty data points removed.

-----------------
Reading solutions
-----------------

Calliope provides functionality to read a previously-saved model from a single NetCDF file:

.. code-block:: python

   solved_model = calliope.read_netcdf('my_saved_model.nc')

Once loaded, the input and results data can be accessed as above (i.e., :python:`solved_model.inputs` and :python:`solved_model.results`).

.. seealso:: The `xarray documentation <https://docs.xarray.dev/en/v2022.03.0/>`_ should be consulted for further information on dealing with Datasets.

.. note:: Calliope's NetCDF files follow the `CF conventions <https://cfconventions.org/>`_ and can easily be processed with any other tool that can deal with NetCDF. However, certain model attributes are serialised on saving the model to ensure the convention is followed. To view the model as intended, use the Calliope functionality to read the NetCDF.
