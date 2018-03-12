=================
Analysing a model
=================

-------------------
Visualising results
-------------------

**TBA: using plotly**

Refer to the :ref:`API documentation for the analysis module<api_analysis>` for an overview of available analysis functionality.

Refer to the :doc:`tutorials <tutorials>` for some basic analysis techniques.

Saving publication-quality SVG figures
--------------------------------------

**TBA**

-----------------
Reading solutions
-----------------

Calliope provides functionality to read a previously-saved model a single NetCDF file:

.. code-block:: python

   solved_model = calliope.read_netcdf('my_solution.nc')

In the above example, the model's input data will be available under ``solved_model.inputs``, while the results (if the model had previously been solved) are available under ``solved_model.results``.

Both of these are `xarray.Datasets <http://xarray.pydata.org/en/stable/data-structures.html#dataset>`_ and can be further processed with Python. The `xarray documentation <http://xarray.pydata.org/en/stable/>`_ should be consulted for further information.

Calliope's NetCDF files follow the `CF conventions <http://cfconventions.org/>`_ and easily be processed with any other tool that can deal with NetCDF.
