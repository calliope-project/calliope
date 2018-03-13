=================
Analysing a model
=================

Calliope inputs and results are made to be very easily handled. Whatever platform you prefer to use for data processing, our NetCDF or CSV output makes it easy. If you prefer to not worry about writing your own scripts, then we have that covered too! :class:`~calliope.Model.plot` is built on plotly's interactive toolbox to bring your data to life!

-------------------
Visualising results
-------------------

In an interactive Python session, there are four primary visualisation functions: ``capacity``, ``timeseries``, ``transmission``, and ``summary``. ``summary`` can also be accessed from the command line interface, to gain access to result visualisation without the need to interact with Python.

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

Both of these are `xarray.Datasets <http://xarray.pydata.org/en/stable/data-structures.html#dataset>`_ and can be further processed with Python.

.. seealso:: The `xarray documentation <http://xarray.pydata.org/en/stable/>`_ should be consulted for further information on dealing with Datasets. Calliope's NetCDF files follow the `CF conventions <http://cfconventions.org/>`_ and can easily be processed with any other tool that can deal with NetCDF.
