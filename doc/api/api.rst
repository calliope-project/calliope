
=================
API Documentation
=================

.. _api_model:

Model class
===========

.. autoclass:: calliope.Model
    :members:

.. _api_time_masks:

Time series
===========

.. automodule:: calliope.time.clustering
    :members: get_clusters

.. automodule:: calliope.time.masks
    :members: extreme, extreme_diff

.. automodule:: calliope.time.funcs
    :members: resample

.. _api_analysis:

Analyzing models
================

.. autoclass:: calliope.analysis.plotting.plotting.ModelPlotMethods
    :members: timeseries, capacity, transmission, summary

.. _api_backend_interface:

Pyomo backend interface
=======================

.. autoclass:: calliope.backend.pyomo.interface.BackendInterfaceMethods
    :members: access_model_inputs, update_param, activate_constraint, rerun

.. _api_utility_classes:

Utility classes: AttrDict, Exceptions, Logging
==============================================

.. autoclass:: calliope.core.attrdict.AttrDict
    :members:

.. automodule:: calliope.exceptions
    :members:

.. automodule:: calliope.core.util.logging
    :members: set_log_verbosity