
=================
API Documentation
=================

Model class
===========

.. autoclass:: calliope.Model
    :members:

.. _api_time_masks:

Time series
===========

.. automodule:: calliope.core.time.clustering
    :members: get_clusters_kmeans, get_clusters_hierarchical

.. automodule:: calliope.core.time.masks
    :members: extreme, extreme_diff

.. automodule:: calliope.core.time.funcs
    :members: resample

.. _api_analysis:

Analyzing models
================

.. autoclass:: calliope.analysis.plotting.plotting.ModelPlotMethods
    :members: timeseries, capacity, transmission, summary

.. _backend_interface_api:

Pyomo backend interface
=======================

.. autoclass:: calliope.backend.pyomo.interface.BackendInterfaceMethods

Utility classes: AttrDict, Exceptions
=====================================

.. autoclass:: calliope.core.attrdict.AttrDict
    :members:

.. automodule:: calliope.exceptions
    :members:
