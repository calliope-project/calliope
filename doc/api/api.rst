
=================
API Documentation
=================

.. _api_model:

Model class
===========

.. autoclass:: calliope.Model
    :members:

.. _api_backend_interface:

Optimisation backend interface
==============================

.. automodule:: calliope.backend.backends
    :members: BackendModel

.. _api_time_masks:

Time series
===========

.. automodule:: calliope.time.clustering
    :members: get_clusters

.. automodule:: calliope.time.masks
    :members: extreme, extreme_diff

.. automodule:: calliope.time.funcs
    :members: resample

.. _api_backend_interface:

Methods to interface with the optimisation problem (a.k.a., `backend`)
======================================================================

.. autoclass:: calliope.backend.backend_model.BackendModel
    :members:
    :inherited-members:

.. _api_utility_classes:

Utility classes: AttrDict, Exceptions, Logging
==============================================

.. autoclass:: calliope.core.attrdict.AttrDict
    :members:

.. automodule:: calliope.exceptions
    :members:

.. automodule:: calliope.core.util.logging
    :members: set_log_verbosity

Math formulation helper functions
=================================

.. automodule:: calliope.backend.helper_functions
    :members:
    :special-members: __call__
    :inherited-members: