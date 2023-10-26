
=================
API Documentation
=================

.. _api_model:

Model class
===========

.. autoclass:: calliope.Model
    :members:

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