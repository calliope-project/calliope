
-----------------------
Built-in example models
-----------------------

This section gives a listing of all the YAML configuration files included in the built-in example models. Refer to the :doc:`tutorials section <tutorials>` for a brief overview of how these parts together can provide a simple working model.

The example models are accessible in the :mod:`calliope.examples` module. To create an instance of an example model, e.g.::

   urban_model = calliope.examples.UrbanScale()

National-scale example
----------------------

Available as :class:`calliope.examples.NationalScale`.

.. _examplemodels_nationalscale_settings:

Model settings
^^^^^^^^^^^^^^

The layout of the model directory is as follows (``+`` denotes directories, ``-`` files):

.. code-block:: text

    - model.yaml
    + timeseries_data
        - csp_r.csv
        - demand-1.csv
        - demand-2.csv
    + model_config
        - locations.yaml
        - techs.yaml


``model.yaml``:

.. literalinclude:: ../../calliope/example_models/national_scale/model.yaml
   :language: yaml

``techs.yaml``:

.. literalinclude:: ../../calliope/example_models/national_scale/model_config/techs.yaml
   :language: yaml

``locations.yaml``:

.. literalinclude:: ../../calliope/example_models/national_scale/model_config/locations.yaml
   :language: yaml

Urban-scale example
-------------------

Available as :class:`calliope.examples.UrbanScale`.

.. _examplemodels_urbanscale_runsettings:

Model settings
^^^^^^^^^^^^^^

``model.yaml``:

.. literalinclude:: ../../calliope/example_models/urban_scale/model.yaml
   :language: yaml

``techs.yaml``:

.. literalinclude:: ../../calliope/example_models/urban_scale/model_config/techs.yaml
   :language: yaml

``locations.yaml``:

.. literalinclude:: ../../calliope/example_models/urban_scale/model_config/locations.yaml
   :language: yaml

Mixed Integer Linear Programming (MILP) example
-----------------------------------------------

Available as :class:`calliope.examples.MILP`.

This example is based on the Urban scale example, calling the `milp` override group from `overrides.yaml`, which includes the necessary overrides for MILP functionality.

Model settings
^^^^^^^^^^^^^^

``model.yaml``:

.. literalinclude:: ../../calliope/example_models/urban_scale/model.yaml
   :language: yaml

``techs.yaml``:

.. literalinclude:: ../../calliope/example_models/urban_scale/model_config/techs.yaml
   :language: yaml

``locations.yaml``:

.. literalinclude:: ../../calliope/example_models/urban_scale/model_config/locations.yaml
   :language: yaml

Overrides
------------

``overrides.yaml``:

.. literalinclude:: ../../calliope/example_models/urban_scale/overrides.yaml
   :language: yaml
