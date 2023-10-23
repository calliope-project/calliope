
-----------------------
Built-in example models
-----------------------

This section gives a listing of all the YAML configuration files included in the built-in example models. Refer to the :doc:`tutorials section <tutorials>` for a brief overview of how these parts together provide a working model.

The example models are accessible in the :mod:`calliope.examples` module. To create an instance of an example model, call its constructor function, e.g.

  .. code-block:: python

    urban_model = calliope.examples.urban_scale()

The available example models and their constructor functions are:

.. automodule:: calliope.examples
    :members:

National-scale example
----------------------

Available as :class:`calliope.examples.national_scale`.

.. _examplemodels_nationalscale_settings:

Model settings
^^^^^^^^^^^^^^

The layout of the model directory is as follows (``+`` denotes directories, ``-`` files):

.. code-block:: text

    - model.yaml
    - scenarios.yaml
    + timeseries_data
        - csp_resource.csv
        - demand-1.csv
        - demand-2.csv
    + model_config
        - locations.yaml
        - techs.yaml


``model.yaml``:

.. literalinclude:: ../../src/calliope/example_models/national_scale/model.yaml
   :language: yaml

``scenarios.yaml``:

.. literalinclude:: ../../src/calliope/example_models/national_scale/scenarios.yaml
   :language: yaml

``techs.yaml``:

.. literalinclude:: ../../src/calliope/example_models/national_scale/model_config/techs.yaml
   :language: yaml

``locations.yaml``:

.. literalinclude:: ../../src/calliope/example_models/national_scale/model_config/locations.yaml
   :language: yaml

Urban-scale example
-------------------

Available as :class:`calliope.examples.urban_scale`.

.. _examplemodels_urbanscale_runsettings:

Model settings
^^^^^^^^^^^^^^

``model.yaml``:

.. literalinclude:: ../../src/calliope/example_models/urban_scale/model.yaml
   :language: yaml

``scenarios.yaml``:

.. literalinclude:: ../../src/calliope/example_models/urban_scale/scenarios.yaml
   :language: yaml

``techs.yaml``:

.. literalinclude:: ../../src/calliope/example_models/urban_scale/model_config/techs.yaml
   :language: yaml

``locations.yaml``:

.. literalinclude:: ../../src/calliope/example_models/urban_scale/model_config/locations.yaml
   :language: yaml
