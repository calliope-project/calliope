
===========================
The built-in example models
===========================

This section gives a listing of all the YAML configuration files included in the built-in example models. Refer to the :doc:`tutorials section <tutorials>` for a brief overview of how these parts together can provide a simple working model.

----------------------
National-scale example
----------------------

Model settings
--------------

The layout of the model directory is as follows (``+`` denotes directories, ``-`` files):

.. code-block:: text

   + model_config
      + data
         - csp_r.csv
         - demand-1.csv
         - demand-2.csv
         - set_t.csv
      - locations.yaml
      - model.yaml
      - techs.yaml


``model.yaml``:

.. literalinclude:: ../../calliope/example_models/national_scale/model_config/model.yaml
   :language: yaml

``techs.yaml``:

.. literalinclude:: ../../calliope/example_models/national_scale/model_config/techs.yaml
   :language: yaml

``locations.yaml``:

.. literalinclude:: ../../calliope/example_models/national_scale/model_config/locations.yaml
   :language: yaml

.. _examplemodels_nationalscale_runsettings:

Run settings
------------

``run.yaml``:

.. literalinclude:: ../../calliope/example_models/national_scale/run.yaml
   :language: yaml

-------------------
Urban-scale example
-------------------

Model settings
--------------

``model.yaml``:

.. literalinclude:: ../../calliope/example_models/urban_scale/model_config/model.yaml
   :language: yaml

``techs.yaml``:

.. literalinclude:: ../../calliope/example_models/urban_scale/model_config/techs.yaml
   :language: yaml

``locations.yaml``:

.. literalinclude:: ../../calliope/example_models/urban_scale/model_config/locations.yaml
   :language: yaml

Run settings
------------

``run.yaml``:

.. literalinclude:: ../../calliope/example_models/urban_scale/run.yaml
   :language: yaml
