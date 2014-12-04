
==========================
The built-in example model
==========================

This section gives a listing of all the YAML configuration files included in the built-in example model. Refer to the :doc:`tutorial section <tutorial>` for a brief overview of how these parts together provide a simple working model.

Model settings
--------------

The layout of the model directory is as follows (``+`` denotes directories, ``-`` files):

.. code-block:: text

   + model_config
      + data
         - csp_r.csv
         - demand-r1_r.csv
         - demand-r2_r.csv
         - set_t.csv
      - locations.yaml
      - model.yaml
      - techs.yaml
      - transmission.yaml


``model.yaml``:

.. literalinclude:: ../../calliope/example_model/model_config/model.yaml
   :language: yaml

``techs.yaml``:

.. literalinclude:: ../../calliope/example_model/model_config/techs.yaml
   :language: yaml

``locations.yaml``:

.. literalinclude:: ../../calliope/example_model/model_config/locations.yaml
   :language: yaml

``transmission.yaml``:

.. literalinclude:: ../../calliope/example_model/model_config/transmission.yaml
   :language: yaml

Run settings
------------

``run.yaml``:

.. literalinclude:: ../../calliope/example_model/run.yaml
   :language: yaml