# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Schema for Calliope model definition."""

from calliope.schemas.data_table_schema import CalliopeDataTables
from calliope.schemas.dimension_data_schema import (
    CalliopeDataDef,
    CalliopeNodes,
    CalliopeTechs,
)
from calliope.schemas.general import CalliopeBaseModel


class CalliopeModelDef(CalliopeBaseModel):
    """Calliope model definition."""

    model_config = {"title": "Calliope Model Definition"}

    data_definitions: CalliopeDataDef = CalliopeDataDef()
    """All YAML-based model input data definitions."""
    data_tables: CalliopeDataTables = CalliopeDataTables()
    """All tabular-based model input data definitions."""
    nodes: CalliopeNodes = CalliopeNodes()
    """Node-specific model input data definitions."""
    techs: CalliopeTechs = CalliopeTechs()
    """Tech-specific model input data definitions."""
