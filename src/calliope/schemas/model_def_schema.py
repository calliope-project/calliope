# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Schema for Calliope model definition."""

from calliope.schemas.config_schema import CalliopeConfig
from calliope.schemas.data_table_schema import CalliopeDataTables
from calliope.schemas.dimension_data_schema import (
    CalliopeNodes,
    CalliopeParams,
    CalliopeTechs,
)
from calliope.schemas.general import CalliopeBaseModel
from calliope.schemas.math_schema import CalliopeInputMath


class CalliopeModelDef(CalliopeBaseModel):
    """Calliope model definition."""

    model_config = {"title": "Calliope Model Definition"}

    parameters: CalliopeParams = CalliopeParams()
    config: CalliopeConfig = CalliopeConfig()
    data_tables: CalliopeDataTables = CalliopeDataTables()
    nodes: CalliopeNodes = CalliopeNodes()
    techs: CalliopeTechs = CalliopeTechs()
    math: CalliopeInputMath = CalliopeInputMath()
