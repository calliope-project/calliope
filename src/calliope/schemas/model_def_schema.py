# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Schema for Calliope model definition."""

from pydantic import Field

from calliope.schemas.config_schema import CalliopeConfig
from calliope.schemas.data_table_schema import CalliopeDataTable
from calliope.schemas.dimension_data_schema import (
    CalliopeNode,
    CalliopeTech,
    DataValue,
    IndexedParam,
)
from calliope.schemas.general import AttrStr, CalliopeBaseModel


class CalliopeModelDef(CalliopeBaseModel):
    """Calliope model definition."""

    model_config = {"title": "Calliope Model Definition"}

    parameters: dict[AttrStr, DataValue | IndexedParam] = Field(default_factory=dict)
    config: CalliopeConfig = CalliopeConfig()
    data_tables: dict[AttrStr, CalliopeDataTable] = Field(default_factory=dict)
    nodes: dict[AttrStr, CalliopeNode] = Field(default_factory=dict)
    techs: dict[AttrStr, CalliopeTech] = Field(default_factory=dict)
    math: dict[AttrStr, dict] = Field(default_factory=dict)
