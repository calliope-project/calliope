from calliope.schemas import (
    config_schema,
    general,
    math_schema,
    model_def_schema,
    runtime_attrs_schema,
)


class CalliopeInputs(general.CalliopeBaseModel):
    """All Calliope attributes."""

    definition: model_def_schema.CalliopeModelDef = model_def_schema.CalliopeModelDef()
    config: config_schema.CalliopeConfig = config_schema.CalliopeConfig()
    math: math_schema.CalliopeMath = math_schema.CalliopeMath()
    runtime: runtime_attrs_schema.CalliopeRuntime = (
        runtime_attrs_schema.CalliopeRuntime()
    )
