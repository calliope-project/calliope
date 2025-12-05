import os

import calliope


def build_test_model(
    override_dict=None,
    scenario=None,
    model_file="model.yaml",
    data_table_dfs=None,
    pre_validate_math_strings: bool = False,
    math_dict: dict | None = None,
    **init_kwargs,
):
    """Get the Calliope model object of a test model."""
    if math_dict is not None:
        init_kwargs["math_dict"] = {"custom_math": math_dict}
        init_kwargs["extra_math"] = init_kwargs.get("extra_math", []) + ["custom_math"]
    return calliope.read_yaml(
        os.path.join(os.path.dirname(__file__), "test_model", model_file),
        override_dict=override_dict,
        scenario=scenario,
        data_table_dfs=data_table_dfs,
        pre_validate_math_strings=pre_validate_math_strings,
        **init_kwargs,
    )


def check_error_or_warning(error_warning, test_string_or_strings):
    if hasattr(error_warning, "list"):
        output = ",".join(
            str(error_warning.list[i]) for i in range(len(error_warning.list))
        )
    elif hasattr(error_warning, "value"):
        output = str(error_warning.value)
    elif isinstance(error_warning, list | set):
        output = ",".join(error_warning)

    if isinstance(test_string_or_strings, list):
        result = all(test_string in output for test_string in test_string_or_strings)
    else:
        result = test_string_or_strings in output

    return result
