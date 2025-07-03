import os

import xarray as xr

import calliope


def build_test_model(
    override_dict=None,
    scenario=None,
    model_file="model.yaml",
    data_table_dfs=None,
    **init_kwargs,
):
    """Get the Calliope model object of a test model."""
    return calliope.read_yaml(
        os.path.join(os.path.dirname(__file__), "test_model", model_file),
        override_dict=override_dict,
        scenario=scenario,
        data_table_dfs=data_table_dfs,
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


def check_variable_exists(
    expr_or_constr: xr.DataArray | None, variable: str, slices: dict | None = None
) -> bool:
    """
    Search for existence of a decision variable in a Pyomo constraint.
    Args:
        backend_interface :
        constraint :
        variable : str, string to search in the list of variables to check if existing

    Args:
        expr_or_constr (Optional[xr.DataArray]): Array of math expression objects.
        variable (str): Name of variable to search for
        slices (Optional[dict], optional):
            If not None, slice `expr_or_constr` array according to provided key:val pairs.
            `key` is an array dimension name and `val` is a dimension index item or iterable of index items.
            Defaults to None.

    Returns:
        bool: If the variable exists in any of the array expressions (after slicing), returns True.
    """
    if expr_or_constr is None:
        return False

    try:
        var_exists = expr_or_constr.body.astype(str).str.find(variable) > -1
    except (AttributeError, KeyError):
        var_exists = expr_or_constr.astype(str).str.find(variable) > -1

    if slices is not None:
        var_exists = var_exists.sel(**slices)

    return var_exists.any()
