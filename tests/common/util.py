import os
from pathlib import Path
from typing import Literal

import calliope
import xarray as xr
from calliope import backend


def build_test_model(
    override_dict=None,
    scenario=None,
    model_file="model.yaml",
    data_source_dfs=None,
    **init_kwargs,
):
    return calliope.Model(
        os.path.join(os.path.dirname(__file__), "test_model", model_file),
        override_dict=override_dict,
        scenario=scenario,
        data_source_dfs=data_source_dfs,
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


def build_lp(
    model: calliope.Model,
    outfile: str | Path,
    math: dict[str, dict | list] | None = None,
    backend_name: Literal["pyomo"] = "pyomo",
) -> "backend.BackendModel":
    """
    Write a barebones LP file with which to compare in tests.
    All model parameters and variables will be loaded automatically, as well as a dummy objective if one isn't provided as part of `math`.
    Everything else to be added to the LP file must be defined in `math`.

    Args:
        model (calliope.Model): Calliope model.
        outfile (str | Path): Path to LP file.
        math (dict | None, optional): All constraint/global expression/objective math to apply. Defaults to None.
        backend_name (Literal["pyomo"], optional): Backend to use to create the LP file. Defaults to "pyomo".
    """
    backend_instance = backend.get_model_backend(backend_name, model._model_data)

    for name, dict_ in model.math["variables"].items():
        backend_instance.add_variable(name, dict_)
    for name, dict_ in model.math["global_expressions"].items():
        backend_instance.add_global_expression(name, dict_)

    if isinstance(math, dict):
        for component_group, component_math in math.items():
            component = component_group.removesuffix("s")
            if isinstance(component_math, dict):
                for name, dict_ in component_math.items():
                    getattr(backend_instance, f"add_{component}")(name, dict_)
            elif isinstance(component_math, list):
                for name in component_math:
                    dict_ = model.math[component_group][name]
                    getattr(backend_instance, f"add_{component}")(name, dict_)

    # MUST have an objective for a valid LP file
    if math is None or "objectives" not in math.keys():
        backend_instance.add_objective(
            "dummy_obj", {"equations": [{"expression": "1 + 1"}], "sense": "minimize"}
        )
        backend_instance._instance.objectives["dummy_obj"][0].activate()
    elif "objectives" in math.keys():
        if isinstance(math["objectives"], dict):
            objectives = list(math["objectives"].keys())
        else:
            objectives = math["objectives"]
        assert len(objectives) == 1, "Can only test with one objective"
        backend_instance._instance.objectives[objectives[0]][0].activate()

    backend_instance.verbose_strings()

    backend_instance.to_lp(str(outfile))

    # strip trailing whitespace from `outfile` after the fact,
    # so it can be reliably compared other files in future
    with Path(outfile).open("r") as f:
        stripped_lines = []
        while line := f.readline():
            stripped_lines.append(line.rstrip())

    # reintroduce the trailing newline since both Pyomo and file formatters love them.
    Path(outfile).write_text("\n".join(stripped_lines) + "\n")
    return backend_instance
