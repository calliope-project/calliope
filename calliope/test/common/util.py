import os
import sys
import ast

import pytest
from pyomo.core.expr.current import identify_variables
import pyomo.core as po

import calliope
from calliope import AttrDict


constraint_sets = {
    k: [ast.literal_eval(i) for i in v]
    for k, v in AttrDict.from_yaml(
        os.path.join(os.path.dirname(__file__), "constraint_sets.yaml")
    )
    .as_dict_flat()
    .items()
}

defaults = AttrDict.from_yaml(
    os.path.join(os.path.dirname(calliope.__file__), "config", "defaults.yaml")
)

subsets_config = AttrDict.from_yaml(
    os.path.join(os.path.dirname(calliope.__file__), "config", "subsets.yaml")
)

python36_or_higher = pytest.mark.skipif(
    sys.version_info < (3, 6), reason="Requires ordered dicts from Python >= 3.6"
)


def build_test_model(
    override_dict=None,
    scenario=None,
    model_file="model.yaml",
    timeseries_dataframes=None,
):
    return calliope.Model(
        os.path.join(os.path.dirname(__file__), "test_model", model_file),
        override_dict=override_dict,
        scenario=scenario,
        timeseries_dataframes=timeseries_dataframes,
    )


def check_error_or_warning(error_warning, test_string_or_strings):
    if hasattr(error_warning, "list"):
        output = ",".join(
            str(error_warning.list[i]) for i in range(len(error_warning.list))
        )
    else:
        output = str(error_warning.value)

    if isinstance(test_string_or_strings, list):
        result = all(test_string in output for test_string in test_string_or_strings)
    else:
        result = test_string_or_strings in output

    return result


def check_variable_exists(backend_model, constraint, variable, idx=None):
    """
    Search for existence of a decision variable in a Pyomo constraint.

    Parameters
    ----------
    backend_model : Pyomo ConcreteModel
    constraint : str, name of constraint which could exist in the backend
    variable : str, string to search in the list of variables to check if existing
    """
    if getattr(backend_model, constraint) in backend_model.component_objects(
        ctype=po.Constraint
    ):
        expression_accessor = "body"
    elif getattr(backend_model, constraint) in backend_model.component_objects(
        ctype=po.Expression
    ):
        expression_accessor = "value"
    if idx is not None:
        if idx in getattr(backend_model, constraint)._index:
            variables = identify_variables(
                getattr(getattr(backend_model, constraint)[idx], expression_accessor)
            )
            return any(variable in j.getname() for j in list(variables))
        else:
            return False
    else:
        exists = []
        for v in getattr(backend_model, constraint).values():
            variables = identify_variables(getattr(v, expression_accessor))
            exists.append(any(variable in j.getname() for j in list(variables)))
        return any(exists)
