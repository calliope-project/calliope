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

    def _get_body(pyomo_parent_obj, pyomo_child_obj):
        if pyomo_parent_obj in backend_model.component_objects(ctype=po.Constraint):
            return pyomo_child_obj.body
        else:
            return pyomo_child_obj

    pyomo_obj = getattr(backend_model, constraint)
    if idx is not None:
        if idx in pyomo_obj.index_set():
            variables = identify_variables(_get_body(pyomo_obj, pyomo_obj[idx]))
            return any(variable in j.getname() for j in list(variables))
        else:
            return False
    else:
        exists = []
        for v in pyomo_obj.values():
            variables = identify_variables(_get_body(pyomo_obj, v))
            exists.append(any(variable in j.getname() for j in list(variables)))
        return any(exists)
