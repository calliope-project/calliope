import os
import sys

import pytest
from pyomo.core.expr.current import identify_variables

import calliope
from calliope import AttrDict


constraint_sets = AttrDict.from_yaml(
    os.path.join(os.path.dirname(__file__), "constraint_sets.yaml")
)

defaults = AttrDict.from_yaml(
    os.path.join(os.path.dirname(calliope.__file__), "config", "defaults.yaml")
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


def check_variable_exists(backend_model, constraint, variable):
    """
    Search for existence of a decision variable in a Pyomo constraint.

    Parameters
    ----------
    backend_model : Pyomo ConcreteModel
    constraint : str, name of constraint which could exist in the backend
    variable : str, string to search in the list of variables to check if existing
    """
    exists = []
    for v in getattr(backend_model, constraint).values():
        variables = identify_variables(v.body)
        exists.append(any(variable in j.getname() for j in list(variables)))
    return any(exists)


def get_indexed_constraint_body(backend_model, constraint, input_index):
    """
    Return all indeces of a specific decision variable used in a constraint.
    This is useful to check that all expected loc_techs are in a summation.

    Parameters
    ----------
    backend_model : Pyomo ConcreteModel
    constraint : str,
        Name of constraint which could exist in the backend
    input_index : tuple or string,
        The index of the constraint in which to look for a the variable.
        The index may impact the index of the decision ariable

    """
    constraint_index = [
        v
        for v in getattr(backend_model, constraint).values()
        if v.index() == input_index
    ]
    if len(constraint_index) == 0:
        raise KeyError(
            "Unable to find index {} in constraint {}".format(input_index, constraint)
        )
    else:
        return constraint_index[0].body
