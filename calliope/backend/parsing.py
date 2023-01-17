import itertools
from typing import Dict, KeysView, List

from typing_extensions import NotRequired, TypedDict, Required

import pyparsing as pp
import xarray as xr


class ForEach(TypedDict):
    set_item: str
    set_name: str


# TODO: decide if this should simply be typed as a Dict and all the details are left in a YAML schema
class Constraint(TypedDict):
    foreach: Required[list]
    where: List[str]
    equation: NotRequired[str]
    equations: NotRequired[List[str]]
    components: NotRequired[List[Dict[str, str]]]
    index_items: NotRequired[List[Dict[str, str]]]


def process_constraint(constraints, constraint_name):
    constraint = constraints[constraint_name]
    constraint_equations = parse_constraint_equations(constraint, constraint_name)
    return {"name": constraint_name, "equations": constraint_equations}


def parse_constraint_equations(constraint, constraint_name):
    """
    Builds the equations for a given constraint.

    Returns
    -------
    expanded_equations : list of equation dicts

    """
    parsed_constraint = ParsedConstraint(constraint, constraint_name)

    return None


def parse_foreach(
    foreach_string: str,
) -> ForEach:  # TODO: wrap in exception capture to pass more readable message about parsing errors to the user?
    """
    Extract information on the sets to loop over for a constraint and the
    name assigned to the set items in the constraint equation expression(s).

    Args:
        foreach_string (str): String of the form "A in B"

    Returns:
        ForEach: String parsed to dictionary {"set_item": "A", "set_name": "B"}
    """
    object_ = pp.Word(pp.alphas + "_", pp.alphanums + "_")
    dim_expr = object_ + pp.Suppress("in") + object_
    dim_expr.setParseAction(lambda t: {"set_iterator": t[0], "set_name": t[1]})
    parsed_string = dim_expr.parse_string(foreach_string, parseAll=True)
    return parsed_string


class ParsedConstraint:
    def __init__(
        self, constraint: Constraint, constraint_name: str, model_data: xr.Dataset
    ) -> None:
        """Parse a constraint defined in a dictionary of strings loaded from YAML into a series of Python objects that can be passed onto a solver interface like Pyomo or Gurobipy.

        Args:
            constraint (Constraint): Dictionary of the form:
                foreach: List[str]  <- sets over which to iterate the constraint, e.g. ["timestep in timesteps"], where "timesteps" is the set and "timestep" is the reference to the set iterator in the constraint equation(s)
                where: List[str]  <- conditions defining which items in the product of all sets to apply the constraint to
                equation: String  <- if no other conditions, single constraint equation of the form LHS OPERATOR RHS, e.g. "energy_cap[node, tech] >= my_parameter[tech] * 2"
                equations: List[Dict]  <- if different equations for different conditions, list them here with an additional "where" statement and an associated expression: {"where": [...], "expression": "..."}
                components: Dict[List[Dict]]  <- if applying the same expression to multiple equations in the constraint, store them here with optional conditions on their exact composition, e.g. "$energy_prod" in an equation would refer to a component {"energy_prod": [{"where": [...], "expression": "..."}, ...]}
                index_items: Dict[List[Dict]]  <- if indexing a parameter/variable separately to the set iterators given in "foreach", define them here. FIXME: not implemented.
            constraint_name (str): Name of constraint.
            model_data (xr.Dataset): Calliope processed model dataset with all necessary parameters, sets, and run configuration options defined.
        """
        self.name = constraint_name
        self._unparsed = constraint
        self._warnings = []  # capture warnings and errors to dump after processing, to make it easier for a user to fix the constraint YAML.
        self._errors = []  # capture warnings and errors to dump after processing, to make it easier for a user to fix the constraint YAML.

        self.get_sets(constraint, model_data.dims.keys())

    def get_sets(self, constraint: Constraint, model_data_dims: KeysView[str]) -> None:
        """Process "foreach" key in constraint to access the set iterators and the identifier for set items in the constraint equation expessions

        Args:
            constraint (Constraint): Dictionary describing the constraint definition
            model_data_dims (KeysView[str]): List of dimensions in calliope.Model._model_data
        """
        self.sets = [parse_foreach(_string)[0] for _string in constraint["foreach"]]
        self.set_names = [_set["set_name"] for _set in self.sets]
        self.set_iterators = [_set["set_iterator"] for _set in self.sets]

        unknown_sets = set(self.set_names).difference(model_data_dims)
        if unknown_sets:
            self._errors.append(
                f"Constraint sets {set(unknown_sets)} must be given as dimensions in the model dataset"
            )
