from __future__ import annotations

from typing import Dict, KeysView, List, Optional
from typing_extensions import NotRequired, TypedDict, Required
from functools import partial

import pyparsing as pp
import xarray as xr


class ForEach(TypedDict):
    set_iterator: str
    set_name: str


# TODO: decide if this should simply be typed as a Dict and all the details are left in a YAML schema
class Constraint(TypedDict):
    foreach: Required[list]
    where: List[str]
    equation: NotRequired[str]
    equations: NotRequired[List[str]]
    components: NotRequired[List[Dict[str, str]]]
    index_items: NotRequired[List[Dict[str, str]]]


class ParsedConstraint:
    def __init__(self, constraint: Constraint, constraint_name: str) -> None:
        """Parse a constraint defined in a dictionary of strings loaded from YAML into a series of Python objects that can be passed onto a solver interface like Pyomo or Gurobipy.

        Args:
            constraint (Constraint): Dictionary of the form:
                foreach: List[str]  <- sets over which to iterate the constraint, e.g. ["timestep in timesteps"], where "timesteps" is the set and "timestep" is the reference to the set iterator in the constraint equation(s)
                where: List[str]  <- conditions defining which items in the product of all sets to apply the constraint to  FIXME: not implemented.
                equation: String  <- if no other conditions, single constraint equation of the form LHS OPERATOR RHS, e.g. "energy_cap[node, tech] >= my_parameter[tech] * 2"  FIXME: not implemented.
                equations: List[Dict]  <- if different equations for different conditions, list them here with an additional "where" statement and an associated expression: {"where": [...], "expression": "..."}  FIXME: not implemented.
                components: Dict[List[Dict]]  <- if applying the same expression to multiple equations in the constraint, store them here with optional conditions on their exact composition, e.g. "$energy_prod" in an equation would refer to a component {"energy_prod": [{"where": [...], "expression": "..."}, ...]}  FIXME: not implemented.
                index_items: Dict[List[Dict]]  <- if indexing a parameter/variable separately to the set iterators given in "foreach", define them here. FIXME: not implemented.
            constraint_name (str): Name of constraint.
        """
        self.name = constraint_name
        self._unparsed = constraint

        # capture warnings and errors to dump after processing,
        # to make it easier for a user to fix the constraint YAML.
        self._warnings = []
        self._errors = []

        # Initialise data variables
        self.sets = dict()

        # Initialise switches
        self._is_valid = True

    def parse_strings(self, model_data: xr.Dataset) -> None:
        """
        Parse all elements of the constraint: "foreach", "equation(s)", "component".

        Args:
            model_data (xr.Dataset):
                Calliope processed model dataset with all necessary parameters, sets,
                and run configuration options defined.

        """

        self._get_sets_from_foreach(model_data.dims.keys())

        return None

    @staticmethod
    def _foreach_parser() -> pp.ParserElement:
        """
        Returns:
            pp.ParserElement: Parsing grammar for strings of the form "A in B".
        """
        in_ = pp.Keyword("in", caseless=True)
        generic_identifier = pp.Combine(~in_ + pp.pyparsing_common.identifier)

        set_iterator = generic_identifier.set_results_name("set_iterator")
        set_name = generic_identifier.set_results_name("set_name")
        return set_iterator + pp.Suppress(in_) + set_name

    def _get_sets_from_foreach(self, model_data_dims: KeysView[str]) -> None:
        """
        Process "foreach" key in constraint to access the set iterators and the
        identifier for set items in the constraint equation expessions

        Args:
            model_data_dims (KeysView[str]):
                List of dimensions in calliope.Model._model_data
        """
        foreach_parser = self._foreach_parser()
        sets = dict()
        for string_ in self._unparsed["foreach"]:
            error_handler = partial(self._add_error, string_, "foreach")
            parsed_ = self._parse_string(foreach_parser, string_, "foreach")
            if parsed_ is not None:
                set_iterator, set_name = parsed_.as_list()
                if set_iterator in sets.keys():
                    error_handler(f"Found duplicate set iterator `{set_iterator}`.")
                if set_name not in model_data_dims:
                    error_handler(f"`{set_name}` not a valid model set name.")

                sets[set_iterator] = set_name

        if self._is_valid:
            self.sets = sets

        return None

    def _parse_string(
        self, parser: pp.ParserElement, parse_string: str, string_type: str
    ) -> Optional[pp.ParseResults]:
        """
        Parse equation string according to predefined string parsing grammar
        given by `self.parser`

        Args:
            parser (pp.ParserElement): Parsing grammar.
            parse_string (str): String to parse according to parser grammar.
            string_type (str): For error reporting, the constraint dict key corresponding to the parse_string.

        Returns:
            Optional[pp.ParseResults]:
                Parsed string or, if any parsing errors are caught, NoneType.
                Errors will be logged to `self._errors` to raise later.
        """
        try:
            parsed = parser.parse_string(parse_string, parse_all=True)
        except pp.ParseException as excinfo:
            parsed = None
            self._add_error(parse_string, string_type, excinfo)

        return parsed

    def _add_error(self, instring: str, string_type: str, error_message: str) -> None:
        """
        Add error message to the list self._errors following a predefined structure of
        `(string_type, instring): error`, e.g. `(foreach, a in A): Found duplicate set iterator`.

        Also set self._is_valid flag to False since at least one error has been caught.

        Args:
            instring (str): String being parsed where the error was caught.
            string_type (str):
                Location in the constraint definition where the string was defined,
                e.g., "foreach", "equations", "components".
            error_message (str): Description of error.
        """
        self._is_valid = False
        self._errors.append(f"({string_type}, {instring}): {error_message}")
