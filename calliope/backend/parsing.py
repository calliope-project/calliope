import itertools
from typing import KeysView, Optional
from typing_extensions import NotRequired, TypedDict, Required
from functools import partial

import pyparsing as pp
import xarray as xr

from calliope.backend import equation_parser
from calliope.exceptions import print_warnings_and_raise_errors


class ForEach(TypedDict):
    set_iterator: str
    set_name: str


# TODO: decide if this should simply be typed as a dict and all the details are left in a YAML schema
class ConstraintDict(TypedDict):
    foreach: Required[list]
    where: list[str]
    equation: NotRequired[str]
    equations: NotRequired[list[str]]
    components: NotRequired[list[dict[str, str]]]
    index_items: NotRequired[list[dict[str, str]]]


def raise_parsing_errors(self, constraints: list) -> None:
    errors_ = []
    warnings_ = []
    for constraint in constraints:
        errors_.extend(
            [
                f"({constraint.name}) {parse_string}: {error}"
                for parse_string, error in constraint._errors.items()
            ]
        )
        warnings_.extend(
            [
                f"({constraint.name}) {parse_string}: {warning}"
                for parse_string, warning in constraint._warnings.items()
            ]
        )
    print_warnings_and_raise_errors(
        warnings=warnings_, errors=errors_, during="Constraint string parsing"
    )


class ParsedConstraint:
    def __init__(self, constraint: ConstraintDict, constraint_name: str) -> None:
        """Parse a constraint defined in a dictionary of strings loaded from YAML into a series of Python objects that can be passed onto a solver interface like Pyomo or Gurobipy.

        Args:
            constraint (Constraint): Dictionary of the form:
                foreach: list[str]  <- sets over which to iterate the constraint, e.g. ["timestep in timesteps"], where "timesteps" is the set and "timestep" is the reference to the set iterator in the constraint equation(s)
                where: list[str]  <- conditions defining which items in the product of all sets to apply the constraint to  FIXME: not implemented.
                equation: String  <- if no other conditions, single constraint equation of the form LHS OPERATOR RHS, e.g. "energy_cap[node, tech] >= my_parameter[tech] * 2"  FIXME: not implemented.
                equations: list[dict]  <- if different equations for different conditions, list them here with an additional "where" statement and an associated expression: {"where": [...], "expression": "..."}  FIXME: not implemented.
                components: dict[list[dict]]  <- if applying the same expression to multiple equations in the constraint, store them here with optional conditions on their exact composition, e.g. "$energy_prod" in an equation would refer to a component {"energy_prod": [{"where": [...], "expression": "..."}, ...]}  FIXME: not implemented.
                index_items: dict[list[dict]]  <- if indexing a parameter/variable separately to the set iterators given in "foreach", define them here. FIXME: not implemented.
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
                list of dimensions in calliope.Model._model_data
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
                Parsed string. If any parsing errors are caught,
                they will be logged to `self._errors` to raise later.
        """
        try:
            parsed = parser.parse_string(parse_string, parse_all=True)
        except (pp.ParseException, KeyError) as excinfo:
            parsed = None
            self._add_error(parse_string, string_type, excinfo)

        return parsed

    def _parse_where_expression(
        self,
        expression_parser: pp.ParserElement,
        expression_list: list[dict],
        string_type: str,
        id_prefix: Optional[str] = None,
    ) -> list[ConstraintDict]:
        """
        Align user-defined constraint equations/components by parsing expressions,
        specifying a default "where" string if not defined,
        and providing an ID to enable returning to the initial dictionary.

        Args:
            expression_list (list[dict]):
                list of constraint equations or components with arithmetic expression
                string and optional where string.
            string_type (str):
                For error reporting, the constraint dict key corresponding to the parse_string.
            id_prefix (Optional[str]):
                If provided, will extend the ID from a number corresponding to the
                expression_list position `idx` to a tuple of the form (id_prefix, idx).

        Returns:
            list[ConstraintDict]:
                Aligned expression dictionaries with parsed expression strings.
        """

        return [
            {
                "id": idx if id_prefix is None else (id_prefix, idx),
                "where": expression_data.get("where", []),
                "expression": self._parse_string(
                    expression_parser, expression_data["expression"], string_type
                ),
            }
            for idx, expression_data in enumerate(expression_list)
        ]

    def _find_components(self, parser_elements: pp.ParseResults) -> set[str]:
        """
        Recursively find components defined in an equation expression.

        Args:
            parser_elements (pp.ParseResults): list of parser elements to check.

        Returns:
            set[str]: All unique component names.
        """
        components = []
        for parser_element in parser_elements:
            if isinstance(parser_element, equation_parser.EvalComponent):
                components.append(parser_element.name)

            elif isinstance(parser_element, (equation_parser.ArithmeticOperator)):
                components.extend(self._find_components(parser_element.value))
        return set(components)

    def _get_component_product(
        self,
        equation_data: ConstraintDict,
        parsed_components: list[ConstraintDict],
    ) -> list[list[ConstraintDict]]:
        """
        Find all components referenced in an equation expression and return a
        product of the component data.

        Args:
            parser (pp.ParserElement): Parser for arithmetic grammar.
            equation_data (ConstraintDict): Equation data dictionary.

        Returns:
            list[list[ConstraintDict]]:
                Each nested list contains a unique product of component data dictionaries.
        """

        eq_components = set(self._find_components(equation_data["expression"][0].value))

        invalid_components = eq_components.difference(parsed_components.keys())
        if invalid_components:
            self._errors.append(
                "Undefined component(s) found in equation "
                f"#{equation_data['id']}: {invalid_components}"
            )

        eq_components.difference_update(invalid_components)

        component_combinations = list(
            itertools.product(*[parsed_components[k] for k in eq_components])
        )

        return component_combinations

    def _combine_components_with_equation(
        self,
        equation_data: ConstraintDict,
        component_product: list[ConstraintDict],
    ) -> ConstraintDict:
        """Create new equations with components replaced by their underlying data.
        The new equation has an updated ID, `where` list, and expression.

        Args:
            equation_data (ConstraintDict):
                Original equation data dictionary with reference to components.
            component_product (list[ConstraintDict]):
                list of component data dictionaries to use in updating the equation data.

        Returns:
            ConstraintDict:
                Updated equation dictionary with unique ID and component equation
                expressions attached under the key `components`.
        """
        component_ids = [component["id"] for component in component_product]

        component_wheres = [equation_data["where"]]
        for component in component_product:
            component_wheres.extend(["and", component["where"]])

        return {
            "id": (equation_data["id"], *component_ids),
            "where": component_wheres,
            "expression": equation_data["expression"],
            "components": {
                component["id"][0]: component["expression"]
                for component in component_product
            },
        }

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
