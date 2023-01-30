import itertools
from typing import KeysView, Optional, Union, Literal
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
        self._warnings = set()
        self._errors = set()

        # Initialise data variables
        self.sets = dict()
        self.equations: list[ConstraintDict] = []

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

        sets = self._get_sets_from_foreach(model_data.dims.keys())

        if "equation" in self._unparsed.keys():
            equation_expression_list = [{"expression": self._unparsed["equation"]}]
        else:
            equation_expression_list = self._unparsed["equations"]

        equations = self._parse_where_expression(
            expression_parser=equation_parser.generate_equation_parser(),
            expression_list=equation_expression_list,
            expression_group="equations",
        )

        component_dict = {
            c_name: self._parse_where_expression(
                expression_parser=equation_parser.generate_arithmetic_parser(),
                expression_list=c_list,
                expression_group="components",
                id_prefix=c_name,
            )
            for c_name, c_list in self._unparsed.get("components", {}).items()
        }
        index_item_dict = {
            idx_name: self._parse_where_expression(
                expression_parser=equation_parser.generate_index_item_parser(),
                expression_list=idx_list,
                expression_group="index_items",
                id_prefix=idx_name,
            )
            for idx_name, idx_list in self._unparsed.get("index_items", {}).items()
        }

        if self._is_valid:
            equations_with_components = self._add_sub_exprs_per_equation_expr(
                equations, component_dict, "components"
            )
            equations_with_components_and_index_items = (
                self._add_sub_exprs_per_equation_expr(
                    equations_with_components, index_item_dict, "index_items"
                )
            )

        if self._is_valid:
            self.sets = sets
            self.equations = equations_with_components_and_index_items

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

        return sets

    def _add_error(
        self, instring: str, expression_group: str, error_message: str
    ) -> None:
        """
        Add error message to the list self._errors following a predefined structure of
        `(expression_group, instring): error`, e.g. `(foreach, a in A): Found duplicate set iterator`.

        Also set self._is_valid flag to False since at least one error has been caught.

        Args:
            instring (str): String being parsed where the error was caught.
            expression_group (str):
                Location in the constraint definition where the string was defined,
                e.g., "foreach", "equations", "components".
            error_message (str): Description of error.
        """
        self._is_valid = False
        self._errors.add(f"({expression_group}, {instring}): {error_message}")

    def _parse_string(
        self,
        parser: pp.ParserElement,
        parse_string: str,
        expression_group: Literal["foreach", "equations", "components", "index_items"],
    ) -> Optional[pp.ParseResults]:
        """
        Parse equation string according to predefined string parsing grammar
        given by `self.parser`

        Args:
            parser (pp.ParserElement): Parsing grammar.
            parse_string (str): String to parse according to parser grammar.
            expression_group (str): For error reporting, the constraint dict key corresponding to the parse_string.

        Returns:
            Optional[pp.ParseResults]:
                Parsed string. If any parsing errors are caught,
                they will be logged to `self._errors` to raise later.
        """
        try:
            parsed = parser.parse_string(parse_string, parse_all=True)
        except (pp.ParseException, KeyError) as excinfo:
            parsed = None
            self._add_error(parse_string, expression_group, excinfo)

        return parsed

    def _parse_where_expression(
        self,
        expression_parser: pp.ParserElement,
        expression_list: list[dict],
        expression_group: Literal["foreach", "equations", "components", "index_items"],
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
            expression_group (str):
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
                    expression_parser, expression_data["expression"], expression_group
                ),
            }
            for idx, expression_data in enumerate(expression_list)
        ]

    def _find_items_in_expression(
        self,
        parser_elements: Union[list, pp.ParseResults],
        to_find: type[equation_parser.EvalString],
        valid_eval_classes=tuple[type(equation_parser.EvalString)],
    ) -> set[str]:
        """
        Recursively find components / index items defined in an equation expression.

        Args:
            parser_elements (pp.ParseResults): list of parser elements to check.
            to_find (type[equation_parser.EvalString]): type of equation element to search for
            valid_eval_classes (tuple[type(equation_parser.EvalString)]):
                Other expression elements that can be recursively searched

        Returns:
            set[str]: All unique component / index item names.
        """
        items = []
        recursive_func = partial(
            self._find_items_in_expression,
            to_find=to_find,
            valid_eval_classes=valid_eval_classes,
        )
        for parser_element in parser_elements:
            if isinstance(parser_element, to_find):
                items.append(parser_element.name)

            elif isinstance(parser_element, pp.ParseResults):
                items.extend(recursive_func(parser_elements=parser_element))

            elif isinstance(parser_element, valid_eval_classes):
                items.extend(recursive_func(parser_elements=parser_element.value))
        return set(items)

    def _get_expression_group_product(
        self,
        equation_data: ConstraintDict,
        parsed_items: dict[list[ConstraintDict]],
        expression_group: Literal["components", "index_items"],
    ) -> list[list[ConstraintDict]]:
        """
        Find all components referenced in an equation expression and return a
        product of the component data.

        Args:
            equation_data (ConstraintDict): Equation data dictionary.
            parsed_items (dict[list[ConstraintDict]]):
                Dictionary of expressions to replace within the equation data dictionary.
            expression_group (Literal["components", "index_items"]):
                Name of expression group that the parsed_items dict is referencing.

        Returns:
            list[list[ConstraintDict]]:
                Each nested list contains a unique product of parsed_item dictionaries.
        """
        eq_expr = equation_data["expression"][0]
        valid_eval_classes = (
            equation_parser.EvalOperatorOperand,
            equation_parser.EvalFunction,
        )
        elements = eq_expr.value.as_list()
        if expression_group == "components":
            to_find = equation_parser.EvalComponent

        elif expression_group == "index_items":
            elements += list(equation_data.get("components", {}).values())
            to_find = equation_parser.EvalIndexItems
            valid_eval_classes += (equation_parser.EvalIndexedParameterOrVariable,)

        eq_items = self._find_items_in_expression(elements, to_find, valid_eval_classes)

        invalid_items = eq_items.difference(parsed_items.keys())
        if invalid_items:
            self._add_error(
                eq_expr.__repr__(),
                "equation",
                f"Undefined {expression_group} found in equation: {invalid_items}",
            )

        eq_items.difference_update(invalid_items)

        return itertools.product(*[parsed_items[k] for k in eq_items])

    def _add_exprs_to_equation_data(
        self,
        equation_data: ConstraintDict,
        expression_combination: list[ConstraintDict],
        expression_group: Literal["components", "index_items"],
    ) -> ConstraintDict:
        """
        Create new equation dictionaries with evaluatable expressions for components or index items.
        The new equation dict has an updated ID, `where` list, and an additional key
        with the expression in.

        Args:
            equation_data (ConstraintDict):
                Original equation data dictionary with reference to the expression group.
            expression_combination (list[ConstraintDict]):
                list of data dictionaries to use in updating the equation data.
            expression_group (str): Name of the source of replacement expressions.

        Returns:
            ConstraintDict:
                Updated equation dictionary with unique ID and equation
                expressions attached under the key given by expression_group.
        """
        new_equation_data = equation_data.copy()

        expr_ids = [expr["id"] for expr in expression_combination]
        expr_wheres = [new_equation_data.pop("where")]
        for expr in expression_combination:
            expr_wheres.extend(["and", expr["where"]])

        new_equation_data[expression_group] = {
            expr["id"][0]: expr["expression"] for expr in expression_combination
        }

        return {
            "id": (new_equation_data.pop("id"), *expr_ids),
            "where": expr_wheres,
            expression_group: {
                expr["id"][0]: expr["expression"] for expr in expression_combination
            },
            **new_equation_data,
        }

    def _add_sub_exprs_per_equation_expr(
        self,
        equations: list[ConstraintDict],
        expression_dict: dict[str, list[ConstraintDict]],
        expression_group: Literal["components", "index_items"],
    ) -> list[ConstraintDict]:
        """
        Build new list of equation dictionaries with nested expression group information.

        Args:
            equations (list[ConstraintDict]):
                List of original equation data dictionaries with reference to the expression group.
            expression_dict (dict[str, list[ConstraintDict]]):

            expression_group (Literal[&quot;components&quot;, &quot;index_items&quot;]): _description_

        Returns:
            List[ConstraintDict]: _description_
        """
        updated_equations = []
        for equation_dict in equations:
            component_product = self._get_expression_group_product(
                equation_dict, expression_dict, expression_group
            )
            for component_combination in component_product:
                updated_equation_dict = self._add_exprs_to_equation_data(
                    equation_dict, component_combination, expression_group
                )
                updated_equations.append(updated_equation_dict)
        return updated_equations
