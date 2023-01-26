import itertools
from typing import Dict, KeysView, List, Optional, Union, Tuple, Set
from typing_extensions import NotRequired, TypedDict, Required

import pyparsing as pp
import xarray as xr

from calliope.backend import equation_parser
from calliope.exceptions import print_warnings_and_raise_errors


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


class ConstraintExpression(TypedDict):
    id: Union[int, Tuple[str, int]]
    where: List[str]
    expression: pp.ParseResults


def process_constraint(constraints, constraint_name):
    constraint = constraints[constraint_name]
    constraint_equations = parse_constraint_equations(constraint, constraint_name)
    raise_parsing_errors(constraints)
    return {"name": constraint_name, "equations": constraint_equations}


def raise_parsing_errors(constraints: list) -> None:
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


def parse_constraint_equations(constraint, constraint_name):
    """
    Builds the equations for a given constraint.

    Returns
    -------
    expanded_equations : list of equation dicts

    """
    parsed_constraint = ParsedConstraint(constraint, constraint_name)


class ParsedConstraint:
    def __init__(self, constraint: Constraint, constraint_name: str) -> None:
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
        """
        self.name = constraint_name
        self._unparsed = constraint
        # capture warnings and errors to dump after processing,
        # to make it easier for a user to fix the constraint YAML.
        self._warnings = []
        self._errors = []
        self.is_active = True
        self._is_valid = True

    def parse_constraint(self, model_data: xr.Dataset) -> None:
        """
        Parse all elements of the constraint: "foreach", "equation(s)", "component".

        Args:
            model_data (xr.Dataset):
                Calliope processed model dataset with all necessary parameters, sets,
                and run configuration options defined.

        """

        self._get_sets_from_foreach(model_data.dims.keys())

        equation_expression_parser = equation_parser.generate_equation_parser(
            self.set_iterators
        )
        component_expression_parser = equation_parser.generate_arithmetic_parser(
            self.set_iterators
        )

        if "equation" in self._unparsed.keys():
            equations = [{"expression": self._unparsed["equation"]}]
        elif "equations" in self._unparsed.keys():
            equations = self._unparsed["equations"]
        self._parsed_equations = self._parse_where_expression(
            equation_expression_parser, equations, "equation"
        )

        self._parsed_components = {
            c_name: self._parse_where_expression(
                component_expression_parser, c_list, "component", c_name
            )
            for c_name, c_list in self._unparsed.get("components", {}).items()
        }

        self._parsed_equations_with_components = [
            self._combine_components_with_equation(equation_data, component_product)
            for equation_data in self._parsed_equations
            for component_product in self._get_component_product(
                component_expression_parser, equation_data, self._parsed_components
            )
        ]

        return None

    @staticmethod
    def _foreach_parser(model_data_dims):
        set_iterator = pp.pyparsing_common.identifier.set_results_name("set_iterator")
        set_name = pp.one_of(model_data_dims, as_keyword=True).set_results_name(
            "set_name"
        )
        return set_iterator + pp.Suppress("in") + set_name

    def _get_sets_from_foreach(self, model_data_dims: KeysView[str]) -> None:
        """
        Process "foreach" key in constraint to access the set iterators and the
        identifier for set items in the constraint equation expessions

        Args:
            model_data_dims (KeysView[str]):
                List of dimensions in calliope.Model._model_data
        """
        foreach_parser = self._foreach_parser(model_data_dims)

        self.sets = []
        for string_ in self._unparsed["foreach"]:
            parsed_ = self._parse_string(foreach_parser, string_, "foreach")
            if parsed_ is not None:
                self.sets.append(parsed_.asDict())

        self.set_names = [_set["set_name"] for _set in self.sets]
        self.set_iterators = [_set["set_iterator"] for _set in self.sets]

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
            self._is_valid = False
            parsed = None
            self._errors.append(f"({string_type}, {parse_string}): {excinfo}")

        return parsed

    def _parse_where_expression(
        self,
        expression_parser: pp.ParserElement,
        expression_list: List[Dict],
        string_type: str,
        id_prefix: Optional[str] = None,
    ) -> List[ConstraintExpression]:
        """
        Align user-defined constraint equations/components by parsing expressions,
        specifying a default "where" string if not defined,
        and providing an ID to enable returning to the initial dictionary.

        Args:
            expression_list (List[Dict]):
                List of constraint equations or components with arithmetic expression
                string and optional where string.
            string_type (str):
                For error reporting, the constraint dict key corresponding to the parse_string.
            id_prefix (Optional[str]):
                If provided, will extend the ID from a number corresponding to the
                expression_list position `idx` to a tuple of the form (id_prefix, idx).

        Returns:
            List[ConstraintExpression]:
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

    def _find_components(self, parser_elements: pp.ParseResults) -> Set[str]:
        """
        Recursively find components defined in an equation expression.

        Args:
            parser_elements (pp.ParseResults): List of parser elements to check.

        Returns:
            Set[str]: All unique component names.
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
        equation_data: ConstraintExpression,
        parsed_components: List[ConstraintExpression],
    ) -> List[List[ConstraintExpression]]:
        """
        Find all components referenced in an equation expression and return a
        product of the component data.

        Args:
            parser (pp.ParserElement): Parser for arithmetic grammar.
            equation_data (ConstraintExpression): Equation data dictionary.

        Returns:
            List[List[ConstraintExpression]]:
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

        #component_combinations = list(
        #    itertools.product(*[[component_dict["id"] for component in eq_components for component_dict in #parsed_components[component]]])
        #)

        component_combinations = list(
            itertools.product(*[parsed_components[k] for k in eq_components])
        )

        return component_combinations

    def _combine_components_with_equation(
        self,
        equation_data: ConstraintExpression,
        component_product: List[ConstraintExpression],
    ) -> ConstraintExpression:
        """Create new equations with components replaced by their underlying data.
        The new equation has an updated ID, `where` list, and expression.

        Args:
            equation_data (ConstraintExpression):
                Original equation data dictionary with reference to components.
            component_product (List[ConstraintExpression]):
                List of component data dictionaries to use in updating the equation data

        Returns:
            ConstraintExpression:
                Updated equation dictionary with no reference to components
        """
        component_ids = [component["id"] for component in component_product]
        # This works to stop the main expression changing when updating the copy,
        # But the result isn't evaluatable. It has lost a level of parse action,
        # to evaluate the equation.
        eq_expression = equation_data["expression"][0].value.copy()
        eq_expression = self._replace_components(eq_expression, component_product)

        # This works to ensure the final result is evaluatable, but the main expression changes
        # when updating the copy.
        # eq_expression = equation_data["expression"].copy() <- since the copys don't do anything, might as well not have them.
        # eq_expression[0].value = equation_data["expression"][0].value.copy() <- since the copys don't do anything, might as well not have them.
        # eq_expression[0].value = self._replace_components(eq_expression[0].value, component_product)

        # Replace all components at their indexes with the component expression
        eq_expression = self._replace_components(eq_expression, component_product)

        component_wheres = [equation_data["where"]]
        for component in component_product:
            component_wheres.extend(["and", component["where"]])

        return {
            "id": (equation_data["id"], *component_ids),
            "where": component_wheres,
            "expression": eq_expression,
        }

    def _replace_components(
        self, eq_expr: pp.ParseResults, component_exprs: ConstraintExpression
    ) -> pp.ParseResults:
        """
        Recursively iterate through equation expression to find and replace component
        references with expressions.

        Args:
            eq_expr (pp.ParseResults): Equation expression with references to components
            component_exprs (ConstraintExpression): Component data dictionary

        Returns:
            pp.ParseResults: Equation expression with no reference to components
        """

        component_names = [component["id"][0] for component in component_exprs]
        for idx, expression_element in enumerate(eq_expr):
            if (
                isinstance(expression_element, equation_parser.EvalComponent)
                and expression_element.name in component_names
            ):
                component_product_idx = component_names.index(expression_element.name)
                eq_expr[idx] = component_exprs[component_product_idx]["expression"][0]

            elif isinstance(expression_element, (equation_parser.ArithmeticOperator)):
                eq_expr[idx].value = self._replace_components(
                    expression_element.value.copy(), component_exprs
                )
        return eq_expr
