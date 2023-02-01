import itertools
from typing import KeysView, Optional, Union, Literal, Iterable
from typing_extensions import NotRequired, TypedDict, Required
import functools
import operator

import pyparsing as pp
import xarray as xr
import pandas as pd

from calliope.backend import equation_parser, subset_parser
from calliope.exceptions import print_warnings_and_raise_errors


def _inheritance(model_data, **kwargs):
    def __inheritance(tech_group):
        # Only for base tech inheritance
        return model_data.inheritance.str.endswith(tech_group)

    return __inheritance


VALID_HELPER_FUNCTIONS = {
    "inheritance": _inheritance,
}


class UnparsedEquationDict(TypedDict):
    where: NotRequired[str]
    expression: str


class ConstraintDict(TypedDict):
    foreach: Required[list]
    where: str
    equation: NotRequired[str]
    equations: NotRequired[list[UnparsedEquationDict]]
    components: NotRequired[dict[str, list[UnparsedEquationDict]]]
    index_items: NotRequired[dict[str, list[UnparsedEquationDict]]]


class ParsedEquationDict(TypedDict):
    id: Union[int, tuple[str, int]]
    where: list[Optional[pp.ParseResults]]
    expression: Optional[pp.ParseResults]
    components: NotRequired[dict[str, pp.ParseResults]]
    index_items: NotRequired[dict[str, pp.ParseResults]]


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
        self._warnings: set = set()
        self._errors: set = set()

        # Initialise data variables
        self.sets: dict = dict()
        self.equations: list[ParsedEquationDict] = []
        self.top_level_where: Optional[pp.ParseResults] = None

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
        top_level_where = self._parse_string(
            subset_parser.generate_where_string_parser(),
            self._unparsed.get("where", "True"),
            "where",
        )
        equation_expression_list: list[UnparsedEquationDict]

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
            self.top_level_where = top_level_where

        return None

    def build_constraint(self, model_data):
        if not self._is_valid:
            return None
        for constraint_equation in self.equations:
            imask = self._create_valid_subset(model_data, constraint_equation)
            if not self._is_valid:
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

    def _get_sets_from_foreach(self, model_data_dims: KeysView) -> dict[str, str]:
        """
        Process "foreach" key in constraint to access the set iterators and the
        identifier for set items in the constraint equation expessions

        Args:
            model_data_dims (KeysView):
                list of dimensions in calliope.Model._model_data
        """
        foreach_parser = self._foreach_parser()
        sets: dict = dict()
        for string_ in self._unparsed["foreach"]:
            error_handler = functools.partial(self._add_error, string_, "foreach")
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
        expression_group: Literal[
            "foreach", "where", "equations", "components", "index_items"
        ],
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
            self._add_error(parse_string, expression_group, str(excinfo))

        return parsed

    def _parse_where_expression(
        self,
        expression_parser: pp.ParserElement,
        expression_list: list[UnparsedEquationDict],
        expression_group: Literal["foreach", "equations", "components", "index_items"],
        id_prefix: Optional[str] = None,
    ) -> list[ParsedEquationDict]:
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
                "where": [
                    self._parse_string(
                        subset_parser.generate_where_string_parser(),
                        expression_data.get("where", "True"),
                        "where",
                    )
                ],
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
        valid_eval_classes=tuple[type[equation_parser.EvalString]],
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
        items: list = []
        recursive_func = functools.partial(
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
        equation_data: ParsedEquationDict,
        parsed_items: dict[str, list[ParsedEquationDict]],
        expression_group: Literal["components", "index_items"],
    ) -> itertools.product:
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
        eq_expr: pp.ParseResults = equation_data["expression"]  # type: ignore
        valid_eval_classes: tuple = (
            equation_parser.EvalOperatorOperand,
            equation_parser.EvalFunction,
        )
        elements: list = [eq_expr[0].lhs, eq_expr[0].rhs]
        if expression_group == "components":
            to_find = equation_parser.EvalComponent

        elif expression_group == "index_items":
            elements += list(equation_data.get("components", {}).values())
            to_find = equation_parser.EvalIndexItems  # type: ignore
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
        equation_data: ParsedEquationDict,
        expression_combination: Iterable[ParsedEquationDict],
        expression_group: Literal["components", "index_items"],
    ) -> ParsedEquationDict:
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

        return {
            "id": (
                new_equation_data.pop("id"),
                *[expr["id"] for expr in expression_combination],
            ),
            "where": [
                *new_equation_data.pop("where"),
                *[expr["where"][0] for expr in expression_combination],
            ],
            expression_group: {  # type: ignore
                expr["id"][0]: expr["expression"] for expr in expression_combination
            },
            **new_equation_data,
        }

    def _add_sub_exprs_per_equation_expr(
        self,
        equations: list[ParsedEquationDict],
        expression_dict: dict[str, list[ParsedEquationDict]],
        expression_group: Literal["components", "index_items"],
    ) -> list[ParsedEquationDict]:
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

    def _create_valid_subset(
        self, model_data: xr.Dataset, equation_dict: ParsedEquationDict
    ) -> Optional[pd.Index]:
        """
        Returns the subset for which a given constraint, variable or
        expression is valid, based on the given configuration. See `config/subsets.yaml` for
        the configuration definitions.

        Parameters
        ----------

        model_data : xarray.Dataset (calliope.Model._model_data)
        name : str
            Name of the constraint, variable or expression
        config : dict
            Configuration for the constraint, variable or expression

        Returns
        -------
        valid_subset : pandas.MultiIndex

        """

        # Start with a mask that is True where the tech exists at a node (across all timesteps and for a each carrier and cost, where appropriate)
        imask_foreach = self._imask_foreach(model_data)

        evaluated_wheres = [
            where[0].eval(
                model_data=model_data,
                helper_func_dict=VALID_HELPER_FUNCTIONS,
                errors=self._errors,
                imask=imask_foreach,
            )
            for where in [self.top_level_where, *equation_dict["where"]]
        ]
        imask = functools.reduce(operator.and_, [imask_foreach, *evaluated_wheres])

        # Only build and return imask if there are some non-zero elements
        if imask.sum() != 0:
            # Squeeze out any unwanted dimensions
            if len(imask.dims) > len(self.sets):
                unwanted_dims = [i for i in imask.dims if i not in self.sets.values()]
                imask = (imask.sum(unwanted_dims) > 0).astype(bool)
            if len(imask.dims) < len(self.sets):
                raise ValueError(f"Missing dimension(s) in imask for set {self.name}")

            valid_subset = self._get_subset_as_index(imask)
        else:
            self._is_valid = False
            valid_subset = None

        return valid_subset

    def _get_subset_as_index(self, imask: xr.DataArray):
        if len(imask.dims) == 1:
            return imask[imask].coords.to_index()
        else:
            mask_stacked = imask.stack(dim_0=imask.dims)
            return (
                mask_stacked[mask_stacked]
                .coords.to_index()
                .reorder_levels(self.sets.values())
            )

    def _imask_foreach(self, model_data: xr.Dataset) -> xr.DataArray:
        set_names = self.sets.values()
        # Start with (carrier, node, tech) and go from there
        initial_imask = model_data.carrier.notnull() * model_data.node_tech.notnull()
        # Squeeze out any of (carrier, node, tech) not in foreach
        reduced_imask = (
            initial_imask.sum([i for i in initial_imask.dims if i not in set_names]) > 0
        )
        # Add other dimensions (costs, timesteps, etc.)
        imask = functools.reduce(
            operator.and_,
            [
                reduced_imask,
                *[
                    model_data[i].notnull()
                    for i in set_names
                    if i not in reduced_imask.dims
                ],
            ],
        )

        return imask
