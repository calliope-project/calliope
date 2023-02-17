from __future__ import annotations

import itertools
from typing import Optional, Union, Literal, Iterable, Callable, TypeVar, Generic
from typing_extensions import NotRequired, TypedDict, Required
import functools
import operator
from abc import ABC

import pyparsing as pp
import xarray as xr
import numpy as np

from calliope.backend import equation_parser, subset_parser
from calliope.backend.backends import BackendModel
from calliope import exceptions
from calliope.backend import helper_functions

VALID_HELPER_FUNCTIONS: dict[str, Callable] = {
    "inheritance": helper_functions.inheritance,
    "sum": helper_functions.backend_sum,
    "squeeze_carriers": helper_functions.squeeze_carriers,
    "squeeze_primary_carriers": helper_functions.squeeze_primary_carriers,
    "get_connected_link": helper_functions.get_connected_link,
    "get_timestep": helper_functions.get_timestep,
    "roll": helper_functions.roll,
}


class UnparsedEquationDict(TypedDict):
    where: NotRequired[str]
    expression: str


class UnparsedConstraintDict(TypedDict):
    foreach: Required[list]
    where: str
    equation: NotRequired[str]
    equations: NotRequired[list[UnparsedEquationDict]]
    components: NotRequired[dict[str, list[UnparsedEquationDict]]]
    index_slices: NotRequired[dict[str, list[UnparsedEquationDict]]]


class UnparsedVariableBoundDict(TypedDict):
    min: str
    max: str
    equals: str
    scale: NotRequired[str]


class UnparsedVariableDict(TypedDict):
    foreach: list[str]
    where: str
    domain: NotRequired[str]
    bounds: UnparsedVariableBoundDict


class UnparsedObjectiveDict(TypedDict):
    equation: NotRequired[str]
    equations: NotRequired[list[UnparsedEquationDict]]
    components: NotRequired[dict[str, list[UnparsedEquationDict]]]
    domain: str
    sense: str


T = TypeVar(
    "T",
    bound=Union[UnparsedConstraintDict, UnparsedVariableDict, UnparsedObjectiveDict],
)


class ParsedBackendEquation:
    def __init__(
        self,
        equation_name: str,
        sets: list[str],
        expression: pp.ParseResults,
        where_list: list[pp.ParseResults],
        components: Optional[dict[str, pp.ParseResults]] = None,
        index_slices: Optional[dict[str, pp.ParseResults]] = None,
    ) -> None:
        self.name = equation_name
        self.where = where_list
        self.expression = expression
        self.components = components if components is not None else dict()
        self.index_slices = index_slices if index_slices is not None else dict()
        self.sets = sets

    def find_components(self) -> set[str]:
        valid_eval_classes: tuple = (
            equation_parser.EvalOperatorOperand,
            equation_parser.EvalFunction,
        )
        elements: list = [self.expression[0].values]
        to_find = equation_parser.EvalComponent

        return self._find_items_in_expression(elements, to_find, valid_eval_classes)

    def find_index_slices(self) -> set[str]:

        valid_eval_classes = (
            equation_parser.EvalOperatorOperand,
            equation_parser.EvalFunction,
            equation_parser.EvalSlicedParameterOrVariable,
        )
        elements = [self.expression[0].values, *list(self.components.values())]
        to_find = equation_parser.EvalIndexSlices

        return self._find_items_in_expression(elements, to_find, valid_eval_classes)

    @staticmethod
    def _find_items_in_expression(
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
            ParsedBackendEquation._find_items_in_expression,
            to_find=to_find,
            valid_eval_classes=valid_eval_classes,
        )
        for parser_element in parser_elements:
            if isinstance(parser_element, to_find):
                items.append(parser_element.name)

            elif isinstance(parser_element, (pp.ParseResults, list)):
                items.extend(recursive_func(parser_elements=parser_element))

            elif isinstance(parser_element, valid_eval_classes):
                items.extend(recursive_func(parser_elements=parser_element.values))
        return set(items)

    def add_expression_group_combination(
        self,
        expression_group_name: Literal["components", "index_slices"],
        expression_group_combination: Iterable[ParsedBackendEquation],
    ):
        new_where_list = [*self.where]
        for expr in expression_group_combination:
            new_where_list.extend(expr.where)
        new_name = f"{self.name}-{'-'.join([expr.name for expr in expression_group_combination])}"
        expression_group_dict = {
            expression_group_name: {
                expr.name.split(":")[0]: expr.expression
                for expr in expression_group_combination
            }
        }
        return ParsedBackendEquation(
            equation_name=new_name,
            sets=self.sets,
            expression=self.expression,
            where_list=new_where_list,
            **{
                "components": self.components,
                "index_slices": self.index_slices,
                **expression_group_dict,
            },
        )

    def evaluate_where(
        self,
        model_data: xr.Dataset,
        defaults: dict,
        initial_imask: Union[np.bool_, xr.DataArray] = np.True_,
    ) -> Union[np.bool_, xr.DataArray]:
        foreach_imask = self._evaluate_foreach(model_data)
        evaluated_wheres = [
            where[0].eval(  # type: ignore
                model_data=model_data,
                helper_func_dict=VALID_HELPER_FUNCTIONS,
                imask=foreach_imask,
                defaults=defaults,
            )
            for where in self.where
        ]

        imask: xr.DataArray = functools.reduce(
            operator.and_, [foreach_imask, initial_imask, *evaluated_wheres]
        )

        if isinstance(imask, xr.DataArray):
            # Squeeze out any unwanted dimensions
            unwanted_dims = set(imask.dims).difference(self.sets)
            imask = (imask.sum(unwanted_dims) > 0).astype(bool)

        return imask

    def _evaluate_foreach(self, model_data: xr.Dataset) -> xr.DataArray:
        """
        Generate a multi-dimensional imasking array based on the sets
        over which the constraint is to be built (defined by "foreach").
        Irrespective of the sets defined by "foreach", this array will always include
        ["nodes", "techs", "carriers", "carrier_tiers"] to ensure only valid combinations
        of technologies consuming/producing specific carriers at specific nodes are included in later imasking.

        Args:
            model_data (xr.Dataset): Calliope model dataset.

        Returns:
            xr.DataArray: imasking boolean array.
        """
        # Start with (carriers, carrier_tiers, nodes, techs) and go from there
        initial_imask = model_data.carrier.notnull() * model_data.node_tech.notnull()
        # Add other dimensions (costs, timesteps, etc.)
        add_dims = set(self.sets).difference(initial_imask.dims)
        if add_dims.difference(model_data.dims):
            raise exceptions.BackendError(
                "Unidentified model set name(s) defined: "
                f"`{add_dims.difference(model_data.dims)}`."
            )
        all_imasks = [initial_imask, *[model_data[i].notnull() for i in add_dims]]

        return functools.reduce(operator.and_, all_imasks)

    def evaluate_expression(
        self, model_data: xr.Dataset, backend_interface: BackendModel
    ):
        return self.expression[0].eval(
            equation_name=self.name,
            index_slice_dict=self.index_slices,
            component_dict=self.components,
            backend_interface=backend_interface,
            backend_dataset=backend_interface.dataset,
            helper_func_dict=VALID_HELPER_FUNCTIONS,
            model_data=model_data,
            as_dict=False,
        )


class ParsedBackendComponent(ABC, Generic[T]):
    def __init__(self, unparsed_data: T, component_name: str) -> None:
        """
        Parse an optimisation problem configuration - defined in a dictionary of strings
        loaded from YAML - into a series of Python objects that can be passed onto a solver
        interface like Pyomo or Gurobipy.
        """
        self.name: str = component_name
        self._unparsed: dict = dict(unparsed_data)

        # capture errors to dump after processing,
        # to make it easier for a user to fix the constraint YAML.
        self._errors: set = set()

        # Add objects that are used by shared functions
        self.sets: list[str] = unparsed_data.get("foreach", [])  # type:ignore
        top_level_where = self.parse_where_string(self._unparsed.get("where", "True"))
        if top_level_where is not None:
            self.where: list[pp.ParseResults] = [top_level_where]
        else:
            self.where = []
        self.equations: list[ParsedBackendEquation] = []

        # Initialise switches
        self._is_valid: bool = True
        self._is_active: bool = True

    def parse_equations(
        self, equation_expression_parser
    ) -> list[ParsedBackendEquation]:
        equation_expression_list: list[UnparsedEquationDict]
        if "equation" in self._unparsed.keys():
            equation_expression_list = [{"expression": self._unparsed["equation"]}]
        else:
            equation_expression_list = self._unparsed.get("equations", [])

        equations = self.generate_expression_list(
            expression_parser=equation_expression_parser,
            expression_list=equation_expression_list,
            expression_group="equations",
            id_prefix=self.name,
        )

        component_dict = {
            c_name: self.generate_expression_list(
                expression_parser=equation_parser.generate_arithmetic_parser(),
                expression_list=c_list,
                expression_group="components",
                id_prefix=c_name,
            )
            for c_name, c_list in self._unparsed.get("components", {}).items()
        }
        index_slice_dict = {
            idx_name: self.generate_expression_list(
                expression_parser=equation_parser.generate_index_slice_parser(),
                expression_list=idx_list,
                expression_group="index_slices",
                id_prefix=idx_name,
            )
            for idx_name, idx_list in self._unparsed.get("index_slices", {}).items()
        }

        if not self._is_valid:
            exceptions.print_warnings_and_raise_errors(
                errors=self._errors, during="string parsing"
            )

        equations_with_components = []
        for equation in equations:
            equations_with_components.extend(
                self.extend_equation_list_with_expression_group(
                    equation, component_dict, "components"
                )
            )
        equations_with_components_and_index_slices: list[ParsedBackendEquation] = []
        for equation in equations_with_components:
            equations_with_components_and_index_slices.extend(
                self.extend_equation_list_with_expression_group(
                    equation, index_slice_dict, "index_slices"
                )
            )

        return equations_with_components_and_index_slices

    def _parse_string(
        self,
        parser: pp.ParserElement,
        parse_string: str,
        expression_group: Literal[
            "foreach", "where", "equations", "components", "index_slices"
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
            self._is_valid = False
            self._errors.add(f"({expression_group}, {parse_string}): {str(excinfo)}")

        return parsed

    def parse_where_string(
        self, where_string: str = "True"
    ) -> Optional[pp.ParseResults]:
        """Parse a "where" string of the form "CONDITION OPERATOR CONDITION", where the
        operator can be "and"/"or"/"not and"/"not or".

        Args:
            equation_dict (Union[UnparsedEquationDict, UnparsedConstraintDict]):
                Dictionary with optional "where" key.
                If not found, the where string will default to "True", to have no effect
                on the subsequent subsetting.

        Returns:
            pp.ParseResults: Parsed string. If any parsing errors are caught,
                they will be logged to `self._errors` to raise later.
        """
        parser = subset_parser.generate_where_string_parser()
        return self._parse_string(parser, where_string, "where")

    def generate_expression_list(
        self,
        expression_parser: pp.ParserElement,
        expression_list: list[UnparsedEquationDict],
        expression_group: Literal["equations", "components", "index_slices"],
        id_prefix: str = "",
    ) -> list[ParsedBackendEquation]:
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
            list[UnparsedConstraintDict]:
                Aligned expression dictionaries with parsed expression strings.
        """
        parsed_equation_list = []
        for idx, expression_data in enumerate(expression_list):
            parsed_expression = self._parse_string(
                expression_parser, expression_data["expression"], expression_group
            )
            parsed_where = self.parse_where_string(expression_data.get("where", "True"))
            if parsed_expression is not None and parsed_where is not None:
                parsed_equation_list.append(
                    ParsedBackendEquation(
                        equation_name=":".join(filter(None, [id_prefix, str(idx)])),
                        sets=self.sets,
                        where_list=[parsed_where],
                        expression=parsed_expression,
                    )
                )

        return parsed_equation_list

    def extend_equation_list_with_expression_group(
        self,
        parsed_equation: ParsedBackendEquation,
        parsed_items: dict[str, list[ParsedBackendEquation]],
        expression_group: Literal["components", "index_slices"],
    ) -> list[ParsedBackendEquation]:
        """
        Find all components referenced in an equation expression and return a
        product of the component data.

        Args:
            equation_data (UnparsedConstraintDict): Equation data dictionary.
            parsed_items (dict[list[UnparsedConstraintDict]]):
                Dictionary of expressions to replace within the equation data dictionary.
            expression_group (Literal["components", "index_slices"]):
                Name of expression group that the parsed_items dict is referencing.

        Returns:
            list[list[UnparsedConstraintDict]]:
                Each nested list contains a unique product of parsed_item dictionaries.
        """
        if expression_group == "components":
            equation_items = parsed_equation.find_components()
        elif expression_group == "index_slices":
            equation_items = parsed_equation.find_index_slices()
        if not equation_items:
            return [parsed_equation]

        invalid_items = equation_items.difference(parsed_items.keys())
        if invalid_items:
            raise KeyError(
                f"({parsed_equation.expression.__repr__()}, equation): Undefined {expression_group} found in equation: {invalid_items}"
            )

        parsed_item_product = itertools.product(
            *[parsed_items[k] for k in equation_items]
        )

        return [
            parsed_equation.add_expression_group_combination(
                expression_group, parsed_item_combination
            )
            for parsed_item_combination in parsed_item_product
        ]


class ParsedConstraint(ParsedBackendComponent, ParsedBackendEquation):
    def __init__(
        self, constraint: UnparsedConstraintDict, constraint_name: str
    ) -> None:
        """Parse a constraint defined in a dictionary of strings loaded from YAML into a series of Python objects that can be passed onto a solver interface like Pyomo or Gurobipy.

        Args:
            constraint (UnparsedConstraintDict): Dictionary of the form:
                foreach: list[str]  <- sets over which to iterate the constraint, e.g. ["timestep in timesteps"], where "timesteps" is the set and "timestep" is the reference to the set iterator in the constraint equation(s)
                where: list[str]  <- conditions defining which items in the product of all sets to apply the constraint to
                equation: String  <- if no other conditions, single constraint equation of the form LHS OPERATOR RHS, e.g. "energy_cap[node, tech] >= my_parameter[tech] * 2"
                equations: list[dict]  <- if different equations for different conditions, list them here with an additional "where" statement and an associated expression: {"where": [...], "expression": "..."}
                components: dict[list[dict]]  <- if applying the same expression to multiple equations in the constraint, store them here with optional conditions on their exact composition, e.g. "$energy_prod" in an equation would refer to a component {"energy_prod": [{"where": [...], "expression": "..."}, ...]}
                index_slices: dict[list[dict]]  <- if indexing a parameter/variable separately to the set iterators given in "foreach", define them here.
            constraint_name (str): Name of constraint.
        """
        ParsedBackendComponent.__init__(self, constraint, constraint_name)
        self.equations = self.parse_equations(
            equation_parser.generate_equation_parser()
        )


class ParsedVariable(ParsedBackendComponent, ParsedBackendEquation):
    def __init__(self, variable: UnparsedVariableDict, variable_name: str) -> None:
        """Parse a variable configuration dictionary.

        Args:
            variable (UnparsedVariableDict): Dictionary of the form:
                foreach: list[str]  <- sets over which to iterate the constraint, e.g. ["techs", "timesteps"],
                where: list[str]  <- conditions defining which items in the product of all sets build the variable for
                domain: str <- limit on types of numeric values that can be assigned to the variable
                bounds: UnparsedBoundsDict <- link to parameters with which to apply explicit numeric bounds on each item in the variable
                    min: str
                    max: str
                    equals: str
                    scale: str
            variable_name (str): Name of variable.
        """
        ParsedBackendComponent.__init__(self, variable, variable_name)
        self.bounds = self._unparsed["bounds"]


class ParsedObjective(ParsedBackendComponent, ParsedBackendEquation):
    def __init__(self, objective: UnparsedObjectiveDict, objective_name: str) -> None:
        """Parse an objective configuration dictionary.

        Args:
            variable (UnparsedObjectiveDict): Dictionary of the form:
                equations: list[str]
            objective_name (str): Name of objective.
        """
        ParsedBackendComponent.__init__(self, objective, objective_name)
        self.equations: list[ParsedBackendEquation] = self.parse_equations(
            equation_parser.generate_arithmetic_parser()
        )
        self.sense = self._unparsed["sense"]


class ParsedExpression(ParsedBackendComponent, ParsedBackendEquation):
    def __init__(
        self, expression: UnparsedConstraintDict, expression_name: str
    ) -> None:
        """Parse an objective configuration dictionary.

        Args:
            variable (UnparsedObjectiveDict): Dictionary of the form:
                equations: list[str]
            objective_name (str): Name of objective.
        """
        ParsedBackendComponent.__init__(self, expression, expression_name)
        self.equations = self.parse_equations(
            equation_parser.generate_arithmetic_parser()
        )
