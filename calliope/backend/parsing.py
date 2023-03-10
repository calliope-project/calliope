from __future__ import annotations

import itertools
from typing import Optional, Union, Literal, Iterable, Callable, TypeVar
from typing_extensions import NotRequired, TypedDict, Required
import functools
import operator

import pyparsing as pp
import xarray as xr
import numpy as np

from calliope.backend import equation_parser, subset_parser, backends
from calliope import exceptions
from calliope.backend import helper_functions

VALID_EXPRESSION_HELPER_FUNCTIONS: dict[str, Callable] = {
    "sum": helper_functions.expression_sum,
    "squeeze_carriers": helper_functions.squeeze_carriers,
    "squeeze_primary_carriers": helper_functions.squeeze_primary_carriers,
    "get_connected_link": helper_functions.get_connected_link,
    "get_timestep": helper_functions.get_timestep,
    "roll": helper_functions.roll,
}
VALID_IMASK_HELPER_FUNCTIONS: dict[str, Callable] = {
    "inheritance": helper_functions.inheritance,
    "sum": helper_functions.imask_sum,
    "get_timestep": helper_functions.get_timestep,
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
        """
        Object for storing a parsed equation expression and corresponding "where" string,
        with methods to evaluate those elements.

        Args:
            equation_name (str): Name of equation.
            sets (list[str]):
                Model data sets with which to create the initial multi-dimensional masking array
                of the evaluated "where" string.
            expression (pp.ParseResults):
                Parsed arithmetic/equation expression.
            where_list (list[pp.ParseResults]):
                List of parsed where strings.
            components (Optional[dict[str, pp.ParseResults]], optional):
                Dictionary of parsed components with which to replace references to components
                on evaluation of the parsed expression. Defaults to None.
            index_slices (Optional[dict[str, pp.ParseResults]], optional):
                Dictionary of parsed index slices with which to replace references to index slices
                on evaluation of the parsed expression / components. Defaults to None.
        """
        self.name = equation_name
        self.where = where_list
        self.expression = expression
        self.components = components if components is not None else dict()
        self.index_slices = index_slices if index_slices is not None else dict()
        self.sets = sets

    def find_components(self) -> set[str]:
        """Identify all the references to components in the parsed expression.

        Returns:
            set[str]: Unique component references.
        """
        valid_eval_classes: tuple = (
            equation_parser.EvalOperatorOperand,
            equation_parser.EvalFunction,
        )
        elements: list = [self.expression[0].values]
        to_find = equation_parser.EvalComponent

        return self._find_items_in_expression(elements, to_find, valid_eval_classes)

    def find_index_slices(self) -> set[str]:
        """
        Identify all the references to index slices in the parsed expression or in the
        parsed components.

        Returns:
            set[str]: Unique index slice references.
        """

        valid_eval_classes = tuple(
            [
                equation_parser.EvalOperatorOperand,
                equation_parser.EvalFunction,
                equation_parser.EvalSlicedParameterOrVariable,
            ]
        )
        elements = [self.expression[0].values, *list(self.components.values())]
        to_find = equation_parser.EvalIndexSlices

        return self._find_items_in_expression(elements, to_find, valid_eval_classes)

    @staticmethod
    def _find_items_in_expression(
        parser_elements: Union[list, pp.ParseResults],
        to_find: type[equation_parser.EvalString],
        valid_eval_classes: tuple[type[equation_parser.EvalString], ...],
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
    ) -> ParsedBackendEquation:
        """
        Add dictionary of parsed components/index slices to a copy of self, updating
        the name and where list of self in the process.

        Args:
            expression_group_name (Literal[components, index_slices]):
                Which of `components`/`index slices` is being added.
            expression_group_combination (Iterable[ParsedBackendEquation]):
                All items of expression_group_name to be added.

        Returns:
            ParsedBackendEquation:
                Copy of self with added component/index slice dictionary and updated name
                and where list to include those corresponding to the dictionary entries.
        """
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
        initial_imask: Union[np.bool_, xr.DataArray] = np.True_,
    ) -> xr.DataArray:
        foreach_imask = self._evaluate_foreach(model_data)
        evaluated_wheres = [
            where[0].eval(  # type: ignore
                model_data=model_data, helper_func_dict=VALID_IMASK_HELPER_FUNCTIONS
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
        else:
            imask = xr.DataArray(imask)

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
        self,
        model_data: xr.Dataset,
        backend_interface: backends.BackendModel,
        imask: xr.DataArray,
        references: Optional[set] = None,
    ):
        return self.expression[0].eval(
            equation_name=self.name,
            index_slice_dict=self.index_slices,
            component_dict=self.components,
            backend_interface=backend_interface,
            backend_dataset=backend_interface._dataset,
            helper_func_dict=VALID_EXPRESSION_HELPER_FUNCTIONS,
            model_data=model_data,
            imask=imask,
            references=references if references is not None else set(),
            as_dict=False,
        )


class ParsedBackendComponent(ParsedBackendEquation):
    """
    Parse an optimisation problem configuration - defined in a dictionary of strings
    loaded from YAML - into a series of Python objects that can be passed onto a solver
    interface like Pyomo or Gurobipy.
    """

    def __init__(self, unparsed_data: T, component_name: str) -> None:
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
        self,
        equation_expression_parser: Callable,
        valid_arithmetic_components: Iterable,
    ) -> list[ParsedBackendEquation]:
        equation_expression_list: list[UnparsedEquationDict]
        if "equation" in self._unparsed.keys():
            equation_expression_list = [{"expression": self._unparsed["equation"]}]
        else:
            equation_expression_list = self._unparsed.get("equations", [])

        equations = self.generate_expression_list(
            expression_parser=equation_expression_parser(valid_arithmetic_components),
            expression_list=equation_expression_list,
            expression_group="equations",
            id_prefix=self.name,
        )

        component_dict = {
            c_name: self.generate_expression_list(
                expression_parser=equation_parser.generate_arithmetic_parser(
                    valid_arithmetic_components
                ),
                expression_list=c_list,
                expression_group="components",
                id_prefix=c_name,
            )
            for c_name, c_list in self._unparsed.get("components", {}).items()
        }
        index_slice_dict = {
            idx_name: self.generate_expression_list(
                expression_parser=equation_parser.generate_index_slice_parser(
                    valid_arithmetic_components
                ),
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
            parsed_where = self.parse_where_string(expression_data.get("where", "True"))
            parsed_expression = self._parse_string(
                expression_parser, expression_data["expression"], expression_group
            )
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
