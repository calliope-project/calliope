# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).

from __future__ import annotations

import itertools
from typing import Optional, Union, Literal, Iterable, Callable, TypeVar
from typing_extensions import NotRequired, TypedDict, Required
import functools
import operator

import pyparsing as pp
import xarray as xr

from calliope.backend import equation_parser, subset_parser, backends
from calliope import exceptions
from calliope.backend import helper_functions

VALID_EXPRESSION_HELPER_FUNCTIONS: dict[str, Callable] = {
    "sum": helper_functions.expression_sum,
    "squeeze_carriers": helper_functions.squeeze_carriers,
    "squeeze_primary_carriers": helper_functions.squeeze_primary_carriers,
    "get_connected_link": helper_functions.get_connected_link,
    "get_val_at_index": helper_functions.get_val_at_index,
    "roll": helper_functions.roll,
}
VALID_IMASK_HELPER_FUNCTIONS: dict[str, Callable] = {
    "inheritance": helper_functions.inheritance,
    "sum": helper_functions.imask_sum,
    "get_val_at_index": helper_functions.get_val_at_index,
}

TRUE_ARRAY = xr.DataArray(True)


class UnparsedEquationDict(TypedDict):
    where: NotRequired[str]
    expression: str


class UnparsedConstraintDict(TypedDict):
    description: NotRequired[str]
    foreach: Required[list]
    where: str
    equation: NotRequired[str]
    equations: NotRequired[list[UnparsedEquationDict]]
    sub_expressions: NotRequired[dict[str, list[UnparsedEquationDict]]]
    index_slices: NotRequired[dict[str, list[UnparsedEquationDict]]]


class UnparsedExpressionDict(UnparsedConstraintDict):
    unit: NotRequired[str]


class UnparsedVariableBoundDict(TypedDict):
    min: str
    max: str
    equals: str
    scale: NotRequired[str]


class UnparsedVariableDict(TypedDict):
    description: NotRequired[str]
    unit: NotRequired[str]
    foreach: list[str]
    where: str
    domain: NotRequired[str]
    bounds: UnparsedVariableBoundDict


class UnparsedObjectiveDict(TypedDict):
    description: NotRequired[str]
    equation: NotRequired[str]
    equations: NotRequired[list[UnparsedEquationDict]]
    sub_expressions: NotRequired[dict[str, list[UnparsedEquationDict]]]
    domain: str
    sense: str


UNPARSED_DICTS = Union[
    UnparsedConstraintDict,
    UnparsedVariableDict,
    UnparsedExpressionDict,
    UnparsedObjectiveDict,
]
T = TypeVar("T", bound=UNPARSED_DICTS)


class ParsedBackendEquation:
    def __init__(
        self,
        equation_name: str,
        sets: list[str],
        expression: pp.ParseResults,
        where_list: list[pp.ParseResults],
        sub_expressions: Optional[dict[str, pp.ParseResults]] = None,
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
            sub_expressions (Optional[dict[str, pp.ParseResults]], optional):
                Dictionary of parsed sub expressions with which to replace references to components
                on evaluation of the parsed expression. Defaults to None.
            index_slices (Optional[dict[str, pp.ParseResults]], optional):
                Dictionary of parsed index slices with which to replace references to index slices
                on evaluation of the parsed expression / components. Defaults to None.
        """
        self.name = equation_name
        self.where = where_list
        self.expression = expression
        self.sub_expressions = (
            sub_expressions if sub_expressions is not None else dict()
        )
        self.index_slices = index_slices if index_slices is not None else dict()
        self.sets = sets

    def find_sub_expressions(self) -> set[str]:
        """Identify all the references to sub_expressions in the parsed expression.

        Returns:
            set[str]: Unique sub-expressions references.
        """
        valid_eval_classes: tuple = (
            equation_parser.EvalOperatorOperand,
            equation_parser.EvalFunction,
        )
        elements: list = [self.expression[0].values]
        to_find = equation_parser.EvalSubExpressions

        return self._find_items_in_expression(elements, to_find, valid_eval_classes)

    def find_index_slices(self) -> set[str]:
        """
        Identify all the references to index slices in the parsed expression or in the
        parsed sub-expressions.

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
        elements = [self.expression[0].values, *list(self.sub_expressions.values())]
        to_find = equation_parser.EvalIndexSlice

        return self._find_items_in_expression(elements, to_find, valid_eval_classes)

    @staticmethod
    def _find_items_in_expression(
        parser_elements: Union[list, pp.ParseResults],
        to_find: type[equation_parser.EvalString],
        valid_eval_classes: tuple[type[equation_parser.EvalString], ...],
    ) -> set[str]:
        """
        Recursively find sub-expressions / index items defined in an equation expression.

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
        expression_group_name: Literal["sub_expressions", "index_slices"],
        expression_group_combination: Iterable[ParsedBackendEquation],
    ) -> ParsedBackendEquation:
        """
        Add dictionary of parsed sub-expressions/index slices to a copy of self, updating
        the name and where list of self in the process.

        Args:
            expression_group_name (Literal[sub_expressions, index_slices]):
                Which of `sub-expressions`/`index slices` is being added.
            expression_group_combination (Iterable[ParsedBackendEquation]):
                All items of expression_group_name to be added.

        Returns:
            ParsedBackendEquation:
                Copy of self with added sub-expressions/index slice dictionary and updated name
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
                "sub_expressions": self.sub_expressions,
                "index_slices": self.index_slices,
                **expression_group_dict,  # type: ignore
            },
        )

    def evaluate_where(
        self,
        model_data: xr.Dataset,
        initial_imask: xr.DataArray = TRUE_ARRAY,
    ) -> xr.DataArray:
        """Evaluate parsed backend object dictionary `where` string.
        NOTE: imask = inverse mask (application of "np.where" to an array)

        Args:
            model_data (xr.Dataset): Calliope model dataset.
            initial_imask (xr.DataArray, optional):
                If given, the imask resulting from evaluation will be further imasked by this array.
                Defaults to xr.DataArray(True) (i.e., no effect).

        Returns:
            xr.DataArray: _description_
        """

        evaluated_wheres = [
            where[0].eval(
                model_data=model_data, helper_func_dict=VALID_IMASK_HELPER_FUNCTIONS
            )
            for where in self.where
        ]

        imask: xr.DataArray = functools.reduce(
            operator.and_, [initial_imask, *evaluated_wheres]
        )

        return xr.DataArray(imask)

    def align_imask_with_sets(self, imask: xr.DataArray):
        unwanted_dims = set(imask.dims).difference(self.sets)
        return (imask.sum(unwanted_dims) > 0).astype(bool)

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
            sub_expression_dict=self.sub_expressions,
            backend_interface=backend_interface,
            backend_dataset=backend_interface._dataset,
            helper_func_dict=VALID_EXPRESSION_HELPER_FUNCTIONS,
            model_data=model_data,
            imask=imask,
            references=references if references is not None else set(),
            as_dict=False,
        )


class ParsedBackendComponent(ParsedBackendEquation):
    _ERR_BULLET: str = " * "
    _ERR_STRING_ORDER: list[str] = ["expression_group", "id", "expr_or_where"]
    PARSERS: dict[str, Callable] = {
        "constraints": equation_parser.generate_equation_parser,
        "expressions": equation_parser.generate_arithmetic_parser,
        "objectives": equation_parser.generate_arithmetic_parser,
        "variables": lambda x: None,
    }

    def __init__(
        self,
        group: Literal["variables", "expressions", "constraints", "objectives"],
        name: str,
        unparsed_data: T,
    ) -> None:
        """
        Parse an optimisation problem configuration - defined in a dictionary of strings
        loaded from YAML - into a series of Python objects that can be passed onto a solver
        interface like Pyomo or Gurobipy.

        Args:
            group (Literal["variables", "expressions", "constraints", "objectives"]): Optimisation problem component group to which the unparsed data belongs.
            name (str): Name of the optimisation problem component
            unparsed_data (T): Unparsed math formulation. Expected structure depends on the group to which the optimisation problem component belongs.
        """
        self.name = name
        self.group_name = group
        self._unparsed: dict = dict(unparsed_data)

        self.where: list[pp.ParseResults] = []
        self.equations: list[ParsedBackendEquation] = []
        self.equation_expression_parser: Callable = self.PARSERS[group]

        # capture errors to dump after processing,
        # to make it easier for a user to fix the constraint YAML.
        self._errors: list = []
        self._tracker = self._init_tracker()

        # Initialise switches
        self._is_valid: bool = True
        self._is_active: bool = self._unparsed.get("active", True)

        # Add objects that are used by shared functions
        self.sets: list[str] = unparsed_data.get("foreach", [])  # type:ignore

    def get_parsing_position(self):
        """Create "." separated list from tracked strings"""
        return ".".join(
            filter(None, [self._tracker[i] for i in self._ERR_STRING_ORDER])
        )

    def reset_tracker(self):
        """Re-initialise error string tracking"""
        self._tracker = self._init_tracker()

    def _init_tracker(self):
        "Initialise error string tracking as dictionary of `key: None`"
        return {i: None for i in self._ERR_STRING_ORDER}

    def parse_top_level_where(
        self, errors: Literal["raise", "ignore"] = "raise"
    ) -> None:
        """Parse the "where" string that is (optionally) given as a top-level key of the math component dictionary.

        Args:
            errors (Literal["raise", "ignore"], optional):
                Collected parsing errors can be raised directly or ignored.
                If errors exist and are ignored, the parsed component cannot be successfully evaluated. Defaults to "raise".
        """
        top_level_where = self.parse_where_string(self._unparsed.get("where", "True"))

        if errors == "raise":
            self.raise_caught_errors()

        if self._is_valid:
            self.where = [top_level_where]

    def parse_equations(
        self,
        valid_math_element_names: Iterable[str],
        errors: Literal["raise", "ignore"] = "raise",
    ) -> list[ParsedBackendEquation]:
        f"""Parse `expression` and `where` strings of backend object configuration dictionary:

        {self._unparsed}

        Args:
            valid_math_element_names (Iterable[str]):
                strings referring to valid backend objects to allow the parser to differentiate between them and generic strings.
            errors (Literal["raise", "ignore"], optional):
                Collected parsing errors can be raised directly or ignored.
                If errors exist and are ignored, the parsed component cannot be successfully evaluated. Defaults to "raise".

        Returns:
            list[ParsedBackendEquation]:
                List of parsed equations ready to be evaluated.
                The length of the list depends on the product of provided equations and sub-expression/index_slice references.
        """
        equation_expression_list: list[UnparsedEquationDict]
        if "equation" in self._unparsed.keys():
            equation_expression_list = [{"expression": self._unparsed["equation"]}]
        else:
            equation_expression_list = self._unparsed.get("equations", [])

        equations = self.generate_expression_list(
            expression_parser=self.equation_expression_parser(valid_math_element_names),
            expression_list=equation_expression_list,
            expression_group="equations",
            id_prefix=self.name,
        )

        sub_expression_dict = {
            c_name: self.generate_expression_list(
                expression_parser=equation_parser.generate_sub_expression_parser(
                    valid_math_element_names
                ),
                expression_list=c_list,
                expression_group="sub_expressions",
                id_prefix=c_name,
            )
            for c_name, c_list in self._unparsed.get("sub_expressions", {}).items()
        }
        index_slice_dict = {
            idx_name: self.generate_expression_list(
                expression_parser=equation_parser.generate_index_slice_parser(
                    valid_math_element_names
                ),
                expression_list=idx_list,
                expression_group="index_slices",
                id_prefix=idx_name,
            )
            for idx_name, idx_list in self._unparsed.get("index_slices", {}).items()
        }

        if errors == "raise":
            self.raise_caught_errors()

        equations_with_sub_expressions = []
        for equation in equations:
            equations_with_sub_expressions.extend(
                self.extend_equation_list_with_expression_group(
                    equation, sub_expression_dict, "sub_expressions"
                )
            )
        equations_with_sub_expressions_and_index_slices: list[
            ParsedBackendEquation
        ] = []
        for equation in equations_with_sub_expressions:
            equations_with_sub_expressions_and_index_slices.extend(
                self.extend_equation_list_with_expression_group(
                    equation, index_slice_dict, "index_slices"
                )
            )

        return equations_with_sub_expressions_and_index_slices

    def _parse_string(
        self,
        parser: pp.ParserElement,
        parse_string: str,
    ) -> pp.ParseResults:
        """
        Parse equation string according to predefined string parsing grammar
        given by `self.parser`

        Args:
            parser (pp.ParserElement): Parsing grammar.
            parse_string (str): String to parse according to parser grammar.

        Returns:
            Optional[pp.ParseResults]:
                Parsed string. If any parsing errors are caught,
                they will be logged to `self._errors` to raise later.
        """
        try:
            parsed = parser.parse_string(parse_string, parse_all=True)
        except pp.ParseException as excinfo:
            parsed = pp.ParseResults([])
            self._is_valid = False
            pointer = f"{self.get_parsing_position()} (line {excinfo.lineno}, char {excinfo.col}): "
            marker_pos = " " * (
                len(pointer) + 2 * len(self._ERR_BULLET) + excinfo.col - 1
            )
            self._errors.append(f"{pointer}{excinfo.line}\n{marker_pos}^")

        return parsed

    def parse_where_string(self, where_string: str = "True") -> pp.ParseResults:
        """Parse a "where" string of the form "CONDITION OPERATOR CONDITION", where the
        operator can be "and"/"or"/"not and"/"not or".

        Args:
            where_string (str):
                string value from a math dictionary "where" key.
                Defaults to "True", to have no effect on the subsequent subsetting.

        Returns:
            pp.ParseResults: Parsed string. If any parsing errors are caught,
                they will be logged to `self._errors` to raise later.
        """
        parser = subset_parser.generate_where_string_parser()
        self._tracker["expr_or_where"] = "where"
        return self._parse_string(parser, where_string)

    def generate_expression_list(
        self,
        expression_parser: pp.ParserElement,
        expression_list: list[UnparsedEquationDict],
        expression_group: Literal["equations", "sub_expressions", "index_slices"],
        id_prefix: str = "",
    ) -> list[ParsedBackendEquation]:
        """
        Align user-defined constraint equations/sub-expressions by parsing expressions,
        specifying a default "where" string if not defined,
        and providing an ID to enable returning to the initial dictionary.

        Args:
            expression_list (list[dict]):
                list of constraint equations or sub-expressions with arithmetic expression
                string and optional where string.
            expression_group (str):
                For error reporting, the constraint dict key corresponding to the parse_string.
            id_prefix (Optional[str]):
                If provided, will extend the ID from a number corresponding to the
                expression_list position `idx` to a tuple of the form (id_prefix, idx).

        Returns:
            list[ParsedBackendEquation]:
                Aligned expression dictionaries with parsed expression strings.
        """
        parsed_equation_list = []

        if expression_group == "equations":
            to_track = {"expression_group": f"{expression_group}[{{id}}]"}
        else:
            to_track = {
                "expression_group": expression_group,
                "id": f"{id_prefix}[{{id}}]",
            }

        for idx, expression_data in enumerate(expression_list):
            self._tracker.update({k: v.format(id=idx) for k, v in to_track.items()})

            parsed_where = self.parse_where_string(expression_data.get("where", "True"))

            self._tracker["expr_or_where"] = "expression"
            parsed_expression = self._parse_string(
                expression_parser, expression_data["expression"]
            )
            if len(parsed_expression) > 0:
                parsed_equation_list.append(
                    ParsedBackendEquation(
                        equation_name=":".join(filter(None, [id_prefix, str(idx)])),
                        sets=self.sets,
                        where_list=[parsed_where],
                        expression=parsed_expression,
                    )
                )
        self.reset_tracker()

        return parsed_equation_list

    def extend_equation_list_with_expression_group(
        self,
        parsed_equation: ParsedBackendEquation,
        parsed_items: dict[str, list[ParsedBackendEquation]],
        expression_group: Literal["sub_expressions", "index_slices"],
    ) -> list[ParsedBackendEquation]:
        """
        Find all sub-expressions referenced in an equation expression and return a
        product of the sub-expression data.

        Args:
            equation_data (ParsedBackendEquation): Equation data dictionary.
            parsed_items (dict[str, list[ParsedBackendEquation]]):
                Dictionary of expressions to replace within the equation data dictionary.
            expression_group (Literal["sub_expressions", "index_slices"]):
                Name of expression group that the parsed_items dict is referencing.

        Returns:
            list[ParsedBackendEquation]:
                Expanded list of parsed equations with the product of all references to items from the `expression_group` producing a new equation object. E.g., if the input equation object has a reference to an index_slice which itself has two expression options, two equation objects will be added to the return list.
        """
        if expression_group == "sub_expressions":
            equation_items = parsed_equation.find_sub_expressions()
        elif expression_group == "index_slices":
            equation_items = parsed_equation.find_index_slices()
        if not equation_items:
            return [parsed_equation]

        invalid_items = equation_items.difference(parsed_items.keys())
        if invalid_items:
            raise KeyError(
                f"({self.group_name}, {self.name}): Undefined {expression_group} found in equation: {invalid_items}"
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

    def evaluate_foreach(self, model_data: xr.Dataset) -> xr.DataArray:
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
            exceptions.warn(
                f"Not generating optimisation problem object `{self.name}` because it is "
                f"indexed over unidentified set name(s): `{add_dims.difference(model_data.dims)}`.",
                _class=exceptions.BackendWarning,
            )
            return xr.DataArray(False)
        all_imasks = [initial_imask, *[model_data[i].notnull() for i in add_dims]]
        return functools.reduce(operator.and_, all_imasks)

    def raise_caught_errors(self):
        """If there are any parsing errors, pipe them to the ModelError bullet point list generator"""
        if not self._is_valid:
            exceptions.print_warnings_and_raise_errors(
                errors={f"({self.group_name}, {self.name})": self._errors},
                during="math string parsing (marker indicates where parsing stopped, not strictly the equation term that caused the failure)",
                bullet=self._ERR_BULLET,
            )
