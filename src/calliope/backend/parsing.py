# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Methods for math syntax parsing."""

from __future__ import annotations

import functools
import itertools
import logging
import operator
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Literal, TypeVar, overload

import pyparsing as pp
import xarray as xr

from calliope import exceptions
from calliope.backend import expression_parser, helper_functions, where_parser
from calliope.schemas import math_schema

if TYPE_CHECKING:
    from calliope.backend import backend_model


TRUE_ARRAY = xr.DataArray(True)

MATH_DEFS = (
    math_schema.Constraint
    | math_schema.Variable
    | math_schema.GlobalExpression
    | math_schema.Objective
    | math_schema.PiecewiseConstraint
)
T = TypeVar("T", bound=MATH_DEFS)

LOGGER = logging.getLogger(__name__)


class ParsedBackendEquation:
    """Backend equation parser."""

    def __init__(
        self,
        equation_name: str,
        sets: list[str],
        expression: pp.ParseResults,
        where_list: list[pp.ParseResults],
        sub_expressions: dict[str, pp.ParseResults] | None = None,
        slices: dict[str, pp.ParseResults] | None = None,
    ) -> None:
        """For parsing equation expressions and corresponding "where" strings.

        Args:
            equation_name (str): Name of equation.
            sets (list[str]):
                Model data sets with which to create the initial multi-dimensional masking array
                of the evaluated "where" string.
            expression (pp.ParseResults):
                Parsed arithmetic/equation expression.
            where_list (list[pp.ParseResults]):
                List of parsed where strings.
            sub_expressions (dict[str, pp.ParseResults] | None, optional):
                Dictionary of parsed sub-expressions with which to replace sub-expression references
                on evaluation of the parsed expression. Defaults to None.
            slices (dict[str, pp.ParseResults] | None, optional):
                Dictionary of parsed array slices with which to replace slice references
                on evaluation of the parsed expression / sub-expression. Defaults to None.
        """
        self.name = equation_name
        self.where = where_list
        self.expression = expression
        self.sub_expressions = (
            sub_expressions if sub_expressions is not None else dict()
        )
        self.slices = slices if slices is not None else dict()
        self.sets = sets

    def find_sub_expressions(self) -> set[str]:
        """Identify all the references to sub_expressions in the parsed expression.

        Returns:
            set[str]: Unique sub-expression references.
        """
        valid_eval_classes: tuple = (
            expression_parser.EvalOperatorOperand,
            expression_parser.EvalFunction,
        )
        to_find = expression_parser.EvalSubExpressions
        elements: list
        if isinstance(self.expression[0], to_find):
            elements = [self.expression[0]]
        else:
            elements = [self.expression[0].values]

        return self._find_items_in_expression(elements, to_find, valid_eval_classes)

    def find_slices(self) -> set[str]:
        """Finds all references to array slices in the expression and sub-expressions.

        Returns:
            set[str]: Unique slice references.
        """
        valid_eval_classes = tuple(
            [
                expression_parser.EvalOperatorOperand,
                expression_parser.EvalFunction,
                expression_parser.EvalSlicedComponent,
            ]
        )
        to_find = expression_parser.EvalIndexSlice
        elements: list = [
            self.expression[0].values,
            *list(self.sub_expressions.values()),
        ]

        return self._find_items_in_expression(elements, to_find, valid_eval_classes)

    @staticmethod
    def _find_items_in_expression(
        parser_elements: list | pp.ParseResults,
        to_find: type[expression_parser.EvalString],
        valid_eval_classes: tuple[type[expression_parser.EvalString], ...],
    ) -> set[str]:
        """Recursively find sub-expressions / index items defined in an equation expression.

        Args:
            parser_elements (list | pp.ParseResults): list of parser elements to check.
            to_find (type[expression_parser.EvalString]): type of equation element to search for.
            valid_eval_classes (tuple[type[expression_parser.EvalString], ...]): Other expression
                elements that can be recursively searched

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

            elif isinstance(parser_element, pp.ParseResults | list):
                items.extend(recursive_func(parser_elements=parser_element))

            elif isinstance(parser_element, valid_eval_classes):
                items.extend(recursive_func(parser_elements=parser_element.values))
        return set(items)

    def add_expression_group_combination(
        self,
        expression_group_name: Literal["sub_expressions", "slices"],
        expression_group_combination: Iterable[ParsedBackendEquation],
    ) -> ParsedBackendEquation:
        """Add parsed sub-expressions/index slices to a copy of self with updated names and were lists.

        Args:
            expression_group_name (Literal[sub_expressions, slices]):
                Which of `sub-expressions`/`index slices` is being added.
            expression_group_combination (Iterable[ParsedBackendEquation]):
                All items of expression_group_name to be added.

        Returns:
            ParsedBackendEquation: Copy of self with added sub-expressions/index slice dictionary and updated name
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
                "slices": self.slices,
                **expression_group_dict,  # type: ignore
            },
        )

    # Expecting array if not requesting latex string
    @overload
    def evaluate_where(
        self,
        backend_interface: backend_model.BackendModelGenerator,
        *,
        return_type: Literal["array"] = "array",
        references: set | None = None,
        initial_where: xr.DataArray = TRUE_ARRAY,
    ) -> xr.DataArray: ...

    # Expecting string if requesting latex string.
    @overload
    def evaluate_where(
        self,
        backend_interface: backend_model.BackendModelGenerator,
        *,
        return_type: Literal["math_string"],
        references: set | None = None,
    ) -> str: ...

    def evaluate_where(
        self,
        backend_interface: backend_model.BackendModelGenerator,
        *,
        return_type: str = "array",
        references: set | None = None,
        initial_where: xr.DataArray = TRUE_ARRAY,
    ) -> xr.DataArray | str:
        """Evaluate parsed backend object dictionary `where` string.

        Args:
            backend_interface (backend_model.BackendModelGenerator):
                Interface to an optimisation backend.
            return_type (str, optional): If "array", return xarray.DataArray.
                If "math_string", return LaTex math string.
                Defaults to "array".
            references (set | None, optional): List of references to use in evaluation.
                Defaults to None.
            initial_where (xr.DataArray, optional): If given, the where array resulting
                from evaluation will be further where'd by this array.
                Defaults to xr.DataArray(True) (i.e., no effect).

        Returns:
            xr.DataArray | str:
                If return_type == `array`: Boolean array defining on which index items a parsed component should be built.
                If return_type == `math_string`: Valid LaTeX math string defining the "where" conditions using logic notation.
        """
        evaluated_wheres = [
            where[0].eval(
                return_type,
                equation_name=self.name,
                helper_functions=helper_functions._registry["where"],
                input_data=backend_interface.inputs,
                backend_interface=backend_interface,
                math=backend_interface.math,
                build_config=backend_interface.config,
                references=references if references is not None else set(),
                apply_where=True,
            )
            for where in self.where
        ]
        if return_type == "math_string":
            return r"\land{}".join(f"({i})" for i in evaluated_wheres if i != "true")
        else:
            where = xr.DataArray(
                functools.reduce(operator.and_, [initial_where, *evaluated_wheres])
            )
            if not where.any():
                self.log_not_added("'where' does not apply anywhere.")
            return where

    def drop_dims_not_in_foreach(self, where: xr.DataArray) -> xr.DataArray:
        """Remove all dimensions not included in "foreach" from the input array.

        Args:
            where (xr.DataArray): Array with potentially unwanted dimensions

        Returns:
            xr.DataArray:
                Array with same dimensions as the user-defined foreach sets.
                Dimensions are ordered to match the order given by the sets.
        """
        unwanted_dims = set(where.dims).difference(self.sets)
        return (where.sum(unwanted_dims) > 0).astype(bool).transpose(*self.sets)

    # Expecting anything (most likely an array) if not requesting latex string.
    @overload
    def evaluate_expression(
        self,
        backend_interface: backend_model.BackendModelGenerator,
        *,
        return_type: Literal["array"] = "array",
        references: set | None = None,
        where: xr.DataArray = TRUE_ARRAY,
    ) -> xr.DataArray: ...

    # Expecting string if requesting latex string.
    @overload
    def evaluate_expression(
        self,
        backend_interface: backend_model.BackendModelGenerator,
        *,
        return_type: Literal["math_string"],
        references: set | None = None,
    ) -> str: ...

    def evaluate_expression(
        self,
        backend_interface: backend_model.BackendModelGenerator,
        *,
        return_type: Literal["array", "math_string"] = "array",
        references: set | None = None,
        where: xr.DataArray = TRUE_ARRAY,
    ) -> xr.DataArray | str:
        """Evaluate a math string to produce an array backend objects or a LaTex math string.

        Args:
            backend_interface (calliope.backend.backend_model.BackendModel):
                Interface to a optimisation backend.

        Keyword Args:
            return_type (str, optional):
                If "array", return xarray.DataArray. If "math_string", return LaTex math string.
                Defaults to "array".
            references (set | None, optional):
                If given, any references in the math string to other model components
                will be logged here. Defaults to None.
            where (xr.DataArray, optional):
                If given, should be a boolean array with which to mask any produced arrays.
                Defaults to xr.DataArray(True).

        Returns:
            xr.DataArray | str:
                If return_type == `array`: array of backend expression objects.
                If return_type == `math_string`: Valid LaTeX math string defining the
                "where" conditions using logic notation.
        """
        evaluated = self.expression[0].eval(
            return_type,
            equation_name=self.name,
            slice_dict=self.slices,
            sub_expression_dict=self.sub_expressions,
            backend_interface=backend_interface,
            math=backend_interface.math,
            input_data=backend_interface.inputs,
            where_array=where,
            references=references if references is not None else set(),
            helper_functions=helper_functions._registry["expression"],
        )
        if return_type == "array":
            self.raise_error_on_where_expr_mismatch(evaluated, where)
        return evaluated

    def raise_error_on_where_expr_mismatch(
        self, expression: xr.DataArray, where: xr.DataArray
    ) -> None:
        """Checks if an evaluated expression is consistent with the `where` array.

        Args:
            expression (xr.DataArray): array of linear expressions or one side of a constraint equation.
            where (xr.DataArray): where array; there should be a valid expression value for all True elements.

        Raises:
            BackendError:
                Raised if there is a dimension in the expression that is not in the where.
            BackendError:
                Raised if the expression has any NaN where the where applies.
        """
        broadcast_dims_where = set(expression.dims).difference(set(where.dims))
        if broadcast_dims_where:
            raise exceptions.BackendError(
                f"{self.name} | The linear expression array is indexed over dimensions not present in `foreach`: {broadcast_dims_where}"
            )
        # Check whether expression has NaN values in elements where the expression should be valid.
        incomplete_constraints = expression.isnull() & where
        if incomplete_constraints.any():
            raise exceptions.BackendError(
                f"{self.name} | Missing a linear expression for some coordinates selected by 'where'. Adapting 'where' might help."
            )

    def log_not_added(
        self,
        message: str,
        level: Literal["info", "warning", "debug", "error", "critical"] = "debug",
    ):
        """Log to module-level logger with some prettification of the message.

        Args:
            message (str): Message to log.
            level (Literal["info", "warning", "debug", "error", "critical"], optional):
                Log level. Defaults to "debug".
        """
        getattr(LOGGER, level)(
            f"Math parsing | {self.name} | Component not added; {message}"
        )


class ParsedBackendComponent(ParsedBackendEquation):
    """Backend component parser."""

    _ERR_BULLET: str = " * "
    _ERR_STRING_ORDER: list[str] = ["expression_group", "id", "expr_or_where"]
    PARSERS: dict[str, Callable] = {
        "constraints": expression_parser.generate_equation_parser,
        "global_expressions": expression_parser.generate_arithmetic_parser,
        "objectives": expression_parser.generate_arithmetic_parser,
        "variables": lambda x: None,
        "piecewise_constraints": expression_parser.generate_arithmetic_parser,
    }

    def __init__(
        self,
        group: Literal[
            "variables",
            "global_expressions",
            "constraints",
            "piecewise_constraints",
            "objectives",
        ],
        name: str,
        unparsed_data: T,
    ) -> None:
        """Parse an optimisation problem configuration.

        Defined in a dictionary of strings loaded from YAML into a series of Python
        objects that can be passed onto a solver interface like Pyomo or Gurobipy.

        Args:
            group (Literal["variables", "global_expressions", "constraints", "objectives"]):
                Optimisation problem component group to which the unparsed data belongs.
            name (str): Name of the optimisation problem component
            unparsed_data (T): Unparsed math formulation. Expected structure depends on
                the group to which the optimisation problem component belongs.
        """
        self.name = f"{group}:{name}"
        self._unparsed = unparsed_data

        self.where: list[pp.ParseResults] = []
        self.equations: list[ParsedBackendEquation] = []
        self.equation_expression_parser: Callable = self.PARSERS[group]

        # capture errors to dump after processing,
        # to make it easier for a user to fix the constraint YAML.
        self._errors: list = []
        self._tracker = self._init_tracker()

        # Initialise switches
        self._is_valid: bool = True

        # Add objects that are used by shared functions
        self.sets: list[str] = unparsed_data.foreach

    def get_parsing_position(self):
        """Create "." separated list from tracked strings."""
        return ".".join(
            filter(None, [self._tracker[i] for i in self._ERR_STRING_ORDER])
        )

    def reset_tracker(self):
        """Re-initialise error string tracking."""
        self._tracker = self._init_tracker()

    def _init_tracker(self):
        """Initialise error string tracking as dictionary of `key: None`."""
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
        top_level_where = self.parse_where_string(self._unparsed.where)

        if errors == "raise":
            self.raise_caught_errors()

        if self._is_valid:
            self.where = [top_level_where]

    def parse_equations(
        self,
        valid_component_names: Iterable[str],
        errors: Literal["raise", "ignore"] = "raise",
    ) -> list[ParsedBackendEquation]:
        """Parse `expression` and `where` strings of math component dictionary.

        Args:
            valid_component_names (Iterable[str]):
                strings referring to valid backend objects to allow the parser to differentiate between them and generic strings.
            errors (Literal["raise", "ignore"], optional):
                Collected parsing errors can be raised directly or ignored.
                If errors exist and are ignored, the parsed component cannot be successfully evaluated. Defaults to "raise".

        Returns:
            list[ParsedBackendEquation]:
                List of parsed equations ready to be evaluated.
                The length of the list depends on the product of provided equations and sub-expression/slice references.
        """
        equations = self.generate_expression_list(
            expression_parser=self.equation_expression_parser(valid_component_names),
            expression_list=self._unparsed.equations,
            expression_group="equations",
            id_prefix=self.name,
        )

        sub_expression_dict = {
            c_name: self.generate_expression_list(
                expression_parser=expression_parser.generate_sub_expression_parser(
                    valid_component_names
                ),
                expression_list=c_list,
                expression_group="sub_expressions",
                id_prefix=c_name,
            )
            for c_name, c_list in self._unparsed.sub_expressions.root.items()
        }
        slice_dict = {
            idx_name: self.generate_expression_list(
                expression_parser=expression_parser.generate_slice_parser(
                    valid_component_names
                ),
                expression_list=idx_list,
                expression_group="slices",
                id_prefix=idx_name,
            )
            for idx_name, idx_list in self._unparsed.slices.root.items()
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
        equations_with_sub_expressions_and_slices: list[ParsedBackendEquation] = []
        for equation in equations_with_sub_expressions:
            equations_with_sub_expressions_and_slices.extend(
                self.extend_equation_list_with_expression_group(
                    equation, slice_dict, "slices"
                )
            )

        return equations_with_sub_expressions_and_slices

    def _parse_string(
        self, parser: pp.ParserElement, parse_string: str
    ) -> pp.ParseResults:
        """Parse equation string according to predefined parsing grammar.

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
        """Parse a "where" string of the form "CONDITION OPERATOR CONDITION".

        The operator can be "and"/"or"/"not and"/"not or".

        Args:
            where_string (str):
                string value from a math dictionary "where" key.
                Defaults to "True", to have no effect on the subsequent subsetting.

        Returns:
            pp.ParseResults: Parsed string. If any parsing errors are caught,
                they will be logged to `self._errors` to raise later.
        """
        parser = where_parser.generate_where_string_parser()
        self._tracker["expr_or_where"] = "where"
        return self._parse_string(parser, where_string)

    def generate_expression_list(
        self,
        expression_parser: pp.ParserElement,
        expression_list: math_schema.Equations,
        expression_group: Literal["equations", "sub_expressions", "slices"],
        id_prefix: str = "",
    ) -> list[ParsedBackendEquation]:
        """Align user-defined constraint equations/sub-expressions.

        Achieved by parsing expressions, specifying a default "where" string if not
        defined, and providing an ID to enable returning to the initial dictionary.

        Args:
            expression_parser (pp.ParserElement): parser to use.
            expression_list (list[UnparsedEquation]): list of constraint equations
                or sub-expressions with arithmetic expression string and optional
                where string.
            expression_group (Literal["equations", "sub_expressions", "slices"]):
                For error reporting, the constraint dict key corresponding to the parse_string.
            id_prefix (str, optional): Extends the ID from a number corresponding to the
                expression_list position `idx` to a tuple of the form (id_prefix, idx).
                Defaults to "".

        Returns:
            list[ParsedBackendEquation]: Aligned expression dictionaries with parsed
                expression strings.
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

            parsed_where = self.parse_where_string(expression_data.where)

            self._tracker["expr_or_where"] = "expression"
            parsed_expression = self._parse_string(
                expression_parser, expression_data.expression
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
        expression_group: Literal["sub_expressions", "slices"],
    ) -> list[ParsedBackendEquation]:
        """Extend equation expressions with sub-expression data.

        Finds all sub-expressions referenced in an equation expression and returns a
        product of the sub-expression data.

        Args:
            parsed_equation (ParsedBackendEquation): Equation data dictionary.
            parsed_items (dict[str, list[ParsedBackendEquation]]):
                Dictionary of expressions to replace within the equation data dictionary.
            expression_group (Literal["sub_expressions", "slices"]):
                Name of expression group that the parsed_items dict is referencing.

        Returns:
            list[ParsedBackendEquation]: Expanded list of parsed equations with the
                product of all references to items from the `expression_group`
                producing a new equation object. E.g., if the input equation object has
                a reference to an slice which itself has two expression options, two
                equation objects will be added to the return list.
        """
        if expression_group == "sub_expressions":
            equation_items = parsed_equation.find_sub_expressions()
        elif expression_group == "slices":
            equation_items = parsed_equation.find_slices()
        if not equation_items:
            return [parsed_equation]

        invalid_items = equation_items.difference(parsed_items.keys())
        if invalid_items:
            raise KeyError(
                f"{self.name}: Undefined {expression_group} found in equation: {invalid_items}"
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

    def combine_definition_matrix_and_foreach(
        self, input_data: xr.Dataset
    ) -> xr.DataArray:
        """Generate a multi-dimensional array where a constraint will be built.

        The multi-dimensional boolean array is based on the sets over which the
        constraint is to be built (`foreach`) and the model `exists` array.

        The `exists` array is a boolean array defining the structure of the model and
        is True for valid combinations of technologies consuming/producing specific
        carriers at specific nodes.

        It is indexed over ["nodes", "techs", "carriers"].

        Args:
            input_data (xr.Dataset): Calliope model dataset.

        Returns:
            xr.DataArray: boolean array indexed over ["nodes", "techs", "carriers"]
                + any additional dimensions provided by `foreach`.
        """
        # Start with (carriers, nodes, techs) and go from there
        exists = input_data.definition_matrix
        # Add other dimensions (costs, timesteps, etc.)
        add_dims = set(self.sets).difference(exists.dims)
        if add_dims.difference(input_data.dims):
            self.log_not_added(
                f"indexed over unidentified set names: `{add_dims.difference(input_data.dims)}`."
            )
            return xr.DataArray(False)
        exists_and_foreach = [exists, *[input_data[i].notnull() for i in add_dims]]
        return functools.reduce(operator.and_, exists_and_foreach)

    def generate_top_level_where_array(
        self,
        backend_interface: backend_model.BackendModel,
        *,
        align_to_foreach_sets: bool = True,
        break_early: bool = True,
        references: set | None = None,
    ) -> xr.DataArray:
        """Generate a multi-dimentional "where" array.

        The multi-dimensional array is created using model inputs and component sets
        defined in foreach. The component top-level "where" is then applied to the
        array.

        Args:
            backend_interface (backend_model.BackendModel): Interface to a optimisation backend.
            align_to_foreach_sets (bool, optional):
                By default, all foreach arrays have the dimensions ("nodes", "techs", "carriers")
                as well as any additional dimensions provided by the component's "foreach" key.
                If this argument is True, the dimensions not included in "foreach" are removed from the array.
                Defaults to True.
            break_early (bool, optional):
                If any intermediate array has no valid elements (i.e. all are False),
                the function will return that array rather than continuing - saving
                time and memory on large models. Defaults to True.
            references (set | None, optional): references to use during evaluation. Defaults to None.

        Returns:
            xr.DataArray: Boolean array defining on which index items a parsed component should be built.
        """
        input_data = backend_interface.inputs
        foreach_where = self.combine_definition_matrix_and_foreach(input_data)

        if not foreach_where.any():
            self.log_not_added("'foreach' does not apply anywhere.")

        if break_early and not foreach_where.any():
            return foreach_where

        self.parse_top_level_where()
        where = self.evaluate_where(
            backend_interface,
            initial_where=foreach_where,
            references=references if references is not None else set(),
        )
        if break_early and not where.any():
            return where

        if align_to_foreach_sets:
            where = self.drop_dims_not_in_foreach(where)
        return where

    def raise_caught_errors(self):
        """Pipe parsing errors to the ModelError bullet point list generator."""
        if not self._is_valid:
            exceptions.print_warnings_and_raise_errors(
                errors={f"{self.name}": self._errors},
                during=(
                    "math string parsing (marker indicates where parsing stopped, "
                    "but may not point to the root cause of the issue)"
                ),
                bullet=self._ERR_BULLET,
            )
