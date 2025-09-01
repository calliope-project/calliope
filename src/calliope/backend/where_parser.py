# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Parsing for 'where' statements."""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyparsing as pp
import xarray as xr
from typing_extensions import NotRequired, TypedDict

from calliope.backend import expression_parser
from calliope.schemas import config_schema, math_schema
from calliope.util import tools

if TYPE_CHECKING:
    from calliope.backend.backend_model import BackendModel


pp.ParserElement.enablePackrat()
BOOLEANTYPE = np.bool_ | np.typing.NDArray[np.bool_]


class EvalAttrs(TypedDict):
    """Fixed dict checker for `eval_attrs`."""

    equation_name: str
    backend_interface: BackendModel
    math: math_schema.CalliopeBuildMath
    input_data: xr.Dataset
    helper_functions: dict[str, Callable]
    apply_where: NotRequired[bool]
    references: NotRequired[set]
    build_config: config_schema.Build


class EvalWhere(expression_parser.EvalToArrayStr):
    """Update type reference for `eval_attrs` to match `where` evaluation kwargs."""

    eval_attrs: EvalAttrs = {}


class EvalNot(EvalWhere, expression_parser.EvalSignOp):
    """Parse action to process successfully parsed expressions with a leading `not`."""

    def as_math_string(self) -> str:  # noqa: D102, override
        evaluated = self.value.eval("math_string", **self.eval_attrs)
        return rf"\neg ({evaluated})"

    def as_array(self) -> xr.DataArray:  # noqa: D102, override
        evaluated = self.value.eval("array", **self.eval_attrs)
        return ~evaluated


class EvalAndOr(EvalWhere, expression_parser.EvalOperatorOperand):
    """Processing of successfully parsed expressions with and/or operators.

    E.g., "OPERAND OPERATOR OPERAND OPERATOR OPERAND ..."
    """

    LATEX_OPERATOR_LOOKUP: dict[str, str] = {
        "and": r"{val} \land {operand}",
        "or": r"{val} \lor {operand}",
    }
    SKIP_IF = ["and", "or"]

    def _skip_component_on_conditional(self, component: str, operator_: str) -> bool:
        return component == "true" and operator_ in self.SKIP_IF

    @staticmethod
    def _operate(
        val: xr.DataArray, evaluated_operand: xr.DataArray, operator_: str
    ) -> xr.DataArray:
        """Apply bitwise comparison between boolean xarray dataarrays."""
        match operator_:
            case "and":
                val = operator.and_(val, evaluated_operand)
            case "or":
                val = operator.or_(val, evaluated_operand)
        return val

    def _apply_where_array(self, evaluated: xr.DataArray) -> xr.DataArray:
        """Override func from parent class to effectively do nothing."""
        return evaluated

    def as_math_string(self) -> str:  # noqa: D102, override
        return super().as_math_string()

    def as_array(self) -> xr.DataArray:  # noqa: D102, override
        return super().as_array()


class ConfigOptionParser(EvalWhere):
    """Parsing of configuration options."""

    def __init__(self, instring: str, loc: int, tokens: pp.ParseResults) -> None:
        """Parse action to process successfully parsed configuration option names.

        Args:
            instring (str): String that was parsed (used in error message).
            loc (int):
                Location in parsed string where parsing error was logged.
                This is not used; we include it as pyparsing injects it alongside `instring` when setting the parse action.
            tokens (pp.ParseResults):
                Has two parsed elements: config group name (str) and config option (str).
        """
        self.config_option = tokens[0]
        self.instring = instring
        self.loc = loc

    def __repr__(self):
        """Programming / official string representation."""
        return f"CONFIG:{self.config_option}"

    def as_math_string(self) -> str:  # noqa: D102, override
        return rf"\text{{config.{self.config_option}}}"

    def as_array(self) -> xr.DataArray:  # noqa: D102, override
        config_val = tools.get_dot_attr(
            self.eval_attrs["build_config"], self.config_option
        )

        if not isinstance(config_val, int | float | str | bool | np.bool_):
            raise self.error_msg(
                f"where string | Configuration option resolves to invalid "
                f"type `{type(config_val).__name__}`, expected a number, string, or boolean."
            )
        else:
            return xr.DataArray(config_val)


class DataVarParser(EvalWhere):
    """Data variable processing."""

    def __init__(self, instring: str, loc: int, tokens: pp.ParseResults) -> None:
        """Parse action to process successfully parsed model data variable names.

        Args:
            instring (str): String that was parsed (used in error message).
            loc (int):
                Location in parsed string where parsing error was logged.
                This is not used; we include it as pyparsing injects it alongside `instring` when setting the parse action.
            tokens (pp.ParseResults):
                Has one parsed element: model data variable name (str).
        """
        self.data_var = tokens[0]
        self.instring = instring
        self.loc = loc

    def __repr__(self):
        """Programming / official string representation."""
        return f"DATA_VAR:{self.data_var}"

    def _preprocess(self) -> str:
        """Get data variable from the optimisation problem dataset.

        Raises:
            TypeError: Cannot work with math components of type `constraint` or `objective`.
            TypeError: Cannot check array contents (`apply_where=False`) of `variable` or `global_expression` math components.
        """
        self.eval_attrs["references"].add(self.data_var)
        try:
            return self.eval_attrs["input_data"][self.data_var].attrs["obj_type"]
        except KeyError:
            pass
        try:
            return (
                self.eval_attrs["backend_interface"]
                ._dataset[self.data_var]
                .attrs["obj_type"]
            )
        except KeyError:
            raise self.error_msg(
                f"where string | Data variable `{self.data_var}` not found in model dataset."
            )

    def _data_var_exists(
        self, source_dataset: xr.Dataset, resolve_contents: bool
    ) -> xr.DataArray:
        """Mask by setting all (NaN | INF/-INF) to False, otherwise True."""
        var = source_dataset.get(self.data_var, xr.DataArray(np.nan))
        if var.dtype.kind == "b":
            return var
        elif resolve_contents:
            return var.notnull() & (var != np.inf) & (var != -np.inf)
        else:
            return var.notnull()

    def _data_var_with_default(self) -> xr.DataArray:
        """Access data var and fill with default values.

        Return default value as an array if var does not exist.
        """
        var = self.eval_attrs["input_data"][self.data_var]
        if var.attrs["obj_type"] != "dimensions":
            default = self.eval_attrs["math"].find(self.data_var)[1]["default"]
            if pd.notnull(default):
                var = var.fillna(default)
        return var

    def as_math_string(self) -> str:  # noqa: D102, override
        self._preprocess()

        var = self.eval_attrs["backend_interface"]._dataset.get(
            self.data_var, xr.DataArray()
        )

        try:
            data_var_string = var.attrs["math_repr"]
        except (AttributeError, KeyError):
            data_var_string = rf"\text{{{self.data_var}}}"
        if self.eval_attrs.get("apply_where", True):
            data_var_string = rf"\exists ({data_var_string})"
        return data_var_string

    def as_array(self) -> xr.DataArray:  # noqa: D102, override
        data_var_type = self._preprocess()
        if data_var_type == "unknown":
            return xr.DataArray(np.False_)
        elif self.eval_attrs.get("apply_where", True):
            if data_var_type in ["parameters", "lookups", "dimensions"]:
                return self._data_var_exists(
                    self.eval_attrs["input_data"], resolve_contents=True
                )
            elif data_var_type in ["variables", "global_expressions"]:
                return self._data_var_exists(
                    self.eval_attrs["backend_interface"]._dataset,
                    resolve_contents=False,
                )
            else:
                raise self.error_msg(
                    f"where string | Cannot check values in {data_var_type.removesuffix('s')} arrays in math `where` strings. "
                    f"Received {data_var_type.removesuffix('s')}: `{self.data_var}`."
                )
        else:
            if data_var_type not in ["lookups", "parameters", "dimensions"]:
                raise self.error_msg(
                    f"where string | Can only check for existence of values in {data_var_type.removesuffix('s')} arrays in math `where` strings. "
                    "These arrays cannot be used for comparison with expected values. "
                    f"Received `{self.data_var}`."
                )

            return self._data_var_with_default()


class ComparisonParser(EvalWhere, expression_parser.EvalComparisonOp):
    """Parse action to process successfully parsed strings of the form x=y."""

    OP_TRANSLATOR = {
        "<=": r"\mathord{\leq}",
        ">=": r"\mathord{\geq}",
        "==": r"\mathord{==}",
        "<": r"\mathord{<}",
        ">": r"\mathord{>}",
    }

    def __repr__(self):
        """Return string representation of the parsed grammar."""
        return f"{self.lhs}{self.op}{self.rhs}"

    def as_math_string(self) -> str:  # noqa: D102, override
        self.eval_attrs["apply_where"] = False
        lhs, rhs = self._eval("math_string")
        if r"\text" not in rhs:
            rhs = rf"\text{{{rhs}}}"
        return lhs + self.OP_TRANSLATOR[self.op] + rhs

    def as_array(self) -> xr.DataArray:  # noqa: D102, override
        self.eval_attrs["apply_where"] = False
        lhs, rhs = self._eval("array")
        match self.op:
            case "<=":
                comparison = lhs <= rhs
            case ">=":
                comparison = lhs >= rhs
            case "<":
                comparison = lhs < rhs
            case ">":
                comparison = lhs > rhs
            case "==":
                comparison = lhs == rhs
        return xr.DataArray(comparison)


class SubsetParser(EvalWhere):
    """Dimension subset parsing."""

    def __init__(self, instring: str, loc: int, tokens: pp.ParseResults) -> None:
        """Parse action to process successfully parsed dimension subsetting.

        Args:
            instring (str): String that was parsed (used in error message).
            loc (int):
                Location in parsed string where parsing error was logged.
                This is not used; we include it as pyparsing injects it alongside `instring` when setting the parse action.
            tokens (pp.ParseResults):
                Has two parsed elements: model set name (str), set items (Any).
        """
        self.val, self.set_name = tokens
        self.instring = instring
        self.loc = loc

    def __repr__(self):
        """Return string representation of the parsed grammar."""
        return f"SUBSET:{self.set_name}{self.val}"

    def _eval(self) -> list[str | float]:
        """Evaluate each element of the subset list."""
        values = [val.eval("array", **self.eval_attrs) for val in self.val]
        return [val.item() if isinstance(val, xr.DataArray) else val for val in values]

    def as_math_string(self) -> str:  # noqa: D102, override
        subset = self._eval()
        iterator = self.eval_attrs["math"].dimensions[self.set_name].iterator
        subset_string = "[" + ",".join(str(i) for i in subset) + "]"
        return rf"\text{{{iterator}}} \in \text{{{subset_string}}}"

    def as_array(self) -> xr.DataArray:  # noqa: D102, override
        subset = self._eval()
        set_item_in_subset = self.eval_attrs["input_data"][self.set_name].isin(subset)
        return set_item_in_subset


class BoolOperandParser(EvalWhere):
    """Boolean operand parsing."""

    def __init__(self, instring: str, loc: int, tokens: pp.ParseResults) -> None:
        """Parse action to process successfully parsed boolean strings.

        Args:
            instring (str): String that was parsed (used in error message).
            loc (int):
                Location in parsed string where parsing error was logged.
                This is not used; we include it as pyparsing injects it alongside `instring` when setting the parse action.
            tokens (pp.ParseResults): Has one parsed element: boolean (str).
        """
        self.val = tokens[0].lower()
        self.instring = instring

    def __repr__(self):
        """Programming / official string representation."""
        return f"BOOL:{self.val}"

    def as_math_string(self):  # noqa: D102, override
        return self.val

    def as_array(self) -> xr.DataArray:  # noqa: D102, override
        if self.val == "true":
            bool_val = xr.DataArray(np.True_)
        elif self.val == "false":
            bool_val = xr.DataArray(np.False_)
        return bool_val


class GenericStringParser(expression_parser.EvalString):
    """Parsing of generic strings."""

    def __init__(self, instring: str, loc: int, tokens: pp.ParseResults) -> None:
        """Parse action to process successfully parsed generic strings.

        This is required since we call "eval()" on all elements of the where string,
        so even arbitrary strings (used in comparison operations) need to be evaluatable.

        Args:
            instring (str): String that was parsed (used in error message).
            loc (int):
                Location in parsed string where parsing error was logged.
                This is not used; we include it as pyparsing injects it alongside `instring` when setting the parse action.
            tokens (pp.ParseResults): Has one parsed element: string name (str).
        """
        self.val = tokens[0]
        self.instring = instring

    def __repr__(self) -> str:
        """Return string representation of the parsed grammar."""
        return f"STRING:{self.val}"

    def eval(self, *args, **eval_kwargs) -> str:
        """Evaluation just returns the string of values."""
        return str(self.val)


def data_var_parser(generic_identifier: pp.ParserElement) -> pp.ParserElement:
    """Process model data variables which can be any valid python identifier (string + "_").

    Args:
        generic_identifier (pp.ParserElement): parser for valid python variables
            without leading underscore and not called "inf". This parser has no parse action.

    Returns:
        pp.ParserElement: parser for model data variables which will access the data
            variable from the Calliope model dataset.
    """
    protected_strings = (
        pp.Keyword("and", caseless=True)
        | pp.Keyword("or", caseless=True)
        | pp.Keyword("not", caseless=True)
        | pp.Keyword("true", caseless=True)
        | pp.Keyword("false", caseless=True)
    )
    data_var = ~protected_strings + generic_identifier
    data_var.set_parse_action(DataVarParser)

    return data_var


def config_option_parser(generic_identifier: pp.ParserElement) -> pp.ParserElement:
    """Parsing grammar to process model configuration option key names of the form "x.y.z".

    Args:
        generic_identifier (pp.ParserElement):
            Parser for valid python variables without leading underscore and not called "inf".
            This parser has no parse action.

    Returns:
        pp.ParserElement:
            Parser for configuration options which will be accessed from the configuration
            dictionary attached to the attributes of the Calliope model dataset.
    """
    data_var = pp.Suppress("config.") + generic_identifier
    data_var.set_parse_action(ConfigOptionParser)

    return data_var


def bool_parser() -> pp.ParserElement:
    """Parsing grammar for True/False (any case), which will evaluate to np.bool_."""
    TRUE = pp.Keyword("True", caseless=True)
    FALSE = pp.Keyword("False", caseless=True)
    bool_operand = TRUE | FALSE
    bool_operand.set_parse_action(BoolOperandParser)

    return bool_operand


def evaluatable_string_parser(generic_identifier: pp.ParserElement) -> pp.ParserElement:
    """Parsing grammar to make generic strings used in comparison operations evaluatable."""
    evaluatable_identifier = generic_identifier.copy()
    evaluatable_identifier.set_parse_action(GenericStringParser)

    return evaluatable_identifier


def comparison_parser(
    evaluatable_identifier: pp.ParserElement,
    number: pp.ParserElement,
    helper_function: pp.ParserElement,
    bool_operand: pp.ParserElement,
    config_option: pp.ParserElement,
    data_var: pp.ParserElement,
) -> pp.ParserElement:
    """Parsing grammar to process comparisons of the form `variable_or_config=comparator`.

    Args:
        evaluatable_identifier (pp.ParserElement): parser for evaluatable generic strings
        number (pp.ParserElement): parser for numbers (integer, float, scientific notation, "inf"/".inf")
        helper_function (pp.ParserElement): parser for helper functions
        bool_operand (pp.ParserElement): parser for boolean strings
        config_option (pp.ParserElement): parser for attribute dictionary keys of the form "x.y.z"
        data_var (pp.ParserElement): parser for Calliope model dataset variable names

    Returns:
        pp.ParserElement:
            Parser which will return a bool/boolean array as a result of the comparison.
    """
    comparison_operators = pp.oneOf(["<", ">", "==", ">=", "<="])
    comparison_expression = (
        (config_option | data_var)
        + comparison_operators
        + (helper_function | bool_operand | number | evaluatable_identifier)
    )
    comparison_expression.set_parse_action(ComparisonParser)

    return comparison_expression


def subset_parser(
    generic_identifier: pp.ParserElement, *subset_items: pp.ParserElement
) -> pp.ParserElement:
    """Parsing grammar to process subsets.

    Args:
        generic_identifier (pp.ParserElement): generic identifier parser
        *subset_items (pp.ParserElement): parsers that can be included in the subset list; will be matched in the order provided.

    Returns:
        pp.ParserElement: subset parser.
    """
    subset = pp.Group(pp.delimited_list(pp.MatchFirst(subset_items)))
    subset_expression = (
        pp.Suppress("[")
        + subset
        + pp.Suppress("]")
        + pp.Suppress("in")
        + generic_identifier
    )
    subset_expression.set_parse_action(SubsetParser)

    return subset_expression


def where_parser(
    bool_operand: pp.ParserElement,
    helper_function: pp.ParserElement,
    data_var: pp.ParserElement,
    comparison_parser: pp.ParserElement,
    subset: pp.ParserElement,
) -> pp.ParserElement:
    """Parser for strings which use AND/OR/NOT operators to combine other parser elements.

    Args:
        bool_operand (pp.ParserElement): parser for boolean operands.
        helper_function (pp.ParserElement): parser for helper functions.
        data_var (pp.ParserElement): parser for dataset variable names.
        comparison_parser (pp.ParserElement): parser for comparisons.
        subset (pp.ParserElement): parser for subsets.

    Returns:
        pp.ParserElement: where parser.
    """
    notop = pp.Keyword("not", caseless=True)
    andorop = pp.Keyword("and", caseless=True) | pp.Keyword("or", caseless=True)

    where_rules = pp.infixNotation(
        helper_function | comparison_parser | subset | data_var | bool_operand,
        [
            (notop, 1, pp.opAssoc.RIGHT, EvalNot),
            (andorop, 2, pp.opAssoc.LEFT, EvalAndOr),
        ],
    )

    return where_rules


def generate_where_string_parser() -> pp.ParserElement:
    """Creates and executes the where parser.

    Args:
        parse_string (str): Constraint subsetting "where" string.

    Returns:
        pp.ParseResults: evaluatable to a bool/boolean array.
    """
    number, generic_identifier = expression_parser.setup_base_parser_elements()
    data_var = data_var_parser(generic_identifier)
    config_option = config_option_parser(generic_identifier)
    bool_operand = bool_parser()
    evaluatable_string = evaluatable_string_parser(generic_identifier)
    id_list = (
        pp.Suppress("[") + pp.delimited_list(evaluatable_string) + pp.Suppress("]")
    )
    helper_function = expression_parser.helper_function_parser(
        evaluatable_string, number, id_list, generic_identifier=generic_identifier
    )
    comparison = comparison_parser(
        evaluatable_string,
        number,
        helper_function,
        bool_operand,
        config_option,
        data_var,
    )
    subset = subset_parser(
        generic_identifier, config_option, number, evaluatable_string
    )
    return where_parser(bool_operand, helper_function, data_var, comparison, subset)
