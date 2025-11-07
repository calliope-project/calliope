# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Parsing for 'where' statements."""

from __future__ import annotations

import operator
from collections.abc import Iterable
from dataclasses import replace

import numpy as np
import pandas as pd
import pyparsing as pp
import xarray as xr

from calliope.backend import expression_parser
from calliope.util import tools

pp.ParserElement.enablePackrat()
BOOLEANTYPE = np.bool_ | np.typing.NDArray[np.bool_]


class EvalNot(expression_parser.EvalSignOp, expression_parser.EvalArrayOrMath):
    """Parse action to process successfully parsed expressions with a leading `not`."""

    def as_math_string(self) -> str:  # noqa: D102, override
        evaluated = self.value.eval("math_string", self.eval_attrs)
        return rf"\neg ({evaluated})"

    def as_array(self) -> xr.DataArray:  # noqa: D102, override
        evaluated = self.value.eval("array", self.eval_attrs)
        return ~evaluated


class EvalAndOr(
    expression_parser.EvalOperatorOperand, expression_parser.EvalArrayOrMath
):
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


class ConfigOptionParser(expression_parser.EvalArrayOrMath):
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
            self.eval_attrs.build_config, self.config_option
        )

        if not isinstance(config_val, int | float | str | bool | np.bool_):
            raise self.error_msg(
                f"where string | Configuration option resolves to invalid "
                f"type `{type(config_val).__name__}`, expected a number, string, or boolean."
            )
        else:
            return xr.DataArray(config_val)


class ResultArrayParser(expression_parser.EvalArrayOrMath):
    """Variable/Expression array processing."""

    def __init__(self, instring: str, loc: int, tokens: pp.ParseResults) -> None:
        """Parse action to process successfully parsed model variable/global expression names.

        Args:
            instring (str): String that was parsed (used in error message).
            loc (int):
                Location in parsed string where parsing error was logged.
                This is not used; we include it as pyparsing injects it alongside `instring` when setting the parse action.
            tokens (pp.ParseResults):
                Has one parsed element: model data variable name (str).
        """
        self.array_name = tokens[0]
        self.instring = instring
        self.loc = loc

    def __repr__(self):
        """Programming / official string representation."""
        return f"RESULT:{self.array_name}"

    def as_math_string(self) -> str:  # noqa: D102, override
        self.eval_attrs.references.add(self.array_name)

        var = self.eval_attrs.backend_data.get(self.array_name, xr.DataArray())
        data_var_string = rf"\exists ({var.attrs['math_repr']})"
        return data_var_string

    def as_array(self) -> xr.DataArray:  # noqa: D102, override
        self.eval_attrs.references.add(self.array_name)
        da = self.eval_attrs.backend_data[self.array_name]
        if self.eval_attrs.apply_where:
            da = da.notnull()
        return da


class InputArrayParser(expression_parser.EvalArrayOrMath):
    """Input array processing."""

    def __init__(self, instring: str, loc: int, tokens: pp.ParseResults) -> None:
        """Parse action to process successfully parsed model input array names.

        Args:
            instring (str): String that was parsed (used in error message).
            loc (int):
                Location in parsed string where parsing error was logged.
                This is not used; we include it as pyparsing injects it alongside `instring` when setting the parse action.
            tokens (pp.ParseResults):
                Has one parsed element: model data variable name (str).
        """
        self.array_name = tokens[0]
        self.instring = instring
        self.loc = loc

    def __repr__(self):
        """Programming / official string representation."""
        return f"INPUT:{self.array_name}"

    def as_math_string(self) -> str:  # noqa: D102, override
        self.eval_attrs.references.add(self.array_name)

        var = self.eval_attrs.backend_data.get(self.array_name, xr.DataArray())

        data_var_string = var.attrs["math_repr"]
        if self.eval_attrs.apply_where:
            data_var_string = rf"\exists ({data_var_string})"
        return data_var_string

    def as_array(self) -> xr.DataArray:  # noqa: D102, override
        self.eval_attrs.references.add(self.array_name)
        da = self.eval_attrs.input_data.get(self.array_name, xr.DataArray(False))
        if self.eval_attrs.apply_where and da.dtype.kind != "b":
            da = da.notnull() & (da != np.inf) & (da != -np.inf)
        elif da.isnull().any() and pd.notnull(
            default := self.eval_attrs.math.find(self.array_name).default
        ):
            da = da.fillna(default)
        return da


class DimensionArrayParser(expression_parser.EvalArrayOrMath):
    """Dimension array processing."""

    def __init__(self, instring: str, loc: int, tokens: pp.ParseResults) -> None:
        """Parse action to process successfully parsed model dimension names.

        Args:
            instring (str): String that was parsed (used in error message).
            loc (int):
                Location in parsed string where parsing error was logged.
                This is not used; we include it as pyparsing injects it alongside `instring` when setting the parse action.
            tokens (pp.ParseResults):
                Has one parsed element: model data variable name (str).
        """
        self.array_name = tokens[0]
        self.instring = instring
        self.loc = loc

    def __repr__(self):
        """Programming / official string representation."""
        return f"DIM:{self.array_name}"

    def as_math_string(self) -> str:  # noqa: D102, override
        return self.array_name

    def as_array(self) -> xr.DataArray:  # noqa: D102, override
        # We want the where string to evaluate successfully even if a dimension hasn't been defined.
        da = self.eval_attrs.input_data.get(self.array_name, xr.DataArray())
        return da


class ComparisonParser(
    expression_parser.EvalComparisonOp, expression_parser.EvalArrayOrMath
):
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
        self.eval_attrs = replace(self.eval_attrs, apply_where=False)
        lhs, rhs = self._eval("math_string")
        if r"\text" not in rhs:
            rhs = rf"\text{{{rhs}}}"
        return lhs + self.OP_TRANSLATOR[self.op] + rhs

    def as_array(self) -> xr.DataArray:  # noqa: D102, override
        self.eval_attrs = replace(self.eval_attrs, apply_where=False)
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


class SubsetParser(expression_parser.EvalArrayOrMath):
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
        values = [val.eval("array", self.eval_attrs) for val in self.val]
        return [val.item() if isinstance(val, xr.DataArray) else val for val in values]

    def as_math_string(self) -> str:  # noqa: D102, override
        subset = self._eval()
        dim = self.set_name.eval("math_string", self.eval_attrs)
        iterator = self.eval_attrs.math.dimensions[dim].iterator
        subset_string = "[" + ",".join(str(i) for i in subset) + "]"
        return rf"\text{{{iterator}}} \in \text{{{subset_string}}}"

    def as_array(self) -> xr.DataArray:  # noqa: D102, override
        subset = self._eval()
        set_item_in_subset = self.set_name.eval("array", self.eval_attrs).isin(subset)
        return set_item_in_subset


class BoolOperandParser(expression_parser.EvalArrayOrMath):
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

    def eval(self, *args, **kwargs) -> str:
        """Evaluation just returns the string of values."""
        return str(self.val)


def data_var_parser(
    names: Iterable, parse_action: type[expression_parser.EvalArrayOrMath]
) -> pp.ParserElement:
    """Process model data variables which can be any valid python identifier (string + "_").

    Args:
        names (Iterable): List of valid component names.
        parse_action (type[expression_parser.EvalArrayOrMath]): Parse action to evaluate the parsed string.

    Returns:
        pp.ParserElement: parser for model data variables which will access the data
            variable from the Calliope model dataset.
    """
    data_var = pp.one_of(names, as_keyword=True)
    data_var.set_parse_action(parse_action)

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


def evaluatable_string_parser(
    generic_identifier: pp.ParserElement, valid_components: Iterable
) -> pp.ParserElement:
    """Parsing grammar to make generic strings used in comparison operations evaluatable."""
    evaluatable_identifier = (
        ~pp.one_of(valid_components, as_keyword=True) + generic_identifier
    )
    evaluatable_identifier.set_parse_action(GenericStringParser)

    return evaluatable_identifier


def comparison_parser(
    lhs: list[pp.ParserElement], rhs: list[pp.ParserElement]
) -> pp.ParserElement:
    """Parsing grammar to process comparisons of the form `variable_or_config=comparator`.

    Args:
        lhs (list[pp.ParserElement]):
            Parsers that can be included on the left-hand side of the comparison; will be matched in the order provided.
        rhs (list[pp.ParserElement]):
            Parsers that can be included on the right-hand side of the comparison; will be matched in the order provided.

    Returns:
        pp.ParserElement:
            Parser which will return a bool/boolean array as a result of the comparison.
    """
    comparison_operators = pp.oneOf(["<", ">", "==", ">=", "<="])
    comparison_expression = (
        pp.MatchFirst(lhs) + comparison_operators + pp.MatchFirst(rhs)
    )
    comparison_expression.set_parse_action(ComparisonParser)

    return comparison_expression


def subset_parser(
    data_var: pp.ParserElement, *subset_items: pp.ParserElement
) -> pp.ParserElement:
    """Parsing grammar to process subsets.

    Args:
        data_var (pp.ParserElement): data variable parser
        *subset_items (pp.ParserElement): parsers that can be included in the subset list; will be matched in the order provided.

    Returns:
        pp.ParserElement: subset parser.
    """
    subset = pp.Group(pp.delimited_list(pp.MatchFirst(subset_items)))
    subset_expression = (
        pp.Suppress("[")
        + subset
        + pp.Suppress("]")
        + pp.Suppress(pp.White(" ", min=1))
        + pp.Suppress("in")
        + pp.Suppress(pp.White(" ", min=1))
        + data_var
    )
    subset_expression.set_parse_action(SubsetParser)

    return subset_expression


def where_parser(*args: pp.ParserElement) -> pp.ParserElement:
    """Parser for strings which use AND/OR/NOT operators to combine other parser elements.

    Args:
        *args (pp.ParserElement):
            parsers that can be included in the where string; will be matched in the order provided.

    Returns:
        pp.ParserElement: where parser.
    """
    notop = pp.Keyword("not", caseless=True)
    andorop = pp.Keyword("and", caseless=True) | pp.Keyword("or", caseless=True)

    where_rules = pp.infixNotation(
        pp.MatchFirst(args),
        [
            (notop, 1, pp.opAssoc.RIGHT, EvalNot),
            (andorop, 2, pp.opAssoc.LEFT, EvalAndOr),
        ],
    )

    return where_rules


def generate_where_string_parser(
    dimensions: Iterable,
    inputs: Iterable,
    results: Iterable,
    postprocessed: Iterable | None = None,
) -> pp.ParserElement:
    """Creates and executes the where parser.

    Args:
        dimensions (Iterable): List of valid dimension names.
        inputs (Iterable): List of valid input names.
        results (Iterable): List of valid variable/global expression names.
        postprocessed (Iterable | None): List of valid postprocessed expression names. Defaults to None.

    Returns:
        pp.ParseResults: evaluatable to a bool/boolean array.
    """
    postprocessed = postprocessed if postprocessed is not None else set()
    number, generic_identifier = expression_parser.setup_base_parser_elements()
    dimensions_parser = data_var_parser(dimensions, DimensionArrayParser)
    inputs_parser = data_var_parser(inputs, InputArrayParser)
    results_parser = data_var_parser(results | postprocessed, ResultArrayParser)
    config_option = config_option_parser(generic_identifier)
    bool_operand = bool_parser()
    unique_evaluatable_string = evaluatable_string_parser(
        generic_identifier, set().union(dimensions, inputs, results, postprocessed)
    )
    general_evaluatable_string = evaluatable_string_parser(generic_identifier, [])
    id_list = expression_parser.list_parser(
        number, unique_evaluatable_string, dimensions_parser
    )
    subset = subset_parser(
        dimensions_parser, config_option, number, general_evaluatable_string
    )

    arithmetic = pp.Forward()
    comparison_helper_function = expression_parser.helper_function_parser(
        unique_evaluatable_string,
        number,
        id_list,
        arithmetic,
        generic_identifier=generic_identifier,
    )
    arithmetic_elements = [
        comparison_helper_function,
        number,
        dimensions_parser,
        inputs_parser,
        config_option,
    ]
    if postprocessed:
        arithmetic_elements.insert(-2, results_parser)

    comparison_arithmetic = expression_parser.arithmetic_parser(
        *arithmetic_elements, arithmetic=arithmetic
    )
    comparison = comparison_parser(
        lhs=[comparison_arithmetic],
        rhs=[
            comparison_helper_function,
            bool_operand,
            number,
            general_evaluatable_string,
        ],
    )

    helper_function = expression_parser.helper_function_parser(
        unique_evaluatable_string,
        number,
        id_list,
        dimensions_parser,
        inputs_parser,
        results_parser,
        config_option,
        generic_identifier=generic_identifier,
    )
    return where_parser(
        bool_operand, comparison, helper_function, subset, inputs_parser, results_parser
    )
