# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).

from __future__ import annotations

import operator
from abc import abstractmethod
from typing import Any, Union

import numpy as np
import pyparsing as pp
import xarray as xr

from calliope.backend import backend_model, expression_parser
from calliope.exceptions import BackendError

pp.ParserElement.enablePackrat()

BOOLEANTYPE = Union[np.bool_, np.typing.NDArray[np.bool_]]


class EvalWhereString(expression_parser.EvalString):
    @abstractmethod
    def eval(self, *args, **eval_kwargs) -> Union[xr.DataArray, str]:
        """Evaluate math string expression.

        Args:
            as_latex (bool, optional): If True, return a valid LaTex math string. Defaults to False.
        Keyword Args:
            equation_name (str): Name of math component in which expression is defined.
            backend_interface (backend_model.BackendModel): Interface to optimisation backend.
            input_data (xr.Dataset): Input parameter arrays.
            apply_where (bool, optional): If True, transform a parameter into a `where` array, otherwise leave parameter array as-is.
            helper_functions (dict[str, type[ParsingHelperFunction]]): Dictionary of allowed helper functions.

        Returns:
            Union[str, xr.DataArray, Callable, list[str], tuple[xr.DataArray, str, xr.DataArray]]:
                If expression is a helper function, returns Callable.
                If expression is a list of strings, returns a list of strings
                If `as_latex` is True, returns a valid LaTex math string.
                If a string without model reference, returns string.
                If expression is a constraint and not `as_latex`, returns LHS, operator, RHS as a tuple.
                Otherwise, returns xarray DataArray.
        """


class EvalNot(EvalWhereString, expression_parser.EvalSignOp):
    "Parse action to process successfully parsed expressions with a leading `not`"

    def as_latex(self, val: str) -> str:
        """Add sign to stringified data for use in a LaTex math formula"""
        return rf"\neg ({val})"

    def eval(self, as_latex: bool = False, **eval_kwargs) -> Union[xr.DataArray, str]:
        evaluated = self.value.eval(as_latex=as_latex, **eval_kwargs)
        if as_latex:
            return self.as_latex(evaluated)
        else:
            return ~evaluated


class EvalAndOr(EvalWhereString, expression_parser.EvalOperatorOperand):
    """
    Parse action to process successfully parsed expressions with operands separated
    by an and/or operator (OPERAND OPERATOR OPERAND OPERATOR OPERAND ...)
    """

    LATEX_OPERATOR_LOOKUP: dict[str, str] = {
        "and": r"{val} \land {operand}",
        "or": r"{val} \lor {operand}",
    }

    def bool_operate(
        self, val: BOOLEANTYPE, evaluated_operand: BOOLEANTYPE, operator_: str
    ) -> BOOLEANTYPE:
        if operator_ == "and":
            val = operator.and_(val, evaluated_operand)
        elif operator_ == "or":
            val = operator.or_(val, evaluated_operand)
        return val

    def _as_latex(
        self, val: str, operand: str, operator_: str, val_type: Any, operand_type: Any
    ) -> str:
        if val == "true":
            val = operand
        elif operand != "true":
            val = self.as_latex(val, operand, operator_, val_type, operand_type)
        return val

    def eval(self, as_latex: bool = False, **eval_kwargs) -> Union[str, xr.DataArray]:
        val = self.value[0].eval(as_latex=as_latex, **eval_kwargs)
        for operator_, operand in self.operatorOperands(self.value[1:]):
            evaluated_operand = operand.eval(as_latex=as_latex, **eval_kwargs)
            if as_latex:
                val = self._as_latex(
                    val,
                    evaluated_operand,
                    operator_,
                    type(self.value[0]),
                    type(operand),
                )
            else:
                val = self.bool_operate(val, evaluated_operand, operator_)
        return val


class ConfigOptionParser(EvalWhereString):
    def __init__(self, instring: str, loc: int, tokens: pp.ParseResults) -> None:
        """
        Parse action to process successfully parsed configuration option names.

        Args:
            instring (str): String that was parsed (used in error message).
            loc (int):
                Location in parsed string where parsing error was logged.
                This is not used, but comes with `instring` when setting the parse action.
            tokens (pp.ParseResults):
                Has two parsed elements: config group name (str) and config option (str).
        """
        self.config_option = tokens[0]
        self.instring = instring
        self.loc = loc

    def __repr__(self):
        "Return string representation of the parsed grammar"
        return f"CONFIG:{self.config_option}"

    def as_latex(self) -> str:
        """Add return input string for use in a LaTex math formula"""
        return rf"\text{{config.{self.config_option}}}"

    def eval(
        self, as_latex: bool = False, *, input_data: xr.Dataset, **eval_kwargs
    ) -> Union[str, xr.DataArray]:
        if as_latex:
            return self.as_latex()
        else:
            config_val = input_data.attrs["config"].build[self.config_option]

            if not isinstance(config_val, (int, float, str, bool, np.bool_)):
                raise BackendError(
                    f"(where, {self.instring}): Configuration option resolves to invalid "
                    f"type `{type(config_val).__name__}`, expected a number, string, or boolean."
                )
            else:
                return xr.DataArray(config_val)


class DataVarParser(EvalWhereString):
    def __init__(self, instring: str, loc: int, tokens: pp.ParseResults) -> None:
        """
        Parse action to process successfully parsed model data variable names.

        Args:
            instring (str): String that was parsed (used in error message).
            loc (int):
                Location in parsed string where parsing error was logged.
                This is not used, but comes with `instring` when setting the parse action.
            tokens (pp.ParseResults):
                Has one parsed element: model data variable name (str).
        """
        self.data_var = tokens[0]
        self.instring = instring
        self.loc = loc

    def __repr__(self):
        "Return string representation of the parsed grammar"
        return f"DATA_VAR:{self.data_var}"

    def as_latex(
        self, data: xr.Dataset, data_var_type: str, apply_where: bool = True
    ) -> str:
        """stringify conditional for use in a LaTex math formula"""
        # TODO: add dims from a YAML schema of params that includes default dims
        if data_var_type == "parameters":
            data_var_string = rf"\textit{{{self.data_var}}}"
        else:
            data_var_string = rf"\textbf{{{self.data_var}}}"

        var = data.get(self.data_var, None)
        if var is not None and var.shape:
            data_var_string += (
                rf"_\text{{{','.join(str(i).removesuffix('s') for i in var.dims)}}}"
            )
        if apply_where:
            data_var_string = rf"\exists ({data_var_string})"
        return data_var_string

    def _data_var_exists(
        self, input_data: xr.Dataset, data_var_type: str
    ) -> xr.DataArray:
        "mask by setting all (NaN | INF/-INF) to False, otherwise True"
        var = input_data.get(self.data_var, xr.DataArray(np.nan))
        if data_var_type == "parameters":
            return var.notnull() & (var != np.inf) & (var != -np.inf)
        else:
            return var.notnull()

    def _data_var_with_default(self, input_data: xr.Dataset) -> xr.DataArray:
        "Access data var and fill with default values. Return default value as an array if var does not exist"
        default = input_data.attrs["defaults"].get(self.data_var)
        return input_data.get(self.data_var, xr.DataArray(default)).fillna(default)

    def eval(
        self,
        as_latex: bool = False,
        *,
        input_data: xr.Dataset,
        backend_interface: backend_model.BackendModel,
        apply_where: bool = True,
        **eval_kwargs,
    ) -> Union[str, xr.DataArray]:
        if self.data_var in backend_interface._dataset.data_vars.keys():
            data_var_type = backend_interface._dataset[self.data_var].attrs["obj_type"]
        else:
            data_var_type = "parameters"

        if data_var_type not in ["parameters", "global_expressions", "variables"]:
            raise TypeError(
                f"Cannot check values in {data_var_type.removesuffix('s')} arrays in math `where` strings. "
                f"Received {data_var_type.removesuffix('s')}: `{self.data_var}`."
            )
        if data_var_type != "parameters" and not apply_where:
            raise TypeError(
                f"Can only check for existence of values in {data_var_type.removesuffix('s')} arrays in math `where` strings. "
                "These arrays cannot be used for comparison with expected values. "
                f"Received `{self.data_var}`."
            )

        if data_var_type == "parameters":
            source_array = input_data
        else:
            source_array = backend_interface._dataset

        if as_latex:
            return self.as_latex(source_array, data_var_type, apply_where)

        if data_var_type == "parameters" and self.data_var not in input_data:
            return xr.DataArray(np.False_)

        if apply_where:
            return self._data_var_exists(source_array, data_var_type)
        else:
            return self._data_var_with_default(source_array)


class ComparisonParser(EvalWhereString, expression_parser.EvalComparisonOp):
    "Parse action to process successfully parsed strings of the form x=y"
    OP_TRANSLATOR = {
        "<=": r"\mathord{\leq}",
        ">=": r"\mathord{\geq}",
        "=": r"\mathord{=}",
        "<": r"\mathord{<}",
        ">": r"\mathord{>}",
    }

    def __repr__(self):
        "Return string representation of the parsed grammar"
        return f"{self.lhs}{self.op}{self.rhs}"

    def compare(self, lhs: xr.DataArray, rhs: xr.DataArray, **kwargs) -> xr.DataArray:
        "Compare RHS and LHS."
        if self.op == "<=":
            comparison = lhs <= rhs
        elif self.op == ">=":
            comparison = lhs >= rhs
        if self.op == "<":
            comparison = lhs < rhs
        elif self.op == ">":
            comparison = lhs > rhs
        elif self.op == "=":
            comparison = lhs == rhs
        return xr.DataArray(comparison)

    def eval(self, as_latex: bool = False, **eval_kwargs) -> Union[str, xr.DataArray]:
        eval_kwargs["apply_where"] = False
        lhs = self.lhs.eval(as_latex=as_latex, **eval_kwargs)
        rhs = self.rhs.eval(as_latex=as_latex, **eval_kwargs)

        if as_latex:
            if r"\text" not in rhs:
                rhs = rf"\text{{{rhs}}}"
            return self.as_latex(lhs, rhs)
        else:
            return self.compare(lhs, rhs)


class SubsetParser(expression_parser.EvalString):
    def __init__(self, instring: str, loc: int, tokens: pp.ParseResults) -> None:
        """
        Parse action to process successfully parsed dimension subsetting.

        Args:
            instring (str): String that was parsed (used in error message).
            loc (int):
                Location in parsed string where parsing error was logged.
                This is not used, but comes with `instring` when setting the parse action.
            tokens (pp.ParseResults):
                Has two parsed elements: model set name (str), set items (Any).
        """
        self.subset, self.set_name = tokens
        self.instring = instring
        self.loc = loc

    def __repr__(self):
        "Return string representation of the parsed grammar"
        return f"SUBSET:{self.set_name}{self.subset}"

    def as_latex(self, subset: list) -> str:
        """stringify subset for use in a LaTex math formula"""
        set_singular = self.set_name.removesuffix("s")
        subset_string = "[" + ",".join(str(i) for i in subset) + "]"
        return rf"\text{{{set_singular}}} \in \text{{{subset_string}}}"

    def eval(
        self, as_latex: bool = False, *, input_data: xr.Dataset, **eval_kwargs
    ) -> Union[str, xr.DataArray]:
        subset = [i.eval(as_latex=as_latex, **eval_kwargs) for i in self.subset]
        if as_latex:
            set_item_in_subset = self.as_latex(subset)
        else:
            set_item_in_subset = input_data[self.set_name].isin(subset)
        return set_item_in_subset


class BoolOperandParser(EvalWhereString):
    def __init__(self, tokens: pp.ParseResults) -> None:
        """
        Parse action to process successfully parsed boolean strings.

        Args:
            tokens (pp.ParseResults): Has one parsed element: boolean (str).
        """
        self.val = tokens[0].lower()

    def __repr__(self):
        "Return string representation of the parsed grammar"
        return f"BOOL:{self.val}"

    def as_latex(self):
        "Return boolean as a string in the domain {true, false}"
        return self.val

    def eval(self, as_latex: bool = False, **eval_kwargs) -> Union[str, xr.DataArray]:
        if as_latex:
            bool_val = self.as_latex()
        else:
            if self.val == "true":
                bool_val = xr.DataArray(np.True_)
            elif self.val == "false":
                bool_val = xr.DataArray(np.False_)
        return bool_val


def data_var_parser(generic_identifier: pp.ParserElement) -> pp.ParserElement:
    """
    Parsing grammar to process model data variables which can be any valid python
    identifier (string + "_")

    Args:
        generic_identifier (pp.ParserElement):
            Parser for valid python variables without leading underscore and not called "inf".
            This parser has no parse action.

    Returns:
        pp.ParserElement:
            Parser for model data variables which will access the data variable from the
            Calliope model dataset.
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
    """
    Parsing grammar to process model configuration option key names of the form "x.y.z".

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
    "Parsing grammar for True/False (any case), which will evaluate to np.bool_"
    TRUE = pp.Keyword("True", caseless=True)
    FALSE = pp.Keyword("False", caseless=True)
    bool_operand = TRUE | FALSE
    bool_operand.set_parse_action(BoolOperandParser)

    return bool_operand


def evaluatable_string_parser(generic_identifier: pp.ParserElement) -> pp.ParserElement:
    "Parsing grammar to make generic strings used in comparison operations evaluatable"
    evaluatable_identifier = generic_identifier.copy()
    evaluatable_identifier.set_parse_action(expression_parser.GenericStringParser)

    return evaluatable_identifier


def comparison_parser(
    evaluatable_identifier: pp.ParserElement,
    number: pp.ParserElement,
    helper_function: pp.ParserElement,
    bool_operand: pp.ParserElement,
    config_option: pp.ParserElement,
    data_var: pp.ParserElement,
) -> pp.ParserElement:
    """Parsing grammar to process comparisons of the form "variable_or_config=comparator"

    Args:
        evaluatable_identifier (pp.ParserElement): parser for evaluatable generic strings
        number (pp.ParserElement):
            Parser for numbers (integer, float, scientific notation, "inf"/".inf")
        bool_operand (pp.ParserElement): Parser for boolean strings
        config_option (pp.ParserElement):
            Parser for attribute dictionary keys of the form "x.y.z"
        data_var (pp.ParserElement): Parser for Calliope model dataset variable names.

    Returns:
        pp.ParserElement:
            Parser which will return a bool/boolean array as a result of the comparison.
    """
    comparison_operators = pp.oneOf(["<", ">", "=", ">=", "<="])
    comparison_expression = (
        (config_option | data_var)
        + comparison_operators
        + (helper_function | bool_operand | number | evaluatable_identifier)
    )
    comparison_expression.set_parse_action(ComparisonParser)

    return comparison_expression


def subset_parser(
    generic_identifier: pp.ParserElement,
    evaluatable_identifier: pp.ParserElement,
    number: pp.ParserElement,
) -> pp.ParserElement:
    """Parsing grammar to process comparisons of the form "variable_or_config=comparator"

    Args:
        evaluatable_identifier (pp.ParserElement): parser for evaluatable generic strings
        number (pp.ParserElement):
            Parser for numbers (integer, float, scientific notation, "inf"/".inf")
        bool_operand (pp.ParserElement): Parser for boolean strings
        config_option (pp.ParserElement):
            Parser for attribute dictionary keys of the form "x.y.z"
        data_var (pp.ParserElement): Parser for Calliope model dataset variable names.

    Returns:
        pp.ParserElement:
            Parser which will return a bool/boolean array as a result of the comparison.
    """
    subset = pp.Group(pp.delimited_list(number | evaluatable_identifier))
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
    """
    Parsing grammar to combine bools/boolean arrays using (case agnostic) AND/OR operators
    and optional (case agnostic) NOT (to invert the bools).

    Args:
        helper_function (pp.ParserElement):
            Parsing grammar to process helper functions of the form `helper_function(*args, **eval_kwargs)`.
        data_var (pp.ParserElement): Parser for Calliope model dataset variable names.
        comparison_parser (pp.ParserElement): Parser for comparisons of the form "variable_or_config=comparator".

    Returns:
        pp.ParserElement:
            Parser for strings which use AND/OR/NOT operators to combine other parser elements.
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
    """
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
    subset = subset_parser(generic_identifier, evaluatable_string, number)
    return where_parser(bool_operand, helper_function, data_var, comparison, subset)
