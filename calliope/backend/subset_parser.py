from __future__ import annotations

from typing import Union
import operator

import pyparsing as pp
import numpy as np
import xarray as xr
import pandas as pd

from calliope.backend import equation_parser
from calliope.exceptions import BackendError

pp.ParserElement.enablePackrat()

BOOLEANTYPE = Union[np.bool_, np.typing.NDArray[np.bool_]]


class EvalNot(equation_parser.EvalSignOp):
    "Parse action to process successfully parsed expressions with a leading `not`"

    def eval(self, **kwargs) -> BOOLEANTYPE:
        "Return inverted bool / boolean array"
        return ~self.value.eval(**kwargs)


class EvalAndOr(equation_parser.EvalOperatorOperand):
    """
    Parse action to process successfully parsed expressions with operands separated
    by an and/or operator (OPERAND OPERATOR OPERAND OPERATOR OPERAND ...)
    """

    def eval(self, **kwargs) -> BOOLEANTYPE:
        "Return combined bools / boolean arrays"
        val = self.value[0].eval(**kwargs)
        for operator_, operand in self.operatorOperands(self.value[1:]):
            evaluated_operand = operand.eval(**kwargs)
            if operator_ == "and":
                val = operator.and_(val, evaluated_operand)
            elif operator_ == "or":
                val = operator.or_(val, evaluated_operand)
        return val


class ConfigOptionParser(equation_parser.EvalString):
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
        config_group, self.config_option = tokens
        self.config_group = f"{config_group}_config"
        self.instring = instring
        self.loc = loc

    def __repr__(self):
        "Return string representation of the parsed grammar"
        return f"CONFIG:{self.config_group}.{self.config_option}"

    def eval(
        self, model_data: xr.Dataset, **kwargs
    ) -> Union[int, float, str, bool, np.bool_]:
        """
        If the parsed configuration group and configuration option are valid then
        return the option value, otherwise add to provided errors list inplace.

        Args:
            model_data (xr.Dataset): Calliope model data.

        Returns:
            Optional[Union[int, float, str, bool, np.bool_]]: Configuration option value.
        """
        if self.config_group not in model_data.attrs:
            raise BackendError(
                f"(where, {self.instring}): Invalid configuration group defined"
            )
        else:
            config_dict = model_data.attrs[self.config_group]
            # TODO: either remove the default key return or make it optional with
            # a "strict" arg
            config_val = config_dict.get_key(self.config_option, np.nan)

            if not isinstance(config_val, (int, float, str, bool, np.bool_)):
                raise BackendError(
                    f"(where, {self.instring}): Configuration option resolves to invalid "
                    f"type `{type(config_val).__name__}`, expected a number, string, or boolean."
                )
            else:
                return config_val


class DataVarParser(equation_parser.EvalString):
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

    @staticmethod
    def _data_var_exists(model_data_var: xr.DataArray) -> xr.DataArray:
        "mask by setting all (NaN | INF/-INF) to False, otherwise True"
        with pd.option_context("mode.use_inf_as_na", True):
            return model_data_var.where(pd.notnull(model_data_var)).notnull()

    def eval(
        self, model_data: xr.Dataset, defaults: dict, apply_imask: bool = True, **kwargs
    ) -> Union[np.bool_, xr.DataArray]:
        """
        Get parsed model data variable from the Calliope model dataset.
        If it isn't there, return False.

        Args:
            model_data (xr.Dataset): Calliope model dataset.
            apply_imask (bool, optional):
                If True, return boolean array corresponding to whether there is data or
                not in each element of the array. If False, return original array.
                Defaults to True.

        Returns:
            Union[np.bool_, xr.DataArray]:
                False if data variable not in model data, array otherwise.
        """
        if self.data_var not in model_data:
            return np.False_

        if apply_imask:
            return self._data_var_exists(model_data[self.data_var])
        else:
            return model_data[self.data_var].fillna(defaults.get(self.data_var))


class ComparisonParser(equation_parser.EvalComparisonOp):
    "Parse action to process successfully parsed strings of the form x=y"

    def __repr__(self):
        "Return string representation of the parsed grammar"
        return f"{self.lhs}={self.rhs}"

    def eval(self, **kwargs) -> BOOLEANTYPE:
        """
        Compare LHS (any) and RHS (numeric, string, bool) and return a bool/boolean array

        Returns:
            BOOLEANTYPE: Same shape as LHS.
        """
        kwargs["apply_imask"] = False
        lhs = self.lhs.eval(**kwargs)
        rhs = self.rhs.eval(**kwargs)

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

        if isinstance(comparison, bool):
            # enables the "~" operator to later invert `comparison` if required.
            comparison = np.bool_(comparison)
        return comparison


class SubsetParser(equation_parser.EvalString):
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

    def eval(self, imask: xr.DataArray, **kwargs) -> xr.DataArray:
        new_imask = imask.copy(deep=True)
        subset = [i.eval(**kwargs) for i in self.subset]
        new_imask.loc[{self.set_name: ~imask[self.set_name].isin(subset)}] = False
        return new_imask


class BoolOperandParser(equation_parser.EvalString):
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

    def eval(self, **kwargs) -> np.bool_:
        "evaluate string to numpy boolean object."
        if self.val == "true":
            bool_val = np.True_
        elif self.val == "false":
            bool_val = np.False_
        return bool_val


class GenericStringParser:
    def __init__(self, tokens: pp.ParseResults) -> None:
        """
        Parse action to process successfully parsed generic strings.
        This is required since we call "eval()" on all elements of the where string,
        so even arbitrary strings (used in comparison operations) need to be evaluatable.

        Args:
            tokens (pp.ParseResults): Has one parsed element: string name (str).
        """
        self.val = tokens[0]

    def __repr__(self):
        "Return string representation of the parsed grammar"
        return f"STRING:{self.val}"

    def eval(self, **kwargs) -> str:
        "Return input as string."
        return str(self.val)


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
    config_group = generic_identifier + pp.FollowedBy(".")
    config_options = pp.ZeroOrMore("." + generic_identifier)
    data_var = (
        config_group
        + pp.Suppress(".")
        + pp.Combine(generic_identifier + config_options)
    )
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


def imasking_parser(
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
            Parsing grammar to process helper functions of the form `helper_function(*args, **kwargs)`.
        data_var (pp.ParserElement): Parser for Calliope model dataset variable names.
        comparison_parser (pp.ParserElement): Parser for comparisons of the form "variable_or_config=comparator".

    Returns:
        pp.ParserElement:
            Parser for strings which use AND/OR/NOT operators to combine other parser elements.
    """
    notop = pp.Keyword("not", caseless=True)
    andorop = pp.Keyword("and", caseless=True) | pp.Keyword("or", caseless=True)

    imask_rules = pp.infixNotation(
        helper_function | comparison_parser | subset | data_var | bool_operand,
        [
            (notop, 1, pp.opAssoc.RIGHT, EvalNot),
            (andorop, 2, pp.opAssoc.LEFT, EvalAndOr),
        ],
    )

    return imask_rules


def generate_where_string_parser() -> pp.ParserElement:
    """
    Args:
        parse_string (str): Constraint subsetting "where" string.

    Returns:
        pp.ParseResults: evaluatable to a bool/boolean array.
    """
    number, generic_identifier = equation_parser.setup_base_parser_elements()
    data_var = data_var_parser(generic_identifier)
    config_option = config_option_parser(generic_identifier)
    bool_operand = bool_parser()
    evaluatable_string = evaluatable_string_parser(generic_identifier)
    helper_function = equation_parser.helper_function_parser(
        generic_identifier, allowed_parser_elements_in_args=[evaluatable_string, number]
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
    return imasking_parser(bool_operand, helper_function, data_var, comparison, subset)
