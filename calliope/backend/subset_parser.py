from typing import Tuple, List
import operator
import functools

import pyparsing as pp
import numpy as np
import pandas as pd

from calliope.core.attrdict import AttrDict
from calliope.backend import equation_parser

pp.ParserElement.enablePackrat()


def operatorOperands(tokenlist):  # TODO: get from equation_parser.py
    "Generator to extract operators and operands in pairs"

    it = iter(tokenlist)
    while 1:
        try:
            yield (next(it), next(it))
        except StopIteration:
            break


class EvalNotOp:
    "Class to evaluate expressions with a leading `not`"

    def __init__(self, tokens):
        self.sign, self.value = tokens[0]

    def __repr__(self):
        return str(f"{self.sign} {self.value}")

    def eval(self, **kwargs):
        evaluated_ = self.value.eval(**kwargs)
        return ~evaluated_


class EvalAndOrOp:
    "Class to evaluate addition and subtraction expressions"

    def __init__(self, tokens):
        self.value = tokens[0]

    def __repr__(self):
        return str(self.value)

    def eval(self, **kwargs):
        new_imask = self.value[0].eval(**kwargs)
        for op, val in operatorOperands(self.value[1:]):
            if op == "and":
                new_imask = operator.and_(new_imask, val.eval(**kwargs))
            if op == "or":
                new_imask = operator.or_(new_imask, val.eval(**kwargs))
        return new_imask


class ConfigOptionParser:
    def __init__(self, instring, loc, tokens):
        config_group, self.config_option = tokens
        self.config_group = f"{config_group}_config"
        self.instring = instring
        self.loc = loc

    def __repr__(self):
        return f"CONFIG:{self.config_group}.{self.config_option}"

    def eval(self, model_data, **kwargs):
        if self.config_group not in model_data.attrs:
            # FIXME: Maybe shouldn't be a parse exception since it happens on evaluation
            # calliope.exceptions.ModelError? KeyError?
            raise pp.ParseException(
                self.instring, self.loc, "Invalid configuration group defined"
            )
        else:
            config_dict = AttrDict.from_yaml_string(model_data.attrs[self.config_group])
            # TODO: either remove the default key return or make it optional with
            # a "strict" arg
            config_val = config_dict.get_key(self.config_option, np.nan)

            if not isinstance(config_val, (int, float, str, bool, np.bool_)):
                raise TypeError(
                    f"Cannot subset by comparison to `{self.config_group}` option `{self.config_option}` of type `{type(config_val).__name__}`"
                )
            else:
                return config_val


class DataVarParser:
    def __init__(self, instring, loc, tokens):
        self.data_var = tokens[0]
        self.instring = instring
        self.loc = loc

    def __repr__(self):
        return f"DATA_VAR:{self.config_group}{self.config_option}"

    @staticmethod
    def _data_var_exists(model_data_var):
        # mask by NaN and INF/-INF values = False, otherwise True
        with pd.option_context("mode.use_inf_as_na", True):
            return model_data_var.where(pd.notnull(model_data_var)).notnull()

    def eval(self, model_data, apply_imask=True, **kwargs):

        if self.data_var not in model_data.data_vars.keys():
            return np.False_

        if apply_imask:
            return self._data_var_exists(model_data[self.data_var])
        else:
            return model_data[self.data_var]

class ComparisonParser:
    def __init__(self, tokens):
        self.var_or_config = tokens[0]
        self.val_to_compare = tokens[1]

    def __repr__(self):
        return f"{self.var_or_config}={self.val_to_compare}"

    def eval(self, model_data, **kwargs):
        return np.bool_(
            self.var_or_config.eval(apply_imask=False, model_data=model_data)
            == self.val_to_compare.eval()
        )


class BoolOperandParser:
    def __init__(self, tokens):
        self.val = tokens[0].lower()

    def __repr__(self):
        return f"BOOL:{self.val}"

    def eval(self, **kwargs):
        if self.val == "true":
            return np.True_
        elif self.val == "false":
            return np.False_


class GenericStringParser:
    def __init__(self, tokens):
        self.val = tokens[0]

    def __repr__(self):
        return f"STRING:{self.val}"

    def eval(self, **kwargs):
        return str(self.val)


def data_var_parser(generic_identifier: pp.ParserElement):
    protected_variables = (
        pp.Keyword("node_tech") | pp.Keyword("carrier") | pp.Keyword("inheritance")
    )
    protected_strings = (
        pp.Keyword("and", caseless=True)
        | pp.Keyword("or", caseless=True)
        | pp.Keyword("not", caseless=True)
    )
    data_var = ~(protected_variables | protected_strings) + generic_identifier
    data_var.set_parse_action(DataVarParser)

    return data_var


def config_option_parser(generic_identifier: pp.ParserElement):
    config_group = generic_identifier + pp.FollowedBy(".")
    config_options = pp.ZeroOrMore("." + generic_identifier)
    data_var = (
        config_group
        + pp.Suppress(".")
        + pp.Combine(generic_identifier + config_options)
    )
    data_var.set_parse_action(ConfigOptionParser)

    return data_var



def bool_parser():
    TRUE = pp.Keyword("True", caseless=True)
    FALSE = pp.Keyword("False", caseless=True)
    bool_operand = TRUE | FALSE
    bool_operand.set_parse_action(BoolOperandParser)

    return bool_operand


def evaluatable_string_parser(generic_identifier):
    evaluatable_identifier = generic_identifier.copy()
    evaluatable_identifier.set_parse_action(GenericStringParser)

    return evaluatable_identifier


def comparison_parser(
    evaluatable_identifier, number, bool_operand, config_option, data_var
):
    comparison_expression = (
        (config_option | data_var)
        + pp.Suppress("=")
        + (bool_operand | number | evaluatable_identifier)
    )
    comparison_expression.set_parse_action(ComparisonParser)

    return comparison_expression


def imasking_parser(
    helper_function: pp.ParserElement,
    data_var: pp.ParserElement,
    comparison_parser: pp.ParserElement,
) -> pp.ParserElement:
    notop = pp.Keyword("not", caseless=True)
    andorop = pp.Keyword("and", caseless=True) | pp.Keyword("or", caseless=True)

    imask_rules = pp.infixNotation(
        helper_function | comparison_parser | data_var,
        [
            (notop, 1, pp.opAssoc.RIGHT, EvalNotOp),
            (andorop, 2, pp.opAssoc.LEFT, EvalAndOrOp),
        ],
    )

    return imask_rules


def parse_where_string(parse_string):
    number, generic_identifier = equation_parser.setup_base_parser_elements()
    data_var = data_var_parser(generic_identifier)
    config_option = config_option_parser(generic_identifier)
    bool_operand = bool_parser()
    evaluatable_string = evaluatable_string_parser(generic_identifier)
    comparison = comparison_parser(
        evaluatable_string, number, bool_operand, config_option, data_var
    )
    helper_function = equation_parser.helper_function_parser(
        generic_identifier, allowed_parser_elements_in_args=[evaluatable_string, number]
    )
    parser = imasking_parser(helper_function, data_var, comparison)
    parsed = parser.parse_string(parse_string, parse_all=True)

    return parsed[0]
