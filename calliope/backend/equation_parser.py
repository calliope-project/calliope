##
# Part of the code in this file is adapted from
# https://github.com/pyparsing/pyparsing/blob/master/examples/eval_arith.py
# available under the MIT license
##
# Copyright 2009, 2011 Paul McGuire
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
##

from typing import Tuple, List

import pyparsing as pp


pp.ParserElement.enablePackrat()

HELPER_FUNCS = {"dummy_func_1": lambda x: x * 10, "dummy_func_2": lambda x, y: x + y}

COMPONENT_CLASSIFIER = "$"




class EvalFunction:
    def __init__(self, tokens):
        token_dict = tokens.as_dict()
        self.name = token_dict["helper_function_name"]
        self.args = token_dict["args"]
        self.kwargs = token_dict["kwargs"]

    def __repr__(self):
        return f"{str(self.name)}(args={self.args}, kwargs={self.kwargs})"

    def eval(self):
        args_ = []
        for arg in self.args:
            if not isinstance(arg, list):
                args_.append(arg.eval())
            else:  # evaluate nested function
                args_.append(arg[0].eval())

        kwargs_ = {}
        for kwarg_name, kwarg_val in self.kwargs.items():
            if not isinstance(kwarg_val, list):
                kwargs_[kwarg_name] = kwarg_val.eval()
            else:  # evaluate nested function
                kwargs_[kwarg_name] = kwarg_val[0].eval()

        return {"function": self.name.eval(), "args": args_, "kwargs": kwargs_}


class EvalHelperFuncName:
    def __init__(self, instring, loc, tokens):
        self.name = tokens[0]
        self.instring = instring
        self.loc = loc

    def __repr__(self):
        return str(self.name)

    def eval(self):
        if self.name not in HELPER_FUNCS.keys():
            # Maybe shouldn't be a parse exception since it happens on evaluation
            raise pp.ParseException(
                self.instring, self.loc, "Invalid helper function defined"
            )
        return str(self.name)


class EvalIndexedParameterOrVariable:
    def __init__(self, tokens):
        token_dict = tokens.as_dict()
        self.name = token_dict["param_or_var_name"][0]
        self.args = token_dict["index_items"]

    def my_func(self):
        return str(self.name).title()

    def __repr__(self):
        return "INDEXED_PARAM_OR_VAR:" + str(self.name)

    def eval(self):
        return {"param_or_var_name": self.name, "dimensions": self.args}


class EvalComponent:
    def __init__(self, tokens):
        self.name = tokens[0]

    def __repr__(self):
        return "COMPONENT:" + str(self.name)

    def eval(self):
        return {"component": self.name}


class EvalUnindexedParameterOrVariable:
    # TODO: decide whether to loop this into "EvalIndexedParameterOrVariable" directly?
    # ^ tried and the parser doesn't like it. Better to have them share a validation
    # function that checks the param/var name against the model
    def __init__(self, tokens):
        self.name = tokens[0]

    def __repr__(self):
        return "UNINDEXED_PARAM_OR_VAR:" + str(self.name)

    def eval(self):
        return {"param_or_var_name": self.name}

class EvalNumber:
    def __init__(self, tokens):
        self.value = tokens[0]

    def __repr__(self):
        return "NUM:" + str(self.value)

    def eval(self):
        return float(self.value)


def helper_function_parser(
    indexed_param_or_var: pp.ParserElement,
    component: pp.ParserElement,
    unindexed_param_or_var: pp.ParserElement,
    number: pp.ParserElement,
    generic_identifier: pp.ParserElement,
) -> pp.ParserElement:
    """
    Parsing grammar to process helper functions of the form `helper_function(*args, **kwargs)`.

    Helper functions can accept other parser elements as arguments,
    i.e., components, parameters or variables, numbers, and other functions.

    Available helper functions are predefined in calliope.backend.helper_functions.

    Calling an unavailable helper will lead to a raised exception on evaluating the
    parsed element.

    Args:
        indexed_param_or_var (pp.ParserElement):
            Parser for indexed parameters or variables, e.g. "foo[bar]"
        component (pp.ParserElement):
            Parser for constraint components, e.g. "$foo"
        unindexed_param_or_var (pp.ParserElement):
            Parser for unindezed parameters or variables, e.g. "foo"
        number (pp.ParserElement):
            Parser for numbers (integer, float, scientific notation, "inf"/".inf").
        generic_identifier (pp.ParserElement):
            Parser for valid python variables without leading underscore and not called "inf".
            This parser has no parse action.

    Returns:
        pp.ParserElement:
            Parser for functions which will call the function with the specified
            arguments on evaluation.
    """
    # https://stackoverflow.com/questions/61807705/pyparsing-generic-python-function-args-and-kwargs

    helper_function = pp.Forward()

    lpar = pp.Suppress("(")
    rpar = pp.Suppress(")")

    helper_function_name = generic_identifier.set_results_name("helper_function_name")
    helper_function_name.set_parse_action(EvalHelperFuncName)

    arg_values = (
        pp.Group(helper_function)
        | number
        | ((indexed_param_or_var | component | unindexed_param_or_var) + pp.NotAny("="))
    )
    # define function arguments
    arglist = pp.delimitedList(arg_values.copy())
    args = pp.Group(arglist).set_results_name("args")

    # define function keyword arguments
    key = generic_identifier + pp.Suppress("=")
    kwarglist = pp.delimited_list(pp.dict_of(key, arg_values))
    kwargs = pp.Group(kwarglist).set_results_name("kwargs")

    # build generic function
    helper_func_args = args + pp.Suppress(",") + kwargs | pp.Opt(
        args, default=[]
    ) + pp.Opt(kwargs, default={})
    helper_function << (
        pp.Combine(helper_function_name + lpar) + helper_func_args + rpar
    )

    helper_function.setParseAction(EvalFunction)

    return helper_function


def indexed_param_or_var_parser(
    generic_identifier: pp.ParserElement, set_iterators: List[str]
) -> pp.ParserElement:
    """
    Parsing grammar to process strings representing indexed model parameters or variables,
    e.g. "resource[node, tech]".

    If a parameter, must be a data variable in the Model._model_data xarray dataset.

    If a variable, must be an optimisation problem decision variable.

    The parser will not verify whether it has parsed a valid parameter or variable until
    evaluation.

    Args:
        generic_identifier (pp.ParserElement):
            Parser for valid python variables without leading underscore and not called "inf".
            This parser has no parse action.
        set_iterators (List[str]):
            Allowed set iterators on which the parameters can be indexed, e.g. ["node", "tech"].

    Returns:
        pp.ParserElement:
            Parser which returns a dictionary with name of parameter/variable and list
            of index items as separate entries on.
    """

    def _missing_iterator(instring, loc, expr, err):
        # TODO: pass this to the ParsedConstraint error catcher to handle
        raise KeyError(err)

    lspar = pp.Suppress("[")
    rspar = pp.Suppress("]")

    indexed_param_name = generic_identifier.set_results_name("param_or_var_name")

    set_iterator = pp.one_of(set_iterators, as_keyword=True)
    set_iterator.set_fail_action(_missing_iterator)

    index_items = pp.Group(pp.delimitedList(set_iterator)).set_results_name(
        "index_items"
    )
    indexed_param_or_var = pp.Combine(indexed_param_name + lspar) + index_items + rspar

    indexed_param_or_var.setParseAction(EvalIndexedParameterOrVariable)

    return indexed_param_or_var


def component_parser(generic_identifier: pp.ParserElement) -> pp.ParserElement:
    f"""
    Parse strings preppended with the YAML constraint component classifier.
    {COMPONENT_CLASSIFIER}.
    E.g. "{COMPONENT_CLASSIFIER}my_component"

    Args:
        generic_identifier (pp.ParserElement):
            Parser for valid python variables without leading underscore and not called "inf".
            This parser has no parse action.

    Returns:
        pp.ParserElement:
            Parser which produces a dictionary of the form {{"component": "my_component"}}
            on evaluation.
    """

    component = pp.Combine(pp.Suppress(COMPONENT_CLASSIFIER) + generic_identifier)
    component.setParseAction(EvalComponent)

    return component


def unindexed_param_parser(generic_identifier: pp.ParserElement) -> pp.ParserElement:
    """
    Create copy of generic identifier to set a parse action on later.

    Args:
        generic_identifier (pp.ParserElement):
            Parser for valid python variables without leading underscore and not called "inf".
            This parser has no parse action.

    Returns:
        pp.ParserElement:
            Copy of input parser with added parse action to lookup an unindexed
            parameter/variable value
    """

    unindexed_param_or_var = generic_identifier.copy()
    unindexed_param_or_var.setParseAction(EvalUnindexedParameterOrVariable)

    return unindexed_param_or_var


def setup_base_parser_elements() -> Tuple[pp.ParserElement, pp.ParserElement]:
    """
    Setup parser elements that will be components of other parsers.

    Returns:
        number [pp.ParserElement]:
            Parser for numbers (integer, float, scientific notation, "inf"/".inf").
        generic_identifier (pp.ParserElement):
            Parser for valid python variables without leading underscore and not called "inf".
            This parser has no parse action.
    """

    inf_kw = pp.Combine(pp.Opt(pp.Suppress(".")) + pp.Keyword("inf", caseless=True))
    number = pp.pyparsing_common.number | inf_kw
    generic_identifier = ~inf_kw + pp.Word(pp.alphas + "_", pp.alphanums + "_")

    number.setParseAction(EvalNumber)

    return number, generic_identifier
