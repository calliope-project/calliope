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

from typing import Tuple, List, Union, Any, Callable, Dict

import pyparsing as pp


pp.ParserElement.enablePackrat()

COMPONENT_CLASSIFIER = "$"


def operatorOperands(tokenlist):
    "Generator to extract operators and operands in pairs"

    it = iter(tokenlist)
    while 1:
        try:
            yield (next(it), next(it))
        except StopIteration:
            break


class EvalSignOp:
    "Class to evaluate expressions with a leading + or - sign"

    def __init__(self, tokens):
        self.sign, self.value = tokens[0]

    def __repr__(self):
        return str(f"{self.sign}_{self.value}")

    def eval(self, **kwargs):
        mult = {"+": 1, "-": -1}[self.sign]
        return mult * self.value.eval(**kwargs)


class EvalPowerOp:
    "Class to evaluate multiplication and division expressions"

    def __init__(self, tokens):
        self.value = tokens[0]

    def __repr__(self):
        return str(self.value)

    def eval(self, **kwargs):
        res = self.value[-1].eval(**kwargs)
        for val in self.value[-3::-2]:
            res = val.eval(**kwargs) ** res
        return res


class EvalMultOp:
    "Class to evaluate multiplication and division expressions"

    def __init__(self, tokens):
        self.value = tokens[0]

    def __repr__(self):
        return str(self.value)

    def eval(self, **kwargs):
        prod = self.value[0].eval(**kwargs)
        for op, val in operatorOperands(self.value[1:]):
            if op == "*":
                prod *= val.eval(**kwargs)
            if op == "/":
                prod /= val.eval(**kwargs)
        return prod


class EvalAddOp:
    "Class to evaluate addition and subtraction expressions"

    def __init__(self, tokens):
        self.value = tokens[0]

    def __repr__(self):
        return str(self.value)

    def eval(self, **kwargs):
        sum = self.value[0].eval(**kwargs)
        for op, val in operatorOperands(self.value[1:]):
            if op == "+":
                sum += val.eval(**kwargs)
            if op == "-":
                sum -= val.eval(**kwargs)
        return sum


class EvalComparisonOp:
    "Class to evaluate comparison expressions"

    opMap = {
        "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
        ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b,
        "!=": lambda a, b: a != b,
        "=": lambda a, b: a == b,  # FIXME
        "==": lambda a, b: a == b,
    }

    def __init__(self, tokens):
        self.value = tokens[0]

    def __repr__(self):
        return str(self.value)

    def eval(self, **kwargs):
        val1 = self.value[0].eval(**kwargs)
        for op, val in operatorOperands(self.value[1:]):
            fn = self.opMap[op]
            val2 = val.eval(**kwargs)
            if not fn(val1, val2):
                break
            val1 = val2
        else:
            return True
        return False


class EvalFunction:
    def __init__(self, tokens: pp.ParseResults) -> None:
        """
        Parse action to process successfully parsed helper function strings of the form
        helper_function_name(*args, **kwargs).

        Args:
            tokens (pp.ParseResults):
                Has a dictionary component with the parsed elements:
                helper_function_name (pp.ParseResults), args (list), kwargs (dict).
        """
        token_dict = tokens.as_dict()
        self.name = token_dict["helper_function_name"]
        self.args = token_dict["args"]
        self.kwargs = token_dict["kwargs"]

    def __repr__(self):
        """Return string representation of the parsed grammar"""
        return f"{str(self.name)}(args={self.args}, kwargs={self.kwargs})"

    def eval(self, test: bool = False, **kwargs) -> Union[Any, Dict]:
        """

        Args:
            test (bool, optional):
                If True, return a dictionary with parsed components rather than
                calling the helper function with the defined args and kwargs.
                Defaults to False.

        Returns:
            Union[Any, Dict]:
                Either the defined helper function is called, or only a dictionary with
                parsed components is returned (if test=True).
        """
        args_ = []
        for arg in self.args:
            if not isinstance(arg, list):
                args_.append(arg.eval(test=test, **kwargs))
            else:  # evaluate nested function
                args_.append(arg[0].eval(test=test, **kwargs))

        kwargs_ = {}
        for kwarg_name, kwarg_val in self.kwargs.items():
            if not isinstance(kwarg_val, list):
                kwargs_[kwarg_name] = kwarg_val.eval(test=test, **kwargs)
            else:  # evaluate nested function
                kwargs_[kwarg_name] = kwarg_val[0].eval(test=test, **kwargs)

        helper_function = self.name.eval(test=test, **kwargs)
        if test:
            return {
                "function": helper_function,
                "args": args_,
                "kwargs": kwargs_,
            }
        else:
            return helper_function(*args_, **kwargs_)


class EvalHelperFuncName:
    def __init__(self, instring: str, loc: int, tokens: pp.ParseResults) -> None:
        """
        Parse action to process successfully parsed helper function names.
        This is a unique parse action so that we can catch invalid helper functions
        most safely.

        Args:
            instring (str): String that was parsed (used in error message).
            loc (int):
                Location in parsed string where parsing error was logged.
                This is not used, but comes with `instring` when setting the parse action.
            tokens (pp.ParseResults):
                Has one parsed element: helper_function_name (str).
        """
        self.name = tokens[0]
        self.instring = instring
        self.loc = loc

    def __repr__(self):
        """Return string representation of the parsed grammar"""
        return str(self.name)

    def eval(
        self, helper_func_dict: Dict[str, Callable], test: bool = False, **kwargs
    ) -> Union[str, Callable]:
        """

        Args:
            helper_func_dict (Dict[str, Callable]): Allowed helper functions.
            test (bool, optional):
                If True, return a string with the helper function name rather than
                collecting the helper function from the dictionary of functions.
                Defaults to False.

        Raises:
            pp.ParseException:
                If parsed helper function name is not in `helper_func_dict`.

        Returns:
            Union[str, Callable]:
                Helper functions are expected to be two-tiered, with the first level
                taking the generic eval kwargs (e.g. model_data) and the second level
                taking the user-defined input arguments.
                If test=True, only the helper function name is returned.
        """
        if self.name not in helper_func_dict.keys():
            # FIXME: Maybe shouldn't be a parse exception since it happens on evaluation
            # calliope.exceptions.ModelError? KeyError?
            raise pp.ParseException(
                self.instring, self.loc, "Invalid helper function defined"
            )
        if test:
            return str(self.name)
        else:
            return helper_func_dict[self.name](**kwargs)


class EvalIndexedParameterOrVariable:
    def __init__(self, tokens: pp.ParseResults) -> None:
        """
        Parse action to process successfully parsed indexed parameters or decision variables
        of the form param_or_var[*index_items].

        Args:
            tokens (pp.ParseResults):
                Has a dictionary component with the parsed elements:
                param_or_var_name (str), index_items (list of strings).
        """
        token_dict = tokens.as_dict()
        self.name = token_dict["param_or_var_name"][0]
        self.index_items = token_dict["index_items"]

    def __repr__(self):
        """Return string representation of the parsed grammar"""
        return "INDEXED_PARAM_OR_VAR:" + str(self.name)

    def eval(self, **kwargs) -> Dict[str, Union[str, List[str]]]:
        """

        Returns:
            Dict[str, Union[str, List[str]]]:
                Separated key:val pairs for parameter/variable name and index items
                TODO: make this a return only in testing and grab the actual
                parameter/variable from model_data/backend_model on evaluation.
        """
        return {"param_or_var_name": self.name, "dimensions": self.index_items}


class EvalComponent:
    def __init__(self, tokens: pp.ParseResults) -> None:
        """
        Parse action to process successfully parsed expression components of the form
        `$component`.

        Args:
            tokens (pp.ParseResults):
                Has one parsed element containing the component name (str).
        """
        self.name = tokens[0]

    def __repr__(self):
        """Return string representation of the parsed grammar"""
        return "COMPONENT:" + str(self.name)

    def eval(self, **kwargs) -> Dict[str, str]:
        """

        Returns:
            Dict[str, str]:
                TODO: make this a return only in testing and evaluate the actual
                component expression contents on evaluation.
        """
        return {"component": self.name}


class EvalUnindexedParameterOrVariable:
    def __init__(self, tokens: pp.ParseResults) -> None:
        """
        Parse action to process successfully parsed unindexed parameters or decision variables
        of the form `param_or_var`.

        Args:
            tokens (pp.ParseResults):
                Has one parsed element containing the paramater/variable name (str).
        """
        self.name = tokens[0]

    def __repr__(self):
        """Return string representation of the parsed grammar"""
        return "UNINDEXED_PARAM_OR_VAR:" + str(self.name)

    def eval(self, **kwargs) -> Dict[str, str]:
        """

        Returns:
            Dict[str, str]:
                TODO: make this a return only in testing and grab the actual
                parameter/variable from model_data/backend_model on evaluation.
        """
        return {"param_or_var_name": self.name}


class EvalNumber:
    def __init__(self, tokens: pp.ParseResults) -> None:
        """
        Parse action to process successfully parsed numbers described as integers (1),
        floats (1.), or in scientific notation (1e1). Also capture infinity (inf/.inf).

        Args:
            tokens (pp.ParseResults):
                Has one parsed element containing the number (str).
        """
        self.value = tokens[0]

    def __repr__(self):
        """Return string representation of the parsed grammar"""
        return "NUM:" + str(self.value)

    def eval(self, **kwargs) -> float:
        """

        Returns:
            float: Input string as a float, even if given as an integer.
        """
        return float(self.value)


def helper_function_parser(
    generic_identifier: pp.ParserElement,
    allowed_parser_elements_in_args: List[pp.ParserElement],
    allow_function_in_function: bool = True,
) -> pp.ParserElement:
    """
    Parsing grammar to process helper functions of the form `helper_function(*args, **kwargs)`.

    Helper functions can accept other parser elements as arguments,
    i.e., components, parameters or variables, numbers, and other functions.

    Available helper functions are predefined in calliope.backend.helper_functions.

    Calling an unavailable helper will lead to a raised exception on evaluating the
    parsed element.

    Based partially on: # https://stackoverflow.com/questions/61807705/pyparsing-generic-python-function-args-and-kwargs

    Args:
        generic_identifier (pp.ParserElement):
            Parser for valid python variables without leading underscore and not called "inf".
            This parser has no parse action.
        allowed_parser_elements_in_args (List[pp.ParserElement]):
            List of parser elements that can be arguments in the function (e.g., "number", "indexed_param_or_var").
        allow_function_in_function (bool, optional):
            If True, allows functions to be defined inside functions.
            Nested functions are evaluated from the greatest level of nesting up to the main helper function.
            Defaults to True.

    Returns:
        pp.ParserElement:
            Parser for functions which will call the function with the specified
            arguments on evaluation.
    """

    helper_function = pp.Forward()

    lpar = pp.Suppress("(")
    rpar = pp.Suppress(")")

    helper_function_name = generic_identifier.set_results_name("helper_function_name")
    helper_function_name.set_parse_action(EvalHelperFuncName)

    if allow_function_in_function:
        allowed_parser_elements_in_args.insert(0, pp.Group(helper_function))

    arg_values = pp.MatchFirst(allowed_parser_elements_in_args) + pp.NotAny("=")

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

    helper_function.set_parse_action(EvalFunction)

    return helper_function


def indexed_param_or_var_parser(
    generic_identifier: pp.ParserElement,
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

    Returns:
        pp.ParserElement:
            Parser which returns a dictionary with name of parameter/variable and list
            of index items as separate entries on.
    """

    lspar = pp.Suppress("[")
    rspar = pp.Suppress("]")

    indexed_param_name = generic_identifier("param_or_var_name")

    index_items = pp.Group(pp.delimitedList(generic_identifier))("index_items")
    indexed_param_or_var = pp.Combine(indexed_param_name + lspar) + index_items + rspar

    indexed_param_or_var.set_parse_action(EvalIndexedParameterOrVariable)

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
    component.set_parse_action(EvalComponent)

    return component


def unindexed_param_parser(generic_identifier: pp.ParserElement) -> pp.ParserElement:
    """
    Create a copy of the generic identifier and set a parse action to find the string in
    the list of input paramaters or optimisation decision variables.

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
    unindexed_param_or_var.set_parse_action(EvalUnindexedParameterOrVariable)

    return unindexed_param_or_var


def arithmetic_parser(
    helper_function: pp.ParserElement, indexed_param_or_var: pp.ParserElement, component: pp.ParserElement, unindexed_param_or_var: pp.ParserElement, number: pp.ParserElement
) -> pp.ParserElement:
    """
    Parsing grammar to combine equation elements using basic arithmetic (+, -, *, /, **).
    Can handle the difference between a sign (e.g., -1,+1) and a addition/subtraction (0 - 1, 0 + 1).

    Whitespace is ignored on parsing (i.e., "1+1+foo" is equivalent to "1 + 1 + foo").

    Args:
        helper_function (pp.ParserElement):
            Parsing grammar to process helper functions of the form `helper_function(*args, **kwargs)`.
        indexed_param_or_var (pp.ParserElement):
            Parser for indexed parameters or variables, e.g. "foo[bar]"
        component (pp.ParserElement):
            Parser for constraint components, e.g. "$foo"
        unindexed_param_or_var (pp.ParserElement):
            Parser for unindezed parameters or variables, e.g. "foo"
        number (pp.ParserElement):
            Parser for numbers (integer, float, scientific notation, "inf"/".inf").

    Returns:
        pp.ParserElement:
            Parser for strings which use arithmetic operations to combine other parser elements.
    """
    signop = pp.oneOf("+ -")
    multop = pp.oneOf("* /")
    plusop = pp.oneOf("+ -")
    expop = pp.Literal("**")

    arithmetic = pp.infixNotation(
        # the order matters if two could capture the same string, e.g. "inf".
        helper_function
        | indexed_param_or_var
        | component
        | number
        | unindexed_param_or_var,
        [
            (signop, 1, pp.opAssoc.RIGHT, EvalSignOp),
            (expop, 2, pp.opAssoc.LEFT, EvalPowerOp),
            (multop, 2, pp.opAssoc.LEFT, EvalMultOp),
            (plusop, 2, pp.opAssoc.LEFT, EvalAddOp),
        ],
    )

    return arithmetic


def equation_comparison_parser(arithmetic: pp.ParserElement) -> pp.ParserElement:
    """
    Parsing grammar to combine equation elements either side of a comparison operator
    (< <= > >= ==).

    Whitespace is ignored on parsing (i.e., "1+foo==$bar" is equivalent to "1 + 1 == $bar").

    Args:
        arithmetic (pp.ParserElement):
            Parser for arithmetic operations to combine other parser elements.

    Returns:
        pp.ParserElement:
            Parser for strings of the form "LHS OPERATOR RHS".
    """

    comparison_operators = pp.oneOf("< <= > >= ==")
    equation_comparison = pp.infixNotation(
        arithmetic,
        [
            (comparison_operators, 2, pp.opAssoc.LEFT, EvalComparisonOp),
        ],
    )

    return equation_comparison


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
    generic_identifier = ~inf_kw + pp.Word(pp.alphas, pp.alphanums + "_")

    number.set_parse_action(EvalNumber)

    return number, generic_identifier
