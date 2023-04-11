# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).

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

from __future__ import annotations

from typing import Callable, Any, Union, Optional, Iterator, Iterable
from abc import ABC

import pyparsing as pp
import xarray as xr

from calliope.exceptions import BackendError

pp.ParserElement.enablePackrat()

COMPONENT_CLASSIFIER = "$"


class EvalString(ABC):
    "Parent class for all string evaluation classes - used in type hinting"
    name: str


class EvalOperatorOperand(EvalString):
    def __init__(self, tokens: pp.ParseResults) -> None:
        """
        Parse action to process successfully parsed expressions with operands separated
        by an operator (OPERAND OPERATOR OPERAND OPERATOR OPERAND ...)

        Args:
            tokens (pp.ParseResults):
                Contains a list of the form [operand (pp.ParseResults), operator (str),
                operand (pp.ParseResults), operator (str), ...].
        """
        self.value = tokens[0]
        self.values = tokens

    def __repr__(self) -> str:
        "Return string representation of the parsed grammar"
        first_operand = self.value[0].__repr__()
        operand_operator_pairs = " ".join(
            op + " " + val.__repr__()
            for op, val in self.operatorOperands(self.value[1:])
        )
        arithmetic_string = f"({first_operand} {operand_operator_pairs})"
        return arithmetic_string

    def operatorOperands(
        self, tokenlist: list
    ) -> Iterator[tuple[str, pp.ParseResults]]:
        "Generator to extract operators and operands in pairs"

        it = iter(tokenlist)
        while 1:
            try:
                yield (next(it), next(it))
            except StopIteration:
                break

    def eval(self, **eval_kwargs) -> Any:
        """
        Returns:
            Any:
                If all operands are numeric, returns float, otherwise returns an
                expression to use in an optimisation model constraint.
        """
        apply_imask = eval_kwargs.get("apply_imask", True)
        val = xr.DataArray(self.value[0].eval(**eval_kwargs))

        if apply_imask:
            val = val.where(eval_kwargs["imask"])
        for operator_, operand in self.operatorOperands(self.value[1:]):
            evaluated_operand = xr.DataArray(operand.eval(**eval_kwargs))
            if apply_imask:
                evaluated_operand = evaluated_operand.where(eval_kwargs["imask"])
            if operator_ == "**":
                val = val**evaluated_operand
            elif operator_ == "*":
                val = val * evaluated_operand
            elif operator_ == "/":
                val = val / evaluated_operand
            elif operator_ == "+":
                val = val + evaluated_operand
            elif operator_ == "-":
                val = val - evaluated_operand

        return val


class EvalSignOp(EvalString):
    def __init__(self, tokens: pp.ParseResults) -> None:
        """
        Parse action to process successfully parsed expressions with a leading + or - sign.

        Args:
            tokens (pp.ParseResults):
                Contains a list of the form [sign (str), operand (pp.ParseResults)].
        """
        self.sign, self.value = tokens[0]
        self.values = tokens

    def __repr__(self) -> str:
        "Return string representation of the parsed grammar"
        return str(f"({self.sign}){self.value.__repr__()}")

    def eval(self, **eval_kwargs) -> Any:
        val = self.value.eval(**eval_kwargs)
        if self.sign == "+":
            return val
        elif self.sign == "-":
            return -1 * val


class EvalComparisonOp(EvalString):
    def __init__(self, tokens: pp.ParseResults) -> None:
        """
        Parse action to process successfully parsed equations of the form LHS OPERATOR RHS.

        Args:
            tokens (pp.ParseResults):
                Contains a list with an RHS (pp.ParseResults), operator (str), and LHS (pp.ParseResults).
        """
        self.lhs, self.op, self.rhs = tokens
        self.values = tokens

    def __repr__(self) -> str:
        "Return string representation of the parsed grammar"
        return f"{self.lhs.__repr__()} {self.op} {self.rhs.__repr__()}"

    def eval(self, **eval_kwargs) -> Any:
        """
        Returns:
            Any:
                If LHS and RHS are numeric, returns bool, otherwise returns an equation
                to use as an optimisation model constraint.
        """
        lhs = self.lhs.eval(**eval_kwargs)
        rhs = self.rhs.eval(**eval_kwargs)

        return lhs, self.op, rhs


class EvalFunction(EvalString):
    def __init__(self, tokens: pp.ParseResults) -> None:
        """
        Parse action to process successfully parsed helper function strings of the form
        helper_function_name(*args, **eval_kwargs).

        Args:
            tokens (pp.ParseResults):
                Has a dictionary component with the parsed elements:
                helper_function_name (pp.ParseResults), args (list), kwargs (dict).
        """
        token_dict = tokens.as_dict()
        self.func_name: pp.ParseResults = token_dict["helper_function_name"]
        self.args: list = token_dict["args"]
        self.kwargs: dict = token_dict["kwargs"]
        self.values = tokens

    def __repr__(self) -> str:
        "Return string representation of the parsed grammar"
        return f"{str(self.func_name)}(args={self.args}, kwargs={self.kwargs})"

    def arg_eval(self, arg, **eval_kwargs):
        if isinstance(arg, pp.ParseResults):
            evaluated_ = arg[0].eval(**eval_kwargs)
        elif isinstance(arg, list):
            evaluated_ = [self.arg_eval(arg_) for arg_ in arg]
        else:
            evaluated_ = arg.eval(**eval_kwargs)
        return evaluated_

    def eval(self, **eval_kwargs) -> Any:
        """

        Args:
            test (bool, optional):
                If True, return a dictionary with parsed components rather than
                calling the helper function with the defined args and kwargs.
                Defaults to False.

        Returns:
            Any:
                Either the defined helper function is called, or only a dictionary with
                parsed components is returned (if test=True).
        """
        eval_kwargs["apply_imask"] = False

        args_ = []
        for arg in self.args:
            args_.append(self.arg_eval(arg, **eval_kwargs))

        kwargs_ = {}
        for kwarg_name, kwarg_val in self.kwargs.items():
            kwargs_[kwarg_name] = self.arg_eval(kwarg_val, **eval_kwargs)

        helper_function = self.func_name.eval(**eval_kwargs)
        if eval_kwargs.get("as_dict"):
            return {
                "function": helper_function,
                "args": args_,
                "kwargs": kwargs_,
            }
        else:
            eval_func = helper_function(*args_, **kwargs_)
            return eval_func


class EvalHelperFuncName(EvalString):
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
        self.name = self.value = tokens[0]  # type: str
        self.instring = instring
        self.loc = loc
        self.values = tokens

    def __repr__(self) -> str:
        "Return string representation of the parsed grammar"
        return str(self.name)

    def eval(
        self,
        helper_func_dict: dict[str, Callable],
        as_dict: bool = False,
        **eval_kwargs,
    ) -> Optional[Union[str, Callable]]:
        """

        Args:
            helper_func_dict (dict[str, Callable]): Allowed helper functions.
            test (bool, optional):
                If True, return a string with the helper function name rather than
                collecting the helper function from the dictionary of functions.
                Defaults to False.

        Returns:
            str, Callable:
                Helper functions are expected to be two-tiered, with the first level
                taking the generic eval kwargs (e.g. model_data) and the second level
                taking the user-defined input arguments.
                If test=True, only the helper function name is returned.
        """

        if self.name not in helper_func_dict.keys():
            raise BackendError(
                f"({eval_kwargs['equation_name']}, {self.instring}): Invalid helper function defined"
            )
        else:
            if as_dict:
                return str(self.name)
            else:
                return helper_func_dict[self.name](**eval_kwargs)


class EvalSlicedParameterOrVariable(EvalString):
    def __init__(self, tokens: pp.ParseResults) -> None:
        """
        Parse action to process successfully parsed sliced parameters or decision variables
        of the form param_or_var[*index_slices].

        Args:
            tokens (pp.ParseResults):
                Has a dictionary component with the parsed elements:
                param_or_var_name (str), index_slices (list of strings).
        """
        token_dict = tokens.as_dict()
        self.obj_name: pp.ParseResults = token_dict["param_or_var_name"]

        self.index_slices: dict[str, pp.ParseResults] = {
            idx["set_name"][0]: idx["slicer"][0] for idx in token_dict["index_slices"]
        }
        self.values = tokens

    def __repr__(self) -> str:
        "Return string representation of the parsed grammar"
        slices = ", ".join(f"{k}={v.__repr__()}" for k, v in self.index_slices.items())
        return f"SLICED_{self.obj_name}[{slices}]"

    def eval(self, **eval_kwargs) -> Optional[Union[dict, xr.DataArray]]:
        """
        Returns:
            Optional[Union[dict, xr.DataArray]]:
                If `eval_kwargs["as_dict"]` is True, returns separated key:val pairs for parameter/variable name and index items;
                else, `eval_kwargs` has a backend dataset, returns sliced xarray object;
                else, returns None.
        """
        index_slices: dict[str, Any] = {
            k: v.eval(**eval_kwargs) for k, v in self.index_slices.items()
        }

        if eval_kwargs.get("as_dict", False):
            return {"dimensions": index_slices, **self.obj_name.eval(**eval_kwargs)}
        elif eval_kwargs.get("backend_dataset", None) is not None:
            return self.obj_name.eval(**eval_kwargs).sel(**index_slices)
        else:
            return None


class EvalIndexSlice(EvalString):
    def __init__(self, tokens: pp.ParseResults) -> None:
        """
        Parse action to process successfully parsed expression index slice references
        of the form `$index_slice`.

        Args:
            tokens (pp.ParseResults):
                Has one parsed element containing the index slice name (str).
        """
        self.name: str = tokens[0]
        self.values = tokens

    def __repr__(self) -> str:
        "Return string representation of the parsed grammar"
        return "REFERENCE:" + str(self.name)

    def eval(
        self,
        index_slice_dict: Optional[dict[str, pp.ParseResults]] = None,
        **eval_kwargs,
    ) -> Any:
        """
        Args:
            index_slice_dict (Optional[dict[str, pp.ParseResults]]):
                Dictionary mapping the index slice name to a parsed equation expression.
                Default is None.

        Returns:
            Any: If eval_kwargs["as_dict"] is True, returns a dictionary,
            otherwise attempts to evaluate the referenced index slice.
        """
        if eval_kwargs.get("as_dict"):
            return {"index_slice_reference": self.name}
        elif index_slice_dict is not None:
            return index_slice_dict[self.name][0].eval(as_values=True, **eval_kwargs)


class EvalComponent(EvalString):
    def __init__(self, tokens: pp.ParseResults) -> None:
        """
        Parse action to process successfully parsed expression components of the form
        `$component`.

        Args:
            tokens (pp.ParseResults):
                Has one parsed element containing the component name (str).
        """
        self.name: str = tokens[0]
        self.values = tokens

    def __repr__(self) -> str:
        "Return string representation of the parsed grammar"
        return "COMPONENT:" + str(self.name)

    def eval(
        self,
        component_dict: Optional[dict[str, pp.ParseResults]] = None,
        **eval_kwargs,
    ) -> Any:
        """
        Args:
            component_dict (Optional[dict[str, pp.ParseResults]]):
                Dictionary mapping the component name to a parsed equation expression.
                Default is None.

        Returns:
            Any: If component_expressions dictionary is given, find the expression matching
            the component name and evaluate it.
            If not given, return a dictionary giving the component name.
        """
        if eval_kwargs.get("as_dict"):
            return {"component": self.name}
        elif component_dict is not None:
            return component_dict[self.name][0].eval(**eval_kwargs)


class EvalUnslicedParameterOrVariable(EvalString):
    def __init__(self, tokens: pp.ParseResults) -> None:
        """
        Parse action to process successfully parsed unsliced parameters or decision variables
        of the form `param_or_var`.

        Args:
            tokens (pp.ParseResults):
                Has one parsed element containing the paramater/variable name (str).
        """
        self.name: str = tokens[0]
        self.values = tokens

    def __repr__(self) -> str:
        "Return string representation of the parsed grammar"
        return "PARAM_OR_VAR:" + str(self.name)

    def eval(
        self,
        references: set,
        as_dict: bool = False,
        as_values: bool = False,
        **eval_kwargs,
    ) -> Optional[Union[dict, xr.DataArray]]:
        """
        Args:
            references (set):
            as_dict (bool, optional):
            as_values(bool, optional):
                If True, return values rather than backend objects
        Returns:
            Optional[Union[dict, xr.DataArray]]:
                If `eval_kwargs["as_dict"]` is True, returns separated key:val pairs for parameter/variable name and index items;
                else, `eval_kwargs` has a backend dataset, returns sliced xarray object;
                else, returns None.
        """
        references.add(self.name)
        if as_dict:
            return {"param_or_var_name": self.name}
        elif eval_kwargs.get("backend_dataset", None) is not None:
            if as_values:
                return eval_kwargs["backend_interface"].get_parameter(
                    self.name, as_backend_objs=False
                )
            else:
                return eval_kwargs["backend_dataset"][self.name]


class EvalNumber(EvalString):
    def __init__(self, tokens: pp.ParseResults) -> None:
        """
        Parse action to process successfully parsed numbers described as integers (1),
        floats (1.), or in scientific notation (1e1). Also capture infinity (inf/.inf).

        Args:
            tokens (pp.ParseResults):
                Has one parsed element containing the number (str).
        """
        self.value = tokens[0]
        self.values = tokens

    def __repr__(self) -> str:
        "Return string representation of the parsed grammar"
        return "NUM:" + str(self.value)

    def eval(self, **eval_kwargs) -> float:
        """
        Returns:
            float: Input string as a float, even if given as an integer.
        """
        return float(self.value)


class StringListParser(EvalString):
    def __init__(self, tokens: pp.ParseResults) -> None:
        """
        Parse action to process successfully parsed lists of generic strings.
        This is required since we call "eval()" on all elements of the where string,
        so lists of strings need to be evaluatable as a whole "package".

        Args:
            tokens (pp.ParseResults): a list of parsed string elements.
        """
        self.val = tokens

    def __repr__(self) -> str:
        "Return string representation of the parsed grammar"
        return f"{self.val}"

    def eval(self, **kwargs) -> list[str]:
        "Return input as list of strings."

        return [val.eval() for val in self.val]


class GenericStringParser(EvalString):
    def __init__(self, tokens: pp.ParseResults) -> None:
        """
        Parse action to process successfully parsed generic strings.
        This is required since we call "eval()" on all elements of the where string,
        so even arbitrary strings (used in comparison operations) need to be evaluatable.

        Args:
            tokens (pp.ParseResults): Has one parsed element: string name (str).
        """
        self.val = tokens[0]

    def __repr__(self) -> str:
        "Return string representation of the parsed grammar"
        return f"STRING:{self.val}"

    def eval(self, **kwargs) -> str:
        "Return input as string."
        return str(self.val)


def helper_function_parser(
    *args: pp.ParserElement,
    generic_identifier: pp.ParserElement,
    allow_function_in_function: bool = False,
) -> pp.ParserElement:
    """
    Parsing grammar to process helper functions of the form `helper_function(*args, **eval_kwargs)`.

    Helper functions can accept other parser elements as arguments,
    i.e., components, parameters or variables, numbers, and other functions.

    Available helper functions are predefined in calliope.backend.helper_functions.

    Calling an unavailable helper will lead to a raised exception on evaluating the
    parsed element.

    Based partially on: # https://stackoverflow.com/questions/61807705/pyparsing-generic-python-function-args-and-kwargs

    Args (pp.ParserElement):
        Parser elements that can be arguments in the function (e.g., "number", "sliced_param_or_var").
        NOTE: the order of inclusion in the args list matters. The parser will parse based on first matches.

    Kwargs
        generic_identifier (pp.ParserElement):
            Parser for valid python variables without leading underscore and not called "inf".
            This parser has no parse action.
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
    allowed_parser_elements_in_args = list(args)
    lpar = pp.Suppress("(")
    rpar = pp.Suppress(")")

    helper_function_name = generic_identifier.set_results_name("helper_function_name")
    helper_function_name.set_parse_action(EvalHelperFuncName)

    if allow_function_in_function:
        allowed_parser_elements_in_args.insert(0, helper_function)

    arg_values = pp.MatchFirst(allowed_parser_elements_in_args) + pp.NotAny("=")

    # define function arguments
    arglist = pp.delimited_list(arg_values.copy())
    args_ = pp.Group(arglist).set_results_name("args")

    # define function keyword arguments
    key = generic_identifier + pp.Suppress("=")
    kwarglist = pp.delimited_list(pp.dict_of(key, arg_values))
    kwargs_ = pp.Group(kwarglist).set_results_name("kwargs")

    # build generic function
    helper_func_args = args_ + pp.Suppress(",") + kwargs_ | pp.Opt(
        args_, default=[]
    ) + pp.Opt(kwargs_, default={})
    helper_function << (
        pp.Combine(helper_function_name + lpar) + helper_func_args + rpar
    )

    helper_function.set_parse_action(EvalFunction)

    return helper_function


def sliced_param_or_var_parser(
    number: pp.ParserElement,
    generic_identifier: pp.ParserElement,
    evaluatable_identifier: pp.ParserElement,
    unsliced_object: pp.ParserElement,
    allow_slice_references: bool = True,
) -> pp.ParserElement:
    f"""
    Parsing grammar to process strings representing sliced model parameters or variables,
    e.g. "resource[node, tech]".

    If a parameter, must be a data variable in the Model._model_data xarray dataset.

    If a variable, must be an optimisation problem decision variable.

    The parser will not verify whether it has parsed a valid parameter or variable until
    evaluation.

    Args:
        number (pp.ParserElement):
            Parser for numbers (integer, float, scientific notation, "inf"/".inf").
        generic_identifier (pp.ParserElement):
            Parser for valid python variables without leading underscore and not called "inf".
            This parser has no parse action.
        evaluatable_identifier (pp.ParserElement):
            Parser for valid python variables without leading underscore and not called "inf".
            Evaluates to a string.
        unsliced_object (pp.ParserElement):
            Parser for valid backend objects.
            On evaluation, this parser will access the backend object from the backend dataset.
        allow_slice_references (bool):
            If True, allow reference to `index_slice` expressions
            (e.g. `{COMPONENT_CLASSIFIER}bar` in `foo[bars={COMPONENT_CLASSIFIER}bar]`).
            Defaults to True.

    Returns:
        pp.ParserElement:
            Parser which returns a dictionary with name of parameter/variable and list
            of index items as separate entries on.
    """

    lspar = pp.Suppress("[")
    rspar = pp.Suppress("]")

    direct_slicer = number | evaluatable_identifier
    if allow_slice_references:
        slicer_ref = pp.Suppress(COMPONENT_CLASSIFIER) + generic_identifier
        slicer_ref.set_parse_action(EvalIndexSlice)
        slicer = (slicer_ref | direct_slicer)("slicer")
    else:
        slicer = direct_slicer("slicer")

    index_slice = pp.Group(generic_identifier("set_name") + pp.Suppress("=") + slicer)

    index_slices = pp.Group(pp.delimited_list(index_slice))("index_slices")
    sliced_object_name = unsliced_object("param_or_var_name")

    sliced_param_or_var = pp.Combine(sliced_object_name + lspar) + index_slices + rspar
    sliced_param_or_var.set_parse_action(EvalSlicedParameterOrVariable)

    return sliced_param_or_var


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


def unsliced_object_parser(valid_object_names: Iterable[str]) -> pp.ParserElement:
    """
    Create a copy of the generic identifier and set a parse action to find the string in
    the list of input paramaters or optimisation decision variables.

    Args:
        valid_object_names (Iterable[str]): A
            All backend object names, to ensure they are captured by this parser function.

    Returns:
        pp.ParserElement:
            Copy of input parser with added parse action to lookup an unsliced
            parameter/variable value
    """

    unsliced_param_or_var = pp.one_of(valid_object_names, as_keyword=True)
    unsliced_param_or_var.set_parse_action(EvalUnslicedParameterOrVariable)

    return unsliced_param_or_var


def evaluatable_identifier_parser(
    identifier: pp.ParserElement, valid_object_names: Iterable
) -> tuple[pp.ParserElement, pp.ParserElement]:
    """
    Create an evaluatable copy of the generic identifier that will return a string or a
    list of strings.

    Args:
        identifier (pp.ParserElement):
            Parser for valid python variables without leading underscore and not called "inf".
            This parser has no parse action.
        valid_object_names (Iterable[str]): A
            All backend object names, to ensure they are *not* captured by this parser function.

    Returns:
        evaluatable_identifier (pp.ParserElement):
            Parser for valid python variables without leading underscore and not called "inf".
            Evaluates to a string.
        id_list (pp.ParserElement):
            Parser for lists of "evaluatable_identifier", bound by "[]" parentheses
    """
    evaluatable_identifier = (
        ~pp.one_of(valid_object_names, as_keyword=True) + identifier
    ).set_parse_action(GenericStringParser)

    id_list = (
        pp.Suppress("[") + pp.delimited_list(evaluatable_identifier) + pp.Suppress("]")
    )
    id_list.set_parse_action(StringListParser)

    return evaluatable_identifier, id_list


def setup_base_parser_elements() -> tuple[pp.ParserElement, pp.ParserElement]:
    """
    Setup parser elements that will be components of other parsers.

    Returns:
        number (pp.ParserElement):
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


def arithmetic_parser(
    *args,
    arithmetic: Optional[pp.Forward] = None,
) -> pp.Forward:
    """
    Parsing grammar to combine equation elements using basic arithmetic (+, -, *, /, **).
    Can handle the difference between a sign (e.g., -1,+1) and a addition/subtraction (0 - 1, 0 + 1).

    Whitespace is ignored on parsing (i.e., "1+1+foo" is equivalent to "1 + 1 + foo").

    Args:
        helper_function (pp.ParserElement):
            Parsing grammar to process helper functions of the form `helper_function(*args, **eval_kwargs)`.
        sliced_param_or_var (pp.ParserElement):
            Parser for sliced parameters or variables, e.g. "foo[bar]"
        component (pp.ParserElement):
            Parser for constraint components, e.g. "$foo"
        unsliced_param_or_var (pp.ParserElement):
            Parser for unsliced parameters or variables, e.g. "foo"
        number (pp.ParserElement):
            Parser for numbers (integer, float, scientific notation, "inf"/".inf").
    Kwargs:
        arithmetic (Optional[pp.Forward]):
            If given, will add arithmetic rules to this existing parsing rule.
            Defaults to None (i.e., arithmetic rules will be a newly generated rule).
    Returns:
        pp.ParserElement:
            Parser for strings which use arithmetic operations to combine other parser elements.
    """
    signop = pp.one_of(["+", "-"])
    multop = pp.one_of(["*", "/"])
    expop = pp.Literal("**")
    if arithmetic is None:
        arithmetic = pp.Forward()

    arithmetic <<= pp.infixNotation(
        # the order matters if two could capture the same string, e.g. "inf".
        pp.MatchFirst(args),
        [
            (signop, 1, pp.opAssoc.RIGHT, EvalSignOp),
            (expop, 2, pp.opAssoc.LEFT, EvalOperatorOperand),
            (multop, 2, pp.opAssoc.LEFT, EvalOperatorOperand),
            (signop, 2, pp.opAssoc.LEFT, EvalOperatorOperand),
        ],
    )

    return arithmetic


def equation_comparison_parser(arithmetic: pp.ParserElement) -> pp.ParserElement:
    """
    Parsing grammar to combine equation elements either side of a comparison operator
    (<= >= ==).

    Whitespace is ignored on parsing (i.e., "1+foo==$bar" is equivalent to "1 + 1 == $bar").

    Args:
        arithmetic (pp.ParserElement):
            Parser for arithmetic operations to combine other parser elements.

    Returns:
        pp.ParserElement:
            Parser for strings of the form "LHS OPERATOR RHS".
    """

    comparison_operators = pp.one_of(["<=", ">=", "=="])
    equation_comparison = arithmetic + comparison_operators + arithmetic
    equation_comparison.set_parse_action(EvalComparisonOp)

    return equation_comparison


def generate_index_slice_parser(valid_object_names: Iterable) -> pp.ParserElement:
    """
    Create parser for index slice reference expressions. These expressions are linked
    to the equation expression by e.g. `$bar` in `foo[bars=$bar]`.
    Unlike component and equation expressions, these strings cannot contain arithemtic
    nor references to equation components.

    Args:
        valid_object_names (Iterable):
            Allowed names for optimisation problem components (parameters, decision variables, expressions),
            to allow the parser to separate these from generic strings.

    Returns:
        pp.ParserElement: Parser for expression strings under the constraint key "index_slices".
    """
    number, identifier = setup_base_parser_elements()
    evaluatable_identifier, id_list = evaluatable_identifier_parser(
        identifier, valid_object_names
    )
    unsliced_param = unsliced_object_parser(valid_object_names)
    sliced_param = sliced_param_or_var_parser(
        number,
        identifier,
        evaluatable_identifier,
        unsliced_param,
        allow_slice_references=False,
    )

    helper_function = helper_function_parser(
        sliced_param,
        unsliced_param,
        number,
        id_list,
        evaluatable_identifier,
        generic_identifier=identifier,
        allow_function_in_function=True,
    )

    return (
        helper_function
        | sliced_param
        | unsliced_param
        | number
        | id_list
        | evaluatable_identifier
    )


def generate_component_parser(valid_object_names: Iterable) -> pp.Forward:
    """
    Create parser for equation component reference expressions. These expressions are linked
    to the equation expression by e.g. `$bar`.
    This parser allows arbitrarily nested arithmetic and function calls (and arithmetic inside function calls)
    and reference to index slice expressions.

    Args:
        valid_object_names (Iterable):
            Allowed names for optimisation problem components (parameters, decision variables, expressions),
            to allow the parser to separate these from generic strings.

    Returns:
        pp.ParserElement: Parser for expression strings under the constraint key "components".
    """
    number, identifier = setup_base_parser_elements()
    evaluatable_identifier, id_list = evaluatable_identifier_parser(
        identifier, valid_object_names
    )
    unsliced_param = unsliced_object_parser(valid_object_names)
    sliced_param = sliced_param_or_var_parser(
        number, identifier, evaluatable_identifier, unsliced_param
    )

    arithmetic = pp.Forward()
    helper_function = helper_function_parser(
        arithmetic,
        id_list,
        evaluatable_identifier,
        generic_identifier=identifier,
    )
    arithmetic = arithmetic_parser(
        helper_function,
        sliced_param,
        number,
        unsliced_param,
        arithmetic=arithmetic,
    )
    return arithmetic


def generate_arithmetic_parser(valid_object_names: Iterable) -> pp.ParserElement:
    """
    Create parser for left-/right-hand side (LHS/RHS) of equation expressions of the form LHS OPERATOR RHS (e.g. `foo == 1 + bar`).
    This parser allows arbitrarily nested arithmetic and function calls (and arithmetic inside function calls)
    and reference to component and index slice expressions.

    Args:
        valid_object_names (Iterable):
            Allowed names for optimisation problem components (parameters, decision variables, expressions),
            to allow the parser to separate these from generic strings.

    Returns:
        pp.ParserElement: Partial parser for expression strings under the constraint key "equation/equations".
    """
    number, identifier = setup_base_parser_elements()
    evaluatable_identifier, id_list = evaluatable_identifier_parser(
        identifier, valid_object_names
    )
    unsliced_param = unsliced_object_parser(valid_object_names)
    sliced_param = sliced_param_or_var_parser(
        number, identifier, evaluatable_identifier, unsliced_param
    )
    component = component_parser(identifier)

    arithmetic = pp.Forward()
    helper_function = helper_function_parser(
        arithmetic, id_list, evaluatable_identifier, generic_identifier=identifier
    )
    arithmetic = arithmetic_parser(
        helper_function,
        component,
        sliced_param,
        number,
        unsliced_param,
        arithmetic=arithmetic,
    )

    return arithmetic


def generate_equation_parser(valid_object_names: Iterable) -> pp.ParserElement:
    """
    Create parser for equation expressions of the form LHS OPERATOR RHS (e.g. `foo == 1 + bar`).
    This parser allows arbitrarily nested arithmetic and function calls (and arithmetic inside function calls)
    and reference to component and index slice expressions.

    Args:
        valid_object_names (Iterable):
            Allowed names for optimisation problem components (parameters, decision variables, expressions),
            to allow the parser to separate these from generic strings.

    Returns:
        pp.ParserElement: Parser for expression strings under the constraint key "equation/equations".
    """

    arithmetic = generate_arithmetic_parser(valid_object_names)
    equation_comparison = equation_comparison_parser(arithmetic)

    return equation_comparison
