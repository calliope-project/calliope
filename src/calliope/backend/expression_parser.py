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
"""Expression parsing functionality."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
import pandas as pd
import pyparsing as pp
import xarray as xr
from typing_extensions import NotRequired, TypedDict, Unpack

from calliope.backend.helper_functions import ParsingHelperFunction
from calliope.exceptions import BackendError
from calliope.schemas.math_schema import CalliopeBuildMath

if TYPE_CHECKING:
    from calliope.backend.backend_model import BackendModel

pp.ParserElement.enablePackrat()

SUB_EXPRESSION_CLASSIFIER = "$"


class EvalAttrs(TypedDict):
    """Attribute checker class."""

    equation_name: str
    where_array: xr.DataArray
    slice_dict: dict
    sub_expression_dict: dict
    backend_interface: BackendModel
    math: CalliopeBuildMath
    input_data: xr.DataArray
    references: set[str]
    helper_functions: dict[str, Callable]
    as_values: NotRequired[bool]


RETURN_T = Literal["array", "math_string"]


class EvalString(ABC):
    """Parent class for all string evaluation classes - used in type hinting."""

    name: str
    eval_attrs: EvalAttrs
    instring: str

    def __eq__(self, other):
        """Functionality for '==' operations."""
        return self.__repr__() == other

    @abstractmethod
    def __repr__(self) -> str:
        """Return string representation of the parsed grammar."""

    def error_msg(self, message: str) -> BackendError:
        """Raise an error message with context."""
        return BackendError(
            f"({self.eval_attrs['equation_name']}, {self.instring}) | {message}"
        )


class EvalToArrayStr(EvalString):
    """Evaluation for string arrays."""

    @abstractmethod
    def as_math_string(self) -> str:
        """Evaluate and return expression as LaTeX."""

    @abstractmethod
    def as_array(self) -> xr.DataArray | list[str | float]:
        """Evaluate and return expression as a DataArray or list."""

    # Math strings evaluate to strings.
    @overload
    def eval(self, return_type: Literal["math_string"], **eval_kwargs) -> str: ...

    # Arrays evaluate to arrays
    @overload
    def eval(
        self, return_type: Literal["array"], **eval_kwargs
    ) -> xr.DataArray | list[str | float]: ...

    def eval(
        self, return_type: RETURN_T, **eval_kwargs
    ) -> str | list[str | float] | xr.DataArray:
        """Evaluate math string expression.

        Args:
            return_type (Literal[math_string, input, array]):
                Dictates how the expression should be evaluated (see `Returns` section).
            **eval_kwargs: arbitrary keyword arguments.

        Keyword Args:
            equation_name (str): Name of math component in which expression is defined.
            slice_dict (dict): Dictionary mapping the index slice name to a parsed equation expression.
            sub_expression_dict (dict): Dictionary mapping the sub-expression name to a parsed equation expression.
            backend_interface (backend_model.BackendModel): Interface to optimisation backend.
            input_data (xr.Dataset): Input data arrays.
            where_array (xr.DataArray): boolean array with which to mask evaluated expressions.
            references (set): any references in the math string to other model components.
            helper_functions (dict[str, type[ParsingHelperFunction]]): Dictionary of allowed helper functions.
            as_values (bool, optional): Return array as numeric values, not backend objects. Defaults to False.

        Returns:
            str | list[str | float] | xr.DataArray:
                If `math_string` is desired, returns a valid LaTex math string.
                If `array` is desired, returns xarray DataArray or a list of strings/numbers (if the expression represents a list).
        """
        self.eval_attrs = eval_kwargs
        if return_type == "array":
            evaluated = self.as_array()
        elif return_type == "math_string":
            evaluated = self.as_math_string()
        return evaluated


class EvalToCallable(EvalString):
    """Parent class for callable functionality."""

    @abstractmethod
    def as_callable(self, return_type: RETURN_T) -> Callable:
        """Callable processing."""
        ...

    def eval(
        self,
        return_type: RETURN_T,
        **eval_kwargs: Unpack[EvalAttrs],  # type: ignore
    ) -> Callable:
        """Evaluate math string expression.

        Args:
            return_type (str): Whether to return a math string or xarray DataArray.
            **eval_kwargs: passed on directly to helper function..

        Keyword Args:
            equation_name (str): Name of math component in which expression is defined.
            slice_dict (dict): Dictionary mapping the index slice name to a parsed equation expression.
            sub_expression_dict (dict): Dictionary mapping the sub-expression name to a parsed equation expression.
            backend_interface (backend_model.BackendModel): Interface to optimisation backend.
            input_data (xr.Dataset): Input data arrays.
            where_array (xr.DataArray): boolean array with which to mask evaluated expressions.
            references (set): any references in the math string to other model components.
            helper_functions (dict[str, type[ParsingHelperFunction]]): Dictionary of allowed helper functions.
            as_values (bool, optional): Return array as numeric values, not backend objects. Defaults to False.

        Returns:
            Callable: returns helper function.
        """
        self.eval_attrs = eval_kwargs
        evaluated = self.as_callable(return_type)
        return evaluated


class EvalOperatorOperand(EvalToArrayStr):
    """Evaluation of math operands."""

    LATEX_OPERATOR_LOOKUP: dict[str, str] = {
        "**": "{val}^{{{operand}}}",
        "*": r"{val} \times {operand}",
        "/": r"\frac{{ {val} }}{{ {operand} }}",
        "+": "{val} + {operand}",
        "-": "{val} - {operand}",
    }
    SKIP_IF: list[str] = ["+", "-"]

    def __init__(self, instring: str, loc: int, tokens: pp.ParseResults) -> None:
        """Process successfully parsed expressions with operands separated by an operator.

        I.e.: OPERAND OPERATOR OPERAND OPERATOR OPERAND ...

        Args:
            instring (str): String that was parsed (used in error message).
            loc (int):
                Location in parsed string where parsing error was logged.
                This is not used; we include it as pyparsing injects it alongside `instring` when setting the parse action.
            tokens (pp.ParseResults):
                Contains a list of the form [operand (pp.ParseResults), operator (str),
                operand (pp.ParseResults), operator (str), ...].
        """
        self.value: pp.ParseResults = tokens[0]
        self.values = tokens
        self.instring = instring

    def __repr__(self) -> str:
        """Programming / official string representation."""
        first_operand = self.value[0].__repr__()
        operand_operator_pairs = " ".join(
            op + " " + val.__repr__()
            for op, val in self._operator_operands(self.value[1:])
        )
        arithmetic_string = f"({first_operand} {operand_operator_pairs})"
        return arithmetic_string

    def _operator_operands(
        self, token_list: list
    ) -> Iterator[tuple[str, pp.ParseResults]]:
        """Generator to extract operators and operands in pairs."""
        it = iter(token_list)
        while 1:
            try:
                yield (next(it), next(it))
            except StopIteration:
                break

    def _apply_where_array(self, evaluated: xr.DataArray) -> xr.DataArray:
        """Util function to apply where arrays to non-latex strings."""
        where_array = self.eval_attrs["where_array"]
        try:
            evaluated = evaluated.where(where_array)
        except AttributeError:
            evaluated = evaluated.broadcast_like(where_array).where(where_array)

        return evaluated

    def _skip_component_on_conditional(self, component: str, operator_: str) -> bool:
        """Conditional to skip adding to math string if element evaluates to zero.

        E.g., "0 + flow_cap" is better evaluated as simply "flow_cap".
        """
        return component == "0" and operator_ in self.SKIP_IF

    @staticmethod
    def _operate(
        val: xr.DataArray, evaluated_operand: xr.DataArray, operator_: str
    ) -> xr.DataArray:
        """Apply evaluated operation on two DataArrays."""
        match operator_:
            case "**":
                val = val**evaluated_operand
            case "*":
                val = val * evaluated_operand
            case "/":
                val = val / evaluated_operand
            case "+":
                val = val + evaluated_operand
            case "-":
                val = val - evaluated_operand
        return val

    def as_math_string(self) -> str:  # noqa: D102, override
        val = self.value[0].eval(return_type="math_string", **self.eval_attrs)

        for operator_, operand in self._operator_operands(self.value[1:]):
            evaluated_operand = operand.eval(
                return_type="math_string", **self.eval_attrs
            )
            # We ignore zeros that do nothing
            if self._skip_component_on_conditional(evaluated_operand, operator_):
                continue
            if isinstance(self.value[0], type(self)):
                val = "(" + val + ")"
            if isinstance(operand, type(self)):
                evaluated_operand = "(" + evaluated_operand + ")"
            if self._skip_component_on_conditional(val, operator_):
                val = evaluated_operand
            else:
                val = self.LATEX_OPERATOR_LOOKUP[operator_].format(
                    val=val, operand=evaluated_operand
                )
        return val

    def as_array(self) -> xr.DataArray:  # noqa: D102, override
        val = self._apply_where_array(
            self.value[0].eval(return_type="array", **self.eval_attrs)
        )

        for operator_, operand in self._operator_operands(self.value[1:]):
            evaluated_operand = self._apply_where_array(
                operand.eval(return_type="array", **self.eval_attrs)
            )
            val = self._operate(val, evaluated_operand, operator_)
        return val


class EvalSignOp(EvalToArrayStr):
    """Class for processing expressions with + or -."""

    def __init__(self, instring: str, loc: int, tokens: pp.ParseResults) -> None:
        """Parse action to process successfully parsed expressions with a leading + or - sign.

        Args:
            instring (str): String that was parsed (used in error message).
            loc (int):
                Location in parsed string where parsing error was logged.
                This is not used; we include it as pyparsing injects it alongside `instring` when setting the parse action.
            tokens (pp.ParseResults):
                Contains a list of the form [sign (str), operand (pp.ParseResults)].
        """
        self.sign, self.value = tokens[0]
        self.values = tokens
        self.instring = instring

    def __repr__(self) -> str:
        """Programming / official string representation."""
        return str(f"({self.sign}){self.value.__repr__()}")

    # string return
    @overload
    def _eval(self, return_type: Literal["math_string"]) -> str: ...

    # array return
    @overload
    def _eval(self, return_type: Literal["array"]) -> xr.DataArray: ...

    def _eval(self, return_type: RETURN_T) -> xr.DataArray | str:
        """Evaluate the element that will have the sign attached to it."""
        return self.value.eval(return_type, **self.eval_attrs)

    def as_math_string(self) -> str:  # noqa: D102
        return self.sign + self._eval("math_string")

    def as_array(self) -> xr.DataArray:  # noqa: D102, override
        evaluated = self._eval("array")
        if self.sign == "-":
            evaluated = -1 * evaluated
        return evaluated


class EvalComparisonOp(EvalToArrayStr):
    """Class for processing comparison operations."""

    OP_TRANSLATOR = {"<=": r" \leq ", ">=": r" \geq ", "==": " = "}

    def __init__(self, instring: str, loc: int, tokens: pp.ParseResults) -> None:
        """Parse action to process successfully parsed equations of the form LHS OPERATOR RHS.

        Args:
            instring (str): String that was parsed (used in error message).
            loc (int):
                Location in parsed string where parsing error was logged.
                This is not used; we include it as pyparsing injects it alongside `instring` when setting the parse action.
            tokens (pp.ParseResults):
                Contains a list with an RHS (pp.ParseResults), operator (str), and LHS (pp.ParseResults).
        """
        self.lhs, self.op, self.rhs = tokens
        self.instring = instring
        self.loc = loc
        self.values = tokens

    def __repr__(self) -> str:
        """Programming / official string representation."""
        return f"{self.lhs.__repr__()} {self.op} {self.rhs.__repr__()}"

    # string return
    @overload
    def _eval(self, return_type: Literal["math_string"]) -> tuple[str, str]: ...

    # array return
    @overload
    def _eval(
        self, return_type: Literal["array"]
    ) -> tuple[xr.DataArray, xr.DataArray]: ...

    def _eval(
        self, return_type: RETURN_T
    ) -> tuple[str, str] | tuple[xr.DataArray, xr.DataArray]:
        """Evaluate the LHS and RHS of the comparison."""
        lhs = self.lhs.eval(return_type, **self.eval_attrs)
        rhs = self.rhs.eval(return_type, **self.eval_attrs)
        return lhs, rhs

    def as_math_string(self) -> str:  # noqa: D102, override
        lhs, rhs = self._eval("math_string")
        return lhs + self.OP_TRANSLATOR[self.op] + rhs

    def as_array(self) -> xr.DataArray:  # noqa: D102, override
        lhs, rhs = self._eval("array")
        where = self.eval_attrs["where_array"]
        for side, arr in {"left": lhs, "right": rhs}.items():
            extra_dims = set(arr.dims).difference(set(where.dims))
            if extra_dims:
                raise self.error_msg(
                    f"The {side}-hand side of the equation is indexed over dimensions not present in `foreach`: {extra_dims}"
                )
        lhs_where = lhs.broadcast_like(where)
        rhs_where = rhs.broadcast_like(where)

        match self.op:
            case "==":
                op = np.equal
            case "<=":
                op = np.less_equal
            case ">=":
                op = np.greater_equal
        constraint = op(lhs_where, rhs_where, where=where.values, dtype=np.object_)
        return xr.DataArray(constraint)


class EvalFunction(EvalToArrayStr):
    """Class to process parsed functions."""

    def __init__(self, instring: str, loc: int, tokens: pp.ParseResults) -> None:
        """Parse action to process successfully parsed helper function strings.

        Strings must be in the following form: helper_function_name(*args, **eval_kwargs).

        Args:
            instring (str): String that was parsed (used in error message).
            loc (int):
                Location in parsed string where parsing error was logged.
                This is not used; we include it as pyparsing injects it alongside `instring` when setting the parse action.
            tokens (pp.ParseResults):
                Has a dictionary component with the parsed elements:
                helper_function_name (pp.ParseResults), args (list), kwargs (dict).
        """
        token_dict = tokens.as_dict()
        self.func_name: pp.ParseResults = token_dict["helper_function_name"]
        self.args: list = token_dict["args"]
        self.kwargs: dict = token_dict["kwargs"]
        self.values = tokens
        self.instring = instring

    def __repr__(self) -> str:
        """Programming / official string representation."""
        _kwargs = ", ".join(f"{k}={v}" for k, v in self.kwargs.items())
        return f"{str(self.func_name)}(args={self.args}, kwargs={{{_kwargs}}})"

    @overload
    def _arg_eval(self, return_type: Literal["math_string"], arg: Any) -> str: ...

    @overload
    def _arg_eval(
        self, return_type: Literal["array"], arg: Any
    ) -> xr.DataArray | list[str | float]: ...

    def _arg_eval(
        self, return_type: RETURN_T, arg: Any
    ) -> str | xr.DataArray | list[str | float]:
        """Evaluate the arguments of the helper function."""
        if isinstance(arg, pp.ParseResults):
            evaluated = arg[0].eval(return_type, **self.eval_attrs)
        elif isinstance(arg, list):
            evaluated = [self._arg_eval(return_type, arg_) for arg_ in arg]
        elif isinstance(arg, ListParser):
            evaluated = arg.eval("array", **self.eval_attrs)
        else:
            evaluated = arg.eval(return_type, **self.eval_attrs)
        if isinstance(evaluated, xr.DataArray) and isinstance(arg, EvalGenericString):
            evaluated = evaluated.item()
        return evaluated

    @overload
    def _eval(self, return_type: Literal["math_string"]) -> str: ...

    @overload
    def _eval(self, return_type: Literal["array"]) -> xr.DataArray: ...

    def _eval(self, return_type: RETURN_T) -> str | xr.DataArray:
        """Pass evaluated arguments to evaluated helper function."""
        helper_function = self.func_name.eval(return_type, **self.eval_attrs)
        if helper_function.ignore_where:
            self.eval_attrs["where_array"] = xr.DataArray(True)

        args_ = []
        for arg in self.args:
            args_.append(self._arg_eval(return_type, arg))

        kwargs_ = {}
        for kwarg_name, kwarg_val in self.kwargs.items():
            kwargs_[kwarg_name] = self._arg_eval(return_type, kwarg_val)

        evaluated = helper_function(*args_, **kwargs_)
        return evaluated

    def as_math_string(self) -> str:  # noqa: D102, override
        return self._eval("math_string")

    def as_array(self) -> xr.DataArray:  # noqa: D102, override
        return self._eval("array")


class EvalHelperFuncName(EvalToCallable):
    """For processing parsed helper function names."""

    def __init__(self, instring: str, loc: int, tokens: pp.ParseResults) -> None:
        """Parse action to process successfully parsed helper function names.

        This is a unique parse action so that we can catch invalid helper functions
        most safely.

        Args:
            instring (str): String that was parsed (used in error message).
            loc (int):
                Location in parsed string where parsing error was logged.
                This is not used; we include it as pyparsing injects it alongside `instring` when setting the parse action.
            tokens (pp.ParseResults):
                Has one parsed element: helper_function_name (str).
        """
        self.name = self.value = tokens[0]
        self.instring = instring
        self.loc = loc
        self.values = tokens

    def __repr__(self) -> str:
        """Programming / official string representation."""
        return str(self.name)

    def as_callable(self, return_type: RETURN_T) -> Callable:
        """Evalluate and return the callable action of the helper function."""
        helper_functions = self.eval_attrs["helper_functions"]
        if self.name not in helper_functions.keys():
            raise self.error_msg(f"Invalid helper function defined: {self.name}")
        elif not isinstance(helper_functions[self.name], type(ParsingHelperFunction)):
            raise self.error_msg(
                f"Helper function must be subclassed from calliope.backend.helper_functions.ParsingHelperFunction: {self.name}"
            )
        else:
            return helper_functions[self.name](return_type, **self.eval_attrs)


class EvalSlicedComponent(EvalToArrayStr):
    """For processing of sliced parameters / decision variables."""

    def __init__(self, instring: str, loc: int, tokens: pp.ParseResults) -> None:
        """Process successfully parsed sliced parameters or decision variables.

        In the form of param_or_var[*slices].

        Args:
            instring (str): String that was parsed (used in error message).
            loc (int):
                Location in parsed string where parsing error was logged.
                This is not used; we include it as pyparsing injects it alongside `instring` when setting the parse action.
            tokens (pp.ParseResults):
                Has a dictionary component with the parsed elements:
                param_or_var_name (str), slices (list of strings).
        """
        token_dict = tokens.as_dict()
        self.obj_name: pp.ParseResults = token_dict["param_or_var_name"]

        self.slices: dict[str, pp.ParseResults] = {
            idx["set_name"][0]: idx["slicer"][0] for idx in token_dict["slices"]
        }
        self.values = tokens
        self.instring = instring

    def __repr__(self) -> str:
        """Programming / official string representation."""
        slices = ", ".join(f"{k}={v.__repr__()}" for k, v in self.slices.items())
        return f"SLICED_{self.obj_name}[{slices}]"

    @staticmethod
    def _replace_rule(index_slices):
        """String parsing rule to catch and replace dimension names with the names + their slices.

        E.g., `techs` -> `techs=pv`.
        """

        def __replace(term):
            if len(term) == 1:
                return term
            else:
                replacers = {k: f"{k}={v}" for k, v in index_slices.items()}
                return (
                    term[0]
                    + term[1]
                    + ",".join(replacers.get(k, k) for k in term[2])
                    + term[3]
                )

        return __replace

    @overload
    def _eval(self, return_type: Literal["math_string"]) -> tuple[str, dict]: ...

    @overload
    def _eval(self, return_type: Literal["array"]) -> tuple[xr.DataArray, dict]: ...

    def _eval(self, return_type: RETURN_T) -> tuple[str | xr.DataArray, dict]:
        """Evaluate the slice dim and vals of each slice element."""
        slices: dict[str, Any] = {
            k: v.eval(return_type, **self.eval_attrs) for k, v in self.slices.items()
        }

        evaluated = self.obj_name.eval(return_type, **self.eval_attrs)
        return evaluated, slices

    def as_math_string(self) -> str:  # noqa: D102, override
        evaluated, slices = self._eval("math_string")
        singular_slice_refs = {
            self.eval_attrs["math"].dimensions[k].iterator: v for k, v in slices.items()
        }
        id_ = pp.Combine(
            pp.Word(pp.alphas, pp.alphanums)
            + pp.ZeroOrMore("_" + pp.Word(pp.alphanums))
            + pp.Opt("_")
        )
        id_formatted = pp.Combine("\\" + pp.Word(pp.alphas) + "{" + id_ + "}")
        obj_parser = id_formatted + pp.Opt(
            r"_\text{" + pp.Group(pp.DelimitedList(id_)) + "}"
        )
        obj_parser.set_parse_action(self._replace_rule(singular_slice_refs))
        return obj_parser.parse_string(evaluated, parse_all=True)[0]

    def as_array(self) -> xr.DataArray:  # noqa: D102, override
        evaluated, slices = self._eval("array")
        return evaluated.sel(**slices)


class EvalIndexSlice(EvalToArrayStr):
    """For processing `$slice` expressions."""

    def __init__(self, instring: str, loc: int, tokens: pp.ParseResults) -> None:
        """Process successfully parsed expression index `$slice` references.

        Args:
            instring (str): String that was parsed (used in error message).
            loc (int):
                Location in parsed string where parsing error was logged.
                This is not used; we include it as pyparsing injects it alongside `instring` when setting the parse action.
            tokens (pp.ParseResults):
                Has one parsed element containing the index slice name (str).
        """
        self.name: str = tokens[0]
        self.values = tokens
        self.instring = instring

    def __repr__(self) -> str:
        """Programming / official string representation."""
        return "REFERENCE:" + str(self.name)

    @overload
    def _eval(self, return_type: Literal["math_string"], as_values: bool) -> str: ...

    @overload
    def _eval(
        self, return_type: Literal["array"], as_values: bool
    ) -> xr.DataArray | list[str | float]: ...

    def _eval(
        self, return_type: RETURN_T, as_values: bool
    ) -> str | xr.DataArray | list[str | float]:
        """Evaluate the referenced `slice`."""
        self.eval_attrs["as_values"] = as_values
        return self.eval_attrs["slice_dict"][self.name][0].eval(
            return_type, **self.eval_attrs
        )

    def as_math_string(self) -> str:  # noqa: D102, override
        return self._eval("math_string", False)

    def as_array(self) -> xr.DataArray | list[str | float]:  # noqa: D102, override
        evaluated = self._eval("array", True)
        return evaluated


class EvalSubExpressions(EvalToArrayStr):
    """For processing sub-expressions."""

    def __init__(self, instring: str, loc: int, tokens: pp.ParseResults) -> None:
        """Process successfully parsed `$sub_expressions`.

        Args:
            instring (str): String that was parsed (used in error message).
            loc (int):
                Location in parsed string where parsing error was logged.
                This is not used; we include it as pyparsing injects it alongside `instring` when setting the parse action.
            tokens (pp.ParseResults):
                Has one parsed element containing the sub_expression name (str).
        """
        self.name: str = tokens[0]
        self.values = tokens
        self.instring = instring

    def __repr__(self) -> str:
        """Programming / official string representation."""
        return "SUB_EXPRESSION:" + str(self.name)

    @overload
    def _eval(self, return_type: Literal["math_string"]) -> str: ...

    @overload
    def _eval(self, return_type: Literal["array"]) -> xr.DataArray: ...

    def _eval(self, return_type: RETURN_T) -> str | xr.DataArray:
        """Evaluate the referenced sub_expression."""
        return self.eval_attrs["sub_expression_dict"][self.name][0].eval(
            return_type, **self.eval_attrs
        )

    def as_math_string(self) -> str:  # noqa: D102, override
        return self._eval("math_string")

    def as_array(self) -> xr.DataArray:  # noqa: D102, override
        return self._eval("array")


class EvalNumber(EvalToArrayStr):
    """For processing numbers."""

    def __init__(self, instring: str, loc: int, tokens: pp.ParseResults) -> None:
        """Process successfully parsed numbers.

        Catches integers (1), floats (1.), and in scientific notation (1e1).
        Also capture infinity (inf/.inf).

        Args:
            instring (str): String that was parsed (used in error message).
            loc (int):
                Location in parsed string where parsing error was logged.
                This is not used; we include it as pyparsing injects it alongside `instring` when setting the parse action.
            tokens (pp.ParseResults):
                Has one parsed element containing the number (str).
        """
        self.value = tokens[0]
        self.values = tokens
        self.instring = instring

    def __repr__(self) -> str:
        """Programming / official string representation."""
        return "NUM:" + str(self.value)

    def as_math_string(self) -> str:  # noqa: D102, override
        return re.sub(
            r"([\d]+?)e([+-])([\d]+)",
            r"\1\\mathord{\\times}10^{\2\3}",
            f"{float(self.value):.6g}",
        )

    def as_array(self) -> xr.DataArray:  # noqa: D102, override
        return xr.DataArray(float(self.value), attrs={"obj_type": "number"})


class ListParser(EvalToArrayStr):
    """For parsing lists."""

    def __init__(self, instring: str, loc: int, tokens: pp.ParseResults) -> None:
        """Process successfully parsed lists of generic strings.

        This is required since we call "eval()" on all elements of the where string,
        so lists of strings need to be evaluatable as a whole "package".

        Args:
            instring (str): String that was parsed (used in error message).
            loc (int):
                Location in parsed string where parsing error was logged.
                This is not used; we include it as pyparsing injects it alongside `instring` when setting the parse action.
            tokens (pp.ParseResults): a list of parsed string elements.
        """
        self.val = tokens
        self.instring = instring

    def __repr__(self) -> str:
        """Programming / official string representation."""
        return f"{self.val}"

    def as_math_string(self) -> str:  # noqa: D102, override
        input_list = self.as_array()
        return "[" + ",".join(str(i) for i in input_list) + "]"

    def as_array(self) -> list[str | float]:  # noqa: D102, override
        values = [val.eval("array", **self.eval_attrs) for val in self.val]
        # strings and numbers are returned as xarray arrays of size 1,
        # so we extract those values.
        return [da.item() if da.size == 1 else da.name for da in values]


class EvalUnslicedComponent(EvalToArrayStr):
    """Evaluation of unsliced components."""

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
        self.name = str(self.val)
        self.values = tokens
        self.instring = instring

    def __repr__(self) -> str:
        """Programming / official string representation."""
        return f"COMPONENT:{self.name}"

    def as_math_string(self) -> str:  # noqa: D102, override
        self.eval_attrs["as_values"] = False
        evaluated = self.as_array()
        self.eval_attrs["references"].add(self.name)

        if evaluated.attrs["obj_type"] != "string":
            data_var_string = evaluated.attrs["math_repr"]
        else:
            data_var_string = rf"\text{{{self.name}}}"

        return data_var_string

    def as_array(self) -> xr.DataArray:  # noqa: D102, override
        backend_interface = self.eval_attrs["backend_interface"]

        try:
            evaluated = backend_interface._dataset[self.name]
            if (
                self.eval_attrs.get("as_values", False)
                and evaluated.attrs["obj_type"] == "parameters"
            ):
                evaluated = backend_interface.get_parameter(
                    self.name, as_backend_objs=False
                )
        except KeyError:
            raise self.error_msg(
                f"Trying to access a math component that is not yet defined: {self.name}. "
                "If the referenced component is a global expression, set its `order` to have it defined first."
            )
        if pd.notna(evaluated.attrs["default"]):
            evaluated = evaluated.fillna(evaluated.attrs["default"])

        self.eval_attrs["references"].add(self.name)
        return evaluated


class EvalGenericString(EvalToArrayStr):
    """For generic string parsing."""

    def __init__(self, instring: str, loc: int, tokens: pp.ParseResults) -> None:
        """Process successfully parsed generic strings.

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
        """Programming / official string representation."""
        return f"STRING:{self.val}"

    def as_math_string(self):  # noqa: D102, override
        return str(self.val)

    def as_array(self) -> xr.DataArray:  # noqa: D102, override
        return xr.DataArray(str(self.val), attrs={"obj_type": "string"})


def helper_function_parser(
    *args: pp.ParserElement,
    generic_identifier: pp.ParserElement,
    allow_function_in_function: bool = False,
) -> pp.ParserElement:
    """Process helper functions of the form `helper_function(*args, **eval_kwargs)`.

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
    arglist = pp.DelimitedList(arg_values.copy())
    args_ = pp.Group(arglist).set_results_name("args")

    # define function keyword arguments
    key = generic_identifier + pp.Suppress("=")
    kwarg_list = pp.DelimitedList(pp.dict_of(key, arg_values))
    kwargs_ = pp.Group(kwarg_list).set_results_name("kwargs")

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
    """Process strings representing sliced model parameters or variables.

    E.g. "source_use_max[node, tech]".

    If a parameter, must be a data variable in the Model.inputs xarray dataset.

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
            If True, allow reference to `slice` expressions (e.g. `$bar` in `foo[bars=$bar]`).
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
        slicer_ref = pp.Suppress(SUB_EXPRESSION_CLASSIFIER) + generic_identifier
        slicer_ref.set_parse_action(EvalIndexSlice)
        slicer = (slicer_ref | direct_slicer)("slicer")
    else:
        slicer = direct_slicer("slicer")

    slice = pp.Group(generic_identifier("set_name") + pp.Suppress("=") + slicer)

    slices = pp.Group(pp.DelimitedList(slice))("slices")
    sliced_object_name = unsliced_object("param_or_var_name")

    sliced_param_or_var = pp.Combine(sliced_object_name + lspar) + slices + rspar
    sliced_param_or_var.set_parse_action(EvalSlicedComponent)

    return sliced_param_or_var


def sub_expression_parser(generic_identifier: pp.ParserElement) -> pp.ParserElement:
    """Parse strings prepended with the YAML constraint sub-expression classifier `$`.

    E.g. "$my_sub_expr"

    Args:
        generic_identifier (pp.ParserElement):
            Parser for valid python variables without leading underscore and not called "inf".
            This parser has no parse action.

    Returns:
        pp.ParserElement:
            Parser which produces a dictionary of the form {"sub_expression": "my_sub_expression"} on evaluation.
    """
    sub_expression = pp.Combine(
        pp.Suppress(SUB_EXPRESSION_CLASSIFIER) + generic_identifier
    )
    sub_expression.set_parse_action(EvalSubExpressions)

    return sub_expression


def unsliced_object_parser(valid_component_names: Iterable[str]) -> pp.ParserElement:
    """Parse unsliced objects and identify their corresponding parse actions.

    Creates a copy of the generic identifier and sets a parse action to find the string in
    the list of input parameters or optimisation decision variables.

    Args:
        valid_component_names (Iterable[str]): A
            All backend object names, to ensure they are captured by this parser function.

    Returns:
        pp.ParserElement:
            Copy of input parser with added parse action to lookup an unsliced
            parameter/variable value
    """
    unsliced_param_or_var = pp.one_of(valid_component_names, as_keyword=True)
    unsliced_param_or_var.set_parse_action(EvalUnslicedComponent)

    return unsliced_param_or_var


def evaluatable_identifier_parser(
    identifier: pp.ParserElement, valid_component_names: Iterable
) -> pp.ParserElement:
    """Create an evaluatable copy of the generic identifier that will return a string or a model component as an array.

    Args:
        identifier (pp.ParserElement):
            Parser for valid python variables without leading underscore and not called "inf".
            This parser has no parse action.
        valid_component_names (Iterable[str]): A
            All backend object names, to ensure they are *not* captured by this parser function.

    Returns:
        pp.ParserElement:
            Parser for valid python variables without leading underscore and not called "inf".
            Evaluates to a string or an array (if it is a model component).
    """
    evaluatable_identifier = (
        ~pp.one_of(valid_component_names, as_keyword=True) + identifier
    ).set_parse_action(EvalGenericString)

    return evaluatable_identifier


def list_parser(
    number: pp.ParserElement, evaluatable_identifier: pp.ParserElement
) -> pp.ParserElement:
    """Parse strings which define a list of other strings or numbers.

    Lists are defined as anything wrapped in square brackets (`[]`).

    Strings that could be evaluated as model component arrays will still be evaluated as strings (using the model component name).

    Args:
        number (pp.ParserElement): Parser for numbers (integer, float, scientific notation, "inf"/".inf").
        evaluatable_identifier (pp.ParserElement): Parser for valid python variables without leading underscore and not called "inf".

    Returns:
        pp.ParserElement: Parser for valid lists of strings and/or numbers.
    """
    list_elements = pp.MatchFirst([evaluatable_identifier, number])
    id_list = pp.Suppress("[") + pp.DelimitedList(list_elements) + pp.Suppress("]")
    id_list.set_parse_action(ListParser)
    return id_list


def setup_base_parser_elements() -> tuple[pp.ParserElement, pp.ParserElement]:
    """Setup parser elements that will be components of other parsers.

    Returns:
        tuple[pp.ParserElement, pp.ParserElement]: (number, generic_identifier)
            number: parser for numbers (integer, float, scentific notation, "inf"/".inf").
            generic_identifier: parser for valid python variables without leading
                underscore and not called "inf". This parser has no parse action.
    """
    inf_kw = pp.Combine(pp.Opt(pp.Suppress(".")) + pp.Keyword("inf", caseless=True))
    number = pp.pyparsing_common.number | inf_kw
    generic_identifier = ~inf_kw + pp.Word(pp.alphas, pp.alphanums + "_")

    number.set_parse_action(EvalNumber)

    return number, generic_identifier


def arithmetic_parser(*args, arithmetic: pp.Forward | None = None) -> pp.Forward:
    """Parsing grammar to combine equation elements using basic arithmetic (+, -, *, /, **).

    Can handle the difference between a sign (e.g., -1,+1) and a addition/subtraction (0 - 1, 0 + 1).
    Whitespace is ignored on parsing (i.e., "1+1+foo" is equivalent to "1 + 1 + foo").

    Args:
        *args: arguments in the form of a list. These can be:
            helper_function (pp.ParserElement): parsing grammar to process helper functions
                of the form `helper_function(*args, **eval_kwargs)`.
            sliced_param_or_var (pp.ParserElement): parser for sliced parameters or variables, e.g. "foo[bar]"
            sub_expression (pp.ParserElement): parser for constraint sub expressions, e.g. "$foo"
            unsliced_param_or_var (pp.ParserElement): parser for unsliced parameters or variables, e.g. "foo"
            number (pp.ParserElement): parser for numbers (integer, float, scientific notation, "inf"/".inf").
        arithmetic (pp.Forward | None, optional): If given, add arithmetic rules to this
            existing parsing rule (otherwise, arithmetic rules will be a newly generated rule).
            Defaults to None.

    Returns:
        pp.Forward: parser for strings which use arithmetic operations to combine other parser elements.
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
    """Parsing grammar to combine equation elements either side of a comparison operator (<= >= ==).

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


def generate_slice_parser(valid_component_names: Iterable) -> pp.ParserElement:
    """Create parser for index slice reference expressions.

    These expressions are linked to the equation expression by e.g. `$bar` in `foo[bars=$bar]`.
    Unlike sub-expressions and equation expressions, these strings cannot contain arithmetic
    nor references to sub expressions.

    Args:
        valid_component_names (Iterable):
            Allowed names for optimisation problem components (parameters, decision variables, expressions),
            to allow the parser to separate these from generic strings.

    Returns:
        pp.ParserElement: Parser for expression strings under the constraint key "slices".
    """
    number, identifier = setup_base_parser_elements()
    evaluatable_identifier = evaluatable_identifier_parser(
        identifier, valid_component_names
    )
    id_list = list_parser(number, evaluatable_identifier)
    unsliced_param = unsliced_object_parser(valid_component_names)
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


def generate_sub_expression_parser(valid_component_names: Iterable) -> pp.Forward:
    """Create parser for sub expressions.

    These expressions are linked to the equation expression by e.g. `$bar`.
    This parser allows arbitrarily nested arithmetic and function calls (and arithmetic inside function calls)
    and reference to index slice expressions.

    Args:
        valid_component_names (Iterable):
            Allowed names for optimisation problem components (parameters, decision variables, expressions),
            to allow the parser to separate these from generic strings.

    Returns:
        pp.ParserElement: Parser for expression strings under the constraint key "sub_expressions".
    """
    number, identifier = setup_base_parser_elements()
    evaluatable_identifier = evaluatable_identifier_parser(
        identifier, valid_component_names
    )
    id_list = list_parser(number, evaluatable_identifier)
    unsliced_param = unsliced_object_parser(valid_component_names)
    sliced_param = sliced_param_or_var_parser(
        number, identifier, evaluatable_identifier, unsliced_param
    )

    arithmetic = pp.Forward()
    helper_function = helper_function_parser(
        arithmetic, id_list, evaluatable_identifier, generic_identifier=identifier
    )
    arithmetic = arithmetic_parser(
        helper_function, sliced_param, number, unsliced_param, arithmetic=arithmetic
    )
    return arithmetic


def generate_arithmetic_parser(valid_component_names: Iterable) -> pp.ParserElement:
    """Create parser for arithmetic expressions (+, -, /, *, **).

    This parser allows arbitrarily nested arithmetic and function calls (and arithmetic inside function calls)
    and reference to sub-expressions and index slice expressions.

    Args:
        valid_component_names (Iterable):
            Allowed names for optimisation problem components (parameters, decision variables, global_expressions),
            to allow the parser to separate these from generic strings.

    Returns:
        pp.ParserElement: Partial parser for expression strings under the constraint key "equation/equations".
    """
    number, identifier = setup_base_parser_elements()
    evaluatable_identifier = evaluatable_identifier_parser(
        identifier, valid_component_names
    )
    id_list = list_parser(number, evaluatable_identifier)
    unsliced_param = unsliced_object_parser(valid_component_names)
    sliced_param = sliced_param_or_var_parser(
        number, identifier, evaluatable_identifier, unsliced_param
    )
    sub_expression = sub_expression_parser(identifier)

    arithmetic = pp.Forward()
    helper_function = helper_function_parser(
        arithmetic, id_list, evaluatable_identifier, generic_identifier=identifier
    )
    arithmetic = arithmetic_parser(
        helper_function,
        sub_expression,
        sliced_param,
        number,
        unsliced_param,
        arithmetic=arithmetic,
    )

    return arithmetic


def generate_equation_parser(valid_component_names: Iterable) -> pp.ParserElement:
    """Create parser for equation expressions of the form LHS OPERATOR RHS (e.g. `foo == 1 + bar`).

    This parser allows arbitrarily nested arithmetic and function calls (and arithmetic inside function calls)
    and reference to sub-expressions and index slice expressions.

    Args:
        valid_component_names (Iterable):
            Allowed names for optimisation problem components (parameters, decision variables, global_expressions),
            to allow the parser to separate these from generic strings.

    Returns:
        pp.ParserElement: Parser for expression strings under the constraint key "equation/equations".
    """
    arithmetic = generate_arithmetic_parser(valid_component_names)
    equation_comparison = equation_comparison_parser(arithmetic)

    return equation_comparison
