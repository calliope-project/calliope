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

import pyparsing as pp

pp.ParserElement.enablePackrat()


class EvalSignOp:
    "Class to evaluate expressions with a leading + or - sign"

    def __init__(self, tokens):
        self.sign, self.value = tokens[0]

    def __repr__(self):
        return str(f"{self.sign}_{self.value}")

    def eval(self):
        mult = {"+": 1, "-": -1}[self.sign]
        return mult * self.value.eval()


def operatorOperands(tokenlist):
    "Generator to extract operators and operands in pairs"

    it = iter(tokenlist)
    while 1:
        try:
            yield (next(it), next(it))
        except StopIteration:
            break


class EvalPowerOp:
    "Class to evaluate multiplication and division expressions"

    def __init__(self, tokens):
        self.value = tokens[0]

    def __repr__(self):
        return str(self.value)

    def eval(self):
        res = self.value[-1].eval()
        for val in self.value[-3::-2]:
            res = val.eval() ** res
        return res


class EvalMultOp:
    "Class to evaluate multiplication and division expressions"

    def __init__(self, tokens):
        self.value = tokens[0]

    def __repr__(self):
        return str(self.value)

    def eval(self):
        prod = self.value[0].eval()
        for op, val in pp.operatorOperands(self.value[1:]):
            if op == "*":
                prod *= val.eval()
            if op == "/":
                prod /= val.eval()
        return prod


class EvalAddOp:
    "Class to evaluate addition and subtraction expressions"

    def __init__(self, tokens):
        self.value = tokens[0]

    def __repr__(self):
        return str(self.value)

    def eval(self):
        sum = self.value[0].eval()
        for op, val in pp.operatorOperands(self.value[1:]):
            if op == "+":
                sum += val.eval()
            if op == "-":
                sum -= val.eval()
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

    def eval(self):
        val1 = self.value[0].eval()
        for op, val in pp.operatorOperands(self.value[1:]):
            fn = pp.EvalComparisonOp.opMap[op]
            val2 = val.eval()
            if not fn(val1, val2):
                break
            val1 = val2
        else:
            return True
        return False


class EvalConstant:
    def __init__(self, tokens):
        self.value = tokens[0]

    def __repr__(self):
        return "CONST:" + str(self.value)

    def eval(self):
        return float(self.value)


class EvalFunction:
    def __init__(self, tokens):
        self.name = tokens[0]
        self.args = tokens[1]

    def __repr__(self):
        return "FUNC:" + str(self.name)

    def eval(self):
        return {"function": self.name, "arguments": self.args}


class EvalLookup:
    def __init__(self, tokens):
        self.name = tokens[0]
        self.args = tokens[1]

    def __repr__(self):
        return "LOOKUP:" + str(self.name)

    def eval(self):
        return {"component": self.name, "dimensions": self.args}


class EvalComponent:
    def __init__(self, tokens):
        self.name = tokens[0]

    def __repr__(self):
        return "COMPONENT:" + str(self.name)

    def eval(self):
        return {"component": self.name}


def setup_parser(set_iterators=[]):

    integer = pp.Word(pp.nums)
    real = pp.Combine(pp.Word(pp.nums) + "." + pp.Word(pp.nums))
    constant = real | integer

    object_ = pp.Word(pp.alphas + "_", pp.alphanums + "_")

    lpar = pp.Suppress("(")
    rpar = pp.Suppress(")")
    lspar = pp.Suppress("[")
    rspar = pp.Suppress("]")

    set_iterator = pp.oneOf(set_iterators)

    func_expr = (object_ + pp.Suppress("=") + constant) | constant | object_
    func_expr_list = pp.delimitedList(pp.Group(func_expr))
    function = object_ + lpar + pp.Group(func_expr_list) + rpar

    lookup_expr_list = pp.delimitedList(set_iterator | object_)
    lookup = object_ + lspar + pp.Group(lookup_expr_list) + rspar

    component = pp.Suppress("$") + object_

    signop = pp.oneOf("+ -")
    multop = pp.oneOf("* /")
    plusop = pp.oneOf("+ -")
    expop = pp.Literal("**")
    comparisonop = pp.oneOf("< <= > >= != = ==")

    component.setParseAction(EvalComponent)
    constant.setParseAction(EvalConstant)
    function.setParseAction(EvalFunction)
    lookup.setParseAction(EvalLookup)

    arith_expr = pp.infixNotation(
        function | lookup | object_ | constant | component,
        [
            (signop, 1, pp.opAssoc.RIGHT, EvalSignOp),
            (expop, 2, pp.opAssoc.LEFT, EvalPowerOp),
            (multop, 2, pp.opAssoc.LEFT, EvalMultOp),
            (plusop, 2, pp.opAssoc.LEFT, EvalAddOp),
        ],
    )

    comp_expr = pp.infixNotation(
        arith_expr,
        [
            (comparisonop, 2, pp.opAssoc.LEFT, EvalComparisonOp),
        ],
    )

    return comp_expr
