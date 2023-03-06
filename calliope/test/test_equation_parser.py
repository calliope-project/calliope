import operator
import random

import pytest
import numpy as np
import pyparsing

from calliope.backend import equation_parser
from calliope.test.common.util import check_error_or_warning


COMPONENT_CLASSIFIER = equation_parser.COMPONENT_CLASSIFIER
HELPER_FUNCS = {
    "dummy_func_1": lambda **kwargs: lambda x: x * 10,
    "dummy_func_2": lambda **kwargs: lambda x, y: x + y,
}


@pytest.fixture
def base_parser_elements():
    number, identifier = equation_parser.setup_base_parser_elements()
    return number, identifier


@pytest.fixture
def number(base_parser_elements):
    return base_parser_elements[0]


@pytest.fixture
def identifier(base_parser_elements):
    return base_parser_elements[1]


@pytest.fixture
def unindexed_param(identifier):
    return equation_parser.unindexed_param_parser(identifier)


@pytest.fixture
def indexed_param(identifier):
    return equation_parser.indexed_param_or_var_parser(identifier)


@pytest.fixture
def component(identifier):
    return equation_parser.component_parser(identifier)


@pytest.fixture
def helper_function(number, indexed_param, component, unindexed_param, identifier):
    return equation_parser.helper_function_parser(
        identifier,
        allowed_parser_elements_in_args=[
            indexed_param,
            component,
            unindexed_param,
            number,
        ],
    )


@pytest.fixture
def helper_function_no_nesting(
    number, indexed_param, component, unindexed_param, identifier
):
    return equation_parser.helper_function_parser(
        identifier,
        allowed_parser_elements_in_args=[
            indexed_param,
            component,
            unindexed_param,
            number,
        ],
        allow_function_in_function=False,
    )


@pytest.fixture(
    params=[
        ("number", "1.0", ["$foo", "foo", "foo[bar]"]),
        ("indexed_param", "foo[bar]", ["1.0", "foo", "$foo"]),
        ("component", "$foo", ["1.0", "foo", "foo[bar]"]),
        ("unindexed_param", "foo", ["1.0", "$foo", "foo[bar]"]),
    ]
)
def helper_function_one_parser_in_args(identifier, request):
    parser_element, valid_string, invalid_string = request.param
    return (
        equation_parser.helper_function_parser(
            identifier,
            allowed_parser_elements_in_args=[request.getfixturevalue(parser_element)],
            allow_function_in_function=True,
        ),
        valid_string,
        invalid_string,
    )


@pytest.fixture(scope="function")
def eval_kwargs():
    return {
        "helper_func_dict": HELPER_FUNCS,
        "test": True,
        "errors": [],
        "iterator_dict": {},
        "index_item_dict": {},
    }


@pytest.fixture
def arithmetic(helper_function, number, indexed_param, component, unindexed_param):
    return equation_parser.arithmetic_parser(
        helper_function, indexed_param, component, unindexed_param, number
    )


@pytest.fixture
def equation_comparison(arithmetic):
    return equation_parser.equation_comparison_parser(arithmetic)


@pytest.fixture
def generate_equation():
    return equation_parser.generate_equation_parser()


class TestEquationParserElements:
    @pytest.mark.parametrize(
        ["string_val", "expected"],
        [
            ("0", 0.0),
            ("0.11", 0.11),
            ("-1", -1.0),
            ("1.0", 1.0),
            ("20.1", 20.1),
            ("1e2", 100),
            ("-1e2", -100),
            ("1e-2", 0.01),
            ("inf", np.inf),
            ("INF", np.inf),
            (".inf", np.inf),
        ],
    )
    def test_numbers(self, number, string_val, expected):
        parsed_ = number.parse_string(string_val, parse_all=True)
        assert parsed_[0].eval() == expected

    @pytest.mark.parametrize(
        "string_val",
        [
            "1 2",  # whitespace between valid characters
            "1-2",  # we don't capture arithmetic here
            "1/2",  # we don't capture fractions here
            "1+2",  # we don't capture arithmetic here
            "1'200",  # we don't capture thousand separators. FIXME?
            "1,200",  # we don't capture thousand separators. FIXME?
            "foo",  # not a number
            "fo1o",  # some are not a number
            "1foo",  # some are not a number (requires parse_all=True to fail)
            "foo1",  # some are not a number
            "1ee2",  # misspelled scientific notation
            "infinity",  # shortform of "inf" is all we accept
            "pi",  # only "inf" is accepted as a math constant. FIXME?
            "e",  # only "inf" is accepted as a math constant. FIXME?
            "one",  # spelled our numbers not dealt with
        ],
    )
    def test_fail_numbers(self, number, string_val):
        with pytest.raises(pyparsing.ParseException):
            number.parse_string(string_val, parse_all=True)

    @pytest.mark.parametrize(
        "string_val",
        [
            "foo",
            "foo_bar",
            "Foo1",
            "FOO_1_BAR",
            "FOO__",
        ],
    )
    @pytest.mark.parametrize("parser", ["identifier", "unindexed_param"])
    def test_identifiers(self, string_val, parser, request):
        parser_ = request.getfixturevalue(parser)
        parsed_ = parser_.parse_string(string_val, parse_all=True)
        if parser == "identifier":
            assert parsed_[0] == string_val
        elif parser == "unindexed_param":
            assert parsed_[0].eval() == {"param_or_var_name": string_val}

    @pytest.mark.parametrize(
        "string_val",
        [
            "foo bar",  # whitespace between characters
            "foo.bar",  # invalid character .
            "foo-bar",  # invalid character -
            "$foo",  # invalid character $
            "foo[bar]",  # invalid character []
            "foo(bar)",  # invalid character ()
            "0",  # number
            "1foo",  # number at start
            "1e2",  # number
            "1/2",  # number + arithmetic
            "2+2",  # number + arithmetic
            "inf",  # We capture infinity and evaluate as a number
            "INF",  # We capture infinity and evaluate as a number
            ".inf",  # We capture infinity and evaluate as a number
            "_foo",  # leading underscores not allowed
            "__type__",  # leading underscores not allowed
        ],
    )
    @pytest.mark.parametrize("parser", ["identifier", "unindexed_param"])
    def test_fail_identifiers(self, string_val, parser, request):
        parser_ = request.getfixturevalue(parser)
        with pytest.raises(pyparsing.ParseException):
            parser_.parse_string(string_val, parse_all=True)

    @pytest.mark.parametrize(
        ["string_val", "expected"],
        [
            ("foo[bar]", ["foo", ["bar"]]),
            ("Foo_Bar_1[foo, bar]", ["Foo_Bar_1", ["foo", "bar"]]),
            ("foo[foo,bar]", ["foo", ["foo", "bar"]]),
            ("foo[ foo ,  bar  ]", ["foo", ["foo", "bar"]]),
            ("foo[techs=tech]", ["foo", [{"tech": "techs"}]]),
            ("foo[techs=tech, bar]", ["foo", [{"tech": "techs"}, "bar"]]),
            ("foo[bar, techs=tech]", ["foo", ["bar", {"tech": "techs"}]]),
            (
                "foo[techs=tech, nodes=node]",
                ["foo", [{"tech": "techs"}, {"node": "nodes"}]],
            ),
        ],
    )
    def test_indexed_param(self, indexed_param, eval_kwargs, string_val, expected):
        parsed_ = indexed_param.parse_string(string_val, parse_all=True)
        assert parsed_[0].eval(**eval_kwargs) == {
            "param_or_var_name": expected[0],
            "dimensions": expected[1],
        }

    @pytest.mark.parametrize(
        "string_val",
        [
            "foo[undefined]",  # unknown set iterator
            "foo[FOO]",  # capitalised set iterator
            "foo[fo]",  # missing part of valid set iterator name
            "foo[]",  # missing set iterators
            "foo[foo, bar, baz]",  # one set iterator is valid, but not the other
            "foo[baz, foo]",  # one set iterator is valid, but not the other
        ],
    )
    @pytest.mark.xfail(
        reason="Moved check for missing iterator out of equation parsing"
    )
    def test_fail_missing_iterator_indexed_param(self, indexed_param, string_val):
        with pytest.raises(KeyError) as excinfo:
            indexed_param.parse_string(string_val, parse_all=True)
        assert check_error_or_warning(excinfo, "Expected foo | bar")

    @pytest.mark.parametrize(
        "string_val",
        [
            "foo [bar]",  # space between param name and set iterator reference
            "foo[foo bar]",  # missing delimination
            "foo[]",  # missing set iterator
            "[bar]",  # missing set name
            "foo(bar)",  # incorrect brackets
        ],
    )
    def test_fail_string_issues_indexed_param(self, indexed_param, string_val):
        with pytest.raises(pyparsing.ParseException):
            indexed_param.parse_string(string_val, parse_all=True)

    @pytest.mark.parametrize(
        ["string_val", "expected"],
        [
            # keeping explicit reference to "$" to ensure something weird doesn't happen
            # with the constant (e.g. is accidentally overwritten)
            ("$foo", "foo"),
            (f"{COMPONENT_CLASSIFIER}foo", "foo"),
            (f"{COMPONENT_CLASSIFIER}Foo_Bar_1", "Foo_Bar_1"),
        ],
    )
    def test_component(self, component, string_val, expected):
        parsed_ = component.parse_string(string_val, parse_all=True)
        assert parsed_[0].eval() == {"component": expected}

    @pytest.mark.parametrize(
        "string_val",
        [
            f"{COMPONENT_CLASSIFIER} foo",  # space between classifier and component name
            "foo",  # no classifier
            "foo$",  # classifier not at start
            "f$oo",  # classifier not at start
            "$",  # missing component name
            "$foo(bar)",  # adding classifer to function
            "$foo[bar]",  # adding classifer to indexed param
            "$1",  # adding classifer to invalid python variable name
        ],
    )
    def test_fail_component(self, component, string_val):
        with pytest.raises(pyparsing.ParseException):
            component.parse_string(string_val, parse_all=True)

    @pytest.mark.parametrize(
        ["string_val", "expected"],
        [
            (
                "dummy_func_1(1)",
                {"function": "dummy_func_1", "args": [1], "kwargs": {}},
            ),
            (
                "dummy_func_1(x=1)",
                {"function": "dummy_func_1", "args": [], "kwargs": {"x": 1}},
            ),
            (
                "dummy_func_2(1, y=2)",
                {"function": "dummy_func_2", "args": [1], "kwargs": {"y": 2}},
            ),
            (
                "dummy_func_2(1, 2)",
                {"function": "dummy_func_2", "args": [1, 2], "kwargs": {}},
            ),
            (
                "dummy_func_2(y=1, x=2)",
                {"function": "dummy_func_2", "args": [], "kwargs": {"x": 2, "y": 1}},
            ),
            (
                "dummy_func_1(dummy_func_2(1, 2))",
                {
                    "function": "dummy_func_1",
                    "args": [
                        {"function": "dummy_func_2", "args": [1, 2], "kwargs": {}}
                    ],
                    "kwargs": {},
                },
            ),
            (
                "dummy_func_1(x=dummy_func_2(1, y=2))",
                {
                    "function": "dummy_func_1",
                    "args": [],
                    "kwargs": {
                        "x": {
                            "function": "dummy_func_2",
                            "args": [1],
                            "kwargs": {"y": 2},
                        }
                    },
                },
            ),
            (
                "dummy_func_1(1, foo[bar])",
                {
                    "function": "dummy_func_1",
                    "args": [1, {"param_or_var_name": "foo", "dimensions": ["bar"]}],
                    "kwargs": {},
                },
            ),
            (
                "dummy_func_1(1, dummy_func_2(1))",
                {
                    "function": "dummy_func_1",
                    "args": [
                        1,
                        {"function": "dummy_func_2", "args": [1], "kwargs": {}},
                    ],
                    "kwargs": {},
                },
            ),
            (
                "dummy_func_1(foo)",
                {
                    "function": "dummy_func_1",
                    "args": [{"param_or_var_name": "foo"}],
                    "kwargs": {},
                },
            ),
            (
                "dummy_func_1(foo[bar])",
                {
                    "function": "dummy_func_1",
                    "args": [{"param_or_var_name": "foo", "dimensions": ["bar"]}],
                    "kwargs": {},
                },
            ),
            (
                "dummy_func_1(1, x=foo[bar])",
                {
                    "function": "dummy_func_1",
                    "args": [1],
                    "kwargs": {
                        "x": {"param_or_var_name": "foo", "dimensions": ["bar"]}
                    },
                },
            ),
            (
                "dummy_func_1($foo)",
                {
                    "function": "dummy_func_1",
                    "args": [{"component": "foo"}],
                    "kwargs": {},
                },
            ),
            (
                "dummy_func_1($foo, x=$foo)",
                {
                    "function": "dummy_func_1",
                    "args": [{"component": "foo"}],
                    "kwargs": {"x": {"component": "foo"}},
                },
            ),
            (
                "dummy_func_1(dummy_func_2(foo[bar], y=$foo))",
                {
                    "function": "dummy_func_1",
                    "args": [
                        {
                            "function": "dummy_func_2",
                            "args": [
                                {"param_or_var_name": "foo", "dimensions": ["bar"]}
                            ],
                            "kwargs": {"y": {"component": "foo"}},
                        }
                    ],
                    "kwargs": {},
                },
            ),
            (
                "dummy_func_1(1, dummy_func_2(1, foo[bar]), foo[foo, bar], $foo, 1, bar)",
                {
                    "function": "dummy_func_1",
                    "args": [
                        1,
                        {
                            "function": "dummy_func_2",
                            "args": [
                                1,
                                {"param_or_var_name": "foo", "dimensions": ["bar"]},
                            ],
                            "kwargs": {},
                        },
                        {"param_or_var_name": "foo", "dimensions": ["foo", "bar"]},
                        {"component": "foo"},
                        1,
                        {"param_or_var_name": "bar"},
                    ],
                    "kwargs": {},
                },
            ),
        ],
    )
    def test_function(self, helper_function, string_val, expected, eval_kwargs):
        parsed_ = helper_function.parse_string(string_val, parse_all=True)
        assert parsed_[0].eval(**eval_kwargs) == expected

    @pytest.mark.parametrize(
        "string_val",
        [
            "not_a_helper_func()",
            "dummy_func_1(not_a_helper_func())",
            "dummy_func_1(x=not_a_helper_func())",
            "dummy_func_1(x=dummy_func_2(not_a_helper_func(), y=2))",
            "dummy_func_10()",
            "dummy_func_()",
            "ummy_func_1()",
        ],
    )
    def test_missing_function(self, string_val, helper_function, eval_kwargs):
        parsed_ = helper_function.parse_string(string_val, parse_all=True)
        error_catcher = []
        eval_kwargs["errors"] = error_catcher
        parsed_[0].eval(**eval_kwargs)
        assert check_error_or_warning(error_catcher, "Invalid helper function defined")

    @pytest.mark.parametrize(
        "string_val",
        [
            "dummy_func_1",  # missing brackets
            "dummy_func_1(x=1, 2)",  # kwargs before args
            "dummy_func_1[]",  # wrong brackets
            "dummy_func_1{}",  # wrong brackets
            "dummy_func_1 ()",  # space between function and brackets
            "dummy_func_1()dummy_func_2()",  # missing operator between
            "()",  # no function name
            "()dummy_func_1",  # function name after brackets
        ],
    )
    def test_function_malformed_string(self, helper_function, string_val):
        with pytest.raises(pyparsing.ParseException) as excinfo:
            helper_function.parse_string(string_val, parse_all=True)
        assert check_error_or_warning(excinfo, "Expected")

    def test_function_nesting_not_allowed_valid_string(
        self, helper_function_no_nesting
    ):
        helper_function_no_nesting.parse_string(
            "dummy_func_1(1, $foo, foo, foo[bar])", parse_all=True
        )

    @pytest.mark.parametrize("args", ["dummy_func_1(1)", "x=dummy_func_1(1)"])
    def test_function_nesting_not_allowed_invalid_string(
        self, helper_function_no_nesting, args
    ):
        with pytest.raises(pyparsing.ParseException) as excinfo:
            helper_function_no_nesting.parse_string(
                f"dummy_func_1({args})", parse_all=True
            )
        assert check_error_or_warning(excinfo, "Expected")

    @pytest.mark.parametrize(
        "helper_func_string", ["dummy_func_1({})", "dummy_func_1(dummy_func_1({}))"]
    )
    def test_function_one_arg_allowed_valid_string(
        self, helper_function_one_parser_in_args, helper_func_string
    ):
        parser_, valid_string, invalid_string = helper_function_one_parser_in_args

        parser_.parse_string(helper_func_string.format(valid_string), parse_all=True)

    @pytest.mark.parametrize(
        "helper_func_string", ["dummy_func_1({})", "dummy_func_1(dummy_func_1({}))"]
    )
    def test_function_one_arg_allowed_invalid_string(
        self, helper_function_one_parser_in_args, helper_func_string
    ):
        parser_, valid_string, invalid_string = helper_function_one_parser_in_args

        for string_ in invalid_string:
            with pytest.raises(pyparsing.ParseException) as excinfo:
                parser_.parse_string(helper_func_string.format(string_), parse_all=True)
            assert check_error_or_warning(excinfo, "Expected")

    @pytest.mark.parametrize(
        ["parser_name", "parse_string", "expected"],
        [
            ("number", "1", "NUM:1"),
            ("number", "inf", "NUM:inf"),
            ("unindexed_param", "foo", "UNINDEXED_PARAM_OR_VAR:foo"),
            (
                "indexed_param",
                "foo[bar, foos=foo]",
                "INDEXED_PARAM_OR_VAR:foo[ITERATOR:bar, FOOS:foo]",
            ),
            ("component", "$foo", "COMPONENT:foo"),
            (
                "helper_function",
                "dummy_func_1(1, x=foo)",
                "dummy_func_1(args=[NUM:1], kwargs={'x': UNINDEXED_PARAM_OR_VAR:foo})",
            ),
        ],
    )
    def test_repr(self, request, parser_name, parse_string, expected):
        parser = request.getfixturevalue(parser_name)
        parsed_ = parser.parse_string(parse_string, parse_all=True)
        assert str(parsed_[0]) == expected


class TestEquationParserArithmetic:
    numbers = [2, 100, 0.02, "1e2", "2e-2", "inf"]

    @pytest.fixture(params=numbers)
    def float1(self, request):
        return float(request.param)

    @pytest.fixture(params=numbers)
    def float2(self, request):
        return float(request.param)

    @pytest.mark.parametrize(
        ["sign", "sign_name"],
        [("+", "add"), ("-", "sub"), ("*", "mul"), ("/", "truediv"), ("**", "pow")],
    )
    def test_addition_multiplication(
        self, float1, float2, sign, sign_name, arithmetic, eval_kwargs
    ):
        string_ = f"{float1} {sign} {float2}"
        parsed_ = arithmetic.parse_string(string_, parse_all=True)
        evaluated_ = parsed_[0].eval(**eval_kwargs)
        if np.isinf(float1) and np.isinf(float2) and sign in ["-", "/"]:
            assert np.isnan(evaluated_)
        else:
            assert evaluated_ == getattr(operator, sign_name)(float1, float2)

    @pytest.mark.parametrize(["sign", "sign_name"], [("+", "pos"), ("-", "neg")])
    def test_sign(self, float1, sign, sign_name, arithmetic, eval_kwargs):
        string_ = f"{sign}{float1}"
        parsed_ = arithmetic.parse_string(string_, parse_all=True)
        evaluated_ = parsed_[0].eval(**eval_kwargs)
        if np.isinf(float1):
            assert np.isinf(evaluated_)
        else:
            assert evaluated_ == getattr(operator, sign_name)(float1)

    @pytest.mark.parametrize(
        ["equation_string", "expected"],
        [
            ("1+2*2", 5),
            ("-1 - 2", -3),
            ("-1 + 2**-2", -0.75),
            ("2**2**2-1", 15),
            ("2 * 3 + -2 - -3", 7),
            ("100 / 10 + 3 * 5 - 2**3 + -5", 12),
            ("(1e5 * 10 / 1000 - 16**0.5) + (10 + 100) * (10) - 1/2", 2095.5),
        ],
    )
    def test_mashup(self, equation_string, expected, arithmetic, eval_kwargs):
        parsed_ = arithmetic.parse_string(equation_string, parse_all=True)
        assert parsed_[0].eval(**eval_kwargs) == expected

    @pytest.mark.parametrize("number_", numbers)
    @pytest.mark.parametrize("component_", ["$foo", "$bar1"])
    @pytest.mark.parametrize("unindexed_param_", ["foo", "bar1"])
    @pytest.mark.parametrize("indexed_param_", ["foo[bar]", "bar1[foo, bar]"])
    @pytest.mark.parametrize(
        "helper_function_", ["foo(1)", "bar1(foo, $foo, bar[foo], x=1)"]
    )
    def test_non_numbers(
        self,
        number_,
        component_,
        unindexed_param_,
        indexed_param_,
        helper_function_,
        arithmetic,
    ):
        items = [
            number_,
            component_,
            unindexed_param_,
            indexed_param_,
            helper_function_,
        ]
        random.shuffle(items)
        equation_string = f"({items[0]} / {items[1]}) + {items[1]} - {items[2]} * {items[3]}**-{items[4]}"
        # We can't evaluate this since not all elements evaluate to numbers.
        # Here we simply test that parsing is successful
        arithmetic.parse_string(equation_string, parse_all=True)

    def test_repr(self, arithmetic):
        parse_string = "1 + foo - foo[bar, foos=foo] + (foo / $foo) ** -2"
        expected = (
            "(NUM:1 + UNINDEXED_PARAM_OR_VAR:foo - INDEXED_PARAM_OR_VAR:foo[ITERATOR:bar, FOOS:foo]"
            " + ((UNINDEXED_PARAM_OR_VAR:foo / COMPONENT:foo) ** (-)NUM:2))"
        )
        parsed_ = arithmetic.parse_string(parse_string, parse_all=True)
        assert str(parsed_[0]) == expected


class TestEquationParserComparison:
    EXPR_PARAMS_AND_EXPECTED_EVAL = {
        0: 0.0,
        -1: -1.0,
        1e2: 100,
        "1/2": 0.5,
        "2**2": 4,
        ".inf": np.inf,
        "param_1": {"param_or_var_name": "param_1"},
        "foo[foo, bar]": {"param_or_var_name": "foo", "dimensions": ["foo", "bar"]},
        "foo[bar]": {"param_or_var_name": "foo", "dimensions": ["bar"]},
        "$foo": {"component": "foo"},
        "dummy_func_1(1, foo[bar], $foo, x=dummy_func_2(1, y=2), foo=bar)": {
            "function": "dummy_func_1",
            "args": [
                1,
                {"param_or_var_name": "foo", "dimensions": ["bar"]},
                {"component": "foo"},
            ],
            "kwargs": {
                "x": {"function": "dummy_func_2", "args": [1], "kwargs": {"y": 2}},
                "foo": {"param_or_var_name": "bar"},
            },
        },
    }

    @pytest.fixture(params=EXPR_PARAMS_AND_EXPECTED_EVAL.keys())
    def var_left(self, request):
        return request.param

    @pytest.fixture(params=EXPR_PARAMS_AND_EXPECTED_EVAL.keys())
    def var_right(self, request):
        return request.param

    @pytest.fixture
    def expected_left(self, var_left):
        return self.EXPR_PARAMS_AND_EXPECTED_EVAL[var_left]

    @pytest.fixture
    def expected_right(self, var_right):
        return self.EXPR_PARAMS_AND_EXPECTED_EVAL[var_right]

    @pytest.fixture(params=["<=", ">=", "=="])
    def operator(self, request):
        return request.param

    @pytest.fixture
    def single_equation_simple(self, var_left, var_right, operator):
        return f"{var_left} {operator} {var_right}"

    def test_simple_equation(
        self,
        single_equation_simple,
        expected_left,
        expected_right,
        operator,
        equation_comparison,
        eval_kwargs,
    ):
        parsed_constraint = equation_comparison.parse_string(
            single_equation_simple, parse_all=True
        )
        evaluated_expression = parsed_constraint[0]

        assert evaluated_expression.lhs.eval(**eval_kwargs) == expected_left
        assert evaluated_expression.rhs.eval(**eval_kwargs) == expected_right
        assert evaluated_expression.op == operator

    @pytest.mark.parametrize(
        ["equation_string", "expected"],
        [
            ("1<=2", True),
            ("1 >= 2", False),
            ("1  ==  2", False),
            ("(1) <= (2)", True),
            ("1 >= 2", False),
            ("1 >= 2", False),
            ("1 * 3 <= 1e2", True),
            ("-1 >= -0.1 / 2", False),
            ("2**2 == 4 * 1 / 1 * 1**1", True),
            ("(1 + 3) * 2 >= 10 + -1", False),
        ],
    )
    # "equation_comparison" and "generate_equation" should yield the same result
    @pytest.mark.parametrize(
        "func_string", ["equation_comparison", "generate_equation"]
    )
    def test_evaluation(
        self, equation_string, expected, func_string, eval_kwargs, request
    ):
        parser_func = request.getfixturevalue(func_string)
        parsed_equation = parser_func.parse_string(equation_string, parse_all=True)
        assert parsed_equation[0].eval(**eval_kwargs) is expected

    @pytest.mark.parametrize(
        "equation_string",
        [
            "1 + 2 =<",  # missing RHS
            "== 1 + 2 ",  # missing LHS
            "1 = 2",  # unallowed operator
            "1 < 2",  # unallowed operator
            "2 > 1",  # unallowed operator
            "1 (<= 2)",  # weird brackets
            "foo.bar <= 2",  # unparsable string
            "1 <= foo <= 2",  # Too many operators
        ],
    )
    # "equation_comparison" and "generate_equation" should yield the same result
    @pytest.mark.parametrize(
        "func_string", ["equation_comparison", "generate_equation"]
    )
    def test_fail_evaluation(self, equation_string, func_string, request):
        parser_func = request.getfixturevalue(func_string)
        with pytest.raises(pyparsing.ParseException):
            parser_func.parse_string(equation_string, parse_all=True)

    def test_repr(self, equation_comparison):
        parse_string = "1 + foo - foo[bar, foos=foo] >= (foo / $foo) ** -2"
        expected = (
            "(NUM:1 + UNINDEXED_PARAM_OR_VAR:foo - INDEXED_PARAM_OR_VAR:foo[ITERATOR:bar, FOOS:foo])"
            " >= ((UNINDEXED_PARAM_OR_VAR:foo / COMPONENT:foo) ** (-)NUM:2)"
        )
        parsed_ = equation_comparison.parse_string(parse_string, parse_all=True)
        assert str(parsed_[0]) == expected
