import operator
import random

import pytest
import numpy as np
import pyparsing

from calliope.backend import equation_parser
from calliope.test.common.util import check_error_or_warning


COMPONENT_CLASSIFIER = equation_parser.COMPONENT_CLASSIFIER
HELPER_FUNCS = {"dummy_func_1": lambda x: x * 10, "dummy_func_2": lambda x, y: x + y}


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
    return equation_parser.indexed_param_or_var_parser(identifier, ["foo", "bar"])


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
        ],
    )
    def test_indexed_param(self, indexed_param, string_val, expected):
        parsed_ = indexed_param.parse_string(string_val, parse_all=True)
        assert parsed_[0].eval() == {
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
            "foo[foo, baz]",  # one set iterator is valid, but not the other
        ],
    )
    def test_fail_missing_iterator_indexed_param(self, indexed_param, string_val):
        with pytest.raises(KeyError):
            indexed_param.parse_string(string_val, parse_all=True)

    @pytest.mark.parametrize(
        "string_val",
        [
            "foo [bar]",  # space between param name and set iterator reference
            "foo[foo bar]",  # missing delimination
            "foo(bar)",  # incorrect brackets
        ],
    )
    def test_fail_string_issues_indexed_param(self, indexed_param, string_val):
        with pytest.raises(pyparsing.ParseException):
            indexed_param.parse_string(string_val, parse_all=True)

    @pytest.mark.parametrize(
        ["string_val", "expected"],
        [
            (
                "$foo",
                "foo",
            ),  # keeping to ensure something weird doesn't happen with the constant (e.g. is accidentally overwritten)
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
    def test_function(self, helper_function, string_val, expected):
        parsed_ = helper_function.parse_string(string_val, parse_all=True)
        assert parsed_[0].eval(HELPER_FUNCS) == expected

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
    def test_missing_function(self, string_val, helper_function):
        parsed_ = helper_function.parse_string(string_val, parse_all=True)
        with pytest.raises(pyparsing.ParseException) as excinfo:
            parsed_[0].eval(HELPER_FUNCS)
        assert check_error_or_warning(excinfo, "Invalid helper function defined")

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
    def test_function_invalid_string(self, helper_function, string_val):
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
    def test_function_one_arg_allowed_valid_string(
        self, helper_function_one_parser_in_args, helper_func_string
    ):
        parser_, valid_string, invalid_string = helper_function_one_parser_in_args

        for string_ in invalid_string:
            with pytest.raises(pyparsing.ParseException) as excinfo:
                parser_.parse_string(helper_func_string.format(string_), parse_all=True)
            assert check_error_or_warning(excinfo, "Expected")
