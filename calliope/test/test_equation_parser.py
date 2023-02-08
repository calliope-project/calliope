import operator
import random

import pytest
import numpy as np
import pyparsing as pp

from calliope.backend import equation_parser
from calliope.test.common.util import check_error_or_warning
from calliope import exceptions


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
def foreach():
    return equation_parser.foreach_parser()


@pytest.fixture
def foreach_list(foreach):
    return pp.Suppress("[") + pp.Group(pp.delimited_list(foreach)) + pp.Suppress("]")


@pytest.fixture
def helper_function(
    number, indexed_param, component, unindexed_param, identifier, foreach_list
):
    return equation_parser.helper_function_parser(
        identifier,
        allowed_parser_elements_in_args=[
            indexed_param,
            component,
            unindexed_param,
            number,
            foreach_list,
        ],
    )


@pytest.fixture
def helper_function_no_nesting(
    number, indexed_param, component, unindexed_param, identifier, foreach_list
):
    return equation_parser.helper_function_parser(
        identifier,
        allowed_parser_elements_in_args=[
            indexed_param,
            component,
            unindexed_param,
            number,
            foreach_list,
        ],
        allow_function_in_function=False,
    )


@pytest.fixture(
    params=[
        ("number", "1.0", ["$foo", "foo", "foo[bar]", "[foo in foos]"]),
        ("indexed_param", "foo[bar]", ["1.0", "foo", "$foo", "[foo in foos]"]),
        ("component", "$foo", ["1.0", "foo", "foo[bar]", "[foo in foos]"]),
        ("unindexed_param", "foo", ["1.0", "$foo", "foo[bar]", "[foo in foos]"]),
        ("foreach_list", "[foo in foos]", ["1.0", "$foo", "foo[bar]", "foo"]),
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
        "as_dict": True,
        "iterator_dict": {},
        "index_item_dict": {},
        "component_dict": {},
        "sets": {"bar": "bars", "foo": "foos"},
        "equation_name": "foobar",
    }


@pytest.fixture
def arithmetic(helper_function, number, indexed_param, component, unindexed_param):
    return equation_parser.arithmetic_parser(
        helper_function, indexed_param, component, unindexed_param, number
    )


@pytest.fixture
def helper_function_allow_arithmetic(
    number,
    indexed_param,
    component,
    unindexed_param,
    identifier,
    arithmetic,
    foreach_list,
):
    arithmetic = pp.Forward()
    helper_func = equation_parser.helper_function_parser(
        identifier,
        allowed_parser_elements_in_args=[arithmetic, foreach_list],
        allow_function_in_function=False,
    )
    return equation_parser.arithmetic_parser(
        helper_func,
        indexed_param,
        component,
        unindexed_param,
        number,
        arithmetic=arithmetic,
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
        with pytest.raises(pp.ParseException):
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
            assert parsed_[0].eval(as_dict=True) == {"param_or_var_name": string_val}

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
        with pytest.raises(pp.ParseException):
            parser_.parse_string(string_val, parse_all=True)

    @pytest.mark.parametrize(
        ["string_val", "expected"],
        [
            ("foo[bar]", ["foo", {"bar": "bars"}]),
            ("Foo_Bar_1[foo, bar]", ["Foo_Bar_1", {"foo": "foos", "bar": "bars"}]),
            ("foo[foo,bar]", ["foo", {"foo": "foos", "bar": "bars"}]),
            ("foo[ foo ,  bar  ]", ["foo", {"foo": "foos", "bar": "bars"}]),
            ("foo[techs=tech]", ["foo", {"tech": "techs"}]),
            ("foo[techs=tech, bar]", ["foo", {"tech": "techs", "bar": "bars"}]),
            ("foo[bar, techs=tech]", ["foo", {"bar": "bars", "tech": "techs"}]),
            (
                "foo[techs=tech, nodes=node]",
                ["foo", {"tech": "techs", "node": "nodes"}],
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
        with pytest.raises(pp.ParseException):
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
        assert parsed_[0].eval(as_dict=True) == {"component": expected}

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
        with pytest.raises(pp.ParseException):
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
                    "args": [
                        1,
                        {"param_or_var_name": "foo", "dimensions": {"bar": "bars"}},
                    ],
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
                    "args": [
                        {"param_or_var_name": "foo", "dimensions": {"bar": "bars"}}
                    ],
                    "kwargs": {},
                },
            ),
            (
                "dummy_func_1(1, x=foo[bar])",
                {
                    "function": "dummy_func_1",
                    "args": [1],
                    "kwargs": {
                        "x": {"param_or_var_name": "foo", "dimensions": {"bar": "bars"}}
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
                                {
                                    "param_or_var_name": "foo",
                                    "dimensions": {"bar": "bars"},
                                }
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
                                {
                                    "param_or_var_name": "foo",
                                    "dimensions": {"bar": "bars"},
                                },
                            ],
                            "kwargs": {},
                        },
                        {
                            "param_or_var_name": "foo",
                            "dimensions": {"foo": "foos", "bar": "bars"},
                        },
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

        with pytest.raises(exceptions.BackendError) as excinfo:
            parsed_[0].eval(**eval_kwargs)

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
    def test_function_malformed_string(self, helper_function, string_val):
        with pytest.raises(pp.ParseException) as excinfo:
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
        with pytest.raises(pp.ParseException) as excinfo:
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
            with pytest.raises(pp.ParseException) as excinfo:
                parser_.parse_string(helper_func_string.format(string_), parse_all=True)
            assert check_error_or_warning(excinfo, "Expected")

    @pytest.mark.parametrize(
        ("input_string", "expected_result"),
        [
            ("a in A", ["a", "A"]),
            ("a1 in A1", ["a1", "A1"]),
            ("a_1 in A_1", ["a_1", "A_1"]),
            # TODO: decide if this should be allowed:
            ("techs in techs", ["techs", "techs"]),
        ],
    )
    def test_parse_foreach(self, foreach, input_string, expected_result):
        parsed_string = foreach.parse_string(input_string, parse_all=True)
        assert parsed_string[0].as_dict() == {
            "set_iterator": expected_result[0],
            "set_name": expected_result[1],
        }

    @pytest.mark.parametrize(
        "input_string",
        [
            "1 in foo",  # number as iterator
            "foo in 1",  # number as set name
            "1 in 2",  # numbers for both
            "in B",  # missing iterator
            "in",  # missing iterator and set name
            "foo bar",  # missing "in"
            "foo.bar in B",  # unallowed character in iterator .
            "a in foo.bar",  # unallowed character in set name .
            "ainA",  # missing whitespace
            "1a in 2b",  # invalid python identifiers
            "a in A b in B",  # missing deliminator between two set items
            "a in in A",  # duplicated "in"
            "a in in"  # Cannot have "in" as a set iterator/name
            "in in A"  # Cannot have "in" as a set iterator/name
            "in in in",  # Cannot have "in" as a set iterator/name
        ],
    )
    def test_parse_foreach_fail(self, foreach, input_string):
        with pytest.raises(pp.ParseException):
            foreach.parse_string(input_string, parse_all=True)

    @pytest.mark.parametrize(
        ("input_string", "expected_result"),
        [
            ("[a in A]", [["a", "A"]]),
            ("[a in A, a1 in A1]", [["a", "A"], ["a1", "A1"]]),
        ],
    )
    def test_parse_foreach_list(self, foreach_list, input_string, expected_result):
        parsed_string = foreach_list.parse_string(input_string, parse_all=True)
        assert all(
            parsed_string[0][idx].as_dict()
            == {
                "set_iterator": expected_[0],
                "set_name": expected_[1],
            }
            for idx, expected_ in enumerate(expected_result)
        )

    @pytest.mark.parametrize(
        "input_string",
        [
            "a in A",  # no closing brackets
            "(a in A)",  # wrong brackets
            "[a in A",  # missing one bracket
            "a in A]",  # missing one bracket
            "[ainA, b in B]",  # missing whitespace
            "[a in A; b in B]",  # wrong delimiter
        ],
    )
    def test_parse_foreach_list_fail(self, foreach_list, input_string):
        with pytest.raises(pp.ParseException):
            foreach_list.parse_string(input_string, parse_all=True)


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
    @pytest.mark.parametrize(
        "func_string", ["arithmetic", "helper_function_allow_arithmetic"]
    )
    def test_addition_multiplication(
        self, float1, float2, sign, sign_name, eval_kwargs, func_string, request
    ):
        parser_func = request.getfixturevalue(func_string)
        string_ = f"{float1} {sign} {float2}"
        parsed_ = parser_func.parse_string(string_, parse_all=True)
        evaluated_ = parsed_[0].eval(**eval_kwargs)
        if np.isinf(float1) and np.isinf(float2) and sign in ["-", "/"]:
            assert np.isnan(evaluated_)
        else:
            assert evaluated_ == getattr(operator, sign_name)(float1, float2)

    @pytest.mark.parametrize(["sign", "sign_name"], [("+", "pos"), ("-", "neg")])
    @pytest.mark.parametrize(
        "func_string", ["arithmetic", "helper_function_allow_arithmetic"]
    )
    def test_sign(self, float1, sign, sign_name, eval_kwargs, func_string, request):
        parser_func = request.getfixturevalue(func_string)
        string_ = f"{sign}{float1}"
        parsed_ = parser_func.parse_string(string_, parse_all=True)
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
    @pytest.mark.parametrize(
        "func_string", ["arithmetic", "helper_function_allow_arithmetic"]
    )
    def test_mashup(self, equation_string, expected, eval_kwargs, func_string, request):
        parser_func = request.getfixturevalue(func_string)
        parsed_ = parser_func.parse_string(equation_string, parse_all=True)
        assert parsed_[0].eval(**eval_kwargs) == expected

    @pytest.mark.parametrize("number_", numbers)
    @pytest.mark.parametrize("component_", ["$foo", "$bar1"])
    @pytest.mark.parametrize("unindexed_param_", ["foo", "bar1"])
    @pytest.mark.parametrize("indexed_param_", ["foo[bar]", "bar1[foo, bar]"])
    @pytest.mark.parametrize(
        "helper_function_", ["foo(1)", "bar1(foo, $foo, bar[foo], x=1)"]
    )
    @pytest.mark.parametrize(
        "func_string", ["arithmetic", "helper_function_allow_arithmetic"]
    )
    def test_non_numbers(
        self,
        number_,
        component_,
        unindexed_param_,
        indexed_param_,
        helper_function_,
        func_string,
        request,
    ):
        items = [
            number_,
            component_,
            unindexed_param_,
            indexed_param_,
            helper_function_,
        ]
        parser_func = request.getfixturevalue(func_string)
        random.shuffle(items)
        equation_string = f"({items[0]} / {items[1]}) + {items[1]} - {items[2]} * {items[3]}**-{items[4]}"
        # We can't evaluate this since not all elements evaluate to numbers.
        # Here we simply test that parsing is successful
        parser_func.parse_string(equation_string, parse_all=True)

    @pytest.mark.parametrize(
        "string_", ["1 + 2", "1 * 2", "x=1/2", "foo + 1, x=1", "foo, x=1+1"]
    )
    def test_helper_function_no_arithmetic(self, helper_function, string_):
        helper_func_string = f"foobar({string_})"
        with pytest.raises(pp.ParseException):
            helper_function.parse_string(helper_func_string, parse_all=True)

    @pytest.mark.parametrize(
        ["string_", "expected"],
        [
            ("1 + 2", {"args": [3], "kwargs": {}}),
            ("1 * 2", {"args": [2], "kwargs": {}}),
            ("x=1/2", {"args": [], "kwargs": {"x": 0.5}}),
            (
                "foo, x=1+1",
                {"args": [{"param_or_var_name": "foo"}], "kwargs": {"x": 2}},
            ),
        ],
    )
    def test_helper_function_allow_arithmetic(
        self, helper_function_allow_arithmetic, eval_kwargs, string_, expected
    ):
        helper_func_string = f"dummy_func_1({string_})"
        parsed = helper_function_allow_arithmetic.parse_string(
            helper_func_string, parse_all=True
        )
        evaluated = parsed[0].eval(**eval_kwargs)
        assert evaluated == {"function": "dummy_func_1", **expected}


class TestEquationParserComparison:

    EXPR_PARAMS_AND_EXPECTED_EVAL = {
        0: 0.0,
        -1: -1.0,
        1e2: 100,
        "1/2": 0.5,
        "2**2": 4,
        ".inf": np.inf,
        "param_1": {"param_or_var_name": "param_1"},
        "foo[foo, bar]": {
            "param_or_var_name": "foo",
            "dimensions": {"foo": "foos", "bar": "bars"},
        },
        "foo[bar]": {"param_or_var_name": "foo", "dimensions": {"bar": "bars"}},
        "$foo": {"component": "foo"},
        "dummy_func_1(1, foo[bar], $foo, x=dummy_func_2(1, y=2), foo=bar)": {
            "function": "dummy_func_1",
            "args": [
                1,
                {"param_or_var_name": "foo", "dimensions": {"bar": "bars"}},
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
        with pytest.raises(pp.ParseException):
            parser_func.parse_string(equation_string, parse_all=True)
