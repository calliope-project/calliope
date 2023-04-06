import operator
import random

import pytest
import numpy as np
import pyparsing as pp

from calliope.backend import equation_parser
from calliope.test.common.util import check_error_or_warning
from calliope import exceptions


COMPONENT_CLASSIFIER = equation_parser.COMPONENT_CLASSIFIER


def dummy_func_1(as_latex=False, **kwargs):
    def _dummy_func_1(x):
        return x * 10

    def _as_latex(x):
        return f"{x} * 10"

    if as_latex:
        return _as_latex
    else:
        return _dummy_func_1


def dummy_func_2(as_latex=False, **kwargs):
    def _dummy_func_2(x, y):
        return x + y

    def _as_latex(x, y):
        return f"{x} + {y}"

    if as_latex:
        return _as_latex
    else:
        return _dummy_func_2


HELPER_FUNCS = {"dummy_func_1": dummy_func_1, "dummy_func_2": dummy_func_2}


@pytest.fixture
def valid_object_names():
    return [
        "foo",
        "foo_bar",
        "with_inf",
        "only_techs",
        "no_dims",
        "multi_dim_var",
        "no_dim_var",
    ]


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
def evaluatable_identifier_elements(identifier, valid_object_names):
    return equation_parser.evaluatable_identifier_parser(identifier, valid_object_names)


@pytest.fixture
def evaluatable_identifier(evaluatable_identifier_elements):
    return evaluatable_identifier_elements[0]


@pytest.fixture
def id_list(evaluatable_identifier_elements):
    return evaluatable_identifier_elements[1]


@pytest.fixture
def unsliced_param():
    def _unsliced_param(valid_object_names):
        return equation_parser.unsliced_object_parser(valid_object_names)

    return _unsliced_param


@pytest.fixture
def unsliced_param_with_obj_names(unsliced_param, valid_object_names):
    return unsliced_param(valid_object_names)


@pytest.fixture
def sliced_param(
    number, identifier, evaluatable_identifier, unsliced_param_with_obj_names
):
    return equation_parser.sliced_param_or_var_parser(
        number, identifier, evaluatable_identifier, unsliced_param_with_obj_names
    )


@pytest.fixture
def component(identifier):
    return equation_parser.component_parser(identifier)


@pytest.fixture
def helper_function(
    number,
    sliced_param,
    component,
    unsliced_param_with_obj_names,
    identifier,
    id_list,
):
    return equation_parser.helper_function_parser(
        sliced_param,
        component,
        unsliced_param_with_obj_names,
        number,
        id_list,
        generic_identifier=identifier,
        allow_function_in_function=True,
    )


@pytest.fixture
def helper_function_no_nesting(
    number,
    sliced_param,
    component,
    unsliced_param_with_obj_names,
    identifier,
    id_list,
):
    return equation_parser.helper_function_parser(
        sliced_param,
        component,
        unsliced_param_with_obj_names,
        number,
        id_list,
        generic_identifier=identifier,
        allow_function_in_function=False,
    )


@pytest.fixture(
    params=[
        ("number", "1.0", ["$foo", "foo", "foo[bars=bar]"]),
        ("sliced_param", "foo[bars=bar]", ["1.0", "foo", "$foo"]),
        ("component", "$foo", ["1.0", "foo", "foo[bars=bar]"]),
        ("unsliced_param_with_obj_names", "foo", ["1.0", "$foo", "foo[bars=bar]"]),
    ]
)
def helper_function_one_parser_in_args(identifier, request):
    parser_element, valid_string, invalid_string = request.param
    return (
        equation_parser.helper_function_parser(
            request.getfixturevalue(parser_element),
            generic_identifier=identifier,
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
        "index_slice_dict": {},
        "component_dict": {},
        "equation_name": "foobar",
        "apply_imask": False,
        "references": set(),
    }


@pytest.fixture
def arithmetic(
    helper_function, number, sliced_param, component, unsliced_param_with_obj_names
):
    return equation_parser.arithmetic_parser(
        helper_function, component, sliced_param, number, unsliced_param_with_obj_names
    )


@pytest.fixture
def helper_function_allow_arithmetic(
    number,
    sliced_param,
    component,
    unsliced_param_with_obj_names,
    identifier,
    arithmetic,
    id_list,
):
    arithmetic = pp.Forward()
    helper_func = equation_parser.helper_function_parser(
        arithmetic,
        id_list,
        generic_identifier=identifier,
    )
    return equation_parser.arithmetic_parser(
        helper_func,
        sliced_param,
        component,
        unsliced_param_with_obj_names,
        number,
        arithmetic=arithmetic,
    )


@pytest.fixture
def equation_comparison(arithmetic):
    return equation_parser.equation_comparison_parser(arithmetic)


@pytest.fixture
def generate_equation(valid_object_names):
    return equation_parser.generate_equation_parser(valid_object_names)


@pytest.fixture
def generate_index_slice(valid_object_names):
    return equation_parser.generate_index_slice_parser(valid_object_names)


@pytest.fixture
def generate_component(valid_object_names):
    return equation_parser.generate_component_parser(valid_object_names)


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
    def test_identifiers(self, string_val, identifier):
        parsed_ = identifier.parse_string(string_val, parse_all=True)
        assert parsed_[0] == string_val

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
    def test_fail_identifiers(self, string_val, identifier):
        with pytest.raises(pp.ParseException):
            identifier.parse_string(string_val, parse_all=True)

    def test_evaluatable_identifier(self, evaluatable_identifier, eval_kwargs):
        parsed_ = evaluatable_identifier.parse_string("hello", parse_all=True)
        assert parsed_[0].eval(**eval_kwargs) == "hello"

    def test_evaluatable_identifier_fail(self, evaluatable_identifier):
        with pytest.raises(pp.ParseException):
            # "foo" is a optimisation problem object, so it is ignored by the evaluatable identifier parser
            evaluatable_identifier.parse_string("foo", parse_all=True)

    def test_id_list(self, id_list, eval_kwargs):
        parsed_ = id_list.parse_string("[hello, there]", parse_all=True)
        assert parsed_[0].eval(**eval_kwargs) == ["hello", "there"]

    @pytest.mark.parametrize(
        "string_val", ["1", "inf", "foo", "$foo", "dummy_func_1(1)"]
    )
    def test_id_list_fail(self, id_list, string_val):
        with pytest.raises(pp.ParseException):
            id_list.parse_string(f"[{string_val}, there]", parse_all=True)

    @pytest.mark.parametrize("string_val", ["foo", "foo_bar"])
    def test_unsliced_param(self, unsliced_param_with_obj_names, string_val):
        parsed_ = unsliced_param_with_obj_names.parse_string(string_val, parse_all=True)
        assert parsed_[0].eval(references=set(), as_dict=True) == {
            "param_or_var_name": string_val
        }

    def test_unsliced_param_references(self, unsliced_param_with_obj_names):
        references = set()
        parsed_ = unsliced_param_with_obj_names.parse_string("foo", parse_all=True)
        parsed_[0].eval(references=references, as_dict=True)
        assert references == {"foo"}

    @pytest.mark.parametrize("string_val", ["Foo", "foobar"])
    def test_unsliced_param_fail(self, unsliced_param_with_obj_names, string_val):
        with pytest.raises(pp.ParseException):
            unsliced_param_with_obj_names.parse_string(string_val, parse_all=True)

    @pytest.mark.parametrize(
        ["string_val", "expected"],
        [
            ("foo[techs=tech]", ["foo", {"techs": "tech"}]),
            (
                f"foo[techs={COMPONENT_CLASSIFIER}tech]",
                ["foo", {"techs": {"index_slice_reference": "tech"}}],
            ),
            (
                f"foo[techs=tech,bars={COMPONENT_CLASSIFIER}bar]",
                ["foo", {"techs": "tech", "bars": {"index_slice_reference": "bar"}}],
            ),
            (
                f"foo[ bars={COMPONENT_CLASSIFIER}bar, techs=tech ]",
                ["foo", {"bars": {"index_slice_reference": "bar"}, "techs": "tech"}],
            ),
            (
                "foo_bar[techs=tech, nodes=node]",
                ["foo_bar", {"techs": "tech", "nodes": "node"}],
            ),
        ],
    )
    def test_sliced_param(self, sliced_param, eval_kwargs, string_val, expected):
        parsed_ = sliced_param.parse_string(string_val, parse_all=True)
        assert parsed_[0].eval(**eval_kwargs) == {
            "param_or_var_name": expected[0],
            "dimensions": expected[1],
        }

    @pytest.mark.parametrize(
        "string_val",
        [
            "foobar[bars=bar]",  # name not in allowed list
            "foo [bars=bar]",  # space between param name and slicing
            "foo[techs=tech bars=bar]",  # missing delimination
            "foo[]",  # missing set and index slice
            "foo[techs]",  # missing index slice/set
            "foo[techs=]",  # missing index slice
            "foo[=techs]",  # missing set
            "[bars=bar]",  # missing component name
            "foo(bars=bar)",  # incorrect brackets
        ],
    )
    def test_fail_string_issues_sliced_param(self, sliced_param, string_val):
        with pytest.raises(pp.ParseException):
            sliced_param.parse_string(string_val, parse_all=True)

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
            "$foo(1)",  # adding classifer to function
            "$foo[bars=bar1]",  # adding classifer to indexed param
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
                "dummy_func_1(1, foo[bars=bar])",
                {
                    "function": "dummy_func_1",
                    "args": [
                        1,
                        {"param_or_var_name": "foo", "dimensions": {"bars": "bar"}},
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
                f"dummy_func_1(foo[bars={COMPONENT_CLASSIFIER}bar])",
                {
                    "function": "dummy_func_1",
                    "args": [
                        {
                            "param_or_var_name": "foo",
                            "dimensions": {"bars": {"index_slice_reference": "bar"}},
                        }
                    ],
                    "kwargs": {},
                },
            ),
            (
                "dummy_func_1(1, x=foo[bars=bar1])",
                {
                    "function": "dummy_func_1",
                    "args": [1],
                    "kwargs": {
                        "x": {
                            "param_or_var_name": "foo",
                            "dimensions": {"bars": "bar1"},
                        }
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
                "dummy_func_1(dummy_func_2(foo[bars=bar1], y=$foo))",
                {
                    "function": "dummy_func_1",
                    "args": [
                        {
                            "function": "dummy_func_2",
                            "args": [
                                {
                                    "param_or_var_name": "foo",
                                    "dimensions": {"bars": "bar1"},
                                }
                            ],
                            "kwargs": {"y": {"component": "foo"}},
                        }
                    ],
                    "kwargs": {},
                },
            ),
            (
                "dummy_func_1(1, dummy_func_2(1, foo[bars=$bar]), foo[foos=foo1, bars=bar1], $foo, 1, foo)",
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
                                    "dimensions": {
                                        "bars": {"index_slice_reference": "bar"}
                                    },
                                },
                            ],
                            "kwargs": {},
                        },
                        {
                            "param_or_var_name": "foo",
                            "dimensions": {"foos": "foo1", "bars": "bar1"},
                        },
                        {"component": "foo"},
                        1,
                        {"param_or_var_name": "foo"},
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
            "dummy_func_1(1, $foo, foo, foo[bars=bar1])", parse_all=True
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
        parser_, valid_string, _ = helper_function_one_parser_in_args

        parser_.parse_string(helper_func_string.format(valid_string), parse_all=True)

    @pytest.mark.parametrize(
        "helper_func_string", ["dummy_func_1({})", "dummy_func_1(dummy_func_1({}))"]
    )
    def test_function_one_arg_allowed_invalid_string(
        self, helper_function_one_parser_in_args, helper_func_string
    ):
        parser_, _, invalid_string = helper_function_one_parser_in_args

        for string_ in invalid_string:
            with pytest.raises(pp.ParseException) as excinfo:
                parser_.parse_string(helper_func_string.format(string_), parse_all=True)
            assert check_error_or_warning(excinfo, "Expected")


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
    @pytest.mark.parametrize("unsliced_param_", ["foo", "foo_bar"])
    @pytest.mark.parametrize(
        "sliced_param_", ["foo[bars=bar1]", "foo_bar[foos=foo1, bars=$bar]"]
    )
    @pytest.mark.parametrize(
        "helper_function_", ["foo(1)", "bar1(foo, $foo, foo_bar[foos=foo1], x=1)"]
    )
    @pytest.mark.parametrize(
        "func_string", ["arithmetic", "helper_function_allow_arithmetic"]
    )
    def test_non_numbers(
        self,
        number_,
        component_,
        unsliced_param_,
        sliced_param_,
        helper_function_,
        func_string,
        request,
    ):
        items = [
            number_,
            component_,
            unsliced_param_,
            sliced_param_,
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

    def test_repr(self, arithmetic):
        parse_string = "1 + foo - foo[foos=foo1, bars=$bar] + (foo / $foo) ** -2"
        expected = (
            "(NUM:1 + PARAM_OR_VAR:foo - SLICED_PARAM_OR_VAR:foo[foos=STRING:foo1, bars=REFERENCE:bar]"
            " + ((PARAM_OR_VAR:foo / COMPONENT:foo) ** (-)NUM:2))"
        )
        parsed_ = arithmetic.parse_string(parse_string, parse_all=True)
        assert str(parsed_[0]) == expected


class TestIndexSliceParser:
    @pytest.mark.parametrize(
        "instring", ["foo", "1", "[FOO, BAR]", "foo[bars=bar1]", "FOO"]
    )
    @pytest.mark.parametrize("func_or_not", ["dummy_func_1({})", "{}"])
    def test_index_slice_expression_parser(
        self, generate_index_slice, instring, func_or_not
    ):
        generate_index_slice.parse_string(func_or_not.format(instring), parse_all=True)

    @pytest.mark.parametrize(
        "instring", ["foo + 1", "foo == 1", "$foo", "foo[bars=$bar]", "[foo]"]
    )
    @pytest.mark.parametrize("func_or_not", ["dummy_func_1({})", "{}"])
    def test_index_slice_expression_parser_fail(
        self, generate_index_slice, instring, func_or_not
    ):
        with pytest.raises(pp.ParseException):
            generate_index_slice.parse_string(
                func_or_not.format(instring), parse_all=True
            )


class TestComponentParser:
    @pytest.mark.parametrize(
        "instring",
        [
            "foo",
            "1",
            "dummy_func_1([FOO, BAR])",
            "dummy_func_1(FOO)",
            "foo[foos=foo1, bars=$bar]",
            "foo + 1",
        ],
    )
    @pytest.mark.parametrize("func_or_not", ["dummy_func_1({})", "{}"])
    def test_component_expression_parser(
        self, generate_component, instring, func_or_not
    ):
        generate_component.parse_string(func_or_not.format(instring), parse_all=True)

    @pytest.mark.parametrize("instring", ["[FOO, BAR]", "foo == 1", "$foo", "[foo]"])
    def test_component_expression_parser_fail(self, generate_component, instring):
        with pytest.raises(pp.ParseException):
            generate_component.parse_string(instring, parse_all=True)


class TestEquationParserComparison:
    EXPR_PARAMS_AND_EXPECTED_EVAL = {
        0: 0.0,
        -1: -1.0,
        1e2: 100,
        "1/2": 0.5,
        "2**2": 4,
        ".inf": np.inf,
        "foo_bar": {"param_or_var_name": "foo_bar"},
        "foo[foos=foo1, bars=bar1]": {
            "param_or_var_name": "foo",
            "dimensions": {"foos": "foo1", "bars": "bar1"},
        },
        "foo[bars=bar1]": {"param_or_var_name": "foo", "dimensions": {"bars": "bar1"}},
        "$foo": {"component": "foo"},
        "dummy_func_1(1, foo[bars=$bar], $foo, x=dummy_func_2(1, y=2), foo=foo_bar)": {
            "function": "dummy_func_1",
            "args": [
                1,
                {
                    "param_or_var_name": "foo",
                    "dimensions": {"bars": {"index_slice_reference": "bar"}},
                },
                {"component": "foo"},
            ],
            "kwargs": {
                "x": {"function": "dummy_func_2", "args": [1], "kwargs": {"y": 2}},
                "foo": {"param_or_var_name": "foo_bar"},
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
        lhs, op, rhs = parsed_equation[0].eval(**eval_kwargs)
        comparison_dict = {
            "==": lhs == rhs,
            ">=": lhs >= rhs,
            "<=": lhs <= rhs,
        }
        assert comparison_dict[op] if expected else not comparison_dict[op]

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

    def test_repr(self, equation_comparison):
        parse_string = "1 + foo - foo[foos=foo1, bars=$bar] >= (foo / $foo) ** -2"
        expected = (
            "(NUM:1 + PARAM_OR_VAR:foo - SLICED_PARAM_OR_VAR:foo[foos=STRING:foo1, bars=REFERENCE:bar])"
            " >= ((PARAM_OR_VAR:foo / COMPONENT:foo) ** (-)NUM:2)"
        )
        parsed_ = equation_comparison.parse_string(parse_string, parse_all=True)
        assert str(parsed_[0]) == expected


class TestAsLatex:
    @pytest.fixture
    def latex_eval_kwargs(self, dummy_latex_backend_model, dummy_model_data):
        return {
            "helper_func_dict": HELPER_FUNCS,
            "as_dict": False,
            "as_latex": True,
            "index_slice_dict": {},
            "component_dict": {},
            "equation_name": "foobar",
            "apply_imask": False,
            "references": set(),
            "backend_interface": dummy_latex_backend_model,
            "backend_dataset": dummy_latex_backend_model._dataset,
            "model_data": dummy_model_data,
        }

    @pytest.mark.parametrize(
        ["parser", "instring", "expected"],
        [
            ("number", "1", "1"),
            ("number", "1.0", "1"),
            ("number", "0.01", "0.01"),
            ("number", "inf", "inf"),
            ("number", "-1", "-1"),
            ("number", "2000000", "2\\mathord{\\times}10^{+06}"),
            ("evaluatable_identifier", "hello_there", "hello_there"),
            ("id_list", "[hello, hello_there]", ["hello", "hello_there"]),
            ("unsliced_param_with_obj_names", "no_dims", r"\textit{no_dims}"),
            (
                "unsliced_param_with_obj_names",
                "with_inf",
                r"\textit{with_inf}_\text{node,tech}",
            ),
            ("unsliced_param_with_obj_names", "no_dim_var", r"\textbf{no_dim_var}"),
            (
                "unsliced_param_with_obj_names",
                "multi_dim_var",
                r"\textbf{multi_dim_var}_\text{node,tech}",
            ),
            (
                "sliced_param",
                "with_inf[node=bar]",
                r"\textit{with_inf}_\text{node=bar,tech}",
            ),
            (
                "sliced_param",
                "only_techs[tech=foobar]",
                r"\textit{only_techs}_\text{tech=foobar}",
            ),
            (
                "sliced_param",
                "multi_dim_var[node=bar]",
                r"\textbf{multi_dim_var}_\text{node=bar,tech}",
            ),
            (
                "sliced_param",
                "with_inf[node=bar, tech=foobar]",
                r"\textit{with_inf}_\text{node=bar,tech=foobar}",
            ),
            (
                "sliced_param",
                "multi_dim_var[node=bar, tech=foobar]",
                r"\textbf{multi_dim_var}_\text{node=bar,tech=foobar}",
            ),
            ("helper_function", "dummy_func_1(1)", r"1 * 10"),
            (
                "helper_function",
                "dummy_func_2(1, with_inf)",
                r"1 + \textit{with_inf}_\text{node,tech}",
            ),
            (
                "helper_function",
                "dummy_func_2(dummy_func_1(1), with_inf)",
                r"1 * 10 + \textit{with_inf}_\text{node,tech}",
            ),
            ("arithmetic", "1 + with_inf", r"1 + \textit{with_inf}_\text{node,tech}"),
            (
                "arithmetic",
                "multi_dim_var[node=bar] + with_inf",
                r"\textbf{multi_dim_var}_\text{node=bar,tech} + \textit{with_inf}_\text{node,tech}",
            ),
            # We ignore zeros that make no difference
            ("arithmetic", "0 + with_inf", r"\textit{with_inf}_\text{node,tech}"),
            ("arithmetic", "0 - with_inf", r"\textit{with_inf}_\text{node,tech}"),
            ("arithmetic", "with_inf - 0", r"\textit{with_inf}_\text{node,tech}"),
            # We DO NOT ignore zeros that make a difference
            ("arithmetic", "with_inf**0", r"\textit{with_inf}_\text{node,tech}^{0}"),
            (
                "arithmetic",
                "0 * with_inf",
                r"0 \times \textit{with_inf}_\text{node,tech}",
            ),
            (
                "arithmetic",
                "0 / with_inf",
                r"\frac{ 0 }{ \textit{with_inf}_\text{node,tech} }",
            ),
            (
                "arithmetic",
                "(no_dims * no_dim_var) + (with_inf + 2)",
                r"(\textit{no_dims} \times \textbf{no_dim_var}) + (\textit{with_inf}_\text{node,tech} + 2)",
            ),
            (
                "equation_comparison",
                "no_dim_var >= with_inf",
                r"\textbf{no_dim_var} \geq \textit{with_inf}_\text{node,tech}",
            ),
            (
                "equation_comparison",
                "no_dim_var == with_inf",
                r"\textbf{no_dim_var} = \textit{with_inf}_\text{node,tech}",
            ),
        ],
    )
    def test_latex_eval(self, request, latex_eval_kwargs, parser, instring, expected):
        parser_func = request.getfixturevalue(parser)
        parsed_ = parser_func.parse_string(instring, parse_all=True)
        evaluated_ = parsed_[0].eval(**latex_eval_kwargs)
        assert evaluated_ == expected
