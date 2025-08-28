import operator
import random

import numpy as np
import pandas as pd
import pyparsing as pp
import pytest
import xarray as xr

from calliope import exceptions
from calliope.backend import expression_parser, helper_functions

from .common.util import check_error_or_warning

SUB_EXPRESSION_CLASSIFIER = expression_parser.SUB_EXPRESSION_CLASSIFIER


class DummyFunc1(helper_functions.ParsingHelperFunction):
    NAME = "dummy_func_1"
    ALLOWED_IN = ["expression"]

    def as_math_string(self, x):
        return f"{x} * 10"

    def as_array(self, x):
        return x * 10


class DummyFunc2(helper_functions.ParsingHelperFunction):
    NAME = "dummy_func_2"
    ALLOWED_IN = ["expression"]

    def as_math_string(self, x, y):
        return f"{x} + {y}"

    def as_array(self, x, y):
        return x * 10 + y


@pytest.fixture
def valid_component_names():
    return [
        "foo",
        "with_inf",
        "only_techs",
        "no_dims",
        "multi_dim_var",
        "no_dim_var",
        "all_true",
        "only_techs_as_bool",
    ]


@pytest.fixture
def base_parser_elements():
    number, identifier = expression_parser.setup_base_parser_elements()
    return number, identifier


@pytest.fixture
def number(base_parser_elements):
    return base_parser_elements[0]


@pytest.fixture
def identifier(base_parser_elements):
    return base_parser_elements[1]


@pytest.fixture
def evaluatable_identifier(identifier, valid_component_names):
    return expression_parser.evaluatable_identifier_parser(
        identifier, valid_component_names
    )


@pytest.fixture
def id_list(number, evaluatable_identifier):
    return expression_parser.list_parser(number, evaluatable_identifier)


@pytest.fixture
def unsliced_param():
    def _unsliced_param(valid_component_names):
        return expression_parser.unsliced_object_parser(valid_component_names)

    return _unsliced_param


@pytest.fixture
def unsliced_param_with_obj_names(unsliced_param, valid_component_names):
    return unsliced_param(valid_component_names)


@pytest.fixture
def sliced_param(
    number, identifier, evaluatable_identifier, unsliced_param_with_obj_names
):
    return expression_parser.sliced_param_or_var_parser(
        number, identifier, evaluatable_identifier, unsliced_param_with_obj_names
    )


@pytest.fixture
def sub_expression(identifier):
    return expression_parser.sub_expression_parser(identifier)


@pytest.fixture
def helper_function(
    number,
    sliced_param,
    sub_expression,
    unsliced_param_with_obj_names,
    identifier,
    id_list,
):
    return expression_parser.helper_function_parser(
        sliced_param,
        sub_expression,
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
    sub_expression,
    unsliced_param_with_obj_names,
    identifier,
    id_list,
):
    return expression_parser.helper_function_parser(
        sliced_param,
        sub_expression,
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
        ("sub_expression", "$foo", ["1.0", "foo", "foo[bars=bar]"]),
        ("unsliced_param_with_obj_names", "foo", ["1.0", "$foo", "foo[bars=bar]"]),
    ]
)
def helper_function_one_parser_in_args(identifier, request):
    parser_element, valid_string, invalid_string = request.param
    return (
        expression_parser.helper_function_parser(
            request.getfixturevalue(parser_element),
            generic_identifier=identifier,
            allow_function_in_function=True,
        ),
        valid_string,
        invalid_string,
    )


@pytest.fixture
def eval_kwargs(dummy_pyomo_backend_model):
    return {
        "helper_functions": helper_functions._registry["expression"],
        "slice_dict": {},
        "sub_expression_dict": {},
        "equation_name": "foobar",
        "where_array": xr.DataArray(True),
        "references": set(),
        "backend_interface": dummy_pyomo_backend_model,
        "math": dummy_pyomo_backend_model.math,
        "input_data": dummy_pyomo_backend_model.inputs,
        "return_type": "array",
    }


@pytest.fixture
def arithmetic(
    helper_function, number, sliced_param, sub_expression, unsliced_param_with_obj_names
):
    return expression_parser.arithmetic_parser(
        helper_function,
        sub_expression,
        sliced_param,
        number,
        unsliced_param_with_obj_names,
    )


@pytest.fixture
def helper_function_allow_arithmetic(
    number,
    sliced_param,
    sub_expression,
    unsliced_param_with_obj_names,
    identifier,
    arithmetic,
    id_list,
):
    arithmetic = pp.Forward()
    helper_func = expression_parser.helper_function_parser(
        arithmetic, id_list, generic_identifier=identifier
    )
    return expression_parser.arithmetic_parser(
        helper_func,
        sliced_param,
        sub_expression,
        unsliced_param_with_obj_names,
        number,
        arithmetic=arithmetic,
    )


@pytest.fixture
def equation_comparison(arithmetic):
    return expression_parser.equation_comparison_parser(arithmetic)


@pytest.fixture
def generate_equation(valid_component_names):
    return expression_parser.generate_equation_parser(valid_component_names)


@pytest.fixture
def generate_slice(valid_component_names):
    return expression_parser.generate_slice_parser(valid_component_names)


@pytest.fixture
def generate_sub_expression(valid_component_names):
    return expression_parser.generate_sub_expression_parser(valid_component_names)


class TestEquationParserElements:
    @pytest.mark.parametrize(
        ("string_val", "expected"),
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
        assert parsed_[0].eval(return_type="array") == expected

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
        "string_val", ["foo", "foo_bar", "Foo1", "FOO_1_BAR", "FOO__"]
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

    def test_id_list_with_numeric(self, id_list, eval_kwargs):
        parsed_ = id_list.parse_string("[hello, 1, 1.0, there]", parse_all=True)
        assert parsed_[0].eval(**eval_kwargs) == ["hello", 1.0, 1.0, "there"]

    @pytest.mark.parametrize("string_val", ["foo", "$foo", "dummy_func_1(1)"])
    def test_id_list_fail(self, id_list, string_val):
        with pytest.raises(pp.ParseException):
            id_list.parse_string(f"[{string_val}, there]", parse_all=True)

    @pytest.mark.parametrize("string_val", ["with_inf", "no_dims"])
    def test_unsliced_param(
        self, unsliced_param_with_obj_names, eval_kwargs, string_val
    ):
        parsed_ = unsliced_param_with_obj_names.parse_string(string_val, parse_all=True)
        default = eval_kwargs["input_data"][string_val].attrs["default"]
        assert (
            parsed_[0]
            .eval(**eval_kwargs)
            .equals(
                eval_kwargs["backend_interface"]._dataset[string_val].fillna(default)
            )
        )

    def test_unsliced_param_references(
        self, unsliced_param_with_obj_names, eval_kwargs
    ):
        references = eval_kwargs.pop("references")
        parsed_ = unsliced_param_with_obj_names.parse_string("with_inf", parse_all=True)
        parsed_[0].eval(references=references, **eval_kwargs)
        assert references == {"with_inf"}

    @pytest.mark.parametrize("string_val", ["Foo", "foobar"])
    def test_unsliced_param_fail(self, unsliced_param_with_obj_names, string_val):
        with pytest.raises(pp.ParseException):
            unsliced_param_with_obj_names.parse_string(string_val, parse_all=True)

    @pytest.mark.parametrize(
        ("string_val", "expected"),
        [
            ("foo[techs=tech]", "SLICED_COMPONENT:foo[techs=STRING:tech]"),
            (
                f"foo[techs={SUB_EXPRESSION_CLASSIFIER}tech]",
                "SLICED_COMPONENT:foo[techs=REFERENCE:tech]",
            ),
            (
                f"foo[techs=tech,bars={SUB_EXPRESSION_CLASSIFIER}bar]",
                "SLICED_COMPONENT:foo[techs=STRING:tech,bars=REFERENCE:bar]",
            ),
            (
                f"foo[ bars={SUB_EXPRESSION_CLASSIFIER}bar, techs=tech ]",
                "SLICED_COMPONENT:foo[bars=REFERENCE:bar,techs=STRING:tech]",
            ),
            (
                "with_inf[techs=tech, nodes=node]",
                "SLICED_COMPONENT:with_inf[techs=STRING:tech,nodes=STRING:node]",
            ),
        ],
    )
    def test_sliced_param(self, sliced_param, string_val, expected):
        parsed_ = sliced_param.parse_string(string_val, parse_all=True)
        parsed_[0] == expected

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
        "string_val",
        [
            # keeping explicit reference to "$" to ensure something weird doesn't happen
            # with the constant (e.g. is accidentally overwritten)
            "$foo",
            f"{SUB_EXPRESSION_CLASSIFIER}foo",
            f"{SUB_EXPRESSION_CLASSIFIER}Foo_Bar_1",
        ],
    )
    def test_sub_expression(self, sub_expression, string_val):
        parsed_ = sub_expression.parse_string(string_val, parse_all=True)
        assert (
            str(parsed_[0])
            == f"SUB_EXPRESSION:{string_val.removeprefix(SUB_EXPRESSION_CLASSIFIER)}"
        )

    @pytest.mark.parametrize(
        "string_val",
        [
            f"{SUB_EXPRESSION_CLASSIFIER} foo",  # space between classifier and sub-expression name
            "foo",  # no classifier
            "foo$",  # classifier not at start
            "f$oo",  # classifier not at start
            "$",  # missing sub-expression name
            "$foo(1)",  # adding classifier to function
            "$foo[bars=bar1]",  # adding classifier to indexed param
            "$1",  # adding classifier to invalid python variable name
        ],
    )
    def test_fail_sub_expression(self, sub_expression, string_val):
        with pytest.raises(pp.ParseException):
            sub_expression.parse_string(string_val, parse_all=True)

    @pytest.mark.parametrize(
        ("string_val", "expected"),
        [
            (
                f"dummy_func_1(foo[bars={SUB_EXPRESSION_CLASSIFIER}bar])",
                "dummy_func_1(args=[SLICED_COMPONENT:foo[bars=REFERENCE:bar]], kwargs={})",
            ),
            (
                "dummy_func_1(1, x=foo[bars=bar1])",
                "dummy_func_1(args=[NUM:1], kwargs={x=SLICED_COMPONENT:foo[bars=STRING:bar1]})",
            ),
            (
                "dummy_func_1($foo)",
                "dummy_func_1(args=[SUB_EXPRESSION:foo], kwargs={})",
            ),
            (
                "dummy_func_1($foo, x=$foo)",
                "dummy_func_1(args=[SUB_EXPRESSION:foo], kwargs={x=SUB_EXPRESSION:foo})",
            ),
            (
                "dummy_func_1(dummy_func_2(foo[bars=bar1], y=$foo))",
                "dummy_func_1(args=[dummy_func_2(args=[SLICED_COMPONENT:foo[bars=STRING:bar1]], kwargs={y=SUB_EXPRESSION:foo})], kwargs={})",
            ),
            (
                "dummy_func_1(1, dummy_func_2(1, foo[bars=$bar]), $foo, 1, foo, x=foo[foos=foo1, bars=bar1])",
                "dummy_func_1(args=[NUM:1, dummy_func_2(args=[NUM:1, SLICED_COMPONENT:foo[bars=REFERENCE:bar]], kwargs={}), SUB_EXPRESSION:foo, NUM:1, COMPONENT:foo], kwargs={x=SLICED_COMPONENT:foo[foos=STRING:foo1, bars=STRING:bar1]})",
            ),
        ],
    )
    def test_function_parse(self, helper_function, string_val, expected):
        parsed_ = helper_function.parse_string(string_val, parse_all=True)
        assert parsed_[0] == expected

    @pytest.mark.parametrize(
        ("string_val", "expected"),
        [
            ("dummy_func_1(1)", 10),
            ("dummy_func_1(x=1)", 10),
            ("dummy_func_2(1, y=2)", 12),
            ("dummy_func_2(1, 2)", 12),
            ("dummy_func_2(y=1, x=2)", 21),
            ("dummy_func_1(dummy_func_2(1, 2))", 120),
            ("dummy_func_1(x=dummy_func_2(1, y=2))", 120),
        ],
    )
    def test_function_eval(self, helper_function, string_val, expected, eval_kwargs):
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

    def test_function_mistype(self, helper_function, eval_kwargs):
        parsed_ = helper_function.parse_string("dummy_func_1(1)", parse_all=True)

        eval_kwargs["helper_functions"] = {"dummy_func_1": lambda **kwargs: lambda x: x}
        with pytest.raises(exceptions.BackendError) as excinfo:
            parsed_[0].eval(**eval_kwargs)

        assert check_error_or_warning(excinfo, "Helper function must be subclassed")

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


class TestEquationParserHelper:
    @pytest.mark.parametrize(
        ("where", "expected_notnull"),
        [
            ("all_true", [[True, True, True, True], [True, True, True, True]]),
            ("only_techs_as_bool", [False, True, True, True]),
        ],
    )
    def test_helper_function_where(
        self, helper_function, eval_kwargs, where, expected_notnull
    ):
        """Test that `where` helper function works as expected when passed a backend interface object."""
        string_ = f"where(no_dims, {where})"
        parsed_ = helper_function.parse_string(string_, parse_all=True)
        evaluated_ = parsed_[0].eval(**eval_kwargs)
        np.testing.assert_array_equal(evaluated_.notnull(), expected_notnull)


class TestEquationParserArithmetic:
    numbers = [2, 100, 0.02, "1e2", "2e-2", "inf"]

    @pytest.fixture(params=numbers)
    def float1(self, request):
        return float(request.param)

    @pytest.fixture(params=numbers)
    def float2(self, request):
        return float(request.param)

    @pytest.mark.parametrize(
        ("sign", "sign_name"),
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
            assert pd.isnull(evaluated_)
        else:
            assert evaluated_ == pytest.approx(
                getattr(operator, sign_name)(float1, float2)
            )

    @pytest.mark.parametrize(("sign", "sign_name"), [("+", "pos"), ("-", "neg")])
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
        ("equation_string", "expected"),
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
    @pytest.mark.parametrize("sub_expr_", ["$foo", "$bar1"])
    @pytest.mark.parametrize("unsliced_param_", ["foo", "with_inf"])
    @pytest.mark.parametrize(
        "sliced_param_", ["foo[bars=bar1]", "with_inf[foos=foo1, bars=$bar]"]
    )
    @pytest.mark.parametrize(
        "helper_function_", ["foo(1)", "bar1(foo, $foo, with_inf[foos=foo1], x=1)"]
    )
    @pytest.mark.parametrize(
        "func_string", ["arithmetic", "helper_function_allow_arithmetic"]
    )
    def test_non_numbers(
        self,
        number_,
        sub_expr_,
        unsliced_param_,
        sliced_param_,
        helper_function_,
        func_string,
        request,
    ):
        items = [number_, sub_expr_, unsliced_param_, sliced_param_, helper_function_]
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
        ("string_", "expected"), [("1 + 2", 30), ("1 * 2", 20), ("x=1/2", 5)]
    )
    def test_helper_function_allow_arithmetic(
        self, helper_function_allow_arithmetic, eval_kwargs, string_, expected
    ):
        helper_func_string = f"dummy_func_1({string_})"
        parsed = helper_function_allow_arithmetic.parse_string(
            helper_func_string, parse_all=True
        )
        evaluated = parsed[0].eval(**eval_kwargs)
        assert evaluated == expected

    def test_repr(self, arithmetic):
        parse_string = "1 + foo - foo[foos=foo1, bars=$bar] + (foo / $foo) ** -2"
        expected = (
            "(NUM:1 + COMPONENT:foo - SLICED_COMPONENT:foo[foos=STRING:foo1, bars=REFERENCE:bar]"
            " + ((COMPONENT:foo / SUB_EXPRESSION:foo) ** (-)NUM:2))"
        )
        parsed_ = arithmetic.parse_string(parse_string, parse_all=True)
        assert str(parsed_[0]) == expected


class TestIndexSliceParser:
    @pytest.mark.parametrize(
        "instring", ["foo", "1", "[FOO, BAR]", "foo[bars=bar1]", "FOO"]
    )
    @pytest.mark.parametrize("func_or_not", ["dummy_func_1({})", "{}"])
    def test_slice_expression_parser(self, generate_slice, instring, func_or_not):
        generate_slice.parse_string(func_or_not.format(instring), parse_all=True)

    @pytest.mark.parametrize(
        "instring", ["foo + 1", "foo == 1", "$foo", "foo[bars=$bar]", "[foo]"]
    )
    @pytest.mark.parametrize("func_or_not", ["dummy_func_1({})", "{}"])
    def test_slice_expression_parser_fail(self, generate_slice, instring, func_or_not):
        with pytest.raises(pp.ParseException):
            generate_slice.parse_string(func_or_not.format(instring), parse_all=True)


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
    def test_sub_expression_parser(
        self, generate_sub_expression, instring, func_or_not
    ):
        generate_sub_expression.parse_string(
            func_or_not.format(instring), parse_all=True
        )

    @pytest.mark.parametrize("instring", ["[FOO, BAR]", "foo == 1", "$foo", "[foo]"])
    def test_sub_expression_parser_fail(self, generate_sub_expression, instring):
        with pytest.raises(pp.ParseException):
            generate_sub_expression.parse_string(instring, parse_all=True)


class TestEquationParserComparison:
    EXPR_PARAMS_AND_EXPECTED_EVAL = {
        0: "NUM:0",
        -1: "(-)NUM:1",
        1e2: "NUM:100.0",
        "1/2": "(NUM:1 / NUM:2)",
        "2**2.0": "(NUM:2 ** NUM:2.0)",
        ".inf": "NUM:inf",
        "with_inf": "COMPONENT:with_inf",
        "foo[foos=foo1, bars=bar1]": "SLICED_COMPONENT:foo[foos=STRING:foo1, bars=STRING:bar1]",
        "$foo": "SUB_EXPRESSION:foo",
        "dummy_func_1(1, y=foo[bars=$bar])": "dummy_func_1(args=[NUM:1], kwargs={y=SLICED_COMPONENT:foo[bars=REFERENCE:bar]})",
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
    ):
        parsed_constraint = equation_comparison.parse_string(
            single_equation_simple, parse_all=True
        )
        evaluated_expression = parsed_constraint[0]

        assert evaluated_expression.lhs == expected_left
        assert evaluated_expression.rhs == expected_right
        assert evaluated_expression.op == operator

    @pytest.mark.parametrize(
        ("equation_string", "expected"),
        [
            ("1<=2", True),
            ("1 >= 2", False),
            ("1  ==  2", False),
            ("(1) <= (2)", True),
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
        evaluated = parsed_equation[0].eval(**eval_kwargs)
        assert evaluated if expected else not evaluated

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
            "(NUM:1 + COMPONENT:foo - SLICED_COMPONENT:foo[foos=STRING:foo1, bars=REFERENCE:bar])"
            " >= ((COMPONENT:foo / SUB_EXPRESSION:foo) ** (-)NUM:2)"
        )
        parsed_ = equation_comparison.parse_string(parse_string, parse_all=True)
        assert str(parsed_[0]) == expected


class TestAsMathString:
    @pytest.fixture
    def latex_eval_kwargs(self, dummy_latex_backend_model, dummy_model_math):
        return {
            "helper_functions": helper_functions._registry["expression"],
            "return_type": "math_string",
            "index_slice_dict": {},
            "component_dict": {},
            "equation_name": "foobar",
            "where_array": None,
            "references": set(),
            "backend_interface": dummy_latex_backend_model,
            "math": dummy_model_math,
            "input_data": dummy_latex_backend_model.inputs,
        }

    @pytest.mark.parametrize(
        ("parser", "instring", "expected"),
        [
            ("number", "1", "1"),
            ("number", "1.0", "1"),
            ("number", "0.01", "0.01"),
            ("number", "inf", "inf"),
            ("number", "-1", "-1"),
            ("number", "2000000", "2\\mathord{\\times}10^{+06}"),
            ("evaluatable_identifier", "hello_there", "hello_there"),
            ("id_list", "[hello, hello_there]", "[hello,hello_there]"),
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
                "with_inf[nodes=bar]",
                r"\textit{with_inf}_\text{node=bar,tech}",
            ),
            (
                "sliced_param",
                "only_techs[techs=foobar]",
                r"\textit{only_techs}_\text{tech=foobar}",
            ),
            (
                "sliced_param",
                "multi_dim_var[nodes=bar]",
                r"\textbf{multi_dim_var}_\text{node=bar,tech}",
            ),
            (
                "sliced_param",
                "with_inf[nodes=bar, techs=foobar]",
                r"\textit{with_inf}_\text{node=bar,tech=foobar}",
            ),
            (
                "sliced_param",
                "multi_dim_var[nodes=bar, techs=foobar]",
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
                "multi_dim_var[nodes=bar] + with_inf",
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
