from dataclasses import replace

import numpy as np
import pyparsing
import pyparsing as pp
import pytest
import xarray as xr

from calliope.backend import (
    eval_attrs,
    expression_parser,
    helper_functions,
    where_parser,
)
from calliope.exceptions import BackendError

from .common.util import check_error_or_warning

SUB_EXPRESSION_CLASSIFIER = expression_parser.SUB_EXPRESSION_CLASSIFIER

BASE_DIMS = ["nodes", "techs", "carriers", "costs", "timesteps"]


@pytest.fixture
def results():
    return ["multi_dim_var", "multi_dim_expr", "no_dim_var"]


@pytest.fixture
def input_names():
    return [
        "foo_bar",
        "with_inf",
        "all_inf",
        "all_nan",
        "only_techs",
        "no_dims",
        "all_true",
        "all_ones",
        "only_techs_as_bool",
    ]


@pytest.fixture
def dim_names():
    return ["techs", "nodes", "carriers", "costs", "timesteps", "dummy_dim"]


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
def dim_arr(dim_names):
    return where_parser.data_var_parser(dim_names, where_parser.DimensionArrayParser)


@pytest.fixture
def input_arr(input_names):
    return where_parser.data_var_parser(input_names, where_parser.InputArrayParser)


@pytest.fixture
def result_arr(results):
    return where_parser.data_var_parser(results, where_parser.ResultArrayParser)


@pytest.fixture
def config_option(identifier):
    return where_parser.config_option_parser(identifier)


@pytest.fixture
def bool_operand():
    return where_parser.bool_parser()


@pytest.fixture
def evaluatable_string(identifier, results, input_names, dim_names):
    valid_component_names = set(results) | set(input_names) | set(dim_names)
    return where_parser.evaluatable_string_parser(identifier, valid_component_names)


@pytest.fixture
def id_list(number, identifier, dim_arr):
    return expression_parser.list_parser(number, identifier, dim_arr)


@pytest.fixture
def subset(dim_arr, evaluatable_string, number):
    return where_parser.subset_parser(dim_arr, evaluatable_string, number)


@pytest.fixture
def arithmetic(
    number,
    evaluatable_string,
    subset,
    dim_arr,
    input_arr,
    config_option,
    id_list,
    identifier,
):
    arithmetic = pp.Forward()
    helper_func = expression_parser.helper_function_parser(
        arithmetic, evaluatable_string, id_list, generic_identifier=identifier
    )
    return expression_parser.arithmetic_parser(
        helper_func,
        subset,
        number,
        dim_arr,
        input_arr,
        config_option,
        arithmetic=arithmetic,
    )


@pytest.fixture
def helper_function(
    number, identifier, evaluatable_string, id_list, dim_arr, input_arr
):
    return expression_parser.helper_function_parser(
        dim_arr,
        input_arr,
        evaluatable_string,
        number,
        id_list,
        generic_identifier=identifier,
    )


@pytest.fixture
def comparison(evaluatable_string, number, bool_operand, arithmetic):
    return where_parser.comparison_parser(
        lhs=[arithmetic], rhs=[bool_operand, number, evaluatable_string]
    )


@pytest.fixture
def where(bool_operand, helper_function, input_arr, result_arr, comparison, subset):
    return where_parser.where_parser(
        bool_operand, helper_function, comparison, subset, input_arr, result_arr
    )


@pytest.fixture
def dummy_build_config():
    return {
        "foo": True,
        "FOO": "baz",
        "foo1": np.inf,
        "bar": {"foobar": "baz"},
        "a_b": 0,
        "b_a": [1, 2],
    }


@pytest.fixture
def eval_kwargs(dummy_pyomo_backend_model, dummy_build_config, dummy_model_math):
    attrs = eval_attrs.EvalAttrs(
        input_data=dummy_pyomo_backend_model.inputs,
        backend_data=dummy_pyomo_backend_model._dataset,
        math=dummy_model_math,
        helper_functions=helper_functions._registry["where"],
        equation_name="foo",
        build_config=dummy_build_config,
    )

    return {"return_type": "array", "eval_attrs": attrs}


@pytest.fixture
def eval_where_string(eval_kwargs, where):
    def _parse_where_string(where_string):
        parsed_ = where.parse_string(where_string, parse_all=True)
        return parsed_[0].eval(**eval_kwargs)

    return _parse_where_string


class TestDimensionParser:
    @pytest.mark.parametrize(("dim_string"), ["nodes", "techs", "carriers"])
    def test_dimension_lookup(self, dim_arr, dim_string, eval_kwargs, dummy_model_data):
        parsed_ = dim_arr.parse_string(dim_string, parse_all=True)
        evaluated_ = parsed_[0].eval(**eval_kwargs)
        assert evaluated_.equals(dummy_model_data.coords[dim_string])

    @pytest.mark.parametrize(
        "data_var_string", ["_foo", "__type__", "1foo", "[techs]", "tech"]
    )
    def test_dimension_lookup_fail_malformed_string(self, dim_arr, data_var_string):
        with pytest.raises(pyparsing.ParseException) as excinfo:
            dim_arr.parse_string(data_var_string, parse_all=True)
        assert check_error_or_warning(excinfo, "Expected")

    @pytest.mark.parametrize(
        "data_var_string", ["config.foo", "all_inf", "no_dims", "multi_dim_expr"]
    )
    def test_dimension_lookup_fail_not_a_dimension(self, dim_arr, data_var_string):
        with pytest.raises(pyparsing.ParseException) as excinfo:
            dim_arr.parse_string(data_var_string, parse_all=True)
        assert check_error_or_warning(excinfo, "Expected")

    def test_dimension_lookup_returns_empty(self, dim_arr, eval_kwargs):
        parsed_ = dim_arr.parse_string("dummy_dim", parse_all=True)
        evaluated = parsed_[0].eval(**eval_kwargs)
        assert evaluated.equals(xr.DataArray())


class TestInputParser:
    @pytest.mark.parametrize(
        ("data_var_string", "expected"),
        [("with_inf", "with_inf"), ("all_inf", "all_inf"), ("all_nan", "all_nan")],
    )
    def test_param_lookup(
        self, input_arr, dummy_model_data, data_var_string, expected, eval_kwargs
    ):
        parsed_ = input_arr.parse_string(data_var_string, parse_all=True)
        default = eval_kwargs["eval_attrs"].math.parameters[expected].default
        assert (
            parsed_[0]
            .eval(
                eval_kwargs["return_type"],
                replace(eval_kwargs["eval_attrs"], apply_where=False),
            )
            .equals(dummy_model_data[expected].fillna(default))
        )

    @pytest.mark.parametrize(
        ("data_var_string", "expected"),
        [
            ("with_inf", "with_inf_as_bool"),
            ("all_inf", "all_false"),
            ("all_nan", "all_false"),
        ],
    )
    @pytest.mark.parametrize("kwarg", [{"apply_where": True}, {}])
    def test_param_lookup_with_where(
        self, input_arr, dummy_model_data, data_var_string, expected, eval_kwargs, kwarg
    ):
        parsed_ = input_arr.parse_string(data_var_string, parse_all=True)
        eval_attrs_ = replace(eval_kwargs["eval_attrs"], **kwarg)

        assert (
            parsed_[0]
            .eval(eval_kwargs["return_type"], eval_attrs_)
            .equals(dummy_model_data[expected])
        )

    @pytest.mark.parametrize(
        "data_var_string",
        ["_foo", "__type__", "1foo", "with _ inf", "bar", "nodes", "multi_dim_var"],
    )
    def test_param_lookup_fail_malformed_string(self, input_arr, data_var_string):
        with pytest.raises(pyparsing.ParseException) as excinfo:
            input_arr.parse_string(data_var_string, parse_all=True)
        assert check_error_or_warning(excinfo, "Expected")

    @pytest.mark.parametrize("data_var_string", ["foo_bar"])
    def test_param_lookup_fail_not_in_model(
        self, input_arr, data_var_string, eval_kwargs
    ):
        parsed_ = input_arr.parse_string(data_var_string, parse_all=True)
        evaluated_ = parsed_[0].eval(**eval_kwargs)
        assert evaluated_.equals(xr.DataArray(False))


class TestResultArrayParser:
    @pytest.mark.parametrize(
        ("data_var_string", "expected_similar"),
        [("multi_dim_var", "with_inf_as_bool"), ("multi_dim_expr", "all_true")],
    )
    def test_data_var_with_where_decision_variable_or_expr(
        self,
        result_arr,
        dummy_model_data,
        data_var_string,
        expected_similar,
        eval_kwargs,
    ):
        """Can't quite compare in the same way for decision variables / global expressions
        as with params, because there is a random element to the `definition_matrix` array
        """
        parsed_ = result_arr.parse_string(data_var_string, parse_all=True)
        evaluated = parsed_[0].eval(**eval_kwargs)

        # There's a chance that some values that *should* be True in evaluated are made False by a NaN value in `definition_matrix`,
        # #so we check that at least all the remaining True values match
        assert (evaluated & dummy_model_data[expected_similar]).equals(evaluated)


class TestConfigOptionParser:
    @pytest.mark.parametrize(
        ("config_string", "expected_val"),
        [
            ("config.foo", True),
            ("config.FOO", "baz"),
            ("config.foo1", np.inf),
            ("config.a_b", 0),
        ],
    )
    def test_config_option_valid(
        self, config_option, config_string, expected_val, eval_kwargs
    ):
        parsed_ = config_option.parse_string(config_string, parse_all=True)
        assert parsed_[0].eval(**eval_kwargs) == expected_val

    @pytest.mark.parametrize("config_string", ["config.a"])
    def test_config_option_missing(self, config_option, config_string, eval_kwargs):
        parsed_ = config_option.parse_string(config_string, parse_all=True)
        with pytest.raises(KeyError):
            parsed_[0].eval(**eval_kwargs)

    @pytest.mark.parametrize(
        "config_string",
        ["config.", "config.bar.foobar", "RUN", "r un.foo", "model,a_b", "scenarios"],
    )
    def test_config_fail_malformed_string(self, config_option, config_string):
        with pytest.raises(pyparsing.ParseException) as excinfo:
            config_option.parse_string(config_string, parse_all=True)
        assert check_error_or_warning(excinfo, "Expected")

    @pytest.mark.parametrize(
        "config_string", ["CONFIG.bar", "foo.bar", "all_inf.is_result"]
    )
    def test_config_missing_config_identifier(self, config_option, config_string):
        with pytest.raises(pyparsing.ParseException) as excinfo:
            config_option.parse_string(config_string, parse_all=True)
        assert check_error_or_warning(excinfo, "Expected")

    @pytest.mark.parametrize("config_string", ["config.nonexistent", "config.config"])
    def test_config_missing_from_data(self, config_option, eval_kwargs, config_string):
        parsed_ = config_option.parse_string(config_string, parse_all=True)
        with pytest.raises(KeyError):
            parsed_[0].eval(**eval_kwargs)

    @pytest.mark.parametrize(
        ("config_string", "type_"), [("config.b_a", "list"), ("config.bar", "dict")]
    )
    def test_config_fail_datatype(
        self, config_option, eval_kwargs, config_string, type_
    ):
        parsed_ = config_option.parse_string(config_string, parse_all=True)
        with pytest.raises(BackendError) as excinfo:
            parsed_[0].eval(**eval_kwargs)
        assert check_error_or_warning(
            excinfo, f"Configuration option resolves to invalid type `{type_}`"
        )


class TestBoolParser:
    @pytest.mark.parametrize(
        ("bool_string", "expected_true"),
        [
            ("true", True),
            ("TRUE", True),
            ("True", True),
            ("false", False),
            ("FALSE", False),
            ("False", False),
        ],
    )
    def test_boolean_parser(self, bool_operand, bool_string, expected_true):
        parsed_ = bool_operand.parse_string(bool_string, parse_all=True)
        evaluated = parsed_[0].eval("array", eval_attrs.EvalAttrs())
        assert evaluated if expected_true else not evaluated

    @pytest.mark.parametrize(
        "bool_string", ["tru e", "_TRUE", "True_", "false1", "1false", "1", "foo"]
    )
    def test_boolean_parser_malformed(self, bool_operand, bool_string):
        with pytest.raises(pyparsing.ParseException) as excinfo:
            bool_operand.parse_string(bool_string, parse_all=True)
        assert check_error_or_warning(excinfo, "Expected")


class TestEvalStringParser:
    @pytest.mark.parametrize("instring", ["foo", "FOO", "foo10", "foo_10"])
    def test_evaluatable_string_parser(self, evaluatable_string, instring):
        parsed_ = evaluatable_string.parse_string(instring, parse_all=True)
        parsed_[0].eval("array", eval_attrs.EvalAttrs()) == instring

    @pytest.mark.parametrize("instring", ["_foo", "1foo", ".foo", "$foo", "__foo__"])
    def test_evaluatable_string_parser_malformed(self, evaluatable_string, instring):
        with pytest.raises(pyparsing.ParseException) as excinfo:
            evaluatable_string.parse_string(instring, parse_all=True)
        assert check_error_or_warning(excinfo, "Expected")

    @pytest.mark.parametrize(
        "instring", ["inf", ".inf", "techs", "foo_bar", "multi_dim_var"]
    )
    def test_evaluatable_string_parser_protected(self, evaluatable_string, instring):
        with pytest.raises(pyparsing.ParseException) as excinfo:
            evaluatable_string.parse_string(instring, parse_all=True)
        assert check_error_or_warning(excinfo, "Found unwanted token")


class TestConfigParser:
    @pytest.mark.parametrize(
        ("var_string", "comparison_val", "n_true"),
        [
            ("all_inf", ".inf", 8),
            ("all_inf", 1, 0),
            ("all_inf", "bar", 0),
            ("all_nan", 1, 0),
            ("all_nan", ".inf", 0),
            ("with_inf", ".inf", 1),
            ("with_inf", 3, 1),
            ("with_inf + 2", 3, 3),  # 1/True in original array will now be equal to 3.
            ("with_inf", 100, 2),  # NaNs filled with default val
        ],
    )
    def test_comparison_parser_data_var(
        self, comparison, eval_kwargs, var_string, comparison_val, n_true
    ):
        comparison_string = f"{var_string}=={comparison_val}"
        parsed_ = comparison.parse_string(comparison_string, parse_all=True)
        evaluated_ = parsed_[0].eval(**eval_kwargs)
        assert evaluated_.dtype.kind == "b"
        assert evaluated_.sum() == n_true

    @pytest.mark.parametrize(
        ("operator", "comparison_val", "n_true"),
        [("==", ".inf", 1), ("<", 3, 4), ("<=", 3, 5), (">", 1, 5), (">=", 1, 8)],
    )
    def test_comparison_parser_data_var_different_ops(
        self, comparison, eval_kwargs, operator, comparison_val, n_true
    ):
        comparison_string = f"with_inf{operator}{comparison_val}"
        parsed_ = comparison.parse_string(comparison_string, parse_all=True)
        evaluated_ = parsed_[0].eval(**eval_kwargs)
        assert evaluated_.sum() == n_true

    @pytest.mark.parametrize(
        ("comparison_string", "n_true"),
        [
            # 1/True in original array will now be equal to 3 as no_dims == 2.
            ("with_inf + no_dims == 3", 3),
            # only_techs has default 5, which fills two gaps and then 1 is added to all values, leaving two > 0
            ("all_ones + only_techs - 5 > 0", 2),
            ("sum(with_inf, over=techs) + sum(all_ones, over=nodes) <= 200", 4),
        ],
    )
    def test_comparison_parser_arithmetic(
        self, comparison, eval_kwargs, comparison_string, n_true
    ):
        parsed_ = comparison.parse_string(comparison_string, parse_all=True)
        evaluated_ = parsed_[0].eval(**eval_kwargs)
        assert evaluated_.sum() == n_true

    @pytest.mark.parametrize(
        ("config_string", "comparison_val", "expected_true"),
        [
            ("config.foo", "True", True),
            ("config.foo", "False", False),
            ("config.foo", 1, True),
            ("config.a_b", 0, True),
            ("config.a_b", 0.0, True),
            ("config.a_b", False, True),  # FIXME: should this be expected (0 == False)?
            ("config.a_b", 1, False),
        ],
    )
    def test_comparison_parser_model_config(
        self, comparison, eval_kwargs, config_string, comparison_val, expected_true
    ):
        comparison_string = f"{config_string}=={comparison_val}"
        parsed_ = comparison.parse_string(comparison_string, parse_all=True)
        evaluated_ = parsed_[0].eval(**eval_kwargs)
        assert evaluated_ if expected_true else not evaluated_

    @pytest.mark.parametrize(
        "comparison_string",
        [
            "config.foo=bar",
            "all_inf==__type__",
            "$foo==bar",
            "foo==$bar",
            "foo==config.bar",
            "config.foo==_bar",
        ],
    )
    def test_comparison_malformed_string(self, comparison, comparison_string):
        with pytest.raises(pyparsing.ParseException) as excinfo:
            comparison.parse_string(comparison_string, parse_all=True)
        assert check_error_or_warning(excinfo, "Expected")


class TestSubsettingParser:
    @pytest.mark.parametrize(
        ("subset_string", "expected_subset"),
        [
            ("[bar]", ["bar"]),
            ("[foobar, bar]", ["foobar", "bar"]),
            ("[ 1 ]", [1]),
            ("[1., 2e2]", [1.0, 200]),
            ("[1, bar]", [1.0, "bar"]),
        ],
    )
    def test_subsetting_parser(self, subset, subset_string, expected_subset):
        parsed_ = subset.parse_string(f"{subset_string} in nodes", parse_all=True)
        assert parsed_[0].set_name == "DIM:nodes"
        assert [
            i.eval("array", eval_attrs.EvalAttrs()) for i in parsed_[0].val
        ] == expected_subset

    @pytest.mark.parametrize(
        "subset_string",
        [
            "[bar] infoo",  # missing whitespace
            "[bar] in",  # missing set name
            "bar in [nodes]",  # Wrong order of subset and set name
            "[foo==bar] in nodes",  # comparison string in subset
            "[defined(techs=[tech1, tech2], within=nodes, how=any)] in nodes",  # helper function in subset
            "(bar) in nodes",  # wrong brackets
        ],
    )
    def test_subsetting_parser_malformed(self, subset, subset_string):
        with pytest.raises(pyparsing.ParseException) as excinfo:
            subset.parse_string(subset_string, parse_all=True)
        assert check_error_or_warning(excinfo, "Expected")


class TestRepr:
    @pytest.mark.parametrize(
        ("parser_name", "parse_string", "expected"),
        [
            ("dim_arr", "techs", "DIM:techs"),
            ("input_arr", "foo_bar", "INPUT:foo_bar"),
            ("result_arr", "multi_dim_var", "RESULT:multi_dim_var"),
            ("config_option", "config.bar", "CONFIG:bar"),
            ("bool_operand", "TRUE", "BOOL:true"),
            ("comparison", "config.bar==True", "CONFIG:bar==BOOL:true"),
            ("subset", "[bar, 1] in nodes", "SUBSET:DIM:nodes[STRING:bar, NUM:1]"),
        ],
    )
    def test_repr(self, request, parser_name, parse_string, expected):
        parser = request.getfixturevalue(parser_name)
        parsed_ = parser.parse_string(parse_string, parse_all=True)
        assert str(parsed_[0]) == expected


class TestParserMasking:
    @pytest.mark.parametrize(
        ("instring", "expected"),
        [
            ("all_inf", "all_false"),
            ("config.foo==True", True),
            ("get_val_at_index(nodes=0)", "foo"),
        ],
    )
    def test_no_aggregation(
        self, eval_where_string, dummy_model_data, instring, expected
    ):
        evaluated_ = eval_where_string(instring)
        if expected in dummy_model_data.data_vars:
            assert evaluated_.equals(dummy_model_data[expected])
        else:
            assert evaluated_ == expected

    @pytest.mark.parametrize(
        ("instring", "expected_true"),
        [
            ("config.foo==True and config.a_b==0", True),
            ("config.foo==False And config.a_b==0", False),
            ("config.foo==True AND config.a_b==1", False),
            ("config.foo==False and config.a_b==1", False),
            ("config.foo==1  and  config.a_b==0", True),
        ],
    )
    def test_where_string_and(self, eval_where_string, instring, expected_true):
        evaluated_ = eval_where_string(instring)
        assert evaluated_ if expected_true else not evaluated_

    @pytest.mark.parametrize(
        ("instring", "expected_true"),
        [
            ("config.foo==True or config.a_b==0", True),
            ("config.foo==False Or config.a_b==0", True),
            ("config.foo==True OR config.a_b==1", True),
            ("config.foo==False or config.a_b==1", False),
            ("config.foo==1 or config.a_b==0", True),
        ],
    )
    def test_where_string_or(self, eval_where_string, instring, expected_true):
        evaluated_ = eval_where_string(instring)
        assert evaluated_ if expected_true else not evaluated_

    @pytest.mark.parametrize(
        ("instring", "expected_true"),
        [
            ("not config.foo==True", False),
            ("Not config.foo==False and config.a_b==0", True),
            ("config.foo==True and NOT config.a_b==1", True),
            ("not config.foo==False and not config.a_b==1", True),
            ("config.foo==False or not config.a_b==0", False),
        ],
    )
    def test_where_string_not(self, eval_where_string, instring, expected_true):
        evaluated_ = eval_where_string(instring)
        assert evaluated_ if expected_true else not evaluated_

    @pytest.mark.parametrize(
        ("instring", "expected"),
        [
            ("all_inf and all_nan", "all_false"),
            ("all_inf or not all_nan", "all_true"),
            ("not all_inf and not all_nan", "all_true"),
            ("with_inf or all_inf or all_nan", "with_inf_as_bool"),
            ("with_inf and only_techs", "with_inf_and_only_techs_as_bool"),
            ("with_inf or only_techs", "with_inf_or_only_techs_as_bool"),
            ("only_techs and with_inf", "with_inf_and_only_techs_as_bool"),
            ("only_techs or with_inf", "with_inf_or_only_techs_as_bool"),
        ],
    )
    def test_where_arrays(
        self, eval_where_string, dummy_model_data, instring, expected
    ):
        evaluated_ = eval_where_string(instring)
        assert evaluated_.transpose(*dummy_model_data[expected].dims).equals(
            dummy_model_data[expected]
        )

    @pytest.mark.parametrize(
        ("instring", "expected"),
        [
            ("[foo, bar] in nodes", "nodes_true"),
            (
                "with_inf and [bar] in nodes",
                "with_inf_as_bool_and_subset_on_bar_in_nodes",
            ),
            (
                "with_inf and ([bar] in nodes)",
                "with_inf_as_bool_and_subset_on_bar_in_nodes",
            ),
            (
                "with_inf or [bar] in nodes",
                "with_inf_as_bool_or_subset_on_bar_in_nodes",
            ),
            (
                "with_inf or ([bar] in nodes)",
                "with_inf_as_bool_or_subset_on_bar_in_nodes",
            ),
        ],
    )
    def test_where_arrays_subsetting(
        self, eval_where_string, dummy_model_data, instring, expected
    ):
        evaluated_ = eval_where_string(instring)
        assert evaluated_.equals(dummy_model_data[expected])

    @pytest.mark.parametrize(
        ("instring", "expected"),
        [
            ("all_inf and all_nan or config.foo==True", "all_true"),
            ("all_inf and (all_nan or config.foo==True)", "all_false"),
            ("not all_inf and not config.foo==False ", "all_true"),
            (
                "(all_inf==inf and with_inf) or (config.foo==True and all_nan)",
                "with_inf_as_bool",
            ),
        ],
    )
    def test_mixed_where(self, eval_where_string, dummy_model_data, instring, expected):
        evaluated_ = eval_where_string(instring)
        if isinstance(evaluated_, xr.DataArray):
            assert evaluated_.equals(dummy_model_data[expected])

    @pytest.mark.parametrize(
        "instring",
        [
            "and",
            "or",
            "not",
            "not and",
            "and or",
            "all_inf and",
            "and all_inf",
            "with_inf not and all_inf",
            "with_inf and or all_inf",
            "config.foo==True and and config.foo==True",
            "config.foo==True andnot all_inf",
        ],
    )
    def test_where_malformed(self, where, instring):
        with pytest.raises(pyparsing.ParseException) as excinfo:
            where.parse_string(instring, parse_all=True)
        assert check_error_or_warning(excinfo, "Expected")


class TestAsMathString:
    @pytest.fixture
    def latex_eval_kwargs(
        self, eval_kwargs, dummy_latex_backend_model, dummy_model_math
    ):
        eval_kwargs["return_type"] = "math_string"
        eval_kwargs["eval_attrs"] = replace(
            eval_kwargs["eval_attrs"],
            backend_data=dummy_latex_backend_model._dataset,
            math=dummy_model_math,
        )
        return eval_kwargs

    @pytest.mark.parametrize(
        ("parser", "instring", "expected"),
        [
            ("input_arr", "with_inf", r"\exists (\textit{with_inf}_\text{node,tech})"),
            ("input_arr", "no_dims", r"\exists (\textit{no_dims})"),
            ("config_option", "config.foo", r"\text{config.foo}"),
            ("bool_operand", "True", "true"),
            ("comparison", "config.foo>1", r"\text{config.foo}\mathord{>}\text{1}"),
            (
                "comparison",
                "with_inf==True",
                r"\textit{with_inf}_\text{node,tech}\mathord{==}\text{true}",
            ),
            (
                "subset",
                "[foobar, bar] in techs",
                r"\text{tech} \in \text{[foobar,bar]}",
            ),
            ("where", "NOT no_dims", r"\neg (\exists (\textit{no_dims}))"),
            (
                "where",
                "true AND with_inf",
                r"\exists (\textit{with_inf}_\text{node,tech})",
            ),
            (
                "where",
                "with_inf AND true",
                r"\exists (\textit{with_inf}_\text{node,tech})",
            ),
            (
                "where",
                "no_dims AND (with_inf OR config.foo>1)",
                r"\exists (\textit{no_dims}) \land (\exists (\textit{with_inf}_\text{node,tech}) \lor \text{config.foo}\mathord{>}\text{1})",
            ),
        ],
    )
    def test_latex_eval(self, request, latex_eval_kwargs, parser, instring, expected):
        parser_func = request.getfixturevalue(parser)
        parsed_ = parser_func.parse_string(instring, parse_all=True)
        evaluated_ = parsed_[0].eval(**latex_eval_kwargs)
        assert evaluated_ == expected
