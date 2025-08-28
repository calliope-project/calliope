import numpy as np
import pyparsing
import pytest
import xarray as xr

from calliope.backend import expression_parser, helper_functions, where_parser
from calliope.exceptions import BackendError

from .common.util import check_error_or_warning

SUB_EXPRESSION_CLASSIFIER = expression_parser.SUB_EXPRESSION_CLASSIFIER

BASE_DIMS = ["nodes", "techs", "carriers", "costs", "timesteps"]


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
def data_var(identifier):
    return where_parser.data_var_parser(identifier)


@pytest.fixture
def config_option(identifier):
    return where_parser.config_option_parser(identifier)


@pytest.fixture
def bool_operand():
    return where_parser.bool_parser()


@pytest.fixture
def evaluatable_string(identifier):
    return where_parser.evaluatable_string_parser(identifier)


@pytest.fixture
def helper_function(number, identifier, evaluatable_string):
    return expression_parser.helper_function_parser(
        evaluatable_string, number, generic_identifier=identifier
    )


@pytest.fixture
def comparison(
    evaluatable_string, number, helper_function, bool_operand, config_option, data_var
):
    return where_parser.comparison_parser(
        evaluatable_string,
        number,
        helper_function,
        bool_operand,
        config_option,
        data_var,
    )


@pytest.fixture
def subset(identifier, evaluatable_string, number):
    return where_parser.subset_parser(identifier, evaluatable_string, number)


@pytest.fixture
def where(bool_operand, helper_function, data_var, comparison, subset):
    return where_parser.where_parser(
        bool_operand, helper_function, data_var, comparison, subset
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
    return {
        "input_data": dummy_pyomo_backend_model.inputs,
        "backend_interface": dummy_pyomo_backend_model,
        "math": dummy_model_math,
        "helper_functions": helper_functions._registry["where"],
        "equation_name": "foo",
        "return_type": "array",
        "references": set(),
        "build_config": dummy_build_config,
    }


@pytest.fixture
def parse_where_string(eval_kwargs, where):
    def _parse_where_string(where_string):
        parsed_ = where.parse_string(where_string, parse_all=True)
        return parsed_[0].eval(**eval_kwargs)

    return _parse_where_string


class TestParserElements:
    @pytest.mark.parametrize(
        ("data_var_string", "expected"),
        [("with_inf", "with_inf"), ("all_inf", "all_inf"), ("all_nan", "all_nan")],
    )
    def test_data_var(
        self, data_var, dummy_model_data, data_var_string, expected, eval_kwargs
    ):
        parsed_ = data_var.parse_string(data_var_string, parse_all=True)
        default = eval_kwargs["input_data"][expected].attrs["default"]
        assert (
            parsed_[0]
            .eval(apply_where=False, **eval_kwargs)
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
    def test_data_var_with_where(
        self, data_var, dummy_model_data, data_var_string, expected, eval_kwargs, kwarg
    ):
        parsed_ = data_var.parse_string(data_var_string, parse_all=True)

        assert (
            parsed_[0].eval(**kwarg, **eval_kwargs).equals(dummy_model_data[expected])
        )

    @pytest.mark.parametrize(
        ("data_var_string", "expected_similar"),
        [("multi_dim_var", "with_inf_as_bool"), ("multi_dim_expr", "all_true")],
    )
    def test_data_var_with_where_decision_variable_or_expr(
        self, data_var, dummy_model_data, data_var_string, expected_similar, eval_kwargs
    ):
        """Can't quite compare in the same way for decision variables / global expressions
        as with params, because there is a random element to the `definition_matrix` array
        """
        parsed_ = data_var.parse_string(data_var_string, parse_all=True)
        evaluated = parsed_[0].eval(**eval_kwargs)

        # There's a chance that some values that *should* be True in evaluated are made False by a NaN value in `definition_matrix`,
        # #so we check that at least all the remaining True values match
        assert (evaluated & dummy_model_data[expected_similar]).equals(evaluated)

    @pytest.mark.parametrize(
        "data_var_string", ["_foo", "__type__", "1foo", "with _ inf"]
    )
    def test_data_var_fail_malformed_string(self, data_var, data_var_string):
        with pytest.raises(pyparsing.ParseException) as excinfo:
            data_var.parse_string(data_var_string, parse_all=True)
        assert check_error_or_warning(excinfo, "Expected")

    @pytest.mark.parametrize("data_var_string", ["foo", "with_INF", "all_infs"])
    def test_data_var_fail_not_in_model(self, data_var, data_var_string, eval_kwargs):
        parsed_ = data_var.parse_string(data_var_string, parse_all=True)
        with pytest.raises(
            BackendError,
            match=f"Data variable `{data_var_string}` not found in model dataset",
        ):
            parsed_[0].eval(**eval_kwargs)

    @pytest.mark.parametrize(
        "data_var_string", ["multi_dim_var", "no_dim_var", "multi_dim_expr"]
    )
    def test_data_var_fail_not_parameter_where_false(
        self, data_var, data_var_string, eval_kwargs
    ):
        parsed_ = data_var.parse_string(data_var_string, parse_all=True)
        with pytest.raises(BackendError) as excinfo:
            parsed_[0].eval(apply_where=False, **eval_kwargs)
        assert check_error_or_warning(
            excinfo,
            ["Can only check for existence of values", f"Received `{data_var_string}`"],
        )

    def test_data_var_fail_cannot_handle_constraint(self, data_var, eval_kwargs):
        parsed_ = data_var.parse_string("no_dim_constr", parse_all=True)
        with pytest.raises(BackendError) as excinfo:
            parsed_[0].eval(**eval_kwargs)
        assert check_error_or_warning(
            excinfo, ["Cannot check values", "Received constraint: `no_dim_constr`"]
        )

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
        evaluated = parsed_[0].eval(return_type="array")
        assert evaluated if expected_true else not evaluated

    @pytest.mark.parametrize(
        "bool_string", ["tru e", "_TRUE", "True_", "false1", "1false", "1", "foo"]
    )
    def test_boolean_parser_malformed(self, bool_operand, bool_string):
        with pytest.raises(pyparsing.ParseException) as excinfo:
            bool_operand.parse_string(bool_string, parse_all=True)
        assert check_error_or_warning(excinfo, "Expected")

    @pytest.mark.parametrize("instring", ["foo", "foo_bar", "FOO", "foo10", "foo_10"])
    def test_evaluatable_string_parser(self, evaluatable_string, instring):
        parsed_ = evaluatable_string.parse_string(instring, parse_all=True)
        parsed_[0].eval(return_type="array") == instring

    @pytest.mark.parametrize(
        "instring", ["_foo", "1foo", ".foo", "$foo", "__foo__", "foo bar", "foo-bar"]
    )
    def test_evaluatable_string_parser_malformed(self, evaluatable_string, instring):
        with pytest.raises(pyparsing.ParseException) as excinfo:
            evaluatable_string.parse_string(instring, parse_all=True)
        assert check_error_or_warning(excinfo, "Expected")

    @pytest.mark.parametrize("instring", ["inf", ".inf"])
    def test_evaluatable_string_parser_protected(self, evaluatable_string, instring):
        with pytest.raises(pyparsing.ParseException) as excinfo:
            evaluatable_string.parse_string(instring, parse_all=True)
        assert check_error_or_warning(excinfo, "Found unwanted token")

    @pytest.mark.parametrize(
        ("var_string", "comparison_val", "n_true"),
        [
            ("all_inf", ".inf", 8),
            ("all_inf", 1, 0),
            ("all_inf", "foo", 0),
            ("all_nan", 1, 0),
            ("all_nan", ".inf", 0),
            ("with_inf", ".inf", 1),
            ("with_inf", 3, 1),
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
            "1==1",
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

    @pytest.mark.parametrize(
        ("subset_string", "expected_subset"),
        [
            ("[bar]", ["bar"]),
            ("[foo, bar]", ["foo", "bar"]),
            ("[ 1 ]", [1]),
            ("[1., 2e2]", [1.0, 200]),
            ("[1, bar]", [1.0, "bar"]),
        ],
    )
    def test_subsetting_parser(self, subset, subset_string, expected_subset):
        parsed_ = subset.parse_string(f"{subset_string} in foo", parse_all=True)
        assert parsed_[0].set_name == "foo"
        assert [i.eval(return_type="array") for i in parsed_[0].val] == expected_subset

    @pytest.mark.parametrize(
        "subset_string",
        [
            "[bar] infoo",  # missing whitespace
            "[bar] in",  # missing set name
            "foo in [bar]",  # Wrong order of subset and set name
            "[foo==bar] in foo",  # comparison string in subset
            "[defined(techs=[tech1, tech2], within=nodes, how=any)] in foo",  # helper function in subset
            "(bar) in foo",  # wrong brackets
        ],
    )
    def test_subsetting_parser_malformed(self, subset, subset_string):
        with pytest.raises(pyparsing.ParseException) as excinfo:
            subset.parse_string(f"{subset_string} in foo", parse_all=True)
        assert check_error_or_warning(excinfo, "Expected")

    @pytest.mark.parametrize(
        ("parser_name", "parse_string", "expected"),
        [
            ("data_var", "foo", "DATA_VAR:foo"),
            ("config_option", "config.bar", "CONFIG:bar"),
            ("bool_operand", "TRUE", "BOOL:true"),
            ("comparison", "config.bar==True", "CONFIG:bar==BOOL:true"),
            ("subset", "[foo, 1] in foos", "SUBSET:foos[STRING:foo, NUM:1]"),
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
        self, parse_where_string, dummy_model_data, instring, expected
    ):
        evaluated_ = parse_where_string(instring)
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
    def test_where_string_and(self, parse_where_string, instring, expected_true):
        evaluated_ = parse_where_string(instring)
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
    def test_where_string_or(self, parse_where_string, instring, expected_true):
        evaluated_ = parse_where_string(instring)
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
    def test_where_string_not(self, parse_where_string, instring, expected_true):
        evaluated_ = parse_where_string(instring)
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
        self, parse_where_string, dummy_model_data, instring, expected
    ):
        evaluated_ = parse_where_string(instring)
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
        self, parse_where_string, dummy_model_data, instring, expected
    ):
        evaluated_ = parse_where_string(instring)
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
    def test_mixed_where(
        self, parse_where_string, dummy_model_data, instring, expected
    ):
        evaluated_ = parse_where_string(instring)
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
        eval_kwargs["backend_interface"] = dummy_latex_backend_model
        eval_kwargs["math"] = dummy_model_math
        return eval_kwargs

    @pytest.mark.parametrize(
        ("parser", "instring", "expected"),
        [
            ("data_var", "with_inf", r"\exists (\textit{with_inf}_\text{node,tech})"),
            ("data_var", "no_dims", r"\exists (\textit{no_dims})"),
            ("config_option", "config.foo", r"\text{config.foo}"),
            ("bool_operand", "True", "true"),
            ("comparison", "config.foo>1", r"\text{config.foo}\mathord{>}\text{1}"),
            (
                "comparison",
                "with_inf==True",
                r"\textit{with_inf}_\text{node,tech}\mathord{==}\text{true}",
            ),
            ("subset", "[foo, bar] in techs", r"\text{tech} \in \text{[foo,bar]}"),
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
