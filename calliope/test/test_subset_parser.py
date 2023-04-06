import pytest
import numpy as np
import pyparsing
import xarray as xr

from calliope.backend import equation_parser, subset_parser, subsets
from calliope.test.common.util import check_error_or_warning
from calliope.core.attrdict import AttrDict
from calliope.exceptions import BackendError


COMPONENT_CLASSIFIER = equation_parser.COMPONENT_CLASSIFIER
HELPER_FUNCS = {"dummy_func_1": lambda x: x * 10, "dummy_func_2": lambda x, y: x + y}

BASE_DIMS = ["nodes", "techs", "carriers", "costs", "timesteps", "carrier_tiers"]


def parse_yaml(yaml_string):
    return AttrDict.from_yaml_string(yaml_string)


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
def data_var(identifier):
    return subset_parser.data_var_parser(identifier)


@pytest.fixture
def config_option(identifier):
    return subset_parser.config_option_parser(identifier)


@pytest.fixture
def bool_operand():
    return subset_parser.bool_parser()


@pytest.fixture
def evaluatable_string(identifier):
    return subset_parser.evaluatable_string_parser(identifier)


@pytest.fixture
def helper_function(number, identifier, evaluatable_string):
    return equation_parser.helper_function_parser(
        evaluatable_string, number, generic_identifier=identifier
    )


@pytest.fixture
def comparison(
    evaluatable_string, number, helper_function, bool_operand, config_option, data_var
):
    return subset_parser.comparison_parser(
        evaluatable_string,
        number,
        helper_function,
        bool_operand,
        config_option,
        data_var,
    )


@pytest.fixture
def subset(identifier, evaluatable_string, number):
    return subset_parser.subset_parser(identifier, evaluatable_string, number)


@pytest.fixture
def imasking(bool_operand, helper_function, data_var, comparison, subset):
    return subset_parser.imasking_parser(
        bool_operand, helper_function, data_var, comparison, subset
    )


@pytest.fixture(scope="function")
def eval_kwargs(dummy_model_data):
    return {
        "model_data": dummy_model_data,
        "helper_func_dict": subsets.VALID_HELPER_FUNCTIONS,
        "test": True,
        "errors": set(),
        "warnings": set(),
    }


@pytest.fixture
def parse_where_string(eval_kwargs, imasking):
    def _parse_where_string(imasking_string):
        parsed_ = imasking.parse_string(imasking_string, parse_all=True)
        return parsed_[0].eval(**eval_kwargs)

    return _parse_where_string


class TestParserElements:
    @pytest.mark.parametrize(
        ["data_var_string", "expected"],
        [("with_inf", "with_inf"), ("all_inf", "all_inf"), ("all_nan", "all_nan")],
    )
    def test_data_var(
        self, data_var, dummy_model_data, data_var_string, expected, eval_kwargs
    ):
        parsed_ = data_var.parse_string(data_var_string, parse_all=True)
        default = dummy_model_data.attrs["defaults"][expected]
        assert (
            parsed_[0]
            .eval(apply_imask=False, **eval_kwargs)
            .equals(dummy_model_data[expected].fillna(default))
        )

    @pytest.mark.parametrize(
        ["data_var_string", "expected"],
        [
            ("with_inf", "with_inf_as_bool"),
            ("all_inf", "all_false"),
            ("all_nan", "all_false"),
        ],
    )
    def test_data_var_with_imask(
        self, data_var, dummy_model_data, data_var_string, expected, eval_kwargs
    ):
        parsed_ = data_var.parse_string(data_var_string, parse_all=True)

        # apply_imask=True is the default, but we also test being explicit.
        assert (
            parsed_[0]
            .eval(apply_imask=True, **eval_kwargs)
            .equals(dummy_model_data[expected])
        )
        assert parsed_[0].eval(**eval_kwargs).equals(dummy_model_data[expected])

    @pytest.mark.xfail(
        reason="No longer protected; there doesn't seem a reason to keep this."
    )
    @pytest.mark.parametrize("data_var_string", ["carrier", "node_tech", "inheritance"])
    def test_data_var_fail_protected(self, data_var, data_var_string):
        with pytest.raises(pyparsing.ParseException) as excinfo:
            data_var.parse_string(data_var_string, parse_all=True)
        assert check_error_or_warning(excinfo, "Found unwanted token")

    @pytest.mark.parametrize(
        "data_var_string", ["_foo", "__type__", "1foo", "with _ inf"]
    )
    def test_data_var_fail_malformed_string(self, data_var, data_var_string):
        with pytest.raises(pyparsing.ParseException) as excinfo:
            data_var.parse_string(data_var_string, parse_all=True)
        assert check_error_or_warning(excinfo, "Expected")

    @pytest.mark.parametrize("data_var_string", ["foo", "with_INF", "all_infs"])
    def test_data_var_fail_not_in_model(
        self, data_var, dummy_model_data, data_var_string, eval_kwargs
    ):
        parsed_ = data_var.parse_string(data_var_string, parse_all=True)
        evaluated_ = parsed_[0].eval(**eval_kwargs)
        assert not evaluated_

    @pytest.mark.parametrize(
        ["config_string", "expected_val"],
        [
            ("run.foo", True),
            ("run.bar.foobar", "baz"),
            ("run.foobar.baz.foo", np.inf),
            ("model.a_b", 0),
        ],
    )
    def test_config_option_valid(
        self, config_option, config_string, expected_val, eval_kwargs
    ):
        parsed_ = config_option.parse_string(config_string, parse_all=True)
        assert parsed_[0].eval(**eval_kwargs) == expected_val

    @pytest.mark.parametrize("config_string", ["run.a", "run.a.b", "model.a.b.c"])
    def test_config_option_missing_but_valid(
        self, config_option, config_string, eval_kwargs
    ):
        parsed_ = config_option.parse_string(config_string, parse_all=True)
        assert np.isnan(parsed_[0].eval(**eval_kwargs))

    @pytest.mark.parametrize(
        "config_string", ["run.", "run.", "RUN", "r un.foo", "model,a_b", "scenarios"]
    )
    def test_config_fail_malformed_string(self, config_option, config_string):
        with pytest.raises(pyparsing.ParseException) as excinfo:
            config_option.parse_string(config_string, parse_all=True)
        assert check_error_or_warning(excinfo, "Expected")

    @pytest.mark.parametrize("config_string", ["foo.bar", "all_inf.is_result"])
    def test_config_missing_config_group(
        self, config_option, eval_kwargs, config_string
    ):
        parsed_ = config_option.parse_string(config_string, parse_all=True)
        with pytest.raises(BackendError) as excinfo:
            parsed_[0].eval(**eval_kwargs)
        assert check_error_or_warning(excinfo, "Invalid configuration group")

    @pytest.mark.parametrize(
        ["config_string", "type_"], [("model.b_a", "list"), ("run.bar", "AttrDict")]
    )
    def test_config_fail_datatype(
        self, config_option, eval_kwargs, config_string, type_
    ):
        parsed_ = config_option.parse_string(config_string, parse_all=True)
        with pytest.raises(BackendError) as excinfo:
            parsed_[0].eval(**eval_kwargs)
        assert check_error_or_warning(
            excinfo,
            f"Configuration option resolves to invalid type `{type_}`",
        )

    @pytest.mark.parametrize(
        ["bool_string", "expected_true"],
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
        assert parsed_[0].eval() if expected_true else not parsed_[0].eval()

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
        parsed_[0].eval() == instring

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
        ["var_string", "comparison_val", "n_true"],
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
        comparison_string = f"{var_string}={comparison_val}"
        parsed_ = comparison.parse_string(comparison_string, parse_all=True)
        evaluated_ = parsed_[0].eval(**eval_kwargs)
        assert evaluated_.dtype.kind == "b"
        assert evaluated_.sum() == n_true

    @pytest.mark.parametrize(
        ["operator", "comparison_val", "n_true"],
        [
            ("=", ".inf", 1),
            ("<", 3, 4),
            ("<=", 3, 5),
            (">", 1, 5),
            (">=", 1, 8),
        ],
    )
    def test_comparison_parser_data_var_different_ops(
        self, comparison, eval_kwargs, operator, comparison_val, n_true
    ):
        comparison_string = f"with_inf{operator}{comparison_val}"
        parsed_ = comparison.parse_string(comparison_string, parse_all=True)
        evaluated_ = parsed_[0].eval(**eval_kwargs)
        assert evaluated_.sum() == n_true

    @pytest.mark.parametrize(
        ["config_string", "comparison_val", "expected_true"],
        [
            ("run.foo", "True", True),
            ("run.foo", "False", False),
            ("run.foo", 1, True),
            ("run.bar.foobar", "baz", True),
            ("run.bar.foobar", True, False),
            ("run.bar.foobar", "aaa", False),
            ("model.a_b", 0, True),
            ("model.a_b", 0.0, True),
            ("model.a_b", False, True),  # FIXME: should this be expected (0 == False)?
            ("model.a_b", 1, False),
        ],
    )
    def test_comparison_parser_model_config(
        self, comparison, eval_kwargs, config_string, comparison_val, expected_true
    ):
        comparison_string = f"{config_string}={comparison_val}"
        parsed_ = comparison.parse_string(comparison_string, parse_all=True)
        evaluated_ = parsed_[0].eval(**eval_kwargs)
        assert evaluated_ if expected_true else not evaluated_

    @pytest.mark.parametrize(
        "comparison_string",
        [
            "1=1",
            "run.foo==bar",
            "all_inf=__type__",
            "$foo=bar",
            "foo=$bar",
            "foo=run.bar",
            "run.foo=_bar",
        ],
    )
    def test_comparison_malformed_string(self, comparison, comparison_string):
        with pytest.raises(pyparsing.ParseException) as excinfo:
            comparison.parse_string(comparison_string, parse_all=True)
        assert check_error_or_warning(excinfo, "Expected")

    @pytest.mark.parametrize(
        ["subset_string", "expected_subset"],
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
        assert [i.eval() for i in parsed_[0].subset] == expected_subset

    @pytest.mark.parametrize(
        "subset_string",
        [
            "[bar] infoo",  # missing whitespace
            "[bar] in",  # missing set name
            "foo in [bar]",  # Wrong order of subset and set name
            "[foo=bar] in foo",  # comparison string in subset
            "[inheritance(a)] in foo"  # helper function in subset
            "(bar) in foo",  # wrong brackets
        ],
    )
    def test_subsetting_parser_malformed(self, subset, subset_string):
        with pytest.raises(pyparsing.ParseException) as excinfo:
            subset.parse_string(f"{subset_string} in foo", parse_all=True)
        assert check_error_or_warning(excinfo, "Expected")

    @pytest.mark.parametrize(
        ["parser_name", "parse_string", "expected"],
        [
            ("data_var", "foo", "DATA_VAR:foo"),
            ("config_option", "model.bar", "CONFIG:model_config.bar"),
            ("bool_operand", "TRUE", "BOOL:true"),
            ("comparison", "model.bar=True", "CONFIG:model_config.bar=BOOL:true"),
            ("subset", "[foo, 1] in foos", "SUBSET:foos[STRING:foo, NUM:1]"),
        ],
    )
    def test_repr(self, request, parser_name, parse_string, expected):
        parser = request.getfixturevalue(parser_name)
        parsed_ = parser.parse_string(parse_string, parse_all=True)
        assert str(parsed_[0]) == expected


class TestParserMasking:
    @pytest.mark.parametrize(
        ["instring", "expected"],
        [
            ("all_inf", "all_false"),
            ("run.foo=True", True),
            ("inheritance(boo)", "boo_inheritance_bool"),
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
        ["instring", "expected_true"],
        [
            ("run.foo=True and model.a_b=0", True),
            ("run.foo=False And model.a_b=0", False),
            ("run.foo=True AND model.a_b=1", False),
            ("run.foo=False and model.a_b=1", False),
            ("run.foo=1  and  model.a_b=0", True),
        ],
    )
    def test_imasking_and(self, parse_where_string, instring, expected_true):
        evaluated_ = parse_where_string(instring)
        assert evaluated_ if expected_true else not evaluated_

    @pytest.mark.parametrize(
        ["instring", "expected_true"],
        [
            ("run.foo=True or model.a_b=0", True),
            ("run.foo=False Or model.a_b=0", True),
            ("run.foo=True OR model.a_b=1", True),
            ("run.foo=False or model.a_b=1", False),
            ("run.foo=1 or model.a_b=0", True),
        ],
    )
    def test_imasking_or(self, parse_where_string, instring, expected_true):
        evaluated_ = parse_where_string(instring)
        assert evaluated_ if expected_true else not evaluated_

    @pytest.mark.parametrize(
        ["instring", "expected_true"],
        [
            ("not run.foo=True", False),
            ("Not run.foo=False and model.a_b=0", True),
            ("run.foo=True and NOT model.a_b=1", True),
            ("not run.foo=False and not model.a_b=1", True),
            ("run.foo=False or not model.a_b=0", False),
        ],
    )
    def test_imasking_not(self, parse_where_string, instring, expected_true):
        evaluated_ = parse_where_string(instring)
        assert evaluated_ if expected_true else not evaluated_

    @pytest.mark.parametrize(
        ["instring", "expected"],
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
    def test_imasking_arrays(
        self, parse_where_string, dummy_model_data, instring, expected
    ):
        evaluated_ = parse_where_string(instring)
        assert evaluated_.transpose(*dummy_model_data[expected].dims).equals(
            dummy_model_data[expected]
        )

    @pytest.mark.parametrize(
        ["instring", "expected"],
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
    def test_imasking_arrays_subsetting(
        self, parse_where_string, dummy_model_data, instring, expected
    ):
        evaluated_ = parse_where_string(instring)
        assert evaluated_.equals(dummy_model_data[expected])

    @pytest.mark.parametrize(
        ["instring", "expected"],
        [
            ("all_inf and all_nan or run.foo=True", "all_true"),
            ("all_inf and (all_nan or run.foo=True)", "all_false"),
            ("not all_inf and not run.foo=False ", "all_true"),
            (
                "(all_inf=inf and with_inf) or (run.foo=True and all_nan)",
                "with_inf_as_bool",
            ),
        ],
    )
    def test_mixed_imasking(
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
            "run.foo=True and and run.foo=True",
            "run.foo=True andnot all_inf",
        ],
    )
    def test_imasking_malformed(self, imasking, instring):
        with pytest.raises(pyparsing.ParseException) as excinfo:
            imasking.parse_string(instring, parse_all=True)
        assert check_error_or_warning(excinfo, "Expected")


class TestAsLatex:
    @pytest.fixture
    def latex_eval_kwargs(self, eval_kwargs):
        eval_kwargs["as_latex"] = True
        return eval_kwargs

    @pytest.mark.parametrize(
        ["parser", "instring", "expected"],
        [
            ("data_var", "with_inf", r"\exists (\textit{with_inf}_\text{node,tech})"),
            ("data_var", "foo", r"\exists (\textit{foo})"),
            ("data_var", "no_dims", r"\exists (\textit{no_dims})"),
            ("config_option", "run.foo", r"\text{run_config.foo}"),
            ("bool_operand", "True", "true"),
            ("comparison", "run.foo>1", r"\text{run_config.foo}\mathord{>}\text{1}"),
            (
                "comparison",
                "with_inf=True",
                r"\textit{with_inf}_\text{node,tech}\mathord{=}\text{true}",
            ),
            ("subset", "[foo, bar] in foos", r"\text{foo} \in \text{[foo,bar]}"),
            ("imasking", "NOT no_dims", r"\neg (\exists (\textit{no_dims}))"),
            (
                "imasking",
                "true AND with_inf",
                r"\exists (\textit{with_inf}_\text{node,tech})",
            ),
            (
                "imasking",
                "with_inf AND true",
                r"\exists (\textit{with_inf}_\text{node,tech})",
            ),
            (
                "imasking",
                "no_dims AND (with_inf OR run.foo>1)",
                r"\exists (\textit{no_dims}) \land (\exists (\textit{with_inf}_\text{node,tech}) \lor \text{run_config.foo}\mathord{>}\text{1})",
            ),
        ],
    )
    def test_latex_eval(self, request, latex_eval_kwargs, parser, instring, expected):
        parser_func = request.getfixturevalue(parser)
        parsed_ = parser_func.parse_string(instring, parse_all=True)
        evaluated_ = parsed_[0].eval(**latex_eval_kwargs)
        assert evaluated_ == expected
