from __future__ import annotations

from io import StringIO
from itertools import chain, combinations

import pytest
import ruamel.yaml as yaml
import pyparsing as pp
import xarray as xr
import numpy as np
import pandas as pd

from calliope.backend import parsing, equation_parser
from calliope.test.common.util import check_error_or_warning
from calliope import AttrDict

BASE_DIMS = {"carriers", "carrier_tiers", "nodes", "techs"}
ALL_DIMS = {"nodes", "techs", "carriers", "costs", "timesteps", "carrier_tiers"}


def string_to_dict(yaml_string):
    yaml_loader = yaml.YAML(typ="safe", pure=True)
    return yaml_loader.load(StringIO(yaml_string))


@pytest.fixture
def dummy_model_data():
    d = {
        "A": {"dims": ("A"), "data": [1, 2]},
        "A1": {"dims": ("A1"), "data": [10, 20, 30]},
        "A_1": {"dims": ("A_1"), "data": [-1, -2, -3]},
        "techs": {"dims": ("techs"), "data": ["foo1", "bar1", "foobar1"]},
        "foo": {"dims": ("A", "A1"), "data": [["a", "b", "c"], ["d", "e", "f"]]},
        "bar": {"dims": ("A"), "data": [100, 200]},
    }
    return xr.Dataset.from_dict(d)


@pytest.fixture
def model_data():
    model_data = xr.Dataset(
        coords={
            dim: ["foo", "bar"]
            if dim != "techs"
            else ["foo", "bar", "foobar", "foobaz"]
            for dim in ALL_DIMS
        },
        data_vars={
            "node_tech": (
                ["nodes", "techs"],
                np.random.choice(a=[np.nan, True], p=[0.05, 0.95], size=(2, 4)),
            ),
            "carrier": (
                ["carrier_tiers", "carriers", "techs"],
                np.random.choice(a=[np.nan, True], p=[0.05, 0.95], size=(2, 2, 4)),
            ),
            "with_inf": (
                ["nodes", "techs"],
                [[1.0, np.nan, 1.0, 3], [np.inf, 2.0, True, np.nan]],
            ),
            "only_techs": (["techs"], [np.nan, 1, 2, 3]),
            "all_inf": (["nodes", "techs"], np.ones((2, 4)) * np.inf, {"is_result": 1}),
            "all_nan": (["nodes", "techs"], np.ones((2, 4)) * np.nan),
            "all_false": (["nodes", "techs"], np.zeros((2, 4)).astype(bool)),
            "all_true": (["nodes", "techs"], np.ones((2, 4)).astype(bool)),
            "with_inf_as_bool": (
                ["nodes", "techs"],
                [[True, False, True, True], [False, True, True, False]],
            ),
            "with_inf_as_bool_and_subset_on_bar_in_nodes": (
                ["nodes", "techs"],
                [[False, False, False, False], [False, True, True, False]],
            ),
            "with_inf_as_bool_or_subset_on_bar_in_nodes": (
                ["nodes", "techs"],
                [[True, False, True, True], [True, True, True, True]],
            ),
            "only_techs_as_bool": (["techs"], [False, True, True, True]),
            "with_inf_and_only_techs_as_bool": (
                ["nodes", "techs"],
                [[False, False, True, True], [False, True, True, False]],
            ),
            "with_inf_or_only_techs_as_bool": (
                ["nodes", "techs"],
                [[True, True, True, True], [False, True, True, True]],
            ),
            "inheritance": (
                ["nodes", "techs"],
                [
                    ["foo.bar", "boo", "baz", "boo"],
                    ["bar", "ar", "baz.boo", "foo.boo"],
                ],
            ),
        },
        attrs={"scenarios": ["foo"]},
    )
    model_data.attrs["run_config"] = AttrDict(
        {"foo": True, "bar": {"foobar": "baz"}, "foobar": {"baz": {"foo": np.inf}}}
    )
    model_data.attrs["model_config"] = AttrDict({"a_b": 0, "b_a": [1, 2]})

    model_data.attrs["defaults"] = AttrDict(
        {"all_inf": np.inf, "all_nan": np.nan, "with_inf": 100}
    )

    return model_data


class ComponentObj(parsing.ParsedBackendComponent):
    def parse_strings(self) -> None:
        return super().parse_strings()


@pytest.fixture(scope="function")
def component_obj():
    setup_string = """
    foreach: [a in A, a1 in A1]
    equation: 1 == 1
    """
    variable_data = string_to_dict(setup_string)
    return ComponentObj(variable_data, "foo")


@pytest.fixture
def expression_parser():
    return equation_parser.generate_equation_parser()


@pytest.fixture
def index_item_parser():
    return equation_parser.generate_index_item_parser()


@pytest.fixture
def component_parser():
    return equation_parser.generate_arithmetic_parser()


@pytest.fixture
def foreach_parser():
    return equation_parser.foreach_parser()


@pytest.fixture
def expression_generator():
    def _expression_generator(parse_string, where_string=None):
        expression_dict = {"expression": parse_string}
        if where_string is not None:
            expression_dict["where"] = where_string
        return expression_dict

    return _expression_generator


@pytest.fixture
def parse_where_expression(component_obj, expression_parser):
    def _parse_where_expression(expression_list, **kwargs):
        return component_obj._parse_where_expression(
            expression_parser, expression_list, "my_expr", **kwargs
        )

    return _parse_where_expression


def parse_components_and_index_items(
    parser, expression_list, expression_group, component_obj
):
    return {
        _name: component_obj._parse_where_expression(
            parser, _list, expression_group, _name
        )
        for _name, _list in expression_list.items()
    }


@pytest.fixture
def parsed_component_dict(component_obj, component_parser):
    def _parsed_component_dict(n_foo, n_bar):
        foos = ", ".join(
            [f"{{where: foo, expression: '{i + 1}'}}" for i in range(n_foo)]
        )
        bars = ", ".join(
            [f"{{where: bar, expression: '{i + 1}0'}}" for i in range(n_bar)]
        )
        setup_string = f"""
        foo: [{foos}]
        bar: [{bars}]
        """

        components = string_to_dict(setup_string)

        return parse_components_and_index_items(
            component_parser, components, "components", component_obj
        )

    return _parsed_component_dict


@pytest.fixture
def parsed_index_item_dict(component_obj, index_item_parser):
    def _parsed_component_dict(n_tech1, n_tech2):
        techs1 = ", ".join(["{where: techs, expression: foo}" for i in range(n_tech1)])
        techs2 = ", ".join(["{where: techs, expression: bar}" for i in range(n_tech2)])
        setup_string = f"""
        tech1: [{techs1}]
        tech2: [{techs2}]
        """

        index_items = string_to_dict(setup_string)

        return parse_components_and_index_items(
            index_item_parser, index_items, "index_items", component_obj
        )

    return _parsed_component_dict


@pytest.fixture
def parsed_idx_item_component_dict(component_obj, index_item_parser, component_parser):
    string_dict = {
        "components": """
        foo:
            - expression: 1 + foo
              where: foo1
            - expression: 2 + foo[techs=tech2]
              where: foo2
        bar:
            - expression: 1 + foo[techs=tech1]
              where: bar1
            - expression: 2 + foo[techs=tech2]
              where: bar2
        """,
        "index_items": """
        tech1:
            - expression: dummy_func_1(wind)
              where: techs1
            - expression: dummy_func_1(pv)
              where: techs2
        tech2:
            - expression: lookup_table[a]
              where: techs3
            - expression: lookup_table[a1]
              where: techs4
        """,
    }

    parser = {"index_items": index_item_parser, "components": component_parser}
    return (
        parse_components_and_index_items(
            parser[expr], string_to_dict(string_dict[expr]), expr, component_obj
        )
        for expr in ["components", "index_items"]
    )


class TestParsedConstraintForEach:
    @pytest.mark.parametrize(
        ["foreach_string", "expected"],
        [
            (["a in A"], {"a": "A"}),
            (["a in A", "b in A"], {"a": "A", "b": "A"}),
            (["a in A", "a1 in A1"], {"a": "A", "a1": "A1"}),
            (["a in A", "a_2 in A_2"], {"a": "A", "a_2": "A_2"}),
            (["a in A", "a_2 in A_2", "1 in 2"], {"a": "A", "a_2": "A_2"}),
        ],
    )
    def test_get_sets_from_foreach(self, component_obj, foreach_string, expected):
        sets = component_obj._get_sets_from_foreach(foreach_string)
        assert sets == expected

    def test_get_sets_from_foreach_duplicate_iterators(self):
        setup_string = """
        foreach: [a in A, a in A1]
        where: []
        equation: foo == 0
        """
        component_obj = ComponentObj(string_to_dict(setup_string), "foo")
        component_obj._get_sets_from_foreach(component_obj._unparsed["foreach"])
        assert check_error_or_warning(
            component_obj._errors,
            "(foreach, a in A1): Found duplicate set iterator `a`",
        )

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
    def test_parse_foreach_fail(self, foreach_parser, input_string):
        with pytest.raises(pp.ParseException):
            foreach_parser.parse_string(input_string, parse_all=True)


class TestParsedConstraintAddParseError:
    def test_add_parse_error_no_errors(self, component_obj):
        assert component_obj._is_valid
        assert not component_obj._errors

    def test_add_parse_error(self, component_obj):
        component_obj._add_parse_error("foo", "bar", "foobar")
        assert not component_obj._is_valid
        assert component_obj._errors == {"(bar, foo): foobar"}


class TestParsedConstraintParseString:
    @pytest.mark.parametrize(
        "parse_string",
        [
            "foo + bar == 1",
            "foo - $bar + baz[a, a1] <= 1",
            "-1**foo + dummy_func_1(2) + baz[a, a1] >= foobar",
        ],
    )
    def test_parse_string(self, component_obj, expression_parser, parse_string):
        parsed_ = component_obj._parse_string(expression_parser, parse_string, "foo")
        assert isinstance(parsed_, pp.ParseResults)
        assert not component_obj._errors

    @pytest.mark.parametrize(
        "parse_string",
        ["foo bar == 1", "foo - $bar + baz[foobar] = 1", "1foo == 1", "_foo >= foobar"],
    )
    def test_parse_string_malformed(
        self, component_obj, expression_parser, parse_string
    ):
        parsed_ = component_obj._parse_string(expression_parser, parse_string, "foo")
        assert parsed_ is None
        assert check_error_or_warning(component_obj._errors, "Expected")


class TestParsedConstraintParseWhereExpression:
    @pytest.mark.parametrize(
        "parse_string",
        ["foo == 1", "$foo + (bar + foobar[a1])**2 >= (dummy_func_1(foo) + 1)"],
    )
    @pytest.mark.parametrize(
        ["where_string", "expected_where_eval"],
        [
            (None, True),
            ("False", False),
            ("True or False", True),
        ],
    )
    def test_parse_where_expression(
        self,
        component_obj,
        expression_parser,
        expression_generator,
        parse_string,
        where_string,
        expected_where_eval,
    ):
        expression_dict = expression_generator(parse_string, where_string)
        parsed_list = component_obj._parse_where_expression(
            expression_parser, [expression_dict], "foo"
        )

        assert parsed_list[0]["where"][0][0].eval() == expected_where_eval
        assert isinstance(parsed_list[0]["expression"], pp.ParseResults)

    @pytest.mark.parametrize("n_dicts", [1, 2, 3])
    @pytest.mark.parametrize("id_prefix", ["foo", 1])
    def test_parse_where_expression_id_prefix(
        self,
        parse_where_expression,
        expression_generator,
        n_dicts,
        id_prefix,
    ):
        expression_dict = expression_generator("foo == 1", "bar")
        parsed_list = parse_where_expression(
            [expression_dict] * n_dicts, id_prefix=id_prefix
        )

        for expr_num in range(n_dicts):
            assert parsed_list[expr_num]["id"] == (id_prefix, expr_num)

    @pytest.mark.parametrize("n_dicts", [1, 2, 3])
    def test_parse_where_expression_no_id_prefix(
        self,
        parse_where_expression,
        expression_generator,
        n_dicts,
    ):
        expression_dict = expression_generator("foo == 1", "bar")
        parsed_list = parse_where_expression([expression_dict] * n_dicts)
        for expr_num in range(n_dicts):
            assert parsed_list[expr_num]["id"] == (expr_num,)

    def test_parse_where_expression_error(
        self, component_obj, parse_where_expression, expression_generator
    ):
        expression_dict = expression_generator("foo = 1")
        parsed_list = parse_where_expression([expression_dict])

        assert parsed_list[0]["expression"] is None
        assert check_error_or_warning(
            component_obj._errors, "(my_expr, foo = 1): Expected"
        )

    @pytest.mark.parametrize("error_position", [0, 1])
    def test_parse_where_expression_one_error(
        self,
        component_obj,
        parse_where_expression,
        expression_generator,
        error_position,
    ):
        expression_list = [expression_generator("foo == 1")]
        expression_list.insert(error_position, expression_generator("foo = 1"))

        parsed_list = parse_where_expression(expression_list)

        assert parsed_list[error_position]["expression"] is None
        assert isinstance(
            parsed_list[error_position - 1]["expression"], pp.ParseResults
        )

        assert len(component_obj._errors) == 1
        assert check_error_or_warning(
            component_obj._errors, "(my_expr, foo = 1): Expected"
        )

    def test_parse_where_expression_two_error(
        self, component_obj, parse_where_expression, expression_generator
    ):
        expression_list = [
            expression_generator("foo = 1"),
            expression_generator("foo = 2"),
        ]
        parsed_list = parse_where_expression(expression_list)

        assert all(parsed_list[i]["expression"] is None for i in range(2))

        assert len(component_obj._errors) == 2
        assert check_error_or_warning(
            component_obj._errors,
            ["(my_expr, foo = 1): Expected", "(my_expr, foo = 2): Expected"],
        )


class TestParsingEquationFindItemsInExpression:
    @pytest.mark.parametrize(
        "parse_string",
        [
            "$foo == $bar",
            "$foo + $bar >= 1",
            "$foo * $bar == 1",
            "($foo * 1) + $bar == 1",
            "(1**$bar) <= $foo + $bar",
            "(1 / $bar) <= $foo",
            "($foo - $bar) * ($foo + $bar) <= 2",
        ],
    )
    def test_find_components(self, expression_parser, component_obj, parse_string):
        parsed = expression_parser.parse_string(parse_string, parse_all=True)
        found_components = component_obj._find_items_in_expression(
            [parsed[0].lhs, parsed[0].rhs],
            equation_parser.EvalComponent,
            (equation_parser.EvalOperatorOperand),
        )
        assert found_components == set(["foo", "bar"])

    @pytest.mark.parametrize(
        "parse_string",
        [
            "foo[techs=tech1] == bar[techs=tech2]",
            "foo[techs=tech1] + bar[techs=tech2] >= 1",
            "foo[techs=tech1] * bar[techs=tech2] == 1",
            "(foo[techs=tech1] * 1) + bar[techs=tech2] == 1",
            "(1**bar[techs=tech2]) <= foo[techs=tech1] + bar[techs=tech2]",
            "(1 / bar[techs=tech2]) <= foo[techs=tech1]",
            "(foo[techs=tech1] - bar[techs=tech2]) * (foo[techs=tech1] + bar[techs=tech2]) <= 2",
        ],
    )
    def test_find_index_items(self, expression_parser, component_obj, parse_string):
        parsed = expression_parser.parse_string(parse_string, parse_all=True)
        found_index_items = component_obj._find_items_in_expression(
            [parsed[0].lhs, parsed[0].rhs],
            equation_parser.EvalIndexItems,
            (
                equation_parser.EvalOperatorOperand,
                equation_parser.EvalIndexedParameterOrVariable,
            ),
        )
        assert found_index_items == set(["tech1", "tech2"])


class TestParsedConstraintGetExpressionGroupProduct:
    @pytest.mark.parametrize("n_foos", [0, 1, 2])
    @pytest.mark.parametrize("n_bars", [0, 1, 2])
    def test_get_expression_group_product_components(
        self,
        component_obj,
        parsed_component_dict,
        parse_where_expression,
        expression_generator,
        n_foos,
        n_bars,
    ):
        equation_list = parse_where_expression([expression_generator("$foo == $bar")])
        component_product = list(
            component_obj._get_expression_group_product(
                equation_list[0], parsed_component_dict(n_foos, n_bars), "components"
            )
        )
        assert len(component_product) == n_foos * n_bars
        assert not component_obj._errors

    def test_get_expression_group_product_missing_component(
        self,
        component_obj,
        parse_where_expression,
        parsed_component_dict,
        expression_generator,
    ):
        equation_ = parse_where_expression([expression_generator("$foo == $bar + $ba")])
        component_product = component_obj._get_expression_group_product(
            equation_[0], parsed_component_dict(1, 2), "components"
        )
        assert len(list(component_product)) == 2
        assert check_error_or_warning(
            component_obj._errors, "Undefined components found in equation: {'ba'}"
        )

    @pytest.mark.parametrize(
        ["equation_", "expected"],
        [("$foo == $bar", False), ("$foo <= $bar", True), ("$foo * 20 >= $bar", True)],
    )
    def test_add_exprs_to_equation_data_multi(
        self,
        component_obj,
        parse_where_expression,
        parsed_component_dict,
        expression_generator,
        equation_,
        expected,
    ):
        component_dict = parsed_component_dict(2, 2)

        equation_dict = parse_where_expression([expression_generator(equation_)])[0]
        component_product = component_obj._get_expression_group_product(
            equation_dict, component_dict, "components"
        )
        combined_expression_list = [
            component_obj._add_exprs_to_equation_data(
                equation_dict, component_, "components"
            )
            for component_ in component_product
        ]
        # All IDs should be unique
        assert len(set(expr["id"] for expr in combined_expression_list)) == 4

        for constraint_eq in combined_expression_list:
            component_sub_dict = constraint_eq["components"]
            assert not set(component_sub_dict.keys()).symmetric_difference(
                ["foo", "bar"]
            )
            assert (
                constraint_eq["expression"][0].eval(component_dict=component_sub_dict)
                is expected
            )

    @pytest.mark.parametrize("n_1", [0, 1, 2])
    @pytest.mark.parametrize("n_2", [0, 1, 2])
    def test_get_expression_group_product_index_items(
        self,
        component_obj,
        parsed_index_item_dict,
        parse_where_expression,
        expression_generator,
        n_1,
        n_2,
    ):
        equation_ = parse_where_expression(
            [expression_generator("foo[techs=tech1] == bar[techs=tech2]")]
        )
        product_ = list(
            component_obj._get_expression_group_product(
                equation_[0], parsed_index_item_dict(n_1, n_2), "index_items"
            )
        )
        assert len(product_) == n_1 * n_2
        assert not component_obj._errors

    def test_get_expression_group_product_missing_index_items(
        self,
        component_obj,
        parse_where_expression,
        parsed_index_item_dict,
        expression_generator,
    ):
        equation_ = parse_where_expression(
            [expression_generator("foo[techs=tech1] == bar[techs=tech2, nodes=node1]")]
        )
        index_item_product = component_obj._get_expression_group_product(
            equation_[0], parsed_index_item_dict(1, 2), "index_items"
        )
        assert len(list(index_item_product)) == 2
        assert check_error_or_warning(
            component_obj._errors,
            "Undefined index_items found in equation: {'node1'}",
        )


class TestParsedConstraintAddExprsToEquationData:
    @pytest.mark.parametrize(
        ["equation_", "expected"],
        [("$foo == $bar", False), ("$foo <= $bar", True), ("$foo + 10 >= $bar", True)],
    )
    def test_add_exprs_to_equation_data(
        self,
        component_obj,
        parse_where_expression,
        parsed_component_dict,
        expression_generator,
        equation_,
        expected,
    ):
        component_dict = parsed_component_dict(1, 1)
        component_product = [component_dict["foo"][0], component_dict["bar"][0]]
        equation_list = parse_where_expression([expression_generator(equation_)])[0]

        combined_expression_dict = component_obj._add_exprs_to_equation_data(
            equation_list, component_product, "components"
        )
        component_sub_dict = combined_expression_dict["components"]
        assert not set(component_sub_dict.keys()).symmetric_difference(["foo", "bar"])
        assert (
            combined_expression_dict["expression"][0].eval(
                component_dict=component_sub_dict
            )
            is expected
        )


class TestParsedConstraintAddSubExprsPerEquationExpr:
    def test_add_sub_exprs_per_equation_expr_components(
        self,
        component_obj,
        parse_where_expression,
        parsed_component_dict,
        expression_generator,
    ):
        equation_list = parse_where_expression(
            [
                expression_generator("$foo == 1"),  # contributes 3
                expression_generator("$bar == 1"),  # contributes 2
                expression_generator("$foo == $bar"),  # contributes 6
                expression_generator("$foo == ($bar * $foo)**2"),  # contributes 6
            ]
        )

        final_equation_list = component_obj._add_sub_exprs_per_equation_expr(
            equations=equation_list,
            expression_dict=parsed_component_dict(3, 2),
            expression_group="components",
        )
        assert len(final_equation_list) == 17
        assert len(set(eq_["id"] for eq_ in final_equation_list)) == len(
            final_equation_list
        )

    @pytest.mark.parametrize(
        ["eq_string", "expected_n_equations"],
        [
            ("1 == bar[techs=tech2]", 2),
            ("$foo == bar[techs=tech2]", 4),
            ("$bar == 1", 4),
            ("$bar == bar[techs=tech2]", 6),
            ("$bar + $foo == bar[techs=tech2]", 12),
            ("$bar + $foo == bar[techs=tech2] + foo[techs=tech1]", 16),
        ],
    )
    def test_add_sub_exprs_per_equation_expr_components_and_index_items(
        self,
        component_obj,
        parse_where_expression,
        parsed_idx_item_component_dict,
        expression_generator,
        eq_string,
        expected_n_equations,
    ):
        equation_list = parse_where_expression([expression_generator(eq_string)])
        components, index_items = parsed_idx_item_component_dict
        component_equation_list = component_obj._add_sub_exprs_per_equation_expr(
            equations=equation_list,
            expression_dict=components,
            expression_group="components",
        )
        index_item_and_component_equation_list = (
            component_obj._add_sub_exprs_per_equation_expr(
                equations=component_equation_list,
                expression_dict=index_items,
                expression_group="index_items",
            )
        )

        assert len(index_item_and_component_equation_list) == expected_n_equations
        assert all(
            len(i["where"]) == len(i["index_items"]) + len(i["components"]) + 1
            for i in index_item_and_component_equation_list
        )


class TestParsedConstraintGetSubsetAsIndex:
    @pytest.mark.parametrize(
        "foreach", (["techs"], ["nodes", "techs"], ["nodes", "techs", "carriers"])
    )
    def test_get_subset_as_index(self, model_data, component_obj, foreach):
        component_obj.sets = {f"{i}": foreach_ for i, foreach_ in enumerate(foreach)}
        imask = component_obj._imask_foreach(model_data)
        reduced_imask = imask.sum(set(imask.dims).difference(foreach)) > 0
        idx = component_obj._get_subset_as_index(reduced_imask)
        imask_transposed = reduced_imask.transpose(*foreach)
        assert isinstance(idx, pd.Index)
        for i in imask_transposed.coords.to_index():
            if i in idx:
                assert imask_transposed.loc[i]
            else:
                assert not imask_transposed.loc[i]


class TestParsedConstraintImaskForeach:
    @pytest.mark.parametrize(
        "foreach",
        set(
            chain.from_iterable(
                combinations(ALL_DIMS, i) for i in range(1, len(ALL_DIMS))
            )
        ),
    )
    def test_imask_foreach_all_permutations(self, model_data, component_obj, foreach):
        component_obj.sets = {f"{i}": foreach_ for i, foreach_ in enumerate(foreach)}
        imask = component_obj._imask_foreach(model_data)

        assert not BASE_DIMS.difference(imask.dims)
        assert not set(foreach).difference(imask.dims)


class TestParsedConstraintCreateConstraintIndex:
    def apply_where_to_levels(self, component_obj, where_string, level):
        parsed_where = component_obj._parse_where_string({"where": where_string})
        true_where = component_obj._parse_where_string({"where": "True"})
        if level == "top_level_where":
            component_obj.top_level_where = parsed_where
        else:
            component_obj.top_level_where = true_where
        if level == "equation_dict":
            equation_dict = {"where": [true_where, parsed_where]}
        else:
            equation_dict = {"where": [true_where, true_where]}
        return equation_dict

    @pytest.mark.parametrize(
        "subsets",
        set(
            chain.from_iterable(
                combinations(BASE_DIMS, i) for i in range(1, len(BASE_DIMS))
            )
        ),
    )
    def test_create_subset_from_where_no_subset(
        self, model_data, component_obj, subsets
    ):
        component_obj.sets = {i: subset for i, subset in enumerate(subsets)}
        equation_dict = self.apply_where_to_levels(
            component_obj, "True", "top_level_where"
        )

        expected_imask = component_obj._imask_foreach(model_data)
        expected_imask = expected_imask.sum(BASE_DIMS.difference(subsets)) > 0
        expected_imask_idx = component_obj._get_subset_as_index(expected_imask)
        imask_idx = component_obj._create_subset_from_where(
            model_data, equation_dict["where"]
        )
        assert imask_idx.symmetric_difference(expected_imask_idx).empty

    @pytest.mark.parametrize("false_location", ["top_level_where", "equation_dict"])
    def test_create_subset_from_where_definitely_empty(
        self, model_data, component_obj, false_location
    ):
        component_obj.sets = {"node": "nodes", "tech": "techs"}
        equation_dict = self.apply_where_to_levels(
            component_obj, "False", false_location
        )
        imask = component_obj._create_subset_from_where(
            model_data, equation_dict["where"]
        )
        assert imask is None

    @pytest.mark.parametrize(
        ["where_string", "expected_imasker"],
        [
            ("with_inf", "with_inf_as_bool"),
            ("only_techs", "only_techs_as_bool"),
            ("with_inf and only_techs", "with_inf_and_only_techs_as_bool"),
            ("with_inf or only_techs", "with_inf_or_only_techs_as_bool"),
            ("with_inf or only_techs", "with_inf_or_only_techs_as_bool"),
            (
                "with_inf and [bar] in nodes",
                "with_inf_as_bool_and_subset_on_bar_in_nodes",
            ),
        ],
    )
    @pytest.mark.parametrize("level_", ["top_level_where", "equation_dict"])
    def test_create_subset_from_where_one_level_where(
        self, model_data, component_obj, where_string, expected_imasker, level_
    ):
        component_obj.sets = {"node": "nodes", "tech": "techs"}
        equation_dict = self.apply_where_to_levels(component_obj, where_string, level_)

        initial_expected_imask = component_obj._imask_foreach(model_data)
        added_imask = initial_expected_imask & model_data[expected_imasker]
        expected = component_obj._get_subset_as_index(
            added_imask.sum(BASE_DIMS.difference(["nodes", "techs"])) > 0
        )

        imask = component_obj._create_subset_from_where(
            model_data, equation_dict["where"]
        )
        assert imask.difference(expected).empty

    @pytest.mark.parametrize("level_", ["top_level_where", "equation_dict"])
    def test_create_subset_from_where_trim_dimension(
        self, model_data, component_obj, level_
    ):
        component_obj.sets = {"node": "nodes", "tech": "techs"}
        equation_dict = self.apply_where_to_levels(
            component_obj, "[foo] in carrier_tiers", level_
        )
        imask = component_obj._create_subset_from_where(
            model_data, equation_dict["where"]
        )
        assert not imask.names.difference(["nodes", "techs"])
