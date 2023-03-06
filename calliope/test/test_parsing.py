from io import StringIO
from itertools import chain, combinations

import pytest
import ruamel.yaml as yaml
import pyparsing as pp
import xarray as xr
import numpy as np
import pandas as pd

from calliope.backend import parsing, equation_parser, subset_parser, backends
from calliope.test.common.util import check_error_or_warning

from calliope import AttrDict
import calliope

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
            else ["foobar", "foobaz", "barfoo", "bazfoo"]
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


@pytest.fixture(scope="function")
def component_obj():
    setup_string = """
    foreach: [A, A1]
    equation: 1 == 1
    """
    variable_data = string_to_dict(setup_string)
    return parsing.ParsedBackendComponent(variable_data, "foo")


@pytest.fixture
def expression_parser():
    return equation_parser.generate_equation_parser()


@pytest.fixture
def index_slice_parser():
    return equation_parser.generate_index_slice_parser()


@pytest.fixture
def component_parser():
    return equation_parser.generate_arithmetic_parser()


@pytest.fixture
def where_parser():
    return subset_parser.generate_where_string_parser()


@pytest.fixture
def expression_generator():
    def _expression_generator(parse_string, where_string=None):
        expression_dict = {"expression": parse_string}
        if where_string is not None:
            expression_dict["where"] = where_string
        return expression_dict

    return _expression_generator


@pytest.fixture
def generate_expression_list(component_obj, expression_parser):
    def _generate_expression_list(expression_list, **kwargs):
        return component_obj.generate_expression_list(
            expression_parser, expression_list, "my_expr", **kwargs
        )

    return _generate_expression_list


def parse_components_and_index_slices(
    parser, expression_list, expression_group, component_obj
):
    return {
        _name: component_obj.generate_expression_list(
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

        return parse_components_and_index_slices(
            component_parser, components, "components", component_obj
        )

    return _parsed_component_dict


@pytest.fixture
def parsed_index_slice_dict(component_obj, index_slice_parser):
    def _parsed_index_slice_dict(n_tech1, n_tech2):
        techs1 = ", ".join(["{where: techs, expression: foo}" for i in range(n_tech1)])
        techs2 = ", ".join(["{where: techs, expression: bar}" for i in range(n_tech2)])
        setup_string = f"""
        tech1: [{techs1}]
        tech2: [{techs2}]
        """

        index_slices = string_to_dict(setup_string)

        return parse_components_and_index_slices(
            index_slice_parser, index_slices, "index_slices", component_obj
        )

    return _parsed_index_slice_dict


@pytest.fixture
def obj_with_components_and_index_slices():
    def _obj_with_components_and_index_slices(equation_string):
        if isinstance(equation_string, list):
            equation_string = f"equations: {equation_string}"
        elif isinstance(equation_string, str):
            equation_string = f"equation: {equation_string}"
        string_ = f"""
            foreach: [A, techs, A1]
            {equation_string}
            components:
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
            index_slices:
                tech1:
                    - expression: dummy_func_1(wind)
                      where: techs1
                    - expression: dummy_func_1(pv)
                      where: techs2
                tech2:
                    - expression: hi
                      where: techs3
                    - expression: hi_there
                      where: techs4
            """

        return parsing.ParsedBackendComponent(string_to_dict(string_), "my_constraint")

    return _obj_with_components_and_index_slices


@pytest.fixture(scope="function")
def equation_obj(expression_parser, where_parser):
    return parsing.ParsedBackendEquation(
        equation_name="foo",
        sets=["A", "A1"],
        expression=expression_parser.parse_string("foo == 1", parse_all=True),
        where_list=[where_parser.parse_string("True", parse_all=True)],
    )


@pytest.fixture(scope="function")
def equation_component_obj(component_parser, where_parser):
    def _equation_component_obj(name):
        return parsing.ParsedBackendEquation(
            equation_name=name,
            sets=["A", "A1"],
            expression=component_parser.parse_string("foo + 1", parse_all=True),
            where_list=[where_parser.parse_string("False", parse_all=True)],
        )

    return _equation_component_obj


@pytest.fixture(scope="function")
def equation_index_slice_obj(index_slice_parser, where_parser):
    def _equation_index_slice_obj(name):
        return parsing.ParsedBackendEquation(
            equation_name=name,
            sets=["A", "A1"],
            expression=index_slice_parser.parse_string("bar", parse_all=True),
            where_list=[where_parser.parse_string("False", parse_all=True)],
        )

    return _equation_index_slice_obj


def apply_where_to_levels(component_obj, where_string, level):
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


class TestParsedComponent:
    @pytest.mark.parametrize(
        "parse_string",
        [
            "foo + bar == 1",
            "foo - $bar + baz[A1=a1] <= 1",
            "-1**foo + dummy_func_1(2) + baz[A1=a1] >= foobar",
        ],
    )
    def test_parse_string(self, component_obj, expression_parser, parse_string):
        parsed_ = component_obj._parse_string(expression_parser, parse_string, "foo")
        assert isinstance(parsed_, pp.ParseResults)
        assert not component_obj._errors

    @pytest.mark.parametrize(
        "parse_string",
        ["foo bar == 1", "foo - $bar + baz[A1=a1] = 1", "1foo == 1", "_foo >= foobar"],
    )
    def test_parse_string_malformed(
        self, component_obj, expression_parser, parse_string
    ):
        parsed_ = component_obj._parse_string(expression_parser, parse_string, "foo")
        assert parsed_ is None
        assert check_error_or_warning(component_obj._errors, "Expected")

    @pytest.mark.parametrize(
        "parse_string",
        ["foo == 1", "$foo + (bar + foobar[A1=a1])**2 >= (dummy_func_1(foo) + 1)"],
    )
    @pytest.mark.parametrize(
        ["where_string", "expected_where_eval"],
        [
            (None, True),
            ("False", False),
            ("True or False", True),
        ],
    )
    def test_generate_expression_list(
        self,
        component_obj,
        expression_parser,
        expression_generator,
        parse_string,
        where_string,
        expected_where_eval,
    ):

        expression_dict = expression_generator(parse_string, where_string)
        parsed_list = component_obj.generate_expression_list(
            expression_parser, [expression_dict], "foo"
        )

        assert parsed_list[0].where[0][0].eval() == expected_where_eval
        assert isinstance(parsed_list[0].expression, pp.ParseResults)

    @pytest.mark.parametrize("n_dicts", [1, 2, 3])
    def test_generate_expression_list_id_prefix(
        self,
        generate_expression_list,
        expression_generator,
        n_dicts,
    ):
        expression_dict = expression_generator("foo == 1", "bar")
        parsed_list = generate_expression_list(
            [expression_dict] * n_dicts, id_prefix="foo"
        )

        for expr_num in range(n_dicts):
            assert parsed_list[expr_num].name == f"foo:{expr_num}"

    @pytest.mark.parametrize("n_dicts", [1, 2, 3])
    def test_generate_expression_list_no_id_prefix(
        self,
        generate_expression_list,
        expression_generator,
        n_dicts,
    ):
        expression_dict = expression_generator("foo == 1", "bar")
        parsed_list = generate_expression_list([expression_dict] * n_dicts)
        for expr_num in range(n_dicts):
            assert parsed_list[expr_num].name == str(expr_num)

    def test_generate_expression_list_error(
        self, component_obj, generate_expression_list, expression_generator
    ):
        expression_dict = expression_generator("foo = 1")
        parsed_list = generate_expression_list([expression_dict])

        assert not parsed_list
        assert check_error_or_warning(
            component_obj._errors, "(my_expr, foo = 1): Expected"
        )

    @pytest.mark.parametrize("error_position", [0, 1])
    def test_generate_expression_list_one_error(
        self,
        component_obj,
        generate_expression_list,
        expression_generator,
        error_position,
    ):
        expression_list = [expression_generator("foo == 1")]
        expression_list.insert(error_position, expression_generator("foo = 1"))

        parsed_list = generate_expression_list(expression_list)

        assert len(parsed_list) == 1
        assert isinstance(parsed_list[0].expression, pp.ParseResults)

        assert len(component_obj._errors) == 1
        assert check_error_or_warning(
            component_obj._errors, "(my_expr, foo = 1): Expected"
        )

    def test_generate_expression_list_two_error(
        self, component_obj, generate_expression_list, expression_generator
    ):
        expression_list = [
            expression_generator("foo = 1"),
            expression_generator("foo = 2"),
        ]
        parsed_list = generate_expression_list(expression_list)

        assert not parsed_list

        assert len(component_obj._errors) == 2
        assert check_error_or_warning(
            component_obj._errors,
            ["(my_expr, foo = 1): Expected", "(my_expr, foo = 2): Expected"],
        )

    @pytest.mark.parametrize("n_foos", [0, 1, 2])
    @pytest.mark.parametrize("n_bars", [0, 1, 2])
    def test_extend_equation_list_with_expression_group_components(
        self,
        component_obj,
        parsed_component_dict,
        generate_expression_list,
        expression_generator,
        n_foos,
        n_bars,
    ):
        equation_list = generate_expression_list([expression_generator("$foo == $bar")])
        expression_list = component_obj.extend_equation_list_with_expression_group(
            equation_list[0], parsed_component_dict(n_foos, n_bars), "components"
        )
        assert len(expression_list) == n_foos * n_bars

    def test_extend_equation_list_with_expression_group_missing_component(
        self,
        component_obj,
        generate_expression_list,
        parsed_component_dict,
        expression_generator,
    ):
        equation_ = generate_expression_list(
            [expression_generator("$foo == $bar + $ba")]
        )
        with pytest.raises(KeyError) as excinfo:
            component_obj.extend_equation_list_with_expression_group(
                equation_[0], parsed_component_dict(1, 2), "components"
            )
        assert check_error_or_warning(
            excinfo, "Undefined components found in equation: {'ba'}"
        )

    @pytest.mark.parametrize(
        ["equation_", "expected"],
        [("$foo == $bar", False), ("$foo <= $bar", True), ("$foo * 20 >= $bar", True)],
    )
    def test_add_exprs_to_equation_data_multi(
        self,
        component_obj,
        generate_expression_list,
        parsed_component_dict,
        expression_generator,
        equation_,
        expected,
    ):
        component_dict = parsed_component_dict(2, 2)

        equation_dict = generate_expression_list([expression_generator(equation_)])[0]
        expression_list = component_obj.extend_equation_list_with_expression_group(
            equation_dict, component_dict, "components"
        )
        # All IDs should be unique
        assert len(set(expr.name for expr in expression_list)) == 4

        for constraint_eq in expression_list:
            component_sub_dict = constraint_eq.components
            assert not set(component_sub_dict.keys()).symmetric_difference(
                ["foo", "bar"]
            )
            lhs, op, rhs = constraint_eq.expression[0].eval(
                component_dict=component_sub_dict
            )
            comparison_dict = {
                "==": lhs == rhs,
                ">=": lhs >= rhs,
                "<=": lhs <= rhs,
            }
            assert comparison_dict[op] is expected

    @pytest.mark.parametrize("n_1", [0, 1, 2])
    @pytest.mark.parametrize("n_2", [0, 1, 2])
    def test_extend_equation_list_with_expression_group_index_slices(
        self,
        component_obj,
        parsed_index_slice_dict,
        generate_expression_list,
        expression_generator,
        n_1,
        n_2,
    ):
        equation_ = generate_expression_list(
            [expression_generator("foo[techs=tech1] == bar[techs=tech2]")]
        )
        expression_list = component_obj.extend_equation_list_with_expression_group(
            equation_[0], parsed_index_slice_dict(n_1, n_2), "index_slices"
        )
        assert len(expression_list) == n_1 * n_2

    def test_extend_equation_list_with_expression_group_missing_index_slices(
        self,
        component_obj,
        generate_expression_list,
        parsed_index_slice_dict,
        expression_generator,
    ):
        equation_ = generate_expression_list(
            [expression_generator("foo[techs=tech1] == bar[techs=tech2, nodes=node1]")]
        )
        with pytest.raises(KeyError) as excinfo:
            component_obj.extend_equation_list_with_expression_group(
                equation_[0], parsed_index_slice_dict(1, 2), "index_slices"
            )
        assert check_error_or_warning(
            excinfo,
            "Undefined index_slices found in equation: {'node1'}",
        )

    @pytest.mark.parametrize(
        ["eq_string", "expected_n_equations"],
        [
            ("1 == bar", 1),
            ([{"expression": "1 == bar"}], 1),
            ([{"expression": "1 == bar"}, {"expression": "foo == bar"}], 2),
            ("1 == bar[techs=tech2]", 2),
            ("$foo == bar[techs=tech2]", 4),
            ("$bar == 1", 4),
            ("$bar == bar[techs=tech2]", 6),
            ("$bar + $foo == bar[techs=tech2]", 12),
            ("$bar + $foo == bar[techs=tech2] + foo[techs=tech1]", 16),
            (
                [
                    {"expression": "$foo == bar[techs=tech2]"},
                    {"expression": "$bar + $foo == bar[techs=tech2]"},
                ],
                16,
            ),
        ],
    )
    def test_parse_equations(
        self,
        obj_with_components_and_index_slices,
        expression_parser,
        eq_string,
        expected_n_equations,
    ):
        parsed_equation = obj_with_components_and_index_slices(eq_string)
        parsed_equations = parsed_equation.parse_equations(expression_parser)

        assert len(parsed_equations) == expected_n_equations
        assert len(set(eq.name for eq in parsed_equations)) == expected_n_equations


class TestParsedBackendEquation:
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
    def test_find_items_in_expression(
        self, expression_parser, equation_obj, parse_string
    ):
        parsed = expression_parser.parse_string(parse_string, parse_all=True)
        found_components = equation_obj._find_items_in_expression(
            [parsed[0].lhs, parsed[0].rhs],
            equation_parser.EvalComponent,
            (equation_parser.EvalOperatorOperand),
        )
        assert not found_components.symmetric_difference(["foo", "bar"])

    @pytest.mark.parametrize(
        ["parse_string", "expected"],
        [
            # components in comparisons are always seen
            ("$foo == $bar", ["foo", "bar"]),
            # components in arithmetic are missed
            ("1 + $bar >= $foo", ["foo"]),
            # components in arithmetic are missed
            ("$foo * $bar == 1", []),
            # components in arithmetic are missed
            ("($foo * 1) + $bar == 1", []),
            # components in functions are missed
            ("dummy_func_1($foo) == $bar", ["bar"]),
            # components in functions and arithmetic are missed
            ("dummy_func_1($foo) == $bar + 1", []),
        ],
    )
    def test_find_items_in_expression_missing_eval_class(
        self, expression_parser, equation_obj, parse_string, expected
    ):
        parsed = expression_parser.parse_string(parse_string, parse_all=True)
        found_components = equation_obj._find_items_in_expression(
            [parsed[0].lhs, parsed[0].rhs],
            equation_parser.EvalComponent,
            (),  # The above happens because we provide no eval classes to search inside
        )
        assert not found_components.symmetric_difference(expected)

    @pytest.mark.parametrize(
        "parse_string",
        [
            "foo[techs=tech1] == bar[techs=tech2]",
            "foo[techs=tech1] + bar[techs=tech2] >= 1",
            "(foo[techs=tech1] * 1) + bar[techs=tech2] == 1",
            "(1**bar[techs=tech2]) <= foo[techs=tech1] + 7",
            "(foo[techs=tech1] - bar[techs=tech2]) * 3 <= 2",
            "dummy_func_1(bar[techs=tech2]) <= dummy_func_2(x=foo[techs=tech1])",
            "dummy_func_1(bar[techs=tech2], x=foo[techs=tech1]) <= dummy_func_2(1)",
            "foo[techs=tech1] + 1 <= dummy_func_2(x=bar[techs=tech2])",
            "foo[techs=tech1] + dummy_func_2(bar[techs=tech2]) <= $foo",
        ],
    )
    def test_find_index_slices(self, expression_parser, equation_obj, parse_string):
        parsed = expression_parser.parse_string(parse_string, parse_all=True)
        equation_obj.expression = parsed
        found_index_slices = equation_obj.find_index_slices()
        assert not found_index_slices.symmetric_difference(["tech1", "tech2"])

    @pytest.mark.parametrize(
        "parse_string",
        [
            "$foo == $bar",
            "$foo + $bar >= 1",
            "dummy_func_1($foo) == $bar",
            "($foo * 1) + dummy_func_1($bar) == 1",
            "dummy_func_1($bar, x=$foo) <= 2",
        ],
    )
    def test_find_components(self, expression_parser, equation_obj, parse_string):
        parsed = expression_parser.parse_string(parse_string, parse_all=True)
        equation_obj.expression = parsed
        found_index_slices = equation_obj.find_components()
        assert not found_index_slices.symmetric_difference(["foo", "bar"])

    @pytest.mark.parametrize(
        ["equation_expr", "component_exprs"],
        [
            ("$foo == 1", {"foo": "foo[techs=tech1] + bar[techs=tech2]"}),
            ("$foo == $bar", {"foo": "foo[techs=tech1]", "bar": "bar[techs=tech2]"}),
            ("foo[techs=tech1] + $bar >= 1", {"bar": "bar[techs=tech2]"}),
            (
                "foo[techs=tech1] + $bar == $foo",
                {"foo": "10", "bar": "bar[techs=tech2]"},
            ),
        ],
    )
    def test_find_index_slices_in_expr_and_components(
        self,
        expression_parser,
        component_parser,
        equation_obj,
        equation_expr,
        component_exprs,
    ):
        equation_obj.expression = expression_parser.parse_string(
            equation_expr, parse_all=True
        )
        equation_obj.components = {
            component: component_parser.parse_string(expr_, parse_all=True)
            for component, expr_ in component_exprs.items()
        }
        found_index_slices = equation_obj.find_index_slices()
        assert not found_index_slices.symmetric_difference(["tech1", "tech2"])

    @pytest.mark.parametrize("expression_group", ["components", "index_slices"])
    def test_add_expression_group_combination(
        self, equation_obj, request, expression_group
    ):
        obj_ = request.getfixturevalue(
            f"equation_{expression_group.removesuffix('s')}_obj"
        )
        not_expression_group = [
            i for i in ["components", "index_slices"] if i != expression_group
        ][0]
        obj1 = obj_("bar:0")
        obj2 = obj_("baz:0")
        obj3 = obj_("bam:0")
        new_expression = equation_obj.add_expression_group_combination(
            expression_group, [obj1, obj2, obj3]
        )
        assert new_expression.expression == equation_obj.expression
        assert new_expression.sets == equation_obj.sets
        assert new_expression.name == "-".join(
            i.name for i in [equation_obj, obj1, obj2, obj3]
        )
        assert new_expression.where == [
            i.where[0] for i in [equation_obj, obj1, obj2, obj3]
        ]
        assert getattr(new_expression, not_expression_group) == {}
        assert getattr(new_expression, expression_group) == {
            "bar": obj1.expression,
            "baz": obj2.expression,
            "bam": obj3.expression,
        }

    def test_add_index_slices_after_components(
        self, equation_obj, equation_component_obj, equation_index_slice_obj
    ):
        equation_obj.components = {"bar": equation_component_obj("bar:0")}
        obj1 = equation_index_slice_obj("baz:0")
        obj2 = equation_index_slice_obj("bam:0")
        new_expression = equation_obj.add_expression_group_combination(
            "index_slices", [obj1, obj2]
        )

        assert new_expression.components == equation_obj.components
        assert new_expression.index_slices == {
            "baz": obj1.expression,
            "bam": obj2.expression,
        }

    @pytest.mark.parametrize(
        "foreach",
        set(
            chain.from_iterable(
                combinations(ALL_DIMS, i) for i in range(1, len(ALL_DIMS))
            )
        ),
    )
    def test_evaluate_foreach_all_permutations(self, model_data, equation_obj, foreach):
        equation_obj.sets = foreach
        imask = equation_obj._evaluate_foreach(model_data)

        assert not BASE_DIMS.difference(imask.dims)
        assert not set(foreach).difference(imask.dims)

    def test_imask_foreach_unidentified_name(self, model_data, equation_obj):
        equation_obj.sets = ["nodes", "techs", "foos"]
        with pytest.raises(calliope.exceptions.BackendError) as excinfo:
            equation_obj._evaluate_foreach(model_data)
        assert check_error_or_warning(
            excinfo, "Unidentified model set name(s) defined: `{'foos'}`"
        )
    def apply_where_to_levels(self, component_obj, where_string, level):
        parsed_where = component_obj._parse_where_string({"where": where_string})
        true_where = component_obj._parse_where_string({"where": "True"})
        if level == "top_level_where":
            component_obj.where = parsed_where
        else:
            component_obj.where = true_where
        if level == "equation_dict":
            equation_dict = {"where": [true_where, parsed_where]}
        else:
            equation_dict = {"where": [true_where, true_where]}
        return equation_dict


    @pytest.mark.parametrize(
        "foreach",
        set(
            chain.from_iterable(
                combinations(BASE_DIMS, i) for i in range(1, len(BASE_DIMS))
            )
        ),
    )
    def test_evaluate_where_no_imasking(self, model_data, equation_obj, foreach):
        equation_obj.sets = foreach
        expected_imask = (
            equation_obj._evaluate_foreach(model_data).sum(
                BASE_DIMS.difference(foreach)
            )
            > 0
        )
        imask = equation_obj.evaluate_where(model_data, defaults={})

        assert expected_imask.reindex_like(imask).equals(imask)

    @pytest.mark.parametrize("false_location", [0, -1])
    def test_create_subset_from_where_definitely_empty(
        self, model_data, equation_obj, where_parser, false_location
    ):
        equation_obj.sets = ["nodes", "techs"]
        equation_obj.where.insert(
            false_location, where_parser.parse_string("False", parse_all=True)
        )
        imask = equation_obj.evaluate_where(model_data, defaults={})

        assert not imask.all()

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
    @pytest.mark.parametrize("level_", ["initial_imask", "where"])
    def test_create_subset_from_where_one_level_where(
        self,
        model_data,
        equation_obj,
        where_parser,
        where_string,
        expected_imasker,
        level_,
    ):
        equation_obj.sets = ["nodes", "techs"]
        if level_ == "where":
            equation_obj.where = [
                where_parser.parse_string(where_string, parse_all=True)
            ]
            imask = equation_obj.evaluate_where(model_data, defaults={})
        if level_ == "initial_imask":
            equation_obj.where = [
                where_parser.parse_string(where_string, parse_all=True)
            ]
            initial_imask = equation_obj.evaluate_where(model_data, defaults={})
            equation_obj.where = [where_parser.parse_string("True", parse_all=True)]
            imask = equation_obj.evaluate_where(
                model_data, defaults={}, initial_imask=initial_imask
            )

        initial_expected_imask = equation_obj._evaluate_foreach(model_data)
        added_imask = initial_expected_imask & model_data[expected_imasker]
        expected = added_imask.sum(BASE_DIMS.difference(["nodes", "techs"])) > 0

        assert expected.reindex_like(imask).equals(imask)

    def test_create_subset_from_where_trim_dimension(
        self, model_data, where_parser, equation_obj
    ):
        equation_obj.sets = ["nodes", "techs"]
        equation_obj.where = [
            where_parser.parse_string("[foo] in carrier_tiers", parse_all=True)
        ]
        imask = equation_obj.evaluate_where(model_data, defaults={})

        assert not set(imask.dims).difference(["nodes", "techs"])

class TestEvaluateRule:
    @pytest.fixture
    def backend_interface(self):
        return backends.PyomoBackendModel()

    @pytest.mark.parametrize(
        ["equation_string", "expected"],
        [("2 == 1", False), ("1 == 1", True), ("2 >= 1", True)],
    )
    def test_evaluate_rule(
        self,
        component_obj,
        expression_generator,
        parse_where_expression,
        backend_interface,
        equation_string,
        expected,
    ):
        expression_dict = expression_generator(equation_string, "bar")
        parsed_dict = parse_where_expression([expression_dict])[0]
        parsed_dict["index_items"] = {}
        parsed_dict["components"] = {}
        rule_func = component_obj.evaluate_rule(
            "foobar", parsed_dict, backend_interface
        )
        evaluated_ = rule_func("dummy")
        assert evaluated_ == expected


class TestEvaluateSubset:
    @pytest.fixture
    def expected_subset(self, model_data):
        return (
            (
                (
                    (model_data.carrier.notnull() * model_data.node_tech.notnull()).sum(
                        ["carriers", "carrier_tiers"]
                    )
                    > 0
                )
                * model_data.with_inf_as_bool
            )
            .to_series()
            .where(lambda x: x)
            .dropna()
            .index.reorder_levels(["nodes", "techs"])
        )

    def test_evaluate_subset_no_name(self, component_obj, model_data, expected_subset):
        component_obj.sets = {"node": "nodes", "tech": "techs"}
        equation_dict = apply_where_to_levels(
            component_obj, "with_inf", "top_level_where"
        )
        subset_ = component_obj.evaluate_subset(model_data, equation_dict["where"])

        assert subset_.symmetric_difference(expected_subset).empty

        assert component_obj.index.equals(
            pd.Series(index=subset_, data="foo", dtype=str)
        )

    def test_evaluate_subset_name(self, component_obj, model_data, expected_subset):
        component_obj.sets = {"node": "nodes", "tech": "techs"}
        equation_dict = apply_where_to_levels(
            component_obj, "with_inf", "top_level_where"
        )
        subset_ = component_obj.evaluate_subset(
            model_data, equation_dict["where"], "foobar"
        )

        assert subset_.symmetric_difference(expected_subset).empty
        assert component_obj.index.equals(
            pd.Series(index=subset_, data="foobar", dtype=str)
        )

    def test_evaluate_multiple_subset(self, component_obj, model_data):
        component_obj.sets = {"node": "nodes", "tech": "techs"}
        equation_dict_with = apply_where_to_levels(
            component_obj, "with_inf", "equation_dict"
        )
        equation_dict_not_with = apply_where_to_levels(
            component_obj, "not with_inf", "equation_dict"
        )
        equation_dict_all = apply_where_to_levels(
            component_obj, "True", "equation_dict"
        )
        component_obj.evaluate_subset(model_data, equation_dict_with["where"])
        component_obj.evaluate_subset(model_data, equation_dict_not_with["where"])

        subset_all = component_obj._create_subset_from_where(
            model_data, equation_dict_all["where"]
        )

        assert not component_obj.index.index.duplicated().any()
        assert (component_obj.index == "foo").all()
        assert component_obj.index.index.symmetric_difference(subset_all).empty


class TestParsedConstraint:
    @pytest.fixture
    def constraint_obj(self):
        dict_ = {
            "foreach": ["tech in techs"],
            "where": "with_inf",
            "equation": "$foo == 1",
            "components": {
                "foo": [
                    {"expression": "bar + 2", "where": "False"},
                    {"expression": "bar + 1", "where": "True"},
                ]
            },
        }
        return parsing.ParsedConstraint(dict_, "foo")

    def test_parse_constraint_dict(self, constraint_obj, model_data):
        constraint_obj.parse_strings()

        assert constraint_obj.sets == {"tech": "techs"}
        assert len(constraint_obj.equations) == 2
        assert (
            constraint_obj.evaluate_subset(
                model_data, constraint_obj.equations[0]["where"]
            )
            is None
        )


class TestParsedVariable:
    @pytest.fixture
    def variable_obj(self):
        dict_ = {"foreach": ["tech in techs"], "where": "False"}

        return parsing.ParsedVariable(dict_, "foo")

    def test_parse_variable_dict(self, variable_obj, model_data):
        variable_obj.parse_strings()

        assert variable_obj.sets == {"tech": "techs"}
        assert len(variable_obj.equations) == 0
        assert variable_obj.evaluate_subset(model_data, []) is None


class TestParsedObjective:
    @pytest.fixture
    def objective_obj(self):
        dict_ = {
            "equations": [
                {"expression": "bar + 2", "where": "False"},
                {"expression": "bar + 1", "where": "True"},
            ]
        }

        return parsing.ParsedObjective(dict_, "foo")

    def test_parse_objective_dict(self, objective_obj, model_data):
        objective_obj.parse_strings()
        assert objective_obj.sets == {}
        assert len(objective_obj.equations) == 2
        assert (
            objective_obj.evaluate_subset(
                model_data, objective_obj.equations[0]["where"]
            )
            is None
        )
