from io import StringIO
from itertools import chain, combinations

import pytest
import ruamel.yaml as yaml
import pyparsing as pp

from calliope.backend import parsing, equation_parser, subset_parser, backends
from calliope.test.common.util import check_error_or_warning

import calliope

BASE_DIMS = {"carriers", "carrier_tiers", "nodes", "techs"}


def string_to_dict(yaml_string):
    yaml_loader = yaml.YAML(typ="safe", pure=True)
    return yaml_loader.load(StringIO(yaml_string))


@pytest.fixture(scope="function")
def component_obj():
    setup_string = """
    name: foo
    foreach: [A, A1]
    where: "True"
    equation: 1 == 1
    """
    variable_data = string_to_dict(setup_string)
    return parsing.ParsedBackendComponent(variable_data)


@pytest.fixture(scope="function")
def foreach_imask(component_obj, dummy_model_data):
    component_obj.sets = ["nodes", "techs"]
    return component_obj.evaluate_foreach(dummy_model_data)


@pytest.fixture
def valid_object_names(dummy_model_data):
    return ["foo", "bar", "baz", "foobar", *dummy_model_data.data_vars.keys()]


@pytest.fixture
def expression_parser(valid_object_names):
    return equation_parser.generate_equation_parser(valid_object_names)


@pytest.fixture
def index_slice_parser(valid_object_names):
    return equation_parser.generate_index_slice_parser(valid_object_names)


@pytest.fixture
def component_parser(valid_object_names):
    return equation_parser.generate_component_parser(valid_object_names)


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
            name: my_constraint
            foreach: [A, techs, A1]
            {equation_string}
            components:
                foo:
                    - expression: 1 + foo
                      where: foo1
                    - expression: 2 + foo[techs=$tech2]
                      where: foo2
                bar:
                    - expression: 1 + foo[techs=$tech1]
                      where: bar1
                    - expression: 2 + foo[techs=$tech2]
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

        return parsing.ParsedBackendComponent(string_to_dict(string_))

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


@pytest.fixture
def dummy_backend_interface(dummy_model_data):
    class DummyBackendModel(backends.BackendModel):
        def __init__(self):
            backends.BackendModel.__init__(self, instance=None)

            self._dataset = dummy_model_data.copy(deep=True)
            self._dataset["with_inf"] = self._dataset["with_inf"].fillna(
                dummy_model_data.attrs["defaults"]["with_inf"]
            )
            self._dataset["only_techs"] = self._dataset["only_techs"].fillna(
                dummy_model_data.attrs["defaults"]["only_techs"]
            )

        def add_parameter(self):
            pass

        def add_constraint(self):
            pass

        def add_expression(self):
            pass

        def add_variable(self):
            pass

        def add_objective(self):
            pass

        def get_parameter(self):
            pass

        def get_constraint(self):
            pass

        def get_variable(self):
            pass

        def get_expression(self):
            pass

        def solve(self):
            pass

    return DummyBackendModel()


@pytest.fixture(scope="function")
def evaluatable_component_obj(valid_object_names):
    def _evaluatable_component_obj(equation_expressions):
        if isinstance(equation_expressions, list):
            equations = f"equations: {equation_expressions}"
        elif isinstance(equation_expressions, str):
            equations = f"equation: {equation_expressions}"
        setup_string = f"""
        name: foo
        foreach: [techs, nodes]
        where: with_inf
        {equations}
        components:
            foo: [{{expression: with_inf * 2, where: only_techs}}]
        index_slices:
            tech: [{{expression: barfoo, where: "[bar] in nodes"}}]
        """
        component_dict = string_to_dict(setup_string)

        class DummyParsedBackendComponent(parsing.ParsedBackendComponent):
            def __init__(self, dict_):
                parsing.ParsedBackendComponent.__init__(self, dict_)
                self.parse_top_level_where()
                self.equations = self.parse_equations(
                    equation_parser.generate_equation_parser, valid_object_names
                )

        return DummyParsedBackendComponent(component_dict)

    return _evaluatable_component_obj


@pytest.fixture(
    params=[
        ("with_inf <= 100", 7),  # all vals except .inf meet criterion
        ("with_inf == 100", 2),  # only default vals meet criterion
        (
            "$foo <= 100",
            4,
        ),  # only non-default + non-inf values meet criterion (+ only_techs masks one valid value)
        ("$foo == 100", 0),  # no expressions are valid
        ("only_techs + with_inf[techs=$tech] == 2", 1),
    ]
)
def evaluate_component_where(evaluatable_component_obj, dummy_model_data, request):
    component_obj = evaluatable_component_obj(request.param[0])
    foreach_imask = component_obj.evaluate_foreach(dummy_model_data)
    top_level_imask = component_obj.evaluate_where(dummy_model_data, foreach_imask)
    equation_imask = component_obj.equations[0].evaluate_where(
        dummy_model_data, top_level_imask
    )
    equation_imask_squeezed = component_obj.align_imask_with_sets(equation_imask)

    return component_obj, equation_imask_squeezed, request.param[1]


@pytest.fixture
def evaluate_component_expression(
    evaluate_component_where, dummy_model_data, dummy_backend_interface
):
    component_obj, equation_imask, n_true = evaluate_component_where
    return (
        component_obj.equations[0].evaluate_expression(
            dummy_model_data, dummy_backend_interface, equation_imask
        ),
        n_true,
    )


def apply_comparison(comparison_tuple):
    lhs, op, rhs = comparison_tuple
    if op == "==":
        return lhs == rhs
    if op == ">=":
        return lhs >= rhs
    if op == "<=":
        return lhs <= rhs


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
            comparison_tuple = constraint_eq.expression[0].eval(
                component_dict=component_sub_dict, apply_imask=False
            )

            assert apply_comparison(comparison_tuple) == expected

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
            [expression_generator("foo[techs=$tech1] == bar[techs=$tech2]")]
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
            [
                expression_generator(
                    "foo[techs=$tech1] == bar[techs=$tech2, nodes=$node1]"
                )
            ]
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
            ("1 == bar[techs=$tech2]", 2),
            ("$foo == bar[techs=$tech2]", 4),
            ("$bar == 1", 4),
            ("$bar == bar[techs=$tech2]", 6),
            ("$bar + $foo == bar[techs=$tech2]", 12),
            ("$bar + $foo == bar[techs=$tech2] + foo[techs=$tech1]", 16),
            (
                [
                    {"expression": "$foo == bar[techs=$tech2]"},
                    {"expression": "$bar + $foo == bar[techs=$tech2]"},
                ],
                16,
            ),
        ],
    )
    def test_parse_equations(
        self,
        obj_with_components_and_index_slices,
        valid_object_names,
        eq_string,
        expected_n_equations,
    ):
        parsed_equation = obj_with_components_and_index_slices(eq_string)
        parsed_equations = parsed_equation.parse_equations(
            equation_parser.generate_equation_parser, valid_object_names
        )

        assert len(parsed_equations) == expected_n_equations
        assert len(set(eq.name for eq in parsed_equations)) == expected_n_equations

    def test_evaluate_foreach_all_permutations(
        self, dummy_model_data, component_obj, foreach
    ):
        component_obj.sets = foreach
        imask = component_obj.evaluate_foreach(dummy_model_data)

        assert not BASE_DIMS.difference(imask.dims)
        assert not set(foreach).difference(imask.dims)

    def test_imask_foreach_unidentified_name(self, dummy_model_data, component_obj):
        component_obj.sets = ["nodes", "techs", "foos"]
        with pytest.warns(calliope.exceptions.BackendWarning) as excinfo:
            component_obj.evaluate_foreach(dummy_model_data)
        assert check_error_or_warning(
            excinfo, "Not generating optimisation problem object `foo`"
        )

    def test_evaluate_where_no_imasking(self, dummy_model_data, component_obj):
        component_obj.parse_top_level_where()
        imask = component_obj.evaluate_where(dummy_model_data)
        assert imask.item() == True


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
            "foo[techs=$tech1] == bar[techs=$tech2]",
            "foo[techs=$tech1] + bar[techs=$tech2] >= 1",
            "(foo[techs=$tech1] * 1) + bar[techs=$tech2] == 1",
            "(1**bar[techs=$tech2]) <= foo[techs=$tech1] + 7",
            "(foo[techs=$tech1] - bar[techs=$tech2]) * 3 <= 2",
            "dummy_func_1(bar[techs=$tech2]) <= dummy_func_2(x=foo[techs=$tech1])",
            "dummy_func_1(bar[techs=$tech2], x=foo[techs=$tech1]) <= dummy_func_2(1)",
            "foo[techs=$tech1] + 1 <= dummy_func_2(x=bar[techs=$tech2])",
            "foo[techs=$tech1] + dummy_func_2(bar[techs=$tech2]) <= $foo",
            "foo[techs=$tech1] + dummy_func_2(bar[techs=$tech2, nodes=FOO]) <= $foo",
        ],
    )
    def test_find_index_slice_references(
        self, expression_parser, equation_obj, parse_string
    ):
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
            ("$foo == 1", {"foo": "foo[techs=$tech1] + bar[techs=$tech2]"}),
            ("$foo == $bar", {"foo": "foo[techs=$tech1]", "bar": "bar[techs=$tech2]"}),
            ("foo[techs=$tech1] + $bar >= 1", {"bar": "bar[techs=$tech2]"}),
            (
                "foo[techs=$tech1] + $bar == $foo",
                {"foo": "10", "bar": "bar[techs=$tech2]"},
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

    @pytest.mark.parametrize("false_location", [0, -1])
    def test_create_subset_from_where_definitely_empty(
        self, dummy_model_data, equation_obj, where_parser, false_location
    ):
        equation_obj.sets = ["nodes", "techs"]
        equation_obj.where.insert(
            false_location, where_parser.parse_string("False", parse_all=True)
        )
        imask = equation_obj.evaluate_where(dummy_model_data)

        assert not imask.any()

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
        dummy_model_data,
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
            imask = equation_obj.evaluate_where(dummy_model_data)
        if level_ == "initial_imask":
            equation_obj.where = [
                where_parser.parse_string(where_string, parse_all=True)
            ]
            initial_imask = equation_obj.evaluate_where(dummy_model_data)
            equation_obj.where = [where_parser.parse_string("True", parse_all=True)]
            imask = equation_obj.evaluate_where(
                dummy_model_data, initial_imask=initial_imask
            )

        expected = dummy_model_data[expected_imasker]

        assert expected.reindex_like(imask).equals(dummy_model_data[expected_imasker])

    def test_create_subset_from_where_trim_dimension(
        self, dummy_model_data, where_parser, equation_obj, foreach_imask
    ):
        equation_obj.sets = ["nodes", "techs"]

        equation_obj.where = [
            where_parser.parse_string("[foo] in carrier_tiers", parse_all=True)
        ]
        imask = equation_obj.evaluate_where(dummy_model_data, foreach_imask)
        assert imask.sel(carrier_tiers="foo").any()
        assert not imask.sel(carrier_tiers="bar").any()

    def test_create_subset_align_dims_with_sets(
        self, dummy_model_data, where_parser, equation_obj, foreach_imask
    ):
        equation_obj.sets = ["nodes", "techs"]

        equation_obj.where = [where_parser.parse_string("True", parse_all=True)]
        imask = equation_obj.evaluate_where(dummy_model_data, foreach_imask)
        aligned_imask = equation_obj.align_imask_with_sets(imask)

        assert set(imask.dims).difference(["nodes", "techs"])
        assert not set(aligned_imask.dims).difference(["nodes", "techs"])

    def test_evaluate_expression(self, evaluate_component_expression):
        comparison_tuple, n_true = evaluate_component_expression
        # we can't check for equality since the random generation of NaNs in dummy_model_data carrier/node_tech
        # might nullify an otherwise valid item.
        assert apply_comparison(comparison_tuple).sum() <= n_true


class TestParsedConstraint:
    @pytest.fixture
    def constraint_obj(self):
        dict_ = {
            "name": "foo",
            "foreach": ["techs"],
            "where": "with_inf",
            "equation": "$foo == 1",
            "components": {
                "foo": [
                    {"expression": "only_techs + 2", "where": "False"},
                    {"expression": "only_techs / 3", "where": "True"},
                ]
            },
        }
        parsed_ = parsing.ParsedBackendComponent(dict_)
        parsed_.equations = parsed_.parse_equations(
            equation_parser.generate_equation_parser, ["only_techs"]
        )
        return parsed_

    def test_parse_constraint_dict_sets(self, constraint_obj):
        assert constraint_obj.sets == ["techs"]

    def test_parse_constraint_dict_n_equations(self, constraint_obj):
        assert len(constraint_obj.equations) == 2

    def test_parse_constraint_dict_empty_eq1(self, constraint_obj, dummy_model_data):
        assert not constraint_obj.equations[0].evaluate_where(dummy_model_data).any()

    def test_parse_constraint_dict_evalaute_eq2(
        self, constraint_obj, dummy_model_data, dummy_backend_interface
    ):
        foreach_imask = constraint_obj.evaluate_foreach(dummy_model_data)
        top_level_where_imask = constraint_obj.evaluate_where(
            dummy_model_data, foreach_imask
        )
        valid_imask = constraint_obj.equations[1].evaluate_where(
            dummy_model_data, top_level_where_imask
        )
        aligned_imask = constraint_obj.align_imask_with_sets(valid_imask)
        references = set()
        comparison_tuple = constraint_obj.equations[1].evaluate_expression(
            dummy_model_data, dummy_backend_interface, aligned_imask, references
        )
        assert apply_comparison(comparison_tuple).sum() == 1
        assert references == {"only_techs"}


class TestParsedVariable:
    @pytest.fixture
    def variable_obj(self):
        dict_ = {"name": "foo", "foreach": ["techs"], "where": "False"}

        return parsing.ParsedBackendComponent(dict_)

    def test_parse_variable_dict_sets(self, variable_obj):
        assert variable_obj.sets == ["techs"]

    def test_parse_variable_dict_n_equations(self, variable_obj):
        assert len(variable_obj.equations) == 0

    def test_parse_variable_dict_empty_eq1(self, variable_obj, dummy_model_data):
        foreach_imask = variable_obj.evaluate_foreach(dummy_model_data)
        variable_obj.parse_top_level_where()
        assert not variable_obj.evaluate_where(dummy_model_data, foreach_imask).any()


class TestParsedObjective:
    @pytest.fixture
    def objective_obj(self):
        dict_ = {
            "name": "foo",
            "equations": [
                {"expression": "bar + 2", "where": "False"},
                {"expression": "sum(only_techs, over=[techs]) + 1", "where": "True"},
            ],
        }

        parsed_ = parsing.ParsedBackendComponent(dict_)
        parsed_.equations = parsed_.parse_equations(
            equation_parser.generate_arithmetic_parser, ["only_techs", "bar"]
        )
        return parsed_

    def test_parse_objective_dict_sets(self, objective_obj):
        assert objective_obj.sets == []

    def test_parse_objective_dict_n_equations(self, objective_obj):
        assert len(objective_obj.equations) == 2

    def test_parse_objective_dict_empty_eq1(self, objective_obj, dummy_model_data):
        assert not objective_obj.equations[0].evaluate_where(dummy_model_data).any()

    def test_parse_objective_dict_evalaute_eq2(
        self, objective_obj, dummy_model_data, dummy_backend_interface
    ):
        valid_imask = objective_obj.equations[1].evaluate_where(dummy_model_data)
        objective_expression = objective_obj.equations[1].evaluate_expression(
            dummy_model_data, dummy_backend_interface, valid_imask
        )
        assert objective_expression.sum() == 12
