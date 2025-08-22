import logging
from io import StringIO
from unittest.mock import patch

import pyparsing as pp
import pytest
import ruamel.yaml as yaml
import xarray as xr

import calliope
from calliope.backend import backend_model, expression_parser, parsing, where_parser
from calliope.schemas import math_schema

from .common.util import check_error_or_warning

BASE_DIMS = {"carriers", "nodes", "techs"}


def string_to_def(yaml_string, schema: math_schema.CalliopeBaseModel):
    """Convert a YAML string to its validated equivalent."""
    yaml_loader = yaml.YAML(typ="safe", pure=True)
    validated = schema.model_validate(yaml_loader.load(StringIO(yaml_string)))
    return validated


@pytest.fixture
def component_obj():
    setup_string = """
    foreach: [A, A1]
    where: "True"
    equations:
        - expression: 1 == 1
    """
    variable_data = string_to_def(setup_string, math_schema.Constraint)
    return parsing.ParsedBackendComponent("constraints", "foo", variable_data)


@pytest.fixture
def exists_array(component_obj, dummy_model_data):
    component_obj.sets = ["nodes", "techs"]
    return component_obj.combine_definition_matrix_and_foreach(dummy_model_data)


@pytest.fixture
def valid_component_names(dummy_model_data):
    return ["foo", "bar", "baz", "foobar", *dummy_model_data.data_vars.keys()]


@pytest.fixture
def expression_string_parser(valid_component_names):
    return expression_parser.generate_equation_parser(valid_component_names)


@pytest.fixture
def arithmetic_string_parser(valid_component_names):
    return expression_parser.generate_arithmetic_parser(valid_component_names)


@pytest.fixture
def slice_parser(valid_component_names):
    return expression_parser.generate_slice_parser(valid_component_names)


@pytest.fixture
def sub_expression_parser(valid_component_names):
    return expression_parser.generate_sub_expression_parser(valid_component_names)


@pytest.fixture
def where_string_parser():
    return where_parser.generate_where_string_parser()


@pytest.fixture
def expression_generator():
    def _expression_generator(parse_string, where_string=None):
        expression_dict = {"expression": parse_string}
        if where_string is not None:
            expression_dict["where"] = where_string
        return math_schema.ExpressionItem.model_validate(expression_dict)

    return _expression_generator


@pytest.fixture
def generate_expression_list(component_obj, expression_string_parser):
    def _generate_expression_list(expression_list, **kwargs):
        return component_obj.generate_expression_list(
            expression_string_parser, expression_list, "equations", **kwargs
        )

    return _generate_expression_list


def parse_sub_expressions_and_slices(
    parser, expression_list, expression_group, component_obj
):
    return {
        _name: component_obj.generate_expression_list(
            parser, _list, expression_group, _name
        )
        for _name, _list in expression_list.root.items()
    }


@pytest.fixture
def parsed_sub_expression_dict(component_obj, sub_expression_parser):
    def _parsed_sub_expression_dict(n_foo, n_bar):
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

        sub_expressions = string_to_def(setup_string, math_schema.SubExpressions)

        return parse_sub_expressions_and_slices(
            sub_expression_parser, sub_expressions, "sub_expressions", component_obj
        )

    return _parsed_sub_expression_dict


@pytest.fixture
def parsed_slice_dict(component_obj, slice_parser):
    def _parsed_slice_dict(n_tech1, n_tech2):
        techs1 = ", ".join(["{where: techs, expression: foo}" for i in range(n_tech1)])
        techs2 = ", ".join(["{where: techs, expression: bar}" for i in range(n_tech2)])
        setup_string = f"""
        tech1: [{techs1}]
        tech2: [{techs2}]
        """

        slices = string_to_def(setup_string, math_schema.SubExpressions)

        return parse_sub_expressions_and_slices(
            slice_parser, slices, "slices", component_obj
        )

    return _parsed_slice_dict


@pytest.fixture
def obj_with_sub_expressions_and_slices():
    def _obj_with_sub_expressions_and_slices(equation_string):
        if isinstance(equation_string, str):
            equation_string = f"[{{'expression': '{equation_string}'}}]"

        string_ = f"""
            foreach: [A, techs, A1]
            equations: {equation_string}
            sub_expressions:
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
            slices:
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

        return parsing.ParsedBackendComponent(
            "constraints",
            "my_constraint",
            string_to_def(string_, math_schema.Constraint),
        )

    return _obj_with_sub_expressions_and_slices


@pytest.fixture
def equation_obj(expression_string_parser, where_string_parser):
    return parsing.ParsedBackendEquation(
        equation_name="foo",
        sets=["A", "A1"],
        expression=expression_string_parser.parse_string("foo == 1", parse_all=True),
        where_list=[where_string_parser.parse_string("True", parse_all=True)],
    )


@pytest.fixture
def equation_sub_expression_obj(sub_expression_parser, where_string_parser):
    def _equation_sub_expression_obj(name):
        return parsing.ParsedBackendEquation(
            equation_name=name,
            sets=["A", "A1"],
            expression=sub_expression_parser.parse_string("foo + 1", parse_all=True),
            where_list=[where_string_parser.parse_string("False", parse_all=True)],
        )

    return _equation_sub_expression_obj


@pytest.fixture
def equation_slice_obj(slice_parser, where_string_parser):
    def _equation_slice_obj(name):
        return parsing.ParsedBackendEquation(
            equation_name=name,
            sets=["A", "A1"],
            expression=slice_parser.parse_string("bar", parse_all=True),
            where_list=[where_string_parser.parse_string("False", parse_all=True)],
        )

    return _equation_slice_obj


@pytest.fixture
def dummy_backend_interface(dummy_model_data, dummy_model_math, default_config):
    # ignore the need to define the abstract methods from backend_model.BackendModel
    with patch.multiple(backend_model.BackendModel, __abstractmethods__=set()):

        class DummyBackendModel(backend_model.BackendModel):
            def __init__(self):
                backend_model.BackendModel.__init__(
                    self,
                    dummy_model_data,
                    dummy_model_math,
                    default_config.build,
                    instance=None,
                )

                self._dataset = self.inputs

    return DummyBackendModel()


@pytest.fixture
def evaluatable_component_obj(valid_component_names):
    def _evaluatable_component_obj(equation_expressions):
        setup_string = f"""
        foreach: [techs, nodes]
        where: with_inf
        equations:
            - expression: {equation_expressions}
        sub_expressions:
            foo: [{{expression: with_inf * 2, where: only_techs}}]
        slices:
            tech: [{{expression: barfoo, where: "[bar] in nodes"}}]
        """
        sub_expression_dict = string_to_def(setup_string, math_schema.Constraint)

        class DummyParsedBackendComponent(parsing.ParsedBackendComponent):
            def __init__(self, dict_):
                parsing.ParsedBackendComponent.__init__(
                    self, "constraints", "foo", dict_
                )
                self.parse_top_level_where()
                self.equations = self.parse_equations(valid_component_names)

        return DummyParsedBackendComponent(sub_expression_dict)

    return _evaluatable_component_obj


@pytest.fixture(
    params=[
        ("with_inf <= 100", 7),  # all vals except .inf meet criterion
        ("with_inf == 100", 2),  # only default vals meet criterion
        # only non-default + non-inf values meet criterion (+ only_techs masks one valid value)
        ("$foo <= 100", 4),
        ("$foo == 100", 0),  # no expressions are valid
        ("only_techs + with_inf[techs=$tech] == 2", 1),
    ]
)
def evaluate_component_where(
    evaluatable_component_obj, dummy_pyomo_backend_model, request
):
    component_obj = evaluatable_component_obj(request.param[0])
    top_level_where = component_obj.generate_top_level_where_array(
        dummy_pyomo_backend_model, break_early=False, align_to_foreach_sets=False
    )
    equation_where = component_obj.equations[0].evaluate_where(
        dummy_pyomo_backend_model, initial_where=top_level_where
    )
    equation_where_aligned = component_obj.drop_dims_not_in_foreach(equation_where)
    return component_obj, equation_where_aligned, request.param[1]


@pytest.fixture
def evaluate_component_expression(evaluate_component_where, dummy_backend_interface):
    component_obj, equation_where, n_true = evaluate_component_where

    return (
        component_obj.equations[0].evaluate_expression(
            dummy_backend_interface, where=equation_where
        ),
        n_true,
    )


class TestParsedComponent:
    @pytest.mark.parametrize(
        "parse_string",
        [
            "foo + bar == 1",
            "foo - $bar + baz[A1=a1] <= 1",
            "-1**foo + dummy_func_1(2) + baz[A1=a1] >= foobar",
        ],
    )
    def test_parse_string(self, component_obj, expression_string_parser, parse_string):
        parsed_ = component_obj._parse_string(expression_string_parser, parse_string)
        assert isinstance(parsed_, pp.ParseResults)
        assert not component_obj._errors

    @pytest.mark.parametrize(
        "parse_string",
        ["foo bar == 1", "foo - $bar + baz[A1=a1] = 1", "1foo == 1", "_foo >= foobar"],
    )
    def test_parse_string_malformed(
        self, component_obj, expression_string_parser, parse_string
    ):
        parsed_ = component_obj._parse_string(expression_string_parser, parse_string)
        assert isinstance(parsed_, pp.ParseResults)
        assert len(parsed_) == 0
        assert check_error_or_warning(component_obj._errors, parse_string)

    @pytest.mark.parametrize(
        "parse_string",
        ["foo == 1", "$foo + (bar + foobar[A1=a1])**2 >= (dummy_func_1(foo) + 1)"],
    )
    @pytest.mark.parametrize(
        ("where_string", "expected_where_eval"),
        [(None, True), ("False", False), ("True or False", True)],
    )
    def test_generate_expression_list(
        self,
        component_obj,
        expression_string_parser,
        expression_generator,
        parse_string,
        where_string,
        expected_where_eval,
    ):
        expression_dict = expression_generator(parse_string, where_string)
        parsed_list = component_obj.generate_expression_list(
            expression_string_parser, [expression_dict], "equations", id_prefix="foo"
        )

        assert (
            parsed_list[0].where[0][0].eval(return_type="array") == expected_where_eval
        )
        assert isinstance(parsed_list[0].expression, pp.ParseResults)

    @pytest.mark.parametrize("n_dicts", [1, 2, 3])
    def test_generate_expression_list_id_prefix(
        self, generate_expression_list, expression_generator, n_dicts
    ):
        expression_dict = expression_generator("foo == 1", "bar")
        parsed_list = generate_expression_list(
            [expression_dict] * n_dicts, id_prefix="foo"
        )

        for expr_num in range(n_dicts):
            assert parsed_list[expr_num].name == f"foo:{expr_num}"

    @pytest.mark.parametrize("n_dicts", [1, 2, 3])
    def test_generate_expression_list_no_id_prefix(
        self, generate_expression_list, expression_generator, n_dicts
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
            component_obj._errors, ["equations[0].expression", "foo = 1"]
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
            component_obj._errors, f"equations[{error_position}].expression"
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
            [
                "equations[0].expression (line 1, char 5): foo = 1",
                "equations[1].expression (line 1, char 5): foo = 2",
            ],
        )

    @pytest.mark.parametrize("n_foos", [0, 1, 2])
    @pytest.mark.parametrize("n_bars", [0, 1, 2])
    def test_extend_equation_list_with_expression_group_components(
        self,
        component_obj,
        parsed_sub_expression_dict,
        generate_expression_list,
        expression_generator,
        n_foos,
        n_bars,
    ):
        equation_list = generate_expression_list([expression_generator("$foo == $bar")])
        expression_list = component_obj.extend_equation_list_with_expression_group(
            equation_list[0],
            parsed_sub_expression_dict(n_foos, n_bars),
            "sub_expressions",
        )
        assert len(expression_list) == n_foos * n_bars

    def test_extend_equation_list_with_expression_group_missing_sub_expression(
        self,
        component_obj,
        generate_expression_list,
        parsed_sub_expression_dict,
        expression_generator,
    ):
        equation_ = generate_expression_list(
            [expression_generator("$foo == $bar + $ba")]
        )
        with pytest.raises(KeyError) as excinfo:
            component_obj.extend_equation_list_with_expression_group(
                equation_[0], parsed_sub_expression_dict(1, 2), "sub_expressions"
            )
        assert check_error_or_warning(
            excinfo,
            "constraints:foo: Undefined sub_expressions found in equation: {'ba'}",
        )

    @pytest.mark.parametrize(
        ("equation_", "expected"),
        [("$foo == $bar", False), ("$foo <= $bar", True), ("$foo * 20 >= $bar", True)],
    )
    def test_add_exprs_to_equation_data_multi(
        self,
        dummy_backend_interface,
        component_obj,
        generate_expression_list,
        parsed_sub_expression_dict,
        expression_generator,
        equation_,
        expected,
    ):
        sub_expression_dict = parsed_sub_expression_dict(2, 2)

        equation_dict = generate_expression_list([expression_generator(equation_)])[0]
        expression_list = component_obj.extend_equation_list_with_expression_group(
            equation_dict, sub_expression_dict, "sub_expressions"
        )
        # All IDs should be unique
        assert len(set(expr.name for expr in expression_list)) == 4

        for constraint_eq in expression_list:
            component_sub_dict = constraint_eq.sub_expressions
            assert set(component_sub_dict.keys()) == {"foo", "bar"}
            comparison_expr = constraint_eq.expression[0].eval(
                sub_expression_dict=component_sub_dict,
                backend_interface=dummy_backend_interface,
                where_array=xr.DataArray(True),
                return_type="array",
            )

            assert comparison_expr == expected

    @pytest.mark.parametrize("n_1", [0, 1, 2])
    @pytest.mark.parametrize("n_2", [0, 1, 2])
    def test_extend_equation_list_with_expression_group_slices(
        self,
        component_obj,
        parsed_slice_dict,
        generate_expression_list,
        expression_generator,
        n_1,
        n_2,
    ):
        equation_ = generate_expression_list(
            [expression_generator("foo[techs=$tech1] == bar[techs=$tech2]")]
        )
        expression_list = component_obj.extend_equation_list_with_expression_group(
            equation_[0], parsed_slice_dict(n_1, n_2), "slices"
        )
        assert len(expression_list) == n_1 * n_2

    def test_extend_equation_list_with_expression_group_missing_slices(
        self,
        component_obj,
        generate_expression_list,
        parsed_slice_dict,
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
                equation_[0], parsed_slice_dict(1, 2), "slices"
            )
        assert check_error_or_warning(
            excinfo, "constraints:foo: Undefined slices found in equation: {'node1'}"
        )

    @pytest.mark.parametrize(
        ("eq_string", "expected_n_equations"),
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
        obj_with_sub_expressions_and_slices,
        valid_component_names,
        eq_string,
        expected_n_equations,
    ):
        component_obj = obj_with_sub_expressions_and_slices(eq_string)
        parsed_equations = component_obj.parse_equations(valid_component_names)

        assert len(parsed_equations) == expected_n_equations
        assert len(set(eq.name for eq in parsed_equations)) == expected_n_equations

    @pytest.mark.parametrize("is_valid", [True, False])
    def test_raise_caught_errors(self, component_obj, is_valid):
        component_obj._is_valid = is_valid
        if is_valid:
            component_obj.raise_caught_errors()
        else:
            with pytest.raises(calliope.exceptions.ModelError) as excinfo:
                component_obj.raise_caught_errors()
            assert check_error_or_warning(excinfo, ["\n * constraints:foo:"])

    def test_parse_equations_fail(
        self, obj_with_sub_expressions_and_slices, valid_component_names
    ):
        component_obj = obj_with_sub_expressions_and_slices("bar = 1")
        with pytest.raises(calliope.exceptions.ModelError) as excinfo:
            component_obj.parse_equations(valid_component_names, errors="raise")
        expected_err_string = """
 * constraints:my_constraint:
    * equations[0].expression (line 1, char 5): bar = 1
                                                    ^"""
        assert check_error_or_warning(excinfo, expected_err_string)

    def test_parse_equations_fail_no_raise(
        self, obj_with_sub_expressions_and_slices, valid_component_names
    ):
        component_obj = obj_with_sub_expressions_and_slices("bar = 1")
        component_obj.parse_equations(valid_component_names, errors="ignore")

        expected_err_string = """\
equations[0].expression (line 1, char 5): bar = 1
                                                    ^"""

        assert check_error_or_warning(component_obj._errors, expected_err_string)

    def test_combine_exists_and_foreach_all_permutations(
        self, dummy_model_data, component_obj, foreach
    ):
        component_obj.sets = foreach
        where = component_obj.combine_definition_matrix_and_foreach(dummy_model_data)

        assert not BASE_DIMS.difference(where.dims)
        assert not set(foreach).difference(where.dims)

    def test_foreach_unidentified_name(self, caplog, dummy_model_data, component_obj):
        component_obj.sets = ["nodes", "techs", "foos"]
        caplog.set_level(logging.DEBUG)
        component_obj.combine_definition_matrix_and_foreach(dummy_model_data)
        assert "indexed over unidentified set names" in caplog.text

    def test_evaluate_where_to_false(self, dummy_pyomo_backend_model, component_obj):
        component_obj.parse_top_level_where()
        where = component_obj.evaluate_where(dummy_pyomo_backend_model)
        assert where.item() is True

    def test_parse_top_level_where_fail(self, component_obj):
        component_obj._unparsed = component_obj._unparsed.update({"where": "1"})
        with pytest.raises(calliope.exceptions.ModelError) as excinfo:
            component_obj.parse_top_level_where()

        assert check_error_or_warning(excinfo, "Errors during math string parsing")

    def test_generate_top_level_where_array_break_at_foreach(
        self, caplog, dummy_pyomo_backend_model, component_obj
    ):
        component_obj.sets = ["nodes", "techs", "foos"]
        caplog.set_level(logging.DEBUG)
        where_array = component_obj.generate_top_level_where_array(
            dummy_pyomo_backend_model
        )

        assert "indexed over unidentified set names: `{'foos'}`" in caplog.text
        assert "'foreach' does not apply anywhere." in caplog.text
        assert "'where' does not apply anywhere." not in caplog.text
        assert not where_array.any()
        assert not where_array.shape

    def test_generate_top_level_where_array_break_at_top_level_where(
        self, dummy_pyomo_backend_model, component_obj
    ):
        component_obj.sets = ["nodes", "techs", "timesteps"]
        component_obj._unparsed = component_obj._unparsed.update({"where": "all_nan"})
        where_array = component_obj.generate_top_level_where_array(
            dummy_pyomo_backend_model
        )
        assert not where_array.any()
        assert not set(component_obj.sets).difference(where_array.dims)

    def test_generate_top_level_where_array_no_break_no_align(
        self, caplog, dummy_pyomo_backend_model, component_obj
    ):
        component_obj.sets = ["nodes", "techs", "foos"]
        component_obj._unparsed = component_obj._unparsed.update({"where": "all_nan"})
        caplog.set_level(logging.DEBUG)

        where_array = component_obj.generate_top_level_where_array(
            dummy_pyomo_backend_model, break_early=False, align_to_foreach_sets=False
        )
        assert "indexed over unidentified set names: `{'foos'}`" in caplog.text
        assert "'foreach' does not apply anywhere." in caplog.text
        assert "'where' does not apply anywhere." in caplog.text

        assert not where_array.any()
        assert set(component_obj.sets).difference(where_array.dims) == {"foos"}

    def test_generate_top_level_where_array_no_break_align(
        self, dummy_pyomo_backend_model, component_obj
    ):
        component_obj.sets = ["nodes", "techs"]
        component_obj._unparsed = component_obj._unparsed.update(
            {"where": "all_nan AND all_true_carriers"}
        )
        where_array = component_obj.generate_top_level_where_array(
            dummy_pyomo_backend_model, break_early=False, align_to_foreach_sets=True
        )
        assert not where_array.any()
        assert not set(component_obj.sets).difference(where_array.dims)

    def test_evaluate_where_fail(self, component_obj):
        component_obj._unparsed = component_obj._unparsed.update({"where": "1[]"})
        with pytest.raises(calliope.exceptions.ModelError) as excinfo:
            component_obj.parse_top_level_where()
        expected_err_string = """
 * constraints:foo:
    * where (line 1, char 1): 1[]
                              ^"""
        assert check_error_or_warning(excinfo, expected_err_string)

    def test_evaluate_where_fail_no_raise(self, component_obj):
        component_obj._unparsed = component_obj._unparsed.update({"where": "1[]"})
        component_obj.parse_top_level_where(errors="ignore")
        expected_err_string = """\
where (line 1, char 1): 1[]
                              ^"""
        assert check_error_or_warning(component_obj._errors, expected_err_string)


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
        self, expression_string_parser, equation_obj, parse_string
    ):
        parsed = expression_string_parser.parse_string(parse_string, parse_all=True)
        found_sub_expressions = equation_obj._find_items_in_expression(
            [parsed[0].lhs, parsed[0].rhs],
            expression_parser.EvalSubExpressions,
            (expression_parser.EvalOperatorOperand),
        )
        assert found_sub_expressions == {"foo", "bar"}

    @pytest.mark.parametrize(
        ("parse_string", "expected"),
        [
            # sub-expressions in comparisons are always seen
            ("$foo == $bar", ["foo", "bar"]),
            # sub-expressions in arithmetic are missed
            ("1 + $bar >= $foo", ["foo"]),
            # sub-expressions in arithmetic are missed
            ("$foo * $bar == 1", []),
            # sub-expressions in arithmetic are missed
            ("($foo * 1) + $bar == 1", []),
            # sub-expressions in functions are missed
            ("dummy_func_1($foo) == $bar", ["bar"]),
            # sub-expressions in functions and arithmetic are missed
            ("dummy_func_1($foo) == $bar + 1", []),
        ],
    )
    def test_find_items_in_expression_missing_eval_class(
        self, expression_string_parser, equation_obj, parse_string, expected
    ):
        parsed = expression_string_parser.parse_string(parse_string, parse_all=True)
        found_sub_expressions = equation_obj._find_items_in_expression(
            [parsed[0].lhs, parsed[0].rhs],
            expression_parser.EvalSubExpressions,
            (),  # The above happens because we provide no eval classes to search inside
        )
        assert found_sub_expressions == set(expected)

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
    def test_find_slice_references(
        self, expression_string_parser, equation_obj, parse_string
    ):
        parsed = expression_string_parser.parse_string(parse_string, parse_all=True)
        equation_obj.expression = parsed
        found_slices = equation_obj.find_slices()
        assert found_slices == {"tech1", "tech2"}

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
    def test_find_sub_expressions(
        self, expression_string_parser, equation_obj, parse_string
    ):
        parsed = expression_string_parser.parse_string(parse_string, parse_all=True)
        equation_obj.expression = parsed
        found_slices = equation_obj.find_sub_expressions()
        assert found_slices == {"foo", "bar"}

    def test_find_single_sub_expression(self, expression_string_parser, equation_obj):
        parsed = expression_string_parser.parse_string("$foo == 1", parse_all=True)
        equation_obj.expression = parsed
        found_sub_expressions = equation_obj.find_sub_expressions()
        assert found_sub_expressions == {"foo"}

    def test_find_single_sub_expression_in_global_expression(
        self, arithmetic_string_parser, equation_obj
    ):
        parsed = arithmetic_string_parser.parse_string("$foo", parse_all=True)
        equation_obj.expression = parsed
        found_sub_expressions = equation_obj.find_sub_expressions()
        assert found_sub_expressions == {"foo"}

    @pytest.mark.parametrize(
        ("equation_expr", "sub_expression_exprs"),
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
    def test_find_slices_in_expr_and_sub_expressions(
        self,
        expression_string_parser,
        sub_expression_parser,
        equation_obj,
        equation_expr,
        sub_expression_exprs,
    ):
        equation_obj.expression = expression_string_parser.parse_string(
            equation_expr, parse_all=True
        )
        equation_obj.sub_expressions = {
            sub_expression: sub_expression_parser.parse_string(expr_, parse_all=True)
            for sub_expression, expr_ in sub_expression_exprs.items()
        }
        found_slices = equation_obj.find_slices()
        assert found_slices == {"tech1", "tech2"}

    @pytest.mark.parametrize("expression_group", ["sub_expressions", "slices"])
    def test_add_expression_group_combination(
        self, equation_obj, request, expression_group
    ):
        obj_ = request.getfixturevalue(
            f"equation_{expression_group.removesuffix('s')}_obj"
        )
        not_expression_group = [
            i for i in ["sub_expressions", "slices"] if i != expression_group
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

    def test_add_slices_after_sub_expressions(
        self, equation_obj, equation_sub_expression_obj, equation_slice_obj
    ):
        equation_obj.sub_expressions = {"bar": equation_sub_expression_obj("bar:0")}
        obj1 = equation_slice_obj("baz:0")
        obj2 = equation_slice_obj("bam:0")
        new_expression = equation_obj.add_expression_group_combination(
            "slices", [obj1, obj2]
        )

        assert new_expression.sub_expressions == equation_obj.sub_expressions
        assert new_expression.slices == {"baz": obj1.expression, "bam": obj2.expression}

    @pytest.mark.parametrize("false_location", [0, -1])
    def test_create_subset_from_where_definitely_empty(
        self,
        dummy_pyomo_backend_model,
        equation_obj,
        where_string_parser,
        false_location,
    ):
        equation_obj.sets = ["nodes", "techs"]
        equation_obj.where.insert(
            false_location, where_string_parser.parse_string("False", parse_all=True)
        )
        where = equation_obj.evaluate_where(dummy_pyomo_backend_model)

        assert not where.any()

    @pytest.mark.parametrize(
        ("where_string", "expected_where_array"),
        [
            ("with_inf", "with_inf_as_bool"),
            ("only_techs", "only_techs_as_bool"),
            ("with_inf and only_techs", "with_inf_and_only_techs_as_bool"),
            ("with_inf or only_techs", "with_inf_or_only_techs_as_bool"),
            (
                "with_inf and [bar] in nodes",
                "with_inf_as_bool_and_subset_on_bar_in_nodes",
            ),
        ],
    )
    @pytest.mark.parametrize("level_", ["initial_where", "where"])
    def test_create_subset_from_where_one_level_where(
        self,
        dummy_model_data,
        dummy_pyomo_backend_model,
        equation_obj,
        where_string_parser,
        where_string,
        expected_where_array,
        level_,
    ):
        equation_obj.sets = ["nodes", "techs"]
        if level_ == "where":
            equation_obj.where = [
                where_string_parser.parse_string(where_string, parse_all=True)
            ]
            where = equation_obj.evaluate_where(dummy_pyomo_backend_model)
        if level_ == "initial_where":
            equation_obj.where = [
                where_string_parser.parse_string(where_string, parse_all=True)
            ]
            initial_where = equation_obj.evaluate_where(dummy_pyomo_backend_model)
            equation_obj.where = [
                where_string_parser.parse_string("True", parse_all=True)
            ]
            where = equation_obj.evaluate_where(
                dummy_pyomo_backend_model, initial_where=initial_where
            )

        expected = dummy_model_data[expected_where_array]

        assert expected.reindex_like(where).equals(
            dummy_model_data[expected_where_array]
        )

    def test_create_subset_from_where_trim_dimension(
        self, dummy_pyomo_backend_model, where_string_parser, equation_obj, exists_array
    ):
        equation_obj.sets = ["nodes", "techs"]

        equation_obj.where = [
            where_string_parser.parse_string("[foo] in carriers", parse_all=True)
        ]
        where = equation_obj.evaluate_where(
            dummy_pyomo_backend_model, initial_where=exists_array
        )
        assert where.sel(carriers="foo").any()
        assert not where.sel(carriers="bar").any()

    def test_create_subset_align_dims_with_sets(
        self, dummy_pyomo_backend_model, where_string_parser, equation_obj, exists_array
    ):
        equation_obj.sets = ["nodes", "techs"]

        equation_obj.where = [where_string_parser.parse_string("True", parse_all=True)]
        where = equation_obj.evaluate_where(
            dummy_pyomo_backend_model, initial_where=exists_array
        )
        aligned_where = equation_obj.drop_dims_not_in_foreach(where)

        assert set(where.dims).difference(["nodes", "techs"])
        assert not set(aligned_where.dims).difference(["nodes", "techs"])

    def test_evaluate_expression(self, evaluate_component_expression):
        comparison_expr, n_true = evaluate_component_expression
        # we can't check for equality since the random generation of NaNs in dummy_model_data carrier/node_tech
        # might nullify an otherwise valid item.
        assert comparison_expr.sum() <= n_true


class TestParsedConstraint:
    @pytest.fixture
    def constraint_obj(self):
        constr = math_schema.Constraint.model_validate(
            {
                "foreach": ["techs"],
                "where": "with_inf",
                "equations": [{"expression": "$foo == 1"}],
                "sub_expressions": {
                    "foo": [
                        {"expression": "only_techs + 2", "where": "False"},
                        {"expression": "only_techs / 3", "where": "True"},
                    ]
                },
            }
        )
        parsed_ = parsing.ParsedBackendComponent("constraints", "foo", constr)
        parsed_.equations = parsed_.parse_equations(["only_techs"])
        parsed_.parse_top_level_where()
        return parsed_

    def test_parse_constraint_dict_sets(self, constraint_obj):
        assert constraint_obj.sets == ["techs"]

    def test_parse_constraint_dict_n_equations(self, constraint_obj):
        assert len(constraint_obj.equations) == 2

    def test_parse_constraint_dict_empty_eq1(
        self, constraint_obj, dummy_pyomo_backend_model
    ):
        assert (
            not constraint_obj.equations[0]
            .evaluate_where(dummy_pyomo_backend_model)
            .any()
        )

    def test_parse_constraint_dict_evaluate_eq2(
        self, constraint_obj, dummy_pyomo_backend_model, dummy_backend_interface
    ):
        # We ignore foreach here so we can do "== 1" below. With foreach, there is
        # a random element that might create a where array that masks the only valid index item
        top_level_where = constraint_obj.evaluate_where(dummy_pyomo_backend_model)
        valid_where = constraint_obj.equations[1].evaluate_where(
            dummy_pyomo_backend_model, initial_where=top_level_where
        )
        aligned_where = constraint_obj.drop_dims_not_in_foreach(valid_where)
        references = set()
        comparison_expr = constraint_obj.equations[1].evaluate_expression(
            dummy_backend_interface, where=aligned_where, references=references
        )
        assert comparison_expr.sum() == 1
        assert references == {"only_techs"}


class TestParsedVariable:
    @pytest.fixture
    def variable_obj(self):
        var = math_schema.Variable.model_validate(
            {"foreach": ["techs"], "where": "False", "bounds": {"min": 0, "max": 10}}
        )

        return parsing.ParsedBackendComponent("variables", "foo", var)

    def test_parse_variable_dict_sets(self, variable_obj):
        assert variable_obj.sets == ["techs"]

    def test_parse_variable_dict_n_equations(self, variable_obj):
        assert len(variable_obj.equations) == 0

    def test_parse_variable_dict_empty_eq1(
        self, variable_obj, dummy_pyomo_backend_model
    ):
        top_level_where_where = variable_obj.generate_top_level_where_array(
            dummy_pyomo_backend_model, break_early=False, align_to_foreach_sets=False
        )
        assert not top_level_where_where.any()


class TestParsedObjective:
    @pytest.fixture
    def objective_obj(self):
        obj = math_schema.Objective.model_validate(
            {
                "equations": [
                    {"expression": "bar + 2", "where": "False"},
                    {
                        "expression": "sum(only_techs, over=[techs]) + 1",
                        "where": "True",
                    },
                ],
                "sense": "minimize",
            }
        )

        parsed_ = parsing.ParsedBackendComponent("objectives", "foo", obj)
        parsed_.equations = parsed_.parse_equations(["only_techs", "bar"])
        return parsed_

    def test_parse_objective_dict_sets(self, objective_obj):
        assert objective_obj.sets == []

    def test_parse_objective_dict_n_equations(self, objective_obj):
        assert len(objective_obj.equations) == 2

    def test_parse_objective_dict_empty_eq1(
        self, objective_obj, dummy_pyomo_backend_model
    ):
        assert (
            not objective_obj.equations[0]
            .evaluate_where(dummy_pyomo_backend_model)
            .any()
        )

    def test_parse_objective_dict_evaluate_eq2(
        self, objective_obj, dummy_pyomo_backend_model, dummy_backend_interface
    ):
        valid_where = objective_obj.equations[1].evaluate_where(
            dummy_pyomo_backend_model
        )
        objective_expression = objective_obj.equations[1].evaluate_expression(
            dummy_backend_interface, where=valid_where
        )
        assert objective_expression.sum() == 12
