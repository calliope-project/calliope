import pytest

from calliope.exceptions import ModelError
from calliope.io import read_rich_yaml
from calliope.preprocess.model_definition import (
    TemplateSolver,
    _load_scenario_overrides,
)

from ..common.util import build_test_model as build_model


class TestScenarioOverrides:
    @pytest.mark.parametrize("scenario", ["scenario_1", "scenario_2"])
    def test_valid_scenarios(self, dummy_int, scenario):
        """Direct and nested scenarios apply their overrides to model inputs."""
        override = read_rich_yaml(
            f"""
            scenarios:
                scenario_1: ['one', 'two']
                scenario_2: ['scenario_1', 'new_location']

            overrides:
                one:
                    techs.test_supply_gas.flow_cap_max: {dummy_int}
                two:
                    techs.test_supply_elec.flow_cap_max: {dummy_int / 2}
                new_location:
                    nodes.b.techs:
                        test_supply_elec:

            nodes:
                a:
                    techs:
                        test_supply_gas:
                        test_supply_elec:
                        test_demand_elec:
            """
        )
        model = build_model(override_dict=override, scenario=scenario)

        assert model.inputs.flow_cap_max.sel(techs="test_supply_gas") == dummy_int
        assert model.inputs.flow_cap_max.sel(techs="test_supply_elec") == dummy_int / 2

    @pytest.mark.parametrize("scenario", ["A", "B,C", "A,A", "shared,A"])
    def test_shared_scenario_descendant(self, scenario):
        """Shared descendants are valid and their overrides are applied once."""
        definition = {
            "scenarios": {
                "A": ["B", "C"],
                "B": ["D", "left"],
                "C": ["D", "right"],
                "D": ["shared"],
            },
            "overrides": {
                "shared": {"shared_value": 1},
                "left": {"left_value": 2},
                "right": {"right_value": 3},
            },
        }

        model_def, applied_overrides = _load_scenario_overrides(definition, scenario)

        assert model_def == {"shared_value": 1, "left_value": 2, "right_value": 3}
        assert applied_overrides == "shared;left;right"

    @pytest.mark.parametrize(
        ("entries", "unknown_name"), [({"techs.foo.bar": 1}, "techs"), ("foo", "foo")]
    )
    def test_invalid_scenarios(self, entries, unknown_name):
        definition = {"scenarios": {"A": entries}}

        with pytest.raises(ModelError) as excinfo:
            _load_scenario_overrides(definition, "A")

        assert str(excinfo.value) == (
            f"(scenarios, A) | Unrecognised override name: {unknown_name}."
        )

    @pytest.mark.parametrize(
        ("scenarios", "expected_path"),
        [
            ({"A": ["A"]}, ["A", "A"]),
            ({"A": ["B"], "B": ["A"]}, ["A", "B", "A"]),
            ({"A": ["B"], "B": ["C"], "C": ["B"]}, ["A", "B", "C", "B"]),
            (
                {"A": ["B", "C"], "B": ["D"], "C": ["D", "A"], "D": ["shared"]},
                ["A", "C", "A"],
            ),
        ],
    )
    def test_circular_scenarios(self, scenarios, expected_path):
        """Cycles raise ModelError with the full path, including any noncyclic prefix."""
        definition = {
            "scenarios": scenarios,
            "overrides": {"shared": {"shared_value": 1}},
        }

        with pytest.raises(ModelError) as excinfo:
            _load_scenario_overrides(definition, "A")

        assert str(excinfo.value) == (
            f"(scenarios, {expected_path[-1]}) | Circular scenario reference detected: "
            f"{' -> '.join(expected_path)}."
        )


class TestTemplateSolver:
    @pytest.fixture
    def dummy_solved_template(self) -> TemplateSolver:
        text = """
        templates:
            T1:
                A: ["foo", "bar"]
                B: 1
            T2:
                C: bar
                template: T1
            T3:
                template: T1
                B: 11
            T4:
                template: T3
                A: ["bar", "foobar"]
                B: "1"
                C: {"foo": "bar"}
                D: true
        a:
            template: T1
            a1: 1
        b:
            template: T3
        c:
            template: T4
            D: false
        """
        yaml_data = read_rich_yaml(text)
        return TemplateSolver(yaml_data)

    def test_inheritance_templates(self, dummy_solved_template):
        templates = dummy_solved_template.resolved_templates
        assert all(
            [
                templates.T1 == {"A": ["foo", "bar"], "B": 1},
                templates.T2 == {"A": ["foo", "bar"], "B": 1, "C": "bar"},
                templates.T3 == {"A": ["foo", "bar"], "B": 11},
                templates.T4
                == {"A": ["bar", "foobar"], "B": "1", "C": {"foo": "bar"}, "D": True},
            ]
        )

    def test_template_inheritance_data(self, dummy_solved_template):
        data = dummy_solved_template.resolved_data
        assert all(
            [
                data.a == {"A": ["foo", "bar"], "B": 1, "a1": 1},
                data.b == {"A": ["foo", "bar"], "B": 11},
                data.c
                == {"A": ["bar", "foobar"], "B": "1", "C": {"foo": "bar"}, "D": False},
            ]
        )

    def test_invalid_template_error(self):
        text = read_rich_yaml(
            """
            templates:
                T1: "not_a_yaml_block"
                T2:
                    foo: bar
            a:
                template: T2
            """
        )
        with pytest.raises(
            ModelError, match="Template definitions must be YAML blocks."
        ):
            TemplateSolver(text)

    def test_circular_template_error(self):
        text = read_rich_yaml(
            """
            templates:
                T1:
                    template: T2
                    bar: foo
                T2:
                    template: T1
                    foo: bar
            a:
                template: T2
            """
        )
        with pytest.raises(ModelError, match="Circular template reference detected"):
            TemplateSolver(text)

    def test_incorrect_template_placement_error(self):
        text = read_rich_yaml(
            """
            templates:
                T1:
                    stuff: null
                T2:
                    foo: bar
            a:
                template: T2
            b:
                templates:
                    T3:
                        this: "should not be here"
            """
        )
        with pytest.raises(
            ModelError,
            match="Template definitions must be placed at the top level of the YAML file.",
        ):
            TemplateSolver(text)
