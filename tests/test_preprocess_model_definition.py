import pytest

from calliope.exceptions import ModelError
from calliope.io import read_rich_yaml
from calliope.preprocess.scenarios import TemplateSolver


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
                templates.T4 == {"A": ["bar", "foobar"], "B": "1", "C": {"foo": "bar"}, "D": True}
            ]
        )

    def test_template_inheritance_data(self, dummy_solved_template):
        data = dummy_solved_template.resolved_data
        assert all(
            [
                data.a == {"A": ["foo", "bar"], "B": 1, "a1": 1},
                data.b == {"A": ["foo", "bar"], "B": 11},
                data.c == {"A": ["bar", "foobar"], "B": "1", "C": {"foo": "bar"}, "D": False}
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
        with pytest.raises(ModelError, match="Template definitions must be YAML blocks."):
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
        with pytest.raises(ModelError, match="Template definitions must be placed at the top level of the YAML file."):
            TemplateSolver(text)
