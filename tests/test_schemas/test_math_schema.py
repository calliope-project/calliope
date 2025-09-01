import re

import pytest

from calliope import AttrDict
from calliope.io import read_rich_yaml
from calliope.preprocess import initialise_math
from calliope.schemas import math_schema


class TestGlobalExpressions:
    """Test the global expressions in the math schema."""

    def test_global_expression_order(self):
        """Test that global expressions can be ordered."""
        # Create a math schema with global expressions
        math = {
            "global_expressions": {
                # default = 0
                "expr0": {"equations": [{"expression": "x * y"}]},
                # default = 1
                "expr1": {"equations": [{"expression": "x + y"}], "order": 2},
                # default = 2
                "expr2": {"equations": [{"expression": "x - y"}], "order": 1},
                # default = 3
                "expr3": {"equations": [{"expression": "x / y"}], "order": 3},
            }
        }
        # Validate the schema
        validated_math = math_schema.CalliopeBuildMath(**math)
        # Check the order of global expressions
        assert validated_math.global_expressions["expr0"].order == 0
        assert validated_math.global_expressions["expr1"].order == 2
        assert validated_math.global_expressions["expr2"].order == 1
        assert validated_math.global_expressions["expr3"].order == 3


class TestMathEquationComponent:
    """Test the equation component schema."""

    @pytest.fixture
    def dummy(self) -> AttrDict:
        """A fully defined equation component."""
        return read_rich_yaml(
            """
        title: foo
        description: A dummy equation
        active: true
        equations:
          - where: "True"
            expression: something == $sub_bar
        sub_expressions:
          bar:
            - where: "True"
              expression: something_else + 3
        slices:
          cake:
            - where: "True"
              expression: has_cake
        """
        )

    @pytest.fixture
    def dummy_empty_equation(self, dummy: AttrDict):
        """A case with undefined equations."""
        return dummy | {"equations": []}

    def test_correctness(self, dummy):
        """The schema should not alter the passed data."""
        dumped = math_schema.MathEquationComponent.model_validate(dummy).model_dump()
        assert dumped == dummy

    def test_active_fail_if_empty(self, dummy_empty_equation):
        """Active components must have equations."""
        with pytest.raises(
            ValueError, match="Must have equations defined if component is active."
        ):
            math_schema.MathEquationComponent.model_validate(dummy_empty_equation)

    def test_inactive_empty_pass(self, dummy_empty_equation):
        """Inactive components should silently succed even if equations are empty."""
        dummy_empty_equation |= {"active": False}
        math_schema.MathEquationComponent.model_validate(dummy_empty_equation)


class TestCalliopeBuildMath:
    """Tests related to the built math object."""

    @pytest.fixture
    def base_math_raw(self) -> AttrDict:
        """Unvalidated Calliope base math."""
        return initialise_math()["base"]

    @pytest.fixture
    def base_math_validated(self, base_math_raw) -> AttrDict:
        """Validated Calliope base math.

        Includes edge-cases:
        - Duplicates in several sections (with only one active).
        - Moving variables to parameters (common in operate mode).
        - Fully deactivated components.
        """
        # Duplicates with only one active
        base_math_raw["variables"]["foo"] = math_schema.Variable().model_dump()
        base_math_raw["parameters"]["foo"] = math_schema.Parameter().model_dump()
        base_math_raw["lookups"]["foo"] = math_schema.Lookup().model_dump()
        base_math_raw["parameters"]["foo"]["active"] = False
        base_math_raw["lookups"]["foo"]["active"] = False
        # Simulate variable -> parameter (common case)
        base_math_raw["parameters"]["flow_cap"] = math_schema.Parameter().model_dump()
        base_math_raw["variables"]["flow_cap"]["active"] = False
        # Full deactivation
        base_math_raw["global_expressions"]["flow_in_inc_eff"]["active"] = False
        base_math_raw["dimensions"]["datesteps"]["active"] = False
        return math_schema.CalliopeBuildMath.model_validate(base_math_raw)

    def test_base_math_pass(self, base_math_raw):
        """The base math should pass by default."""
        math_schema.CalliopeBuildMath.model_validate(base_math_raw)

    def test_duplicate_component_error(self, base_math_raw):
        """Duplicate names in math components should raise an error."""
        base_math_raw["variables"]["techs"] = math_schema.Variable().model_dump()
        base_math_raw["dimensions"]["flow_cap"] = math_schema.Dimension().model_dump()
        dupes = sorted(["flow_cap", "techs"])
        with pytest.raises(
            ValueError, match=re.escape(f"Non-unique names in math components: {dupes}")
        ):
            math_schema.CalliopeBuildMath.model_validate(base_math_raw)

    @pytest.mark.parametrize(
        ("to_search", "component"),
        [
            ("foo", "variables"),
            ("flow_out", "variables"),
            ("flow_cap", "parameters"),
            ("bigM", "parameters"),
            ("timesteps", "dimensions"),
            ("cap_method", "lookups"),
            ("force_zero_area_use", "constraints"),
            ("min_cost_optimisation", "objectives"),
            ("cost_operation_variable", "global_expressions"),
        ],
    )
    def test_find(self, to_search, component, base_math_validated):
        """Component searches should return the expected data."""
        location, data = base_math_validated.find(to_search)
        assert location == component
        assert data == base_math_validated[location][to_search]

    def test_find_duplicate_error(self, base_math_validated):
        """Finding duplicate parameters should return an error."""
        m = base_math_validated

        # Brute force a duplicate addition
        foo_active = m.parameters.root["foo"].model_copy(update={"active": True})
        new_param_root = dict(m.parameters.root)
        new_param_root["foo"] = foo_active
        new_params = m.parameters.model_copy(update={"root": new_param_root})
        m_dupe = m.model_copy(update={"parameters": new_params})

        with pytest.raises(ValueError, match=r"found in multiple"):
            m_dupe.find("foo")

    @pytest.mark.parametrize(
        ("to_search", "subset"),
        [
            ("flow_in_inc_eff", None),
            ("datesteps", None),
            ("non_existent", None),
            (3, None),
            (True, None),
            (1.0, None),
            # explicit subset cases
            ("foo", {"parameters", "lookups", "global_expressions"}),
            ("flow_cap", {"variables", "objectives"}),
        ],
    )
    def test_find_not_found_error(self, to_search, subset, base_math_validated):
        """Requesting components that are not there should raise an error."""
        with pytest.raises(
            KeyError, match=f"Component name `{to_search}` not found in math schema"
        ):
            base_math_validated.find(to_search, subset)
