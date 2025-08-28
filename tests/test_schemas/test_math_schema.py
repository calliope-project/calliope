import pytest

from calliope import AttrDict
from calliope.io import read_rich_yaml
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
