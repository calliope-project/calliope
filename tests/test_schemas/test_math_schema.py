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
                "expr3": {"equations": [{"expression": "x / y"}]},
            }
        }
        # Validate the schema
        validated_math = math_schema.MathSchema(**math)
        # Check the order of global expressions
        assert validated_math.global_expressions["expr0"].order == 0
        assert validated_math.global_expressions["expr1"].order == 2
        assert validated_math.global_expressions["expr2"].order == 1
        assert validated_math.global_expressions["expr3"].order == 3
