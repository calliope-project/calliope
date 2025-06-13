from calliope.schemas.config_schema import CalliopeConfig


class TestCalliopeConfig:
    def test_example_models(self, model_def):
        """Test the schema against example and test model definitions."""
        CalliopeConfig(**model_def["config"])
