from calliope.schemas.model_def_schema import CalliopeModelDef


class TestCalliopeModelDef:
    def test_example_models(self, model_def):
        """Test the schema against example and test model definitions."""
        CalliopeModelDef(**model_def)
