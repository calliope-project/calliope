import pytest
from pydantic import ValidationError

from calliope.io import read_rich_yaml
from calliope.preprocess import prepare_model_definition
from calliope.schemas.model_def_schema import CalliopeModelDef

from ..common.util import build_test_model_def
from . import utils


class TestCalliopeModelDef:
    @pytest.mark.parametrize("model_path", utils.EXAMPLE_MODELS + utils.TEST_MODELS)
    def test_example_models(self, model_path):
        """Test the schema against example and test model definitions."""
        model_def, _ = prepare_model_definition(model_path)
        CalliopeModelDef(**model_def)

    def test_undefined_carriers(self):
        """Test that validation tests for technology carriers are enabled."""
        override = read_rich_yaml(
            """
            techs:
                test_undefined_carrier:
                    base_tech: supply
                    name: test
                    source_use_max: .inf
                    flow_cap_max: .inf
            nodes.a.techs.test_undefined_carrier:
            """
        )
        model_def = build_test_model_def(
            override_dict=override, scenario="simple_supply,one_day"
        )
        with pytest.raises(ValidationError, match="Incorrect supply setup"):
            CalliopeModelDef(**model_def)
