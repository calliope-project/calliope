import pydantic
import pytest

from calliope.io import read_rich_yaml
from calliope.preprocess import prepare_model_definition
from calliope.schemas.dimension_data_schema import (
    CalliopeNode,
    CalliopeTech,
    IndexedParam,
)

from ..test_core_util import check_error_or_warning
from . import utils


class TestIndexedParam:
    @pytest.mark.parametrize(
        ("data", "dims", "index"),
        [
            (100, "costs", "monetary"),
            (
                [2, 10],
                ["cost", "carriers"],
                [["monetary", "electricity"], ["monetary", "heat"]],
            ),
        ],
    )
    def test_regular_definition(self, data, dims, index):
        """One dimensional and multi-dimensional definitions should pass."""
        IndexedParam(data=data, dims=dims, index=index)

    def test_broadcasted_definition(self):
        """Broadcasted definitons should be possible."""
        IndexedParam(data=1, dims="my_dim", index=["i1", "i2", "i3", "i4"])

    @pytest.mark.parametrize(
        ("data", "dims", "index"),
        [
            (1, 1, "monetary"),  # dims must be strings
            ("value", "techs", None),  # indexes must be strings or numeric
            (1, ["techs", "techs", "nodes"], ["i1", "i2", "i3"]),  # dims must be unique
            (1, ["techs", "nodes"], ["costs", "costs"]),  # indexes must be unique
            ([], ["techs"], ["i1"]),  # lists must not be empty
            ([1], [], ["i1"]),
            ([1], ["techs"], []),
        ],
    )
    def test_invalid_definition(self, data, dims, index):
        """Catch common user mistakes."""
        with pytest.raises(
            pydantic.ValidationError, match="errors for Indexed parameter definition"
        ):
            IndexedParam(data=data, dims=dims, index=index)


class TestCalliopeTech:
    @pytest.mark.parametrize("model_path", utils.EXAMPLE_MODELS + utils.TEST_MODELS)
    def test_example_model_techs(self, model_path):
        """Test the example model technologies against the schema."""
        model_def, _ = prepare_model_definition(model_path)
        if "techs" in model_def:
            for tech in model_def["techs"].values():
                CalliopeTech(**tech)

    @pytest.mark.parametrize("dims", ["techs", "nodes"])
    def test_invalid_dims(self, dims):
        """Technologies must not use 'techs' or 'nodes' in their indexed params."""
        tech = read_rich_yaml(
            f"""
        name: 'Combined cycle gas turbine'
        color: '#FDC97D'
        base_tech: supply
        carrier_out: power
        foobar:
          data: 0.10
          index: monetary
          dims: {dims}
        """
        )
        with pytest.raises(pydantic.ValidationError, match="`dims` must not contain"):
            CalliopeTech(**tech)

    @pytest.mark.parametrize("base_tech", [None, 1, "foobar"])
    def test_invalid_base_tech_name(self, base_tech):
        """Incorrect `base_tech` settings should be detected."""
        tech = read_rich_yaml(f"""
        name: Supply tech
        carrier_out: gas
        base_tech: {base_tech}
        flow_cap_max: 10
        source_use_max: .inf
        """)
        with pytest.raises(pydantic.ValidationError) as error:
            CalliopeTech(**tech)
        check_error_or_warning(
            error, ["error for Technology dimension data", "base_tech"]
        )


class TestCalliopeNode:
    @pytest.mark.parametrize("model_path", utils.EXAMPLE_MODELS + utils.TEST_MODELS)
    def test_example_models(self, model_path):
        """Test the node schema against example and test model definitions."""
        model_def, _ = prepare_model_definition(model_path)
        if "nodes" in model_def:
            for node in model_def["nodes"].values():
                CalliopeNode(**node)

    @pytest.mark.parametrize(("latitude", "longitude"), [(None, 30), (30, None)])
    def test_dependent_definitions(self, latitude, longitude):
        with pytest.raises(
            pydantic.ValidationError, match="Invalid latitude/longitude definition."
        ):
            CalliopeNode(techs={}, latitude=latitude, longitude=longitude)
