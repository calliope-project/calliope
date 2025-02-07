import importlib

import pydantic
import pytest

from calliope.schemas.dimension_data_schema import CalliopeTech, IndexedParam

EXAMPLES_DIR = importlib.resources.files("calliope") / "example_models"


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
        with pytest.raises(pydantic.ValidationError):
            IndexedParam(data=data, dims=dims, index=index)


class TestCalliopeTech:
    @pytest.mark.parametrize(
        ("link_from", "link_to"),
        [("error", None), (None, "error"), ("error1", "error2")],
    )
    @pytest.mark.parametrize(
        ("base_tech", "carrier_in", "carrier_out"),
        [
            ("conversion", "gas", "elec"),
            ("demand", "gas", None),
            ("storage", "gas", "gas"),
            ("supply", None, "gas"),
        ],
    )
    def test_non_transmission_link_error(
        self, base_tech, carrier_in, carrier_out, link_from, link_to
    ):
        """Non-transmission technologies should not accept links."""
        with pytest.raises(pydantic.ValidationError):
            CalliopeTech(
                base_tech=base_tech,
                carrier_in=carrier_in,
                carrier_out=carrier_out,
                link_from=link_from,
                link_to=link_to,
            )

    @pytest.mark.parametrize("base_tech", ["conversion", "storage"])
    @pytest.mark.parametrize(
        ("carrier_in", "carrier_out"), [(None, None), ("gas", None), (None, "gas")]
    )
    def test_bi_directional_tech_error(self, base_tech, carrier_in, carrier_out):
        """Bi-direcitonal techs need both carrier_in / out."""
        with pytest.raises(pydantic.ValidationError):
            CalliopeTech(
                base_tech=base_tech, carrier_in=carrier_in, carrier_out=carrier_out
            )
