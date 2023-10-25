import pytest  # noqa: F401
from calliope import exceptions

from .common.util import build_test_model as build_model
from .common.util import check_error_or_warning


class TestActiveFalse:
    """
    Test removal of techs, nodes, links, and transmission techs
    with the ``active: False`` configuration option.

    """

    def test_tech_active_false(self):
        overrides = {"techs.test_storage.active": False}
        with pytest.warns(exceptions.ModelWarning) as excinfo:
            model = build_model(overrides, "simple_storage,two_hours,investment_costs")

        # Ensure what should be gone is gone
        assert "test_storage" not in model._model_data.coords["techs"].values

        # Ensure warnings were raised
        assert check_error_or_warning(
            excinfo,
            "Tech test_storage was removed by setting ``active: False`` - not checking the consistency of its constraints at node a.",
        )

    def test_node_active_false(self):
        overrides = {"nodes.b.active": False}
        with pytest.warns(exceptions.ModelWarning) as excinfo:
            model = build_model(overrides, "simple_storage,two_hours,investment_costs")

        # Ensure what should be gone is gone
        assert "b" not in model._model_data.coords["nodes"].values

        # Ensure warnings were raised
        assert check_error_or_warning(
            excinfo,
            "Not building the link a,b because one or both of its nodes have been removed from the model by setting ``active: false``",
        )

    def test_node_tech_active_false(self):
        overrides = {"nodes.b.techs.test_storage.active": False}
        model = build_model(overrides, "simple_storage,two_hours,investment_costs")

        # Ensure what should be gone is gone
        assert (
            model._model_data.node_tech.sel(techs="test_storage", nodes="b")
            .isnull()
            .item()
        )

    def test_link_active_false(self):
        overrides = {"links.a,b.active": False}
        model = build_model(overrides, "simple_storage,two_hours,investment_costs")

        # Ensure what should be gone is gone
        assert not model._model_data.inheritance.str.endswith("transmission").any()

    def test_link_tech_active_false(self):
        overrides = {"links.a,b.techs.test_transmission_elec.active": False}
        model = build_model(overrides, "simple_storage,two_hours,investment_costs")

        # Ensure what should be gone is gone
        assert "test_transmission_elec:b" not in model._model_data.techs
        assert "test_transmission_elec:a" not in model._model_data.techs
        assert (
            model._model_data.node_tech.sel(nodes="a", techs="test_transmission_heat:b")
            .notnull()
            .item()
        )
