import os

import pytest  # noqa: F401

import calliope
from calliope import exceptions
from calliope.test.common.util import check_error_or_warning


def build_model(override_dict, scenario):
    model_path = os.path.join(
        os.path.dirname(__file__), "common", "test_model", "model.yaml"
    )

    return calliope.Model(model_path, override_dict=override_dict, scenario=scenario)


class TestExistsFalse:
    """
    Test removal of techs, nodes, links, and transmission techs
    with the ``exists: False`` configuration option.

    """

    def test_tech_exists_false(self):
        overrides = {"techs.test_storage.exists": False}
        with pytest.warns(exceptions.ModelWarning) as excinfo:
            model = build_model(overrides, "simple_storage,one_day,investment_costs")
        model.run()

        # Ensure what should be gone is gone
        assert "test_storage" not in model.results.coords["techs"].values

        # Ensure warnings were raised
        assert check_error_or_warning(
            excinfo,
            "Tech test_storage was removed by setting ``exists: False`` - not checking the consistency of its constraints at node a.",
        )

    def test_node_exists_false(self):
        overrides = {"nodes.b.exists": False}
        with pytest.warns(exceptions.ModelWarning) as excinfo:
            model = build_model(overrides, "simple_storage,one_day,investment_costs")
        model.run()

        # Ensure what should be gone is gone
        assert "b" not in model._model_data.coords["nodes"].values

        # Ensure warnings were raised
        assert check_error_or_warning(
            excinfo,
            "Not building the link a,b because one or both of its nodes have been removed from the model by setting ``exists: false``",
        )

    def test_node_tech_exists_false(self):
        overrides = {"nodes.b.techs.test_storage.exists": False}
        model = build_model(overrides, "simple_storage,one_day,investment_costs")
        model.run()

        # Ensure what should be gone is gone
        assert "b", "test_storage" not in model._model_data.coords["loc_techs"].values

    def test_link_exists_false(self):
        overrides = {"links.a,b.exists": False}
        model = build_model(overrides, "simple_storage,one_day,investment_costs")
        model.run()

        # Ensure what should be gone is gone
        assert not model._model_data.inheritance.str.endswith("transmission").any()

    def test_link_tech_exists_false(self):
        overrides = {"links.a,b.techs.test_transmission_elec.exists": False}
        model = build_model(overrides, "simple_storage,one_day,investment_costs")
        model.run()
        # Ensure what should be gone is gone
        "test_transmission_elec:b" not in model._model_data.techs
        "test_transmission_elec:a" not in model._model_data.techs
        assert model._model_data.node_tech.loc["a", "test_transmission_heat:b"] == 1
