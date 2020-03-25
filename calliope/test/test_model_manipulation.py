import os

import pytest  # pylint: disable=unused-import

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
    Test removal of techs, locations, links, and transmission techs
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
            "Tech test_storage was removed by setting ``exists: False`` - not checking the consistency of its constraints at location 0.",
        )

    def test_location_exists_false(self):
        overrides = {"locations.1.exists": False}
        with pytest.warns(exceptions.ModelWarning) as excinfo:
            model = build_model(overrides, "simple_storage,one_day,investment_costs")
        model.run()

        # Ensure what should be gone is gone
        assert "1" not in model._model_data.coords["locs"].values

        # Ensure warnings were raised
        assert check_error_or_warning(
            excinfo,
            "Not building the link 0,1 because one or both of its locations have been removed from the model by setting ``exists: false``",
        )

    def test_location_tech_exists_false(self):
        overrides = {"locations.1.techs.test_storage.exists": False}
        model = build_model(overrides, "simple_storage,one_day,investment_costs")
        model.run()

        # Ensure what should be gone is gone
        assert "1::test_storage" not in model._model_data.coords["loc_techs"].values

    def test_link_exists_false(self):
        overrides = {"links.0,1.exists": False}
        model = build_model(overrides, "simple_storage,one_day,investment_costs")
        model.run()

        # Ensure what should be gone is gone
        assert "loc_techs_transmission" not in model._model_data

    def test_link_tech_exists_false(self):
        overrides = {"links.0,1.techs.test_transmission_elec.exists": False}
        model = build_model(overrides, "simple_storage,one_day,investment_costs")
        model.run()

        # Ensure what should be gone is gone
        assert (
            "0::test_transmission_elec:1"
            not in model._model_data.coords["loc_techs_transmission"].values
        )
        assert (
            "0::test_transmission_heat:1"
            in model._model_data.coords["loc_techs_transmission"].values
        )
