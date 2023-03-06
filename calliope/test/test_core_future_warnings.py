import pytest

import calliope
from calliope.test.common.util import build_test_model as build_model
from calliope.test.common.util import check_error_or_warning


class TestDeprecationWarnings:
    def test_get_formatted_array_deprecationwarning(self):
        model = build_model(scenario="simple_supply,one_day,investment_costs")
        model.run()

        with pytest.warns(DeprecationWarning) as warning:
            model.get_formatted_array("carrier_prod")

        assert check_error_or_warning(warning, "get_formatted_array() is deprecated")

    def test_locations_instead_of_nodes(self):
        with pytest.warns(DeprecationWarning) as warning:
            model = build_model(
                scenario="simple_supply_locations,one_day,investment_costs"
            )

        assert check_error_or_warning(warning, "`locations` has been renamed")

        model.run()
        assert model.results.carrier_prod.sum() == 420
