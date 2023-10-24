import pytest

from .common.util import build_test_model as build_model
from .common.util import check_error_or_warning


class TestDeprecationWarnings:
    def test_run_deprecationwarning(self):
        model = build_model(scenario="simple_supply,two_hours,investment_costs")

        with pytest.warns(DeprecationWarning) as warning:
            model.run()

        assert check_error_or_warning(warning, "`run()` is deprecated")
        assert hasattr(model, "backend")
        assert hasattr(model, "results")

    def test_locations_instead_of_nodes(self):
        with pytest.warns(DeprecationWarning) as warning:
            model = build_model(
                scenario="simple_supply_locations,one_day,investment_costs"
            )

        assert check_error_or_warning(warning, "`locations` has been renamed")
        assert set(model._model_data.nodes.values) == {"a", "b"}
