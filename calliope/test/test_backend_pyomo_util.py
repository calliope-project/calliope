import pytest  # pylint: disable=unused-import

import pyomo.core as po

from calliope.test.common.util import build_test_model as build_model
from calliope.backend.pyomo.util import get_domain, get_param, check_value


@pytest.fixture(scope="class")
def model():
    return build_model({}, "simple_supply,two_hours,investment_costs")


class TestGetParam:
    def test_get_param_with_timestep_existing(self):
        """
        """
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m.run()
        param = get_param(
            m._backend_model,
            "resource",
            ("1::test_demand_elec", m._backend_model.timesteps[1]),
        )
        assert po.value(param) == -5  # see demand_elec.csv

    def test_get_param_no_timestep_existing(self):
        """
        """
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m.run()
        param = get_param(
            m._backend_model,
            "energy_eff",
            ("1::test_supply_elec", m._backend_model.timesteps[1]),
        )
        assert po.value(param) == 0.9  # see test model.yaml

    def test_get_param_no_timestep_possible(self):
        """
        """
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m.run()
        param = get_param(m._backend_model, "energy_cap_max", ("1::test_supply_elec"))
        assert po.value(param) == 10  # see test model.yaml

        param = get_param(
            m._backend_model, "cost_energy_cap", ("monetary", "0::test_supply_elec")
        )
        assert po.value(param) == 10

    def test_get_param_from_default(self):
        """
        """
        m = build_model({}, "simple_supply_and_supply_plus,two_hours,investment_costs")
        m.run()

        param = get_param(
            m._backend_model,
            "parasitic_eff",
            ("1::test_supply_plus", m._backend_model.timesteps[1]),
        )
        assert po.value(param) == 1  # see defaults.yaml

        param = get_param(m._backend_model, "resource_cap_min", ("0::test_supply_plus"))
        assert po.value(param) == 0  # see defaults.yaml

        param = get_param(
            m._backend_model, "cost_resource_cap", ("monetary", "1::test_supply_plus")
        )
        assert po.value(param) == 0  # see defaults.yaml

    def test_get_param_no_default_defined(self):
        """
        If a default is not defined, raise KeyError
        """
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m.run()
        with pytest.raises(KeyError):
            get_param(
                m._backend_model,
                "random_param",
                ("1::test_demand_elec", m._backend_model.timesteps[1]),
            )
            get_param(m._backend_model, "random_param", ("1::test_supply_elec"))


class TestGetDomain:
    @pytest.mark.parametrize(
        "var, domain",
        (
            ("energy_cap_max", "NonNegativeReals"),
            ("resource", "Reals"),
            ("cost_energy_cap", "Reals"),
            ("force_resource", "Boolean"),
            ("names", "Any"),
        ),
    )
    def test_dtypes(self, model, var, domain):
        assert get_domain(model._model_data[var]) == domain


class TestCheckValue:
    def test_check_values(self):
        pyomo_model = po.ConcreteModel()
        pyomo_model.new_set = po.Set(initialize=["a", "b"])
        pyomo_model.new_param = po.Param(
            pyomo_model.new_set,
            initialize={"a": 1},
            mutable=True,
            within=po.NonNegativeReals,
        )

        assert check_value(pyomo_model.new_param["a"]) is False
        assert check_value(pyomo_model.new_param["b"]) is True
