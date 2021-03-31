import pytest
from pytest import approx
import pandas as pd
import numpy as np

import calliope
from calliope.test.common.util import (
    check_error_or_warning,
    check_variable_exists,
    get_indexed_constraint_body,
)

TOL = 1 + 1e-6


def get_series(m, var, subset={}):
    return m.get_formatted_array(var).loc[subset].to_series().dropna()


class TestEuroCalliopeConstraints:
    def test_carrier_production_max_time_varying_constraint(self):
        m = calliope.examples.urban_scale(scenario="eurocalliope_base")
        assert "energy_cap_max_time_varying" in m._model_data
        energy_cap_max = get_series(m, "energy_cap_max_time_varying")
        assert energy_cap_max.index.get_level_values("techs").unique() == ["boiler"]

        m.run()
        energy_cap = get_series(m, "energy_cap", {"techs": "boiler"})
        carrier_prod = get_series(m, "carrier_prod", {"techs": "boiler"}).droplevel(
            "carriers"
        )
        assert (
            carrier_prod
            <= energy_cap_max.droplevel("techs").mul(energy_cap, level="locs").mul(TOL)
        ).all()

    def test_chp_extraction_p2h_tech(self):
        m = calliope.examples.urban_scale(scenario="chp_extraction_p2h")
        for param in ["cb", "cv", "energy_cap_ratio"]:
            assert param in m._model_data.data_vars
            assert len(m._model_data[param].to_series().dropna()) == 1

        m.run()

        carrier_prod = get_series(m, "carrier_prod", {"techs": "chp", "locs": "X1"})
        energy_cap = m._model_data.energy_cap.loc["X1::chp"].item()
        slope = 2.5 / (3 - 1)
        assert carrier_prod.sum() > 0
        assert (
            carrier_prod.xs("electricity")
            <= slope * (energy_cap * 2.5 - carrier_prod.xs("heat")) * TOL
        ).all()

        assert (
            carrier_prod.xs("electricity")
            <= (energy_cap - carrier_prod.xs("heat") * 0.4) * TOL
        ).all()

    def test_chp_extraction_simple_tech(self):
        m = calliope.examples.urban_scale(scenario="chp_extraction_simple")
        for param in ["cb", "cv"]:
            assert param in m._model_data.data_vars
            assert len(m._model_data[param].to_series().dropna()) == 1

        m.run()

        carrier_prod = get_series(m, "carrier_prod", {"techs": "chp", "locs": "X1"})
        assert carrier_prod.sum() > 0

        assert (
            TOL * carrier_prod.xs("electricity") >= carrier_prod.xs("heat") * 0.45
        ).all()

        assert (
            carrier_prod.xs("electricity")
            <= (
                m._model_data.energy_cap.loc["X1::chp"].item()
                - carrier_prod.xs("heat") * 0.14
            )
            * TOL
        ).all()

    def test_link_con_to_prod_constraint(self):
        pass

    def test_capacity_factor_constraint(self):
        m = calliope.examples.national_scale(scenario="capacity_factor")
        assert "capacity_factor_min" in m._model_data.data_vars.keys()
        assert "capacity_factor_max" in m._model_data.data_vars.keys()

        m.run()
        cf = (
            m.results.carrier_prod.loc["region1::ccgt::power"].sum()
            / (m.results.energy_cap.loc["region1::ccgt"] * len(m.results.timesteps))
        ).item()
        assert cf >= m._model_data.capacity_factor_min.loc["region1::ccgt"] / TOL
        assert cf <= m._model_data.capacity_factor_max.loc["region1::ccgt"] * TOL

    def test_net_transfer_ratio_constraint(self):
        pass

    @pytest.mark.parametrize(
        "scenario", ("carrier_prod_per_week", "carrier_prod_per_week_ts")
    )
    def test_carrier_prod_per_week_constraint(self, scenario):
        m = calliope.examples.urban_scale(scenario=scenario)
        assert "carrier_prod_per_week_min" in m._model_data.data_vars.keys()
        assert "carrier_prod_per_week_max" in m._model_data.data_vars.keys()
        assert "week_numbers" in m._model_data.data_vars.keys()

        m.run()
        assert check_variable_exists(
            m._backend_model, "carrier_prod_per_week_min_constraint", "carrier_prod"
        )
        assert check_variable_exists(
            m._backend_model, "carrier_prod_per_week_max_constraint", "carrier_prod"
        )
        carrier_prod = m.results.carrier_prod.loc["X1::chp::electricity"]
        for week in m._model_data.weeks.values:
            if "_ts" in scenario:
                _min = (
                    0.0008064516129032258 * (m._model_data.week_numbers == week).sum()
                )
                _max = (
                    0.0014784946236559141 * (m._model_data.week_numbers == week).sum()
                )
            else:
                _min = 0.1
                _max = 0.5
            assert (
                carrier_prod.loc[m._model_data.week_numbers == week].sum()
                / carrier_prod.sum()
                >= _min
            )
            assert (
                carrier_prod.loc[m._model_data.week_numbers == week].sum()
                / carrier_prod.sum()
                <= _max
            )
