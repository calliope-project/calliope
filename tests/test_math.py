import importlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
from calliope import AttrDict
from pyomo.repn.tests import lp_diff

from .common.util import build_lp, build_test_model

CALLIOPE_DIR: Path = importlib.resources.files("calliope")


@pytest.fixture(scope="class")
def compare_lps(tmpdir_factory):
    def _compare_lps(model, custom_math, filename):
        lp_file = filename + ".lp"
        generated_file = Path(tmpdir_factory.mktemp("lp_files")) / lp_file
        backend = build_lp(model, generated_file, custom_math)  # noqa: F841
        expected_file = Path(__file__).parent / "common" / "lp_files" / lp_file
        # Pyomo diff ignores trivial numeric differences (10 == 10.0)
        # But it does not ignore a re-ordering of components
        diff_ordered = lp_diff.load_and_compare_lp_baseline(
            generated_file.as_posix(), expected_file.as_posix()
        )
        # Our unordered comparison ignores component ordering but cannot handle
        # trivial differences in numerics (as everything is a string to it)
        diff_unordered = _diff_files(generated_file, expected_file)

        # If one of the above matches across the board, we're good to go.
        assert diff_ordered == ([], []) or not diff_unordered

    return _compare_lps


def _diff_files(file1, file2):
    file1_lines = file1.read_text().split("\n")
    file2_lines = file2.read_text().split("\n")
    return set(file1_lines).symmetric_difference(file2_lines)


class TestBaseMath:
    TEST_REGISTER: set = set()

    @pytest.fixture(scope="class")
    def base_math(self):
        return AttrDict.from_yaml(CALLIOPE_DIR / "math" / "base.yaml")

    def test_flow_out_cap(self, compare_lps):
        self.TEST_REGISTER.add("variables.flow_out_cap")
        model = build_test_model(
            {
                "nodes.b.techs.test_supply_elec.constraints.flow_out_cap_max": 100,
                "nodes.a.techs.test_supply_elec.constraints.flow_out_cap_min": 1,
                "nodes.a.techs.test_supply_elec.constraints.flow_out_cap_max": np.nan,
            },
            "simple_supply,two_hours,investment_costs",
        )
        custom_math = {
            # need the variable defined in a constraint/objective for it to appear in the LP file bounds
            "objectives": {
                "foo": {
                    "equations": [
                        {
                            "expression": "sum(flow_out_cap[techs=test_supply_elec], over=nodes)"
                        }
                    ],
                    "sense": "minimise",
                }
            }
        }
        compare_lps(model, custom_math, "flow_out_cap")

        # "flow_out_cap" is the name of the lp file

    def test_storage_max(self, compare_lps):
        self.TEST_REGISTER.add("constraints.storage_max")
        model = build_test_model(
            scenario="simple_storage,two_hours,investment_costs",
        )
        custom_math = {
            "constraints": {"storage_max": model.math.constraints.storage_max}
        }
        compare_lps(model, custom_math, "storage_max")

    def test_flow_out_max(self, compare_lps):
        self.TEST_REGISTER.add("constraints.flow_out_max")
        model = build_test_model(
            {
                "nodes.a.techs.test_supply_elec.constraints.flow_out_cap_min": 100,
                "nodes.a.techs.test_supply_elec.constraints.flow_out_cap_max": 100,
            },
            "simple_supply,two_hours,investment_costs",
        )

        custom_math = {
            "constraints": {"flow_out_max": model.math.constraints.flow_out_max}
        }
        compare_lps(model, custom_math, "flow_out_max")

    def test_balance_conversion(self, compare_lps):
        self.TEST_REGISTER.add("constraints.balance_conversion")

        model = build_test_model(
            scenario="simple_conversion,two_hours,investment_costs",
        )
        custom_math = {
            "constraints": {
                "balance_conversion": model.math.constraints.balance_conversion
            }
        }

        compare_lps(model, custom_math, "balance_conversion")

    def test_source_max(self, compare_lps):
        self.TEST_REGISTER.add("constraints.source_max")
        model = build_test_model(
            {},
            "simple_supply_plus,resample_two_days,investment_costs",
        )
        custom_math = {
            "constraints": {"my_constraint": model.math.constraints.source_use_max}
        }
        compare_lps(model, custom_math, "source_max")

    @pytest.mark.xfail(reason="not all base math is in the test config dict yet")
    def test_all_math_registered(self, base_math):
        "After running all the previous tests in the class, the base_math dict should be empty, i.e. all math has been tested"
        for key in self.TEST_REGISTER:
            base_math.del_key(key)
        assert not base_math


class CustomMathExamples(ABC):
    TEST_REGISTER: set = set()

    #: source of all custom math files
    CUSTOM_MATH_DIR = CALLIOPE_DIR.parent.parent / "doc" / "_static" / "custom_math"

    @property
    @abstractmethod
    def YAML_FILEPATH(self) -> str:
        "Source of the specific test class custom math"

    @pytest.fixture(scope="class")
    def abs_filepath(self):
        return (self.CUSTOM_MATH_DIR / self.YAML_FILEPATH).absolute()

    @pytest.fixture(scope="class")
    def custom_math(self):
        return AttrDict.from_yaml(self.CUSTOM_MATH_DIR / self.YAML_FILEPATH)

    @pytest.fixture
    def build_and_compare(self, abs_filepath, compare_lps):
        def _build_and_compare(
            filename: str,
            scenario: str,
            overrides: Optional[dict] = None,
            components: Optional[dict[list[str]]] = None,
        ):
            if components is not None:
                for component_group, component_list in components.items():
                    for component in component_list:
                        self.TEST_REGISTER.add(f"{component_group}.{component}")

                custom_math = {k: v for k, v in components.items() if k != "variables"}
            else:
                self.TEST_REGISTER.add(f"constraints.{filename}")
                custom_math = {"constraints": [filename]}

            if overrides is None:
                overrides = {}

            model = build_test_model(
                {"config.init.custom_math": [abs_filepath], **overrides},
                scenario,
            )

            compare_lps(model, custom_math, filename)

        return _build_and_compare

    @pytest.mark.order(-1)
    def test_all_math_registered(self, custom_math):
        "After running all the previous tests in the class, the register should be full, i.e. all math has been tested"
        for key in self.TEST_REGISTER:
            if custom_math.get_key(key, default=None) is not None:
                custom_math.del_key(key)
        assert not custom_math


@pytest.mark.filterwarnings(
    "ignore:(?s).*defines unrecognised constraint `annual_flow_max`:calliope.exceptions.ModelWarning"
)
class TestAnnualEnergyBalance(CustomMathExamples):
    YAML_FILEPATH = "annual_energy_balance.yaml"

    def test_annual_energy_balance_per_tech_and_node(self, build_and_compare):
        overrides = {
            "nodes.a.techs.test_supply_elec.constraints.annual_flow_max": 10,
            "nodes.b.techs.test_supply_elec.constraints.annual_flow_max": 20,
        }
        build_and_compare(
            "annual_energy_balance_per_tech_and_node",
            "simple_supply,two_hours",
            overrides,
        )

    def test_annual_energy_balance_global_per_tech(self, build_and_compare):
        overrides = {
            "parameters": {
                "annual_flow_max": {
                    "data": 10,
                    "index": ["test_supply_elec"],
                    "dims": "techs",
                },
            }
        }
        build_and_compare(
            "annual_energy_balance_global_per_tech",
            "simple_supply,two_hours",
            overrides,
        )

    def test_annual_energy_balance_global_multi_tech(self, build_and_compare):
        overrides = {
            "parameters": {
                "annual_flow_max": {"data": 10},
                "flow_max_group": {
                    "data": True,
                    "index": ["test_supply_elec", "test_supply_plus"],
                    "dims": "techs",
                },
            }
        }
        build_and_compare(
            "annual_energy_balance_global_multi_tech",
            "simple_supply_and_supply_plus,two_hours",
            overrides,
        )

    def test_annual_energy_balance_total_source_availability(self, build_and_compare):
        overrides = {
            "parameters": {
                "annual_source_max": {
                    "data": 10,
                    "index": ["test_supply_plus"],
                    "dims": "techs",
                },
            }
        }
        build_and_compare(
            "annual_energy_balance_total_source_availability",
            "simple_supply_and_supply_plus,two_hours",
            overrides,
        )

    def test_annual_energy_balance_total_sink_availability(self, build_and_compare):
        overrides = {
            "parameters": {
                "annual_sink_max": {
                    "data": 10,
                    "index": ["test_demand_elec"],
                    "dims": "techs",
                },
            },
        }
        build_and_compare(
            "annual_energy_balance_total_sink_availability",
            "simple_supply,two_hours,demand_elec_max",
            overrides,
        )


class TestMaxTimeVarying(CustomMathExamples):
    YAML_FILEPATH = "max_time_varying.yaml"

    def test_max_time_varying_flow_cap(self, build_and_compare):
        overrides = {
            "parameters": {
                "flow_cap_max_relative_per_ts": {
                    "data": [0.8, 0.5],
                    "index": [
                        ["test_supply_elec", "2005-01-01 00:00"],
                        ["test_supply_elec", "2005-01-01 01:00"],
                    ],
                    "dims": ["techs", "timesteps"],
                },
            },
        }
        build_and_compare(
            "max_time_varying_flow_cap",
            "simple_supply,two_hours",
            overrides,
        )

    def test_max_time_varying_storage(self, build_and_compare):
        overrides = {
            "parameters": {
                "storage_max_relative_per_ts": {
                    "data": [0.8, 0.5],
                    "index": [
                        ["test_storage", "2005-01-01 00:00"],
                        ["test_storage", "2005-01-01 01:00"],
                    ],
                    "dims": ["techs", "timesteps"],
                },
            },
        }
        build_and_compare(
            "max_time_varying_storage",
            "simple_storage,two_hours",
            overrides,
        )


@pytest.mark.filterwarnings(
    "ignore:(?s).*defines unrecognised constraint `turbine_type`:calliope.exceptions.ModelWarning"
)
class TestCHPHTP(CustomMathExamples):
    YAML_FILEPATH = "chp_htp.yaml"

    def test_chp_extraction(self, build_and_compare):
        overrides = {
            "techs.test_chp.constraints.power_loss_factor": 0.1,
            "techs.test_chp.constraints.power_to_heat_ratio": 2,
            "techs.test_chp.constraints.energy_eff": 0.6,
            "techs.test_chp.constraints.turbine_type": "extraction",
        }
        build_and_compare(
            "chp_extraction",
            "simple_chp,two_hours",
            overrides,
            components={
                "constraints": ["chp_extraction_line", "chp_backpressure_line_min"]
            },
        )

    def test_chp_backpressure_and_boiler(self, build_and_compare):
        overrides = {
            "techs.test_chp.constraints.power_to_heat_ratio": 1.5,
            "techs.test_chp.constraints.boiler_eff": 0.8,
            "techs.test_chp.constraints.energy_eff": 0.6,
            "techs.test_chp.constraints.turbine_type": "backpressure",
        }
        build_and_compare(
            "chp_backpressure_and_boiler",
            "simple_chp,two_hours",
            overrides,
            components={
                "constraints": [
                    "chp_divert_fuel_to_boiler",
                    "chp_backpressure_line_max",
                ]
            },
        )

    def test_chp_backpressure_no_boiler(self, build_and_compare):
        overrides = {
            "techs.test_chp.constraints.power_to_heat_ratio": 1.25,
            "techs.test_chp.constraints.turbine_type": "backpressure",
        }
        build_and_compare(
            "chp_backpressure_line_equals",
            "simple_chp,two_hours",
            overrides,
        )


class TestShareAllTimesteps(CustomMathExamples):
    YAML_FILEPATH = "share_all_timesteps.yaml"

    @pytest.mark.filterwarnings(
        "ignore:(?s).*defines unrecognised constraint `demand_share_equals`:calliope.exceptions.ModelWarning"
    )
    def test_demand_share_equals_per_tech(self, build_and_compare):
        overrides = {
            "nodes.a.techs.test_supply_elec.constraints.demand_share_equals": 0.5,
            "nodes.b.techs.test_supply_elec.constraints.demand_share_equals": 0.8,
            "parameters": {"demand_share_tech.data": "test_demand_elec"},
        }
        build_and_compare(
            "demand_share_equals_per_tech",
            "simple_supply,two_hours",
            overrides,
        )

    @pytest.mark.filterwarnings(
        "ignore:(?s).*defines unrecognised constraint `supply_share_equals`:calliope.exceptions.ModelWarning"
    )
    def test_supply_share_equals_per_tech(self, build_and_compare):
        overrides = {
            "nodes.a.techs.test_supply_elec.constraints.supply_share_equals": 0.5,
            "nodes.b.techs.test_supply_elec.constraints.supply_share_equals": 0.8,
            "parameters": {"supply_share_carrier.data": "electricity"},
        }
        build_and_compare(
            "supply_share_equals_per_tech",
            "simple_supply_and_supply_plus,two_hours",
            overrides,
        )


class TestSharePerTimestep(CustomMathExamples):
    YAML_FILEPATH = "share_per_timestep.yaml"

    @pytest.mark.filterwarnings(
        "ignore:(?s).*defines unrecognised constraint `demand_share_per_timestep_equals`:calliope.exceptions.ModelWarning"
    )
    def test_demand_share_per_timestep_equals_per_tech(self, build_and_compare):
        overrides = {
            "nodes.a.techs.test_supply_elec.constraints.demand_share_per_timestep_equals": 0.5,
            "nodes.b.techs.test_supply_elec.constraints.demand_share_per_timestep_equals": 0.8,
            "parameters": {"demand_share_tech.data": "test_demand_elec"},
        }
        build_and_compare(
            "demand_share_per_timestep_equals_per_tech",
            "simple_supply,two_hours",
            overrides,
        )

    @pytest.mark.filterwarnings(
        "ignore:(?s).*defines unrecognised constraint `supply_share_per_timestep_equals`:calliope.exceptions.ModelWarning"
    )
    def test_supply_share_per_timestep_equals_per_tech(self, build_and_compare):
        overrides = {
            "nodes.a.techs.test_supply_elec.constraints.supply_share_per_timestep_equals": 0.5,
            "nodes.b.techs.test_supply_elec.constraints.supply_share_per_timestep_equals": 0.8,
            "parameters": {"supply_share_carrier.data": "electricity"},
        }
        build_and_compare(
            "supply_share_per_timestep_equals_per_tech",
            "simple_supply_and_supply_plus,two_hours",
            overrides,
        )


class TestDemandSharePerTimestepDecision(CustomMathExamples):
    YAML_FILEPATH = "demand_share_per_timestep_decision.yaml"

    def test_demand_share_per_timestep_decision_main(self, build_and_compare):
        overrides = {
            "parameters": {
                "decide_demand_share": {
                    "data": "test_demand_elec",
                    "index": ["test_supply_elec", "test_conversion_plus"],
                    "dims": "techs",
                },
                "demand_share_carrier.data": "electricity",
                "demand_share_relaxation.data": 0.01,
            }
        }
        build_and_compare(
            "demand_share_per_timestep_decision_main",
            "conversion_and_conversion_plus,two_hours",
            overrides,
            components={
                "constraints": [
                    "demand_share_per_timestep_decision_main_min",
                    "demand_share_per_timestep_decision_main_max",
                ],
                "variables": ["demand_share_per_timestep_decision"],
            },
        )

    def test_demand_share_per_timestep_decision_sum(self, build_and_compare):
        overrides = {
            "parameters": {
                "decide_demand_share": {
                    "data": "test_demand_elec",
                    "index": ["test_supply_elec", "test_conversion_plus"],
                    "dims": "techs",
                },
                "demand_share_carrier.data": "electricity",
                "demand_share_limit.data": 0.5,
            },
        }
        build_and_compare(
            "demand_share_per_timestep_decision_sum",
            "conversion_and_conversion_plus,two_hours",
            overrides,
        )


class TestPiecewiseCosts(CustomMathExamples):
    YAML_FILEPATH = "piecewise_linear_costs.yaml"

    def test_piecewise(self, build_and_compare):
        overrides = {
            "techs.test_supply_elec.constraints.lifetime": 10,
            "techs.test_supply_elec.costs.monetary.interest_rate": 0.1,
            "parameters": {
                "cost_flow_out_cap_piecewise_slopes": {
                    "data": [5, 7, 14],
                    "index": [0, 1, 2],
                    "dims": "pieces",
                },
                "cost_flow_out_cap_piecewise_intercept": {
                    "data": [0, -2, -16],
                    "index": [0, 1, 2],
                    "dims": "pieces",
                },
            },
        }
        build_and_compare(
            "piecewise_cost_investment",
            "supply_purchase,two_hours",
            overrides,
            components={
                "constraints": ["piecewise_costs"],
                "variables": ["piecewise_cost_investment"],
                "global_expressions": ["cost_investment", "cost_var", "cost"],
            },
        )


class TestPiecewiseEfficiency(CustomMathExamples):
    YAML_FILEPATH = "piecewise_linear_efficiency.yaml"

    def test_piecewise(self, build_and_compare):
        overrides = {
            "parameters": {
                "flow_out_eff_piecewise_slopes": {
                    "data": [5, 7, 14],
                    "index": [0, 1, 2],
                    "dims": "pieces",
                },
                "flow_out_eff_piecewise_intercept": {
                    "data": [0, -2, -16],
                    "index": [0, 1, 2],
                    "dims": "pieces",
                },
            },
        }
        build_and_compare(
            "piecewise_efficiency",
            "conversion_milp,two_hours",
            overrides,
            components={
                "constraints": [
                    "piecewise_efficiency",
                    "available_flow_cap_binary",
                    "available_flow_cap_continuous",
                    "available_flow_cap_binary_continuous_switch",
                ],
                "variables": ["available_flow_cap"],
            },
        )


@pytest.mark.filterwarnings(
    "ignore:(?s).*`test_conversion_plus` gives a carrier ratio for `heat`:calliope.exceptions.ModelWarning"
)
class TestFuelDist(CustomMathExamples):
    YAML_FILEPATH = "fuel_dist.yaml"

    def test_fuel_distribution(self, build_and_compare):
        overrides = {
            "parameters": {
                "allow_fuel_distribution": {
                    "data": True,
                    "index": ["coal"],
                    "dims": "carriers",
                },
            },
        }
        build_and_compare(
            "fuel_dist_base",
            "fuel_distribution,two_hours",
            overrides,
            components={
                "constraints": [
                    "system_balance",
                    "restrict_total_imports_and_exports",
                ],
                "variables": ["fuel_distributor"],
            },
        )

    def test_fuel_distribution_nodal_limits(self, build_and_compare):
        overrides = {
            "parameters": {
                "allow_fuel_distribution": {
                    "data": True,
                    "index": ["coal"],
                    "dims": "carriers",
                },
                "fuel_import_max": {
                    "data": 5,
                    "index": [["coal", "b"]],
                    "dims": ["carriers", "nodes"],
                },
                "fuel_export_max": {
                    "data": 3,
                    "index": [["coal", "a"]],
                    "dims": ["carriers", "nodes"],
                },
            },
        }
        build_and_compare(
            "fuel_dist_nodal",
            "fuel_distribution,two_hours",
            overrides,
            components={
                "constraints": ["restrict_nodal_imports", "restrict_nodal_exports"],
            },
        )

    def test_fuel_distribution_costs(self, build_and_compare):
        overrides = {
            "parameters": {
                "allow_fuel_distribution": {
                    "data": True,
                    "index": ["coal"],
                    "dims": "carriers",
                },
                "fuel_distributor_costs": {
                    "data": 5,
                    "index": [["coal", "monetary"]],
                    "dims": ["carriers", "costs"],
                },
            },
        }
        build_and_compare(
            "fuel_dist_cost",
            "fuel_distribution,two_hours,investment_costs",
            overrides,
            components={
                "global_expressions": [
                    "cost_investment",  # Need to build these up so that `cost` is available in the objective
                    "cost_var",  # Need to build these up so that `cost` is available in the objective
                    "cost",  # Need to build these up so that `cost` is available in the objective
                    "cost_fuel_distribution",
                ],
                "objectives": ["minmax_cost_optimisation"],
            },
        )


class TestUptimeDowntime(CustomMathExamples):
    YAML_FILEPATH = "uptime_downtime_limits.yaml"

    @pytest.mark.filterwarnings(
        "ignore:(?s).*defines unrecognised constraint `capacity_factor:calliope.exceptions.ModelWarning"
    )
    def test_annual_capacity_factor(self, build_and_compare):
        overrides = {
            "techs.test_supply_elec.constraints.capacity_factor_min": 0.8,
            "techs.test_supply_elec.constraints.capacity_factor_max": 0.9,
        }
        build_and_compare(
            "annual_capacity_factor",
            "simple_supply,two_hours",
            overrides,
            components={
                "constraints": [
                    "annual_capacity_factor_min",
                    "annual_capacity_factor_max",
                ]
            },
        )

    def test_downtime(self, build_and_compare):
        overrides = {
            "parameters": {
                "downtime_periods": {
                    "data": True,
                    "index": [["test_supply_elec", "a", "2005-01-01 00:00"]],
                    "dims": ["techs", "nodes", "timesteps"],
                },
            },
        }
        build_and_compare(
            "downtime_period",
            "simple_supply,two_hours",
            overrides,
            components={"constraints": ["downtime_period"]},
        )

    @pytest.mark.filterwarnings(
        "ignore:(?s).*defines unrecognised constraint `uptime_limit`:calliope.exceptions.ModelWarning"
    )
    def test_downtime_decision(self, build_and_compare):
        overrides = {
            "techs.test_supply_elec.constraints.uptime_limit": 1,
        }
        build_and_compare(
            "downtime_period_decision", "supply_milp,two_hours", overrides
        )
