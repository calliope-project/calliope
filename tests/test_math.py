import importlib
from abc import ABC, abstractmethod
from pathlib import Path

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
        build_lp(model, generated_file, custom_math)
        expected_file = Path(__file__).parent / "common" / "lp_files" / lp_file
        diff = lp_diff.load_and_compare_lp_baseline(
            generated_file.as_posix(), expected_file.as_posix()
        )
        assert diff == ([], [])

    return _compare_lps


class TestBaseMath:
    TEST_REGISTER: set = set()

    @pytest.fixture(scope="class")
    def base_math(self):
        return AttrDict.from_yaml(CALLIOPE_DIR / "math" / "base.yaml")

    def test_flow_cap(self, compare_lps):
        self.TEST_REGISTER.add("variables.flow_cap")
        model = build_test_model(
            {
                "nodes.b.techs.test_supply_elec.constraints.flow_cap_max": 100,
                "nodes.a.techs.test_supply_elec.constraints.flow_cap_min": 1,
                "nodes.a.techs.test_supply_elec.constraints.flow_cap_max": np.nan,
            },
            "simple_supply,two_hours,investment_costs",
        )
        custom_math = {
            # need the variable defined in a constraint/objective for it to appear in the LP file bounds
            "objectives": {
                "foo": {
                    "equations": [
                        {
                            "expression": "sum(flow_cap[techs=test_supply_elec], over=nodes)"
                        }
                    ],
                    "sense": "minimise",
                }
            }
        }
        compare_lps(model, custom_math, "flow_cap")

        # "flow_cap" is the name of the lp file

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
                "nodes.a.techs.test_supply_elec.constraints.flow_cap_min": 100,
                "nodes.a.techs.test_supply_elec.constraints.flow_cap_max": 100,
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
    CUSTOM_MATH_DIR = CALLIOPE_DIR.parent / "doc" / "_static" / "custom_math"

    @property
    @abstractmethod
    def YAML_FILEPATH(self) -> str:
        "Source of the specific test class custom math"

    @pytest.fixture(scope="class")
    def custom_math(self):
        return AttrDict.from_yaml(self.CUSTOM_MATH_DIR / self.YAML_FILEPATH)

    @pytest.mark.order(-1)
    def test_all_math_registered(self, custom_math):
        "After running all the previous tests in the class, the register should be full, i.e. all math has been tested"
        for key in self.TEST_REGISTER:
            custom_math.del_key(key)
        assert not custom_math


class TestAnnualEnergyBalance(CustomMathExamples):
    YAML_FILEPATH = "annual_energy_balance.yaml"

    @pytest.fixture(scope="class")
    def abs_filepath(self):
        return (self.CUSTOM_MATH_DIR / self.YAML_FILEPATH).absolute()

    def test_annual_energy_balance_per_tech_and_node(self, compare_lps, abs_filepath):
        filename = "annual_energy_balance_per_tech_and_node"
        self.TEST_REGISTER.add(f"constraints.{filename}")
        model = build_test_model(
            {
                "nodes.a.techs.test_supply_elec.constraints.annual_flow_max": 10,
                "nodes.b.techs.test_supply_elec.constraints.annual_flow_max": 20,
                "config.init.custom_math": [abs_filepath],
            },
            "simple_supply,two_hours",
        )
        custom_math = {"constraints": [filename]}
        compare_lps(model, custom_math, filename)

    def test_annual_energy_balance_global_per_tech(self, compare_lps, abs_filepath):
        filename = "annual_energy_balance_global_per_tech"
        self.TEST_REGISTER.add(f"constraints.{filename}")
        model = build_test_model(
            {
                "parameters": {
                    "annual_flow_max": {
                        "data": {"test_supply_elec": 10},
                        "dims": ["techs"],
                    },
                },
                "config.init.custom_math": [abs_filepath],
            },
            "simple_supply,two_hours",
        )
        custom_math = {"constraints": [filename]}
        compare_lps(model, custom_math, filename)

    def test_annual_energy_balance_global_multi_tech(self, compare_lps, abs_filepath):
        filename = "annual_energy_balance_global_multi_tech"
        self.TEST_REGISTER.add(f"constraints.{filename}")
        model = build_test_model(
            {
                "parameters": {
                    "annual_flow_max": {"data": 10},
                    "flow_max_group": {
                        "data": {"test_supply_elec": True, "test_supply_plus": True},
                        "dims": "techs",
                    },
                },
                "config.init.custom_math": [abs_filepath],
            },
            "simple_supply_and_supply_plus,two_hours",
        )
        custom_math = {"constraints": [filename]}
        compare_lps(model, custom_math, filename)

    def test_annual_energy_balance_total_source_availability(
        self, compare_lps, abs_filepath
    ):
        filename = "annual_energy_balance_total_source_availability"
        self.TEST_REGISTER.add(f"constraints.{filename}")
        model = build_test_model(
            {
                "parameters": {
                    "annual_source_max": {
                        "data": {"test_supply_plus": 10},
                        "dims": ["techs"],
                    },
                },
                "config.init.custom_math": [abs_filepath],
            },
            "simple_supply_and_supply_plus,two_hours",
        )
        custom_math = {"constraints": [filename]}
        compare_lps(model, custom_math, filename)

    def test_annual_energy_balance_total_sink_availability(
        self, compare_lps, abs_filepath
    ):
        filename = "annual_energy_balance_total_sink_availability"
        self.TEST_REGISTER.add(f"constraints.{filename}")
        model = build_test_model(
            {
                "parameters": {
                    "annual_sink_max": {
                        "data": {"test_demand_elec": 10},
                        "dims": ["techs"],
                    },
                },
                "config.init.custom_math": [abs_filepath],
            },
            "simple_supply,two_hours,demand_elec_max",
        )
        custom_math = {"constraints": [filename]}
        compare_lps(model, custom_math, filename)


class TestMaxTimeVarying(CustomMathExamples):
    YAML_FILEPATH = "max_time_varying.yaml"

    @pytest.fixture(scope="class")
    def abs_filepath(self):
        return (self.CUSTOM_MATH_DIR / self.YAML_FILEPATH).absolute()

    def test_max_time_varying_flow_cap(self, compare_lps, abs_filepath):
        filename = "max_time_varying_flow_cap"
        self.TEST_REGISTER.add(f"constraints.{filename}")
        model = build_test_model(
            {
                "parameters": {
                    "flow_cap_max_relative_per_ts": {
                        "data": {
                            ("test_supply_elec", "2005-01-01 00:00"): 0.8,
                            ("test_supply_elec", "2005-01-01 01:00"): 0.5,
                        },
                        "dims": ["techs", "timesteps"],
                    },
                },
                "config.init.custom_math": [abs_filepath],
            },
            "simple_supply,two_hours",
        )
        custom_math = {"constraints": [filename]}
        compare_lps(model, custom_math, filename)

    def test_max_time_varying_storage(self, compare_lps, abs_filepath):
        filename = "max_time_varying_storage"
        self.TEST_REGISTER.add(f"constraints.{filename}")
        model = build_test_model(
            {
                "parameters": {
                    "flow_cap_max_relative_per_ts": {
                        "data": {
                            ("test_storage", "2005-01-01 00:00"): 0.8,
                            ("test_storage", "2005-01-01 01:00"): 0.5,
                        },
                        "dims": ["techs", "timesteps"],
                    },
                },
                "config.init.custom_math": [abs_filepath],
            },
            "simple_supply,two_hours",
        )
        custom_math = {"constraints": [filename]}
        compare_lps(model, custom_math, filename)
