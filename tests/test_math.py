from pathlib import Path

import calliope
import numpy as np
import pytest
from calliope import AttrDict
from pyomo.repn.tests import lp_diff

from .common.util import build_lp, build_test_model


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


@pytest.fixture(scope="class")
def base_math():
    return AttrDict.from_yaml(Path(calliope.__file__).parent / "math" / "base.yaml")


class TestBaseMath:
    TEST_REGISTER: set = set()

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
