import importlib
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pytest
from pyomo.repn.tests import lp_diff

from calliope import AttrDict, Model, io
from calliope.preprocess import model_math

from .common import util

CALLIOPE_DIR: Path = importlib.resources.files("calliope")  # type: ignore

MATH_TO_KEEP = [
    "variables",
    "global_expressions",
    "dimensions",
    "parameters",
    "lookups",
]


# FIXME: ideally, this should be taken from Model.math after init.
# This should be fixed once math construction is moved to the init stage.
def create_full_math(internal_math: list[str], custom_math: dict | None = None):
    """Simulates the creation of the full math object."""
    math_dataset = {
        name: io.read_rich_yaml((CALLIOPE_DIR / "math" / name).with_suffix(".yaml"))
        for name in internal_math
    }
    if custom_math:
        math_dataset["custom_math"] = AttrDict(custom_math)
    build_math = model_math.build_math(list(math_dataset.keys()), math_dataset)

    return AttrDict(build_math.model_dump())


def create_pruned_math_file(math: dict, filepath: str | Path):
    """Create a barebones calliope math file.

    Setup:
    - No constraints.
    - All variables and expressions in `math`.
    - A standard dummy objective.
    """
    pruned_math = AttrDict({k: math[k] for k in MATH_TO_KEEP})
    pruned_math.set_key(
        "objectives.dummy_obj",
        {"equations": [{"expression": "1 + 1"}], "sense": "minimize"},
    )
    io.to_yaml(pruned_math, filepath)


def build_lp_file(model: Model, outfile: str | Path, objective: str = "dummy_obj"):
    model.build(objective=objective)
    model.backend.verbose_strings()
    model.backend.to_lp(outfile)

    # strip trailing whitespace from `outfile` after the fact,
    # so it can be reliably compared other files in future
    with Path(outfile).open("r") as f:
        stripped_lines = []
        while line := f.readline():
            stripped_lines.append(line.rstrip())
    # reintroduce the trailing newline.
    Path(outfile).write_text("\n".join(stripped_lines) + "\n")


def _diff_files(file1, file2):
    file1_lines = file1.read_text().split("\n")
    file2_lines = file2.read_text().split("\n")
    return set(file1_lines).symmetric_difference(file2_lines)


def compare_lps_new(generated_lp: Path):
    expected_file = Path(__file__).parent / "common" / "lp_files" / generated_lp.name
    diff_ordered = lp_diff.load_and_compare_lp_baseline(
        generated_lp.as_posix(), expected_file.as_posix()
    )
    # Our unordered comparison ignores component ordering but cannot handle
    # trivial differences in numerics (as everything is a string to it)
    diff_unordered = _diff_files(generated_lp, expected_file)
    # If one of the above matches across the board, we're good to go.
    assert diff_ordered == ([], []) or not diff_unordered


class InternalMathFiles(ABC):
    FILENAME: str
    TEST_REGISTER: set
    EXTRA_MATH: list

    @pytest.fixture(scope="class")
    def lp_temp_path(self, tmp_path_factory):
        """Use a single temp. location to make manual checks easier."""
        return tmp_path_factory.mktemp(f"lp_files_{self.FILENAME}")

    @pytest.fixture(scope="class")
    def full_math(self):
        return create_full_math(["base"] + self.EXTRA_MATH)

    @pytest.fixture(scope="class")
    def barebones_math_file(self, full_math, tmp_path_factory) -> str:
        filepath = tmp_path_factory.mktemp("barebones") / "barebones.yaml"
        create_pruned_math_file(full_math, filepath)
        return str(filepath)

    @pytest.fixture(scope="class")
    def barebones_config(self, barebones_math_file):
        return {
            "math_paths": {"base": barebones_math_file},
            "pre_validate_math_strings": False,
        }

    @pytest.mark.order(-1)
    @pytest.mark.xfail(reason="math tests are not fully defined yet")
    def test_all_math_registered(self):
        """If all the math has been evaluated, `file_math` should be empty."""
        file_math = model_math._load_internal_math(self.FILENAME)
        for key in self.TEST_REGISTER:
            file_math.del_key(key)
        assert not file_math


class TestBaseMath(InternalMathFiles):
    FILENAME = "base"
    TEST_REGISTER = set()
    EXTRA_MATH = []

    def test_storage_max(self, full_math, lp_temp_path, barebones_config):
        self.TEST_REGISTER.add("constraints.storage_max")
        custom_math = {
            "constraints": {"storage_max": full_math.constraints["storage_max"]}
        }
        model = util.build_test_model(
            scenario="simple_storage,two_hours,investment_costs",
            math_dict=custom_math,
            **barebones_config,
        )
        lp_file = lp_temp_path / "storage_max.lp"
        build_lp_file(model, lp_file)
        compare_lps_new(lp_file)

    def test_flow_out_max(self, full_math, lp_temp_path, barebones_config):
        self.TEST_REGISTER.add("constraints.flow_out_max")
        custom_math = {
            "constraints": {"flow_out_max": full_math.constraints["flow_out_max"]}
        }
        model = util.build_test_model(
            {},
            "simple_supply,two_hours,investment_costs",
            math_dict=custom_math,
            **barebones_config,
        )

        lp_file = lp_temp_path / "flow_out_max.lp"
        build_lp_file(model, lp_file)
        compare_lps_new(lp_file)

    def test_balance_conversion(self, full_math, lp_temp_path, barebones_config):
        self.TEST_REGISTER.add("constraints.balance_conversion")
        custom_math = {
            "constraints": {
                "balance_conversion": full_math.constraints["balance_conversion"]
            }
        }
        model = util.build_test_model(
            scenario="simple_conversion,two_hours,investment_costs",
            math_dict=custom_math,
            **barebones_config,
        )
        lp_file = lp_temp_path / "balance_conversion.lp"
        build_lp_file(model, lp_file)
        compare_lps_new(lp_file)

    def test_source_max(self, full_math, lp_temp_path, barebones_config):
        self.TEST_REGISTER.add("constraints.source_max")
        custom_math = {
            "constraints": {"my_constraint": full_math.constraints["source_max"]}
        }
        model = util.build_test_model(
            {},
            "simple_supply_plus,resample_two_days,investment_costs",
            math_dict=custom_math,
            **barebones_config,
        )
        lp_file = lp_temp_path / "source_max.lp"
        build_lp_file(model, lp_file)
        compare_lps_new(lp_file)

    def test_balance_transmission(self, full_math, lp_temp_path, barebones_config):
        """Test with the electricity transmission tech being one way only, while the heat transmission tech is the default two-way."""
        self.TEST_REGISTER.add("constraints.balance_transmission")
        custom_math = {
            "constraints": {
                "my_constraint": full_math.constraints["balance_transmission"]
            }
        }
        model = util.build_test_model(
            {"techs.test_link_a_b_elec.one_way": True},
            "simple_conversion,two_hours",
            math_dict=custom_math,
            **barebones_config,
        )
        lp_file = lp_temp_path / "balance_transmission.lp"
        build_lp_file(model, lp_file)
        compare_lps_new(lp_file)

    def test_balance_storage(self, full_math, lp_temp_path, barebones_config):
        """Test balance storage with one tech having and one tech not having per-tech cyclic storage."""
        self.TEST_REGISTER.add("constraints.balance_storage")
        custom_math = {
            "constraints": {"my_constraint": full_math.constraints["balance_storage"]}
        }
        model = util.build_test_model(
            {
                "nodes.a.techs.test_storage.cyclic_storage": True,
                "nodes.b.techs.test_storage.cyclic_storage": False,
            },
            "simple_storage,two_hours",
            math_dict=custom_math,
            **barebones_config,
        )
        lp_file = lp_temp_path / "balance_storage.lp"
        build_lp_file(model, lp_file)
        compare_lps_new(lp_file)

    @pytest.mark.parametrize("with_export", [True, False])
    def test_cost_operation_variable(self, lp_temp_path, barebones_config, with_export):
        """Test variable costs in the objective."""
        self.TEST_REGISTER.add("global_expressions.cost_operation_variable")
        override = {
            "techs.test_conversion_plus.cost_flow_out": {
                "data": [1, 2],
                "index": [["electricity", "monetary"], ["heat", "monetary"]],
                "dims": ["carriers", "costs"],
            },
            "techs.test_conversion_plus.cost_flow_in": {
                "data": 4,
                "index": "monetary",
                "dims": "costs",
            },
        }
        if with_export:
            override.update(
                {
                    "techs.test_conversion_plus": {
                        "carrier_export": "heat",
                        "cost_export": {
                            "data": 3,
                            "index": [["heat", "monetary"]],
                            "dims": ["carriers", "costs"],
                        },
                    },
                    "techs.test_supply_elec": {
                        "carrier_export": "electricity",
                        "cost_export": {
                            "data": 5,
                            "index": "monetary",
                            "dims": "costs",
                        },
                    },
                }
            )
        custom_math = {
            # need the expression defined in a constraint/objective for it to appear in the LP file bounds
            "objectives": {
                "foo": {
                    "equations": [
                        {
                            "expression": "sum(cost_operation_variable, over=[nodes, techs, costs, timesteps])"
                        }
                    ],
                    "sense": "minimise",
                }
            }
        }
        model = util.build_test_model(
            override,
            "conversion_and_conversion_plus,var_costs,two_hours",
            math_dict=custom_math,
            **barebones_config,
        )
        suffix = "_with_export" if with_export else ""
        lp_file = lp_temp_path / f"cost_operation_variable{suffix}.lp"
        build_lp_file(model, lp_file, objective="foo")
        compare_lps_new(lp_file)


class TestMILPMath(InternalMathFiles):
    FILENAME = "milp"
    TEST_REGISTER = set()
    EXTRA_MATH = [FILENAME]

    @pytest.mark.parametrize(
        ("variable", "constraint", "overrides"),
        [
            ("flow_cap", "flow_capacity_minimum", {}),
            (
                "storage_cap",
                "storage_capacity_minimum",
                {"techs.test_supply_elec.include_storage": True},
            ),
            ("area_use", "area_use_minimum", {}),
            ("source_cap", "source_capacity_minimum", {}),
        ],
    )
    def test_capacity_variables_and_bounds(
        self, full_math, lp_temp_path, barebones_config, variable, constraint, overrides
    ):
        """Check that variables are initiated with the appropriate bounds,
        and that the lower bound is updated from zero via a separate constraint if required.
        """
        constraint_full = f"constraints.{constraint}"
        self.TEST_REGISTER.add(f"variables.{variable}")
        self.TEST_REGISTER.add(constraint_full)
        # Custom objective ensures that all variables appear in the LP file.
        # Variables not found in either an objective or constraint will never appear in the LP.
        sum_in_objective = "[nodes]" if variable != "flow_cap" else "[nodes, carriers]"
        custom_objective = {
            "objectives.foo": {
                "equations": [
                    {
                        "expression": f"sum({variable}[techs=test_supply_elec], over={sum_in_objective})"
                    }
                ],
                "sense": "minimise",
            }
        }
        custom_math = AttrDict(
            {constraint_full: full_math.constraints[constraint], **custom_objective}
        )
        model = util.build_test_model(
            {
                f"nodes.b.techs.test_supply_elec.{variable}_max": 100,
                f"nodes.a.techs.test_supply_elec.{variable}_min": 1,
                f"nodes.a.techs.test_supply_elec.{variable}_max": np.nan,
                **overrides,
            },
            "simple_supply,two_hours,investment_costs",
            math_dict=custom_math,
            **barebones_config,
        )

        lp_file = lp_temp_path / (variable + ".lp")
        build_lp_file(model, lp_file, objective="foo")
        compare_lps_new(lp_file)


class CustomMathExamples(ABC):
    TEST_REGISTER: set = set()
    EXTRA_MATH: list = []

    #: source of all example math files
    CUSTOM_MATH_DIR = (
        CALLIOPE_DIR.parent.parent / "docs" / "user_defined_math" / "examples"
    )

    @property
    @abstractmethod
    def YAML_FILEPATH(self) -> str:
        """Source of the specific test class custom math"""

    @pytest.fixture(scope="class")
    def abs_filepath(self):
        return (self.CUSTOM_MATH_DIR / self.YAML_FILEPATH).absolute().as_posix()

    @pytest.fixture(scope="class")
    def custom_math(self):
        return io.read_rich_yaml(self.CUSTOM_MATH_DIR / self.YAML_FILEPATH)

    @pytest.fixture(scope="class")
    def full_math(self, custom_math):
        return create_full_math(["base"] + self.EXTRA_MATH, custom_math)

    @pytest.fixture(scope="class")
    def barebones_math_file(self, tmp_path_factory, full_math) -> str:
        filepath = tmp_path_factory.mktemp("barebones") / "barebones.yaml"
        create_pruned_math_file(full_math, filepath)
        return str(filepath)

    @pytest.fixture(scope="class")
    def lp_temp_path(self, tmp_path_factory):
        """Use a single temp. location to make manual checks easier."""
        return tmp_path_factory.mktemp(
            f"lp_files_{self.YAML_FILEPATH.removesuffix('.yaml')}"
        )

    @pytest.fixture(scope="class")
    def all_custom_components(self, custom_math) -> dict[str, list[str]]:
        """Per-component lists in a custom math file."""
        custom_math_lists = {
            component: list(custom_math[component]) for component in custom_math
        }
        return custom_math_lists

    @pytest.fixture
    def build_and_compare(self, full_math, barebones_math_file, lp_temp_path):
        def _build_and_compare(
            filename: str,
            scenario: str,
            overrides: dict | None = None,
            components: dict[list[str]] | None = None,
        ):
            custom_math = {}
            if components is not None:
                for component_group, component_list in components.items():
                    for component in component_list:
                        self.TEST_REGISTER.add(f"{component_group}.{component}")

                custom_math = {
                    k: {name: full_math[k][name] for name in v}
                    for k, v in components.items()
                    if k not in MATH_TO_KEEP
                }
            else:
                self.TEST_REGISTER.add(f"constraints.{filename}")
                custom_math = {
                    "constraints": {filename: full_math["constraints"][filename]}
                }

            if overrides is None:
                overrides = {}

            objective = "dummy_obj"
            if "objectives" in custom_math:
                objective = list(custom_math["objectives"].keys())[0]

            model = util.build_test_model(
                {
                    "config.init": {
                        "math_paths.base": barebones_math_file,
                        # reset extra math in case scenarios refer to it.
                        # We handle extra math in these tests with self.EXTRA_MATH
                        "extra_math": [],
                    },
                    **overrides,
                },
                scenario,
                math_dict=custom_math,
            )
            lp_file = lp_temp_path / f"{filename}.lp"
            build_lp_file(model, lp_file, objective)
            compare_lps_new(lp_file)

        return _build_and_compare

    @pytest.mark.order(-1)
    def test_all_math_registered(self, custom_math):
        """After running all the previous tests in the class, the register should be full, i.e. all math has been tested"""
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
            "nodes.a.techs.test_supply_elec.annual_flow_max": 10,
            "nodes.b.techs.test_supply_elec.annual_flow_max": 20,
        }
        build_and_compare(
            "annual_energy_balance_per_tech_and_node",
            "simple_supply,two_hours",
            overrides,
            components={
                "constraints": ["annual_energy_balance_per_tech_and_node"],
                "parameters": ["annual_flow_max"],
            },
        )

    def test_annual_energy_balance_global_per_tech(self, build_and_compare):
        overrides = {
            "data_definitions": {
                "annual_flow_max": {
                    "data": 10,
                    "index": ["test_supply_elec"],
                    "dims": "techs",
                }
            }
        }
        build_and_compare(
            "annual_energy_balance_global_per_tech",
            "simple_supply,two_hours",
            overrides,
            components={
                "constraints": ["annual_energy_balance_global_per_tech"],
                "parameters": ["annual_flow_max"],
            },
        )

    def test_annual_energy_balance_global_multi_tech(self, build_and_compare):
        overrides = {
            "data_definitions": {
                "annual_flow_max": 10,
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
            components={
                "constraints": ["annual_energy_balance_global_multi_tech"],
                "parameters": ["annual_flow_max"],
                "lookups": ["flow_max_group"],
            },
        )

    def test_annual_energy_balance_total_source_availability(self, build_and_compare):
        overrides = {"techs": {"test_supply_plus": {"annual_source_max": 10}}}
        build_and_compare(
            "annual_energy_balance_total_source_availability",
            "simple_supply_and_supply_plus,two_hours",
            overrides,
            components={
                "constraints": ["annual_energy_balance_total_source_availability"],
                "parameters": ["annual_source_max"],
            },
        )

    def test_annual_energy_balance_total_sink_availability(self, build_and_compare):
        overrides = {"techs": {"test_demand_elec": {"annual_sink_max": 10}}}
        build_and_compare(
            "annual_energy_balance_total_sink_availability",
            "simple_supply,two_hours,demand_elec_max",
            overrides,
            components={
                "constraints": ["annual_energy_balance_total_sink_availability"],
                "parameters": ["annual_sink_max"],
            },
        )


class TestMaxTimeVarying(CustomMathExamples):
    YAML_FILEPATH = "max_time_varying.yaml"

    def test_max_time_varying_flow_cap(self, build_and_compare, all_custom_components):
        overrides = {
            "data_definitions": {
                "flow_cap_max_relative_per_ts": {
                    "data": [0.8, 0.5],
                    "index": [
                        ["test_supply_elec", "2005-01-01 00:00"],
                        ["test_supply_elec", "2005-01-01 01:00"],
                    ],
                    "dims": ["techs", "timesteps"],
                }
            }
        }
        build_and_compare(
            "max_time_varying_flow_cap",
            "simple_supply,two_hours",
            overrides,
            components=all_custom_components,
        )


class TestCHPHTP(CustomMathExamples):
    YAML_FILEPATH = "chp_htp.yaml"

    def test_chp_extraction(self, build_and_compare):
        overrides = {
            "techs.test_chp.power_loss_factor": 0.1,
            "techs.test_chp.power_to_heat_ratio": 2,
            "techs.test_chp.turbine_type": "extraction",
        }
        build_and_compare(
            "chp_extraction",
            "simple_chp,two_hours",
            overrides,
            components={
                "constraints": [
                    "chp_extraction_line",
                    "chp_backpressure_line_min",
                    "balance_conversion",
                ],
                "lookups": ["turbine_type"],
                "parameters": [
                    "power_loss_factor",
                    "power_to_heat_ratio",
                    "boiler_eff",
                ],
            },
        )

    def test_chp_backpressure_and_boiler(self, build_and_compare):
        overrides = {
            "techs.test_chp.power_to_heat_ratio": 1.5,
            "techs.test_chp.boiler_eff": 0.8,
            "techs.test_chp.turbine_type": "backpressure",
        }
        build_and_compare(
            "chp_backpressure_and_boiler",
            "simple_chp,two_hours",
            overrides,
            components={
                "constraints": [
                    "chp_divert_fuel_to_boiler",
                    "chp_backpressure_line_max",
                    "balance_conversion",
                ],
                "lookups": ["turbine_type"],
                "parameters": ["power_to_heat_ratio", "boiler_eff"],
            },
        )

    def test_chp_backpressure_no_boiler(self, build_and_compare):
        overrides = {
            "techs.test_chp.power_to_heat_ratio": 1.25,
            "techs.test_chp.turbine_type": "backpressure",
        }
        build_and_compare(
            "chp_backpressure_line_equals",
            "simple_chp,two_hours",
            overrides,
            components={
                "constraints": ["chp_backpressure_line_equals", "balance_conversion"],
                "lookups": ["turbine_type"],
                "parameters": ["power_to_heat_ratio", "boiler_eff"],
            },
        )


class TestShareAllTimesteps(CustomMathExamples):
    YAML_FILEPATH = "share_all_timesteps.yaml"

    def test_demand_share_equals_per_tech(self, build_and_compare):
        overrides = {
            "nodes.a.techs.test_supply_elec.demand_share_equals": 0.5,
            "nodes.b.techs.test_supply_elec.demand_share_equals": 0.8,
            "data_definitions": {"demand_share_tech": "test_demand_elec"},
        }
        build_and_compare(
            "demand_share_equals_per_tech",
            "simple_supply,two_hours",
            overrides,
            components={
                "constraints": ["demand_share_equals_per_tech"],
                "parameters": ["demand_share_equals"],
                "lookups": ["demand_share_tech"],
            },
        )

    def test_supply_share_equals_per_tech(self, build_and_compare):
        overrides = {
            "nodes.a.techs.test_supply_elec.supply_share_equals": 0.5,
            "nodes.b.techs.test_supply_elec.supply_share_equals": 0.8,
            "data_definitions": {"supply_share_carrier": "electricity"},
        }
        build_and_compare(
            "supply_share_equals_per_tech",
            "simple_supply_and_supply_plus,two_hours",
            overrides,
            components={
                "constraints": ["supply_share_equals_per_tech"],
                "parameters": ["supply_share_equals"],
                "lookups": ["supply_share_carrier"],
            },
        )


class TestSharePerTimestep(CustomMathExamples):
    YAML_FILEPATH = "share_per_timestep.yaml"

    def test_demand_share_per_timestep_equals_per_tech(self, build_and_compare):
        overrides = {
            "nodes.a.techs.test_supply_elec.demand_share_per_timestep_equals": 0.5,
            "nodes.b.techs.test_supply_elec.demand_share_per_timestep_equals": 0.8,
            "data_definitions": {"demand_share_tech": "test_demand_elec"},
        }
        build_and_compare(
            "demand_share_per_timestep_equals_per_tech",
            "simple_supply,two_hours",
            overrides,
            components={
                "constraints": ["demand_share_per_timestep_equals_per_tech"],
                "lookups": ["demand_share_tech"],
                "parameters": ["demand_share_per_timestep_equals"],
            },
        )

    @pytest.mark.filterwarnings(
        "ignore:(?s).*defines unrecognised constraint `supply_share_per_timestep_equals`:calliope.exceptions.ModelWarning"
    )
    def test_supply_share_per_timestep_equals_per_tech(self, build_and_compare):
        overrides = {
            "nodes.a.techs.test_supply_elec.supply_share_per_timestep_equals": 0.5,
            "nodes.b.techs.test_supply_elec.supply_share_per_timestep_equals": 0.8,
            "data_definitions": {"supply_share_carrier": "electricity"},
        }
        build_and_compare(
            "supply_share_per_timestep_equals_per_tech",
            "simple_supply_and_supply_plus,two_hours",
            overrides,
            components={
                "constraints": ["supply_share_per_timestep_equals_per_tech"],
                "lookups": ["supply_share_carrier"],
                "parameters": ["supply_share_per_timestep_equals"],
            },
        )


class TestDemandSharePerTimestepDecision(CustomMathExamples):
    YAML_FILEPATH = "demand_share_per_timestep_decision.yaml"

    def test_demand_share_per_timestep_decision_main(self, build_and_compare):
        overrides = {
            "techs": {
                "test_supply_elec": {"decide_demand_share": "test_demand_elec"},
                "test_conversion_plus": {"decide_demand_share": "test_demand_elec"},
            },
            "data_definitions": {
                "demand_share_carrier": "electricity",
                "demand_share_relaxation": 0.01,
            },
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
                "lookups": ["decide_demand_share", "demand_share_carrier"],
                "parameters": ["demand_share_relaxation"],
                "variables": ["demand_share_per_timestep_decision"],
            },
        )

    def test_demand_share_per_timestep_decision_sum(self, build_and_compare):
        overrides = {
            "techs": {
                "test_supply_elec": {"decide_demand_share": "test_demand_elec"},
                "test_conversion_plus": {"decide_demand_share": "test_demand_elec"},
            },
            "data_definitions": {
                "demand_share_carrier": "electricity",
                "demand_share_limit": 0.5,
            },
        }
        build_and_compare(
            "demand_share_per_timestep_decision_sum",
            "conversion_and_conversion_plus,two_hours",
            overrides,
            components={
                "constraints": ["demand_share_per_timestep_decision_sum"],
                "variables": ["demand_share_per_timestep_decision"],
                "lookups": ["decide_demand_share"],
                "parameters": ["demand_share_limit"],
                "checks": ["demand_share_is_fraction"],
            },
        )


class TestPiecewiseCosts(CustomMathExamples):
    YAML_FILEPATH = "piecewise_linear_costs.yaml"
    EXTRA_MATH = ["milp"]

    def test_piecewise(self, build_and_compare, all_custom_components):
        overrides = {
            "techs.test_supply_elec.lifetime": 10,
            "techs.test_supply_elec.cost_interest_rate": {
                "data": 0.1,
                "index": "monetary",
                "dims": "costs",
            },
            "data_definitions": {
                "cost_flow_cap_piecewise_slopes": {
                    "data": [5, 7, 14],
                    "index": [0, 1, 2],
                    "dims": "pieces",
                },
                "cost_flow_cap_piecewise_intercept": {
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
            components=all_custom_components,
        )


class TestSOS2PiecewiseCosts(CustomMathExamples):
    YAML_FILEPATH = "sos2_piecewise_linear_costs.yaml"

    def test_piecewise(self, build_and_compare, all_custom_components):
        overrides = {
            "techs.test_supply_elec.lifetime": 10,
            "techs.test_supply_elec.cost_interest_rate": {
                "data": 0.1,
                "index": "monetary",
                "dims": "costs",
            },
            "techs.test_supply_elec": {
                "piecewise_cost_investment_x": {
                    "data": [0, 1, 2, 10],
                    "index": [0, 1, 2, 3],
                    "dims": "breakpoints",
                },
                "piecewise_cost_investment_y": {
                    "data": [0, 5, 8, 20],
                    "index": [0, 1, 2, 3],
                    "dims": "breakpoints",
                },
            },
        }
        build_and_compare(
            "sos2_piecewise_cost_investment",
            "supply_purchase,two_hours",
            overrides,
            components=all_custom_components,
        )


class TestPiecewiseEfficiency(CustomMathExamples):
    YAML_FILEPATH = "piecewise_linear_efficiency.yaml"
    EXTRA_MATH = ["milp"]

    def test_piecewise(self, build_and_compare, all_custom_components):
        overrides = {
            "data_definitions": {
                "flow_eff_piecewise_slopes": {
                    "data": [5, 7, 14],
                    "index": [0, 1, 2],
                    "dims": "pieces",
                },
                "flow_eff_piecewise_intercept": {
                    "data": [0, -2, -16],
                    "index": [0, 1, 2],
                    "dims": "pieces",
                },
            }
        }
        build_and_compare(
            "piecewise_efficiency",
            "conversion_milp,two_hours",
            overrides,
            components=all_custom_components,
        )


@pytest.mark.filterwarnings(
    "ignore:(?s).*`test_conversion_plus` gives a carrier ratio for `heat`:calliope.exceptions.ModelWarning"
)
class TestFuelDist(CustomMathExamples):
    YAML_FILEPATH = "fuel_dist.yaml"

    def test_fuel_distribution(self, build_and_compare):
        overrides = {
            "data_definitions": {
                "allow_fuel_distribution": {
                    "data": True,
                    "index": ["coal"],
                    "dims": "carriers",
                }
            }
        }
        build_and_compare(
            "fuel_dist_base",
            "fuel_distribution,two_hours",
            overrides,
            components={
                "constraints": ["system_balance", "restrict_total_imports_and_exports"],
                "variables": ["fuel_distributor"],
                "lookups": ["allow_fuel_distribution"],
            },
        )

    def test_fuel_distribution_nodal_limits(self, build_and_compare):
        overrides = {
            "data_definitions": {
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
            }
        }
        build_and_compare(
            "fuel_dist_nodal",
            "fuel_distribution,two_hours",
            overrides,
            components={
                "constraints": ["restrict_nodal_imports", "restrict_nodal_exports"],
                "parameters": [
                    "fuel_distributor",
                    "fuel_import_max",
                    "fuel_export_max",
                ],
            },
        )

    def test_fuel_distribution_costs(self, build_and_compare):
        overrides = {
            "data_definitions": {
                "allow_fuel_distribution": {
                    "data": True,
                    "index": ["coal"],
                    "dims": "carriers",
                },
                "cost_fuel_distribution": {
                    "data": 5,
                    "index": [["coal", "monetary"]],
                    "dims": ["carriers", "costs"],
                },
            }
        }
        build_and_compare(
            "fuel_dist_cost",
            "fuel_distribution,two_hours",
            overrides,
            components={
                "objectives": ["min_cost_optimisation"],
                "global_expressions": ["cost_var_fuel_distribution"],
                "parameters": ["cost_fuel_distribution"],
            },
        )


class TestUptimeDowntime(CustomMathExamples):
    YAML_FILEPATH = "uptime_downtime_limits.yaml"
    EXTRA_MATH = ["milp"]

    def test_annual_capacity_factor(self, build_and_compare):
        overrides = {
            "techs.test_supply_elec.capacity_factor_min": 0.8,
            "techs.test_supply_elec.capacity_factor_max": 0.9,
        }
        build_and_compare(
            "annual_capacity_factor",
            "simple_supply,two_hours",
            overrides,
            components={
                "constraints": [
                    "annual_capacity_factor_min",
                    "annual_capacity_factor_max",
                ],
                "parameters": ["capacity_factor_min", "capacity_factor_max"],
            },
        )

    @pytest.mark.filterwarnings(
        "ignore:(?s).*This parameter will only take effect if you have already defined:calliope.exceptions.ModelWarning"
    )
    def test_downtime(self, build_and_compare):
        overrides = {
            "data_definitions": {
                "downtime_periods": {
                    "data": True,
                    "index": [["test_supply_elec", "a", "2005-01-01 00:00"]],
                    "dims": ["techs", "nodes", "timesteps"],
                }
            }
        }
        build_and_compare(
            "downtime_period",
            "simple_supply,two_hours",
            overrides,
            components={
                "constraints": ["downtime_period"],
                "lookups": ["downtime_periods"],
            },
        )

    def test_downtime_decision(self, build_and_compare):
        overrides = {"techs.test_supply_elec.uptime_limit": 1}
        build_and_compare(
            "downtime_period_decision",
            "supply_milp,two_hours",
            overrides,
            components={
                "constraints": ["downtime_period_decision"],
                "lookups": ["downtime_periods"],
                "parameters": ["uptime_limit"],
            },
        )


class TestNetImportShare(CustomMathExamples):
    YAML_FILEPATH = "net_import_share.yaml"
    shared_overrides = {
        "data_definitions.net_import_share": 1.5,
        "data_tables": {
            "demand_heat": {
                "data": "data_tables/demand_heat.csv",
                "rows": "timesteps",
                "columns": "nodes",
                "select": {"nodes": "a"},
                "drop": "nodes",
                "add_dims": {
                    "parameters": "sink_use_equals",
                    "techs": "test_demand_heat",
                    "nodes": "c",
                },
            }
        },
        "techs": {
            "links_a_c_heat": {
                "link_from": "a",
                "link_to": "c",
                "template": "test_transmission_heat",
            },
            "links_a_c_elec": {
                "link_from": "a",
                "link_to": "c",
                "template": "test_transmission_elec",
            },
        },
    }

    def test_net_import_share_max(self, build_and_compare):
        build_and_compare(
            "net_import_share_max",
            "simple_supply,two_hours",
            self.shared_overrides,
            components={
                "constraints": ["net_import_share_max"],
                "global_expressions": ["flow_out_transmission_techs"],
                "parameters": ["net_import_share"],
            },
        )

    def test_net_annual_import_share_max(self, build_and_compare):
        build_and_compare(
            "net_annual_import_share_max",
            "simple_supply,two_hours",
            self.shared_overrides,
            components={
                "constraints": ["net_annual_import_share_max"],
                "global_expressions": ["flow_out_transmission_techs"],
                "parameters": ["net_import_share"],
            },
        )

    def test_net_annual_import_share_max_node_group(self, build_and_compare):
        build_and_compare(
            "net_annual_import_share_max_node_group",
            "simple_supply,two_hours",
            self.shared_overrides,
            components={
                "constraints": ["net_annual_import_share_max_node_group"],
                "global_expressions": ["flow_out_transmission_techs"],
                "parameters": ["net_import_share"],
            },
        )
