import os

import calliope
import calliope.exceptions as exceptions
import numpy as np
import pandas as pd
import pytest
from calliope.attrdict import AttrDict
from pytest import approx

from .common.util import build_test_model as build_model
from .common.util import check_error_or_warning


class TestModelRun:
    def test_model_from_dict(self):
        """
        Test creating a model from dict/AttrDict instead of from YAML
        """
        this_path = os.path.dirname(__file__)
        model_location = os.path.join(this_path, "common", "test_model", "model.yaml")
        model_dict = AttrDict.from_yaml(model_location)
        node_dict = AttrDict(
            {
                "nodes": {
                    "a": {"techs": {"test_supply_elec": {}, "test_demand_elec": {}}},
                    "b": {"techs": {"test_supply_elec": {}, "test_demand_elec": {}}},
                }
            }
        )
        model_dict.union(node_dict)
        model_dict.config.init["time_data_path"] = os.path.join(
            this_path,
            "common",
            "test_model",
            model_dict.config.init["time_data_path"],
        )
        # test as AttrDict
        calliope.Model(model_dict)

        # test as dict
        calliope.Model(model_dict.as_dict())

    @pytest.mark.filterwarnings(
        "ignore:(?s).*Not building the link a,b:calliope.exceptions.ModelWarning"
    )
    def test_valid_scenarios(self):
        """
        Test that valid scenario definition from overrides raises no error and results in applied scenario.
        """
        override = AttrDict.from_yaml_string(
            """
            scenarios:
                scenario_1: ['one', 'two']

            overrides:
                one:
                    techs.test_supply_gas.constraints.flow_cap_max: 20
                two:
                    techs.test_supply_elec.constraints.flow_cap_max: 20

            nodes:
                a:
                    techs:
                        test_supply_gas:
                        test_supply_elec:
                        test_demand_elec:
            """
        )
        model = build_model(override_dict=override, scenario="scenario_1")

        assert (
            model._model_run.nodes["a"].techs.test_supply_gas.constraints.flow_cap_max
            == 20
        )
        assert (
            model._model_run.nodes["a"].techs.test_supply_elec.constraints.flow_cap_max
            == 20
        )

    @pytest.mark.filterwarnings(
        "ignore:(?s).*Not building the link 0,1:calliope.exceptions.ModelWarning"
    )
    def test_valid_scenario_of_scenarios(self):
        """
        Test that valid scenario definition which groups scenarios and overrides raises
        no error and results in applied scenario.
        """
        override = AttrDict.from_yaml_string(
            """
            scenarios:
                scenario_1: ['one', 'two']
                scenario_2: ['scenario_1', 'new_location']

            overrides:
                one:
                    techs.test_supply_gas.constraints.flow_cap_max: 20
                two:
                    techs.test_supply_elec.constraints.flow_cap_max: 20
                new_location:
                    nodes.b.techs:
                        test_supply_elec:

            nodes:
                a:
                    techs:
                        test_supply_gas:
                        test_supply_elec:
                        test_demand_elec:
            """
        )
        model = build_model(override_dict=override, scenario="scenario_2")

        assert (
            model._model_run.nodes["a"].techs.test_supply_gas.constraints.flow_cap_max
            == 20
        )
        assert (
            model._model_run.nodes["b"].techs.test_supply_elec.constraints.flow_cap_max
            == 20
        )

    def test_invalid_scenarios_dict(self):
        """
        Test that invalid scenario definition raises appropriate error
        """
        override = AttrDict.from_yaml_string(
            """
            scenarios:
                scenario_1:
                    techs.foo.bar: 1
            """
        )
        with pytest.raises(exceptions.ModelError) as error:
            build_model(override_dict=override, scenario="scenario_1")

        assert check_error_or_warning(
            error,
            "Scenario definition must be a list of override or other scenario names.",
        )

    def test_invalid_scenarios_str(self):
        """
        Test that invalid scenario definition raises appropriate error
        """
        override = AttrDict.from_yaml_string(
            """
            scenarios:
                scenario_1: 'foo'
            """
        )
        with pytest.raises(exceptions.ModelError) as error:
            build_model(override_dict=override, scenario="scenario_1")

        assert check_error_or_warning(
            error,
            "Scenario definition must be a list of override or other scenario names.",
        )

    def test_scenario_name_overlaps_overrides(self):
        """
        Test that a scenario name which is a list of possibly overrides is not parsed as overrides.
        """
        override = AttrDict.from_yaml_string(
            """
            scenarios:
                'simple_supply,one_day': ['simple_supply', 'one_day']
            """
        )
        with pytest.warns(exceptions.ModelWarning) as warn_info:
            build_model(
                override_dict=override,
                scenario="simple_supply,one_day",
            )

        assert check_error_or_warning(
            warn_info,
            "Scenario name `simple_supply,one_day` includes commas that won't be parsed as a list of overrides",
        )

    def test_undefined_carriers(self):
        """
        Test that user has input either carrier or carrier_in/_out for each tech
        """
        override = AttrDict.from_yaml_string(
            """
            techs:
                test_undefined_carrier:
                    essentials:
                        parent: supply
                        name: test
                    constraints:
                        source_max: .inf
                        flow_cap_max: .inf
            nodes.1.techs.test_undefined_carrier:
            """
        )
        with pytest.raises(exceptions.ModelError):
            build_model(override_dict=override, scenario="simple_supply,one_day")

    def test_conversion_plus_primary_carriers(self):
        """
        Test that user has input input/output primary carriers for conversion_plus techs
        """
        override1 = {
            "techs.test_conversion_plus.essentials.carrier_in": ["gas", "coal"]
        }
        override2 = {"techs.test_conversion_plus.essentials.primary_carrier_in": "coal"}
        override3 = {
            "techs.test_conversion_plus.essentials.primary_carrier_out": "coal"
        }

        model = build_model({}, scenario="simple_conversion_plus,two_hours")
        assert (
            model._model_run.techs.test_conversion_plus.essentials.get_key(
                "primary_carrier_in", None
            )
            == "gas"
        )

        # should fail: multiple carriers in, but no primary_carrier_in assigned
        with pytest.raises(exceptions.ModelError) as error:
            build_model(override1, scenario="simple_conversion_plus,two_hours")
        assert check_error_or_warning(error, "Primary_carrier_in must be assigned")

        # should fail: primary_carrier_in not one of the carriers_in
        with pytest.raises(exceptions.ModelError) as error:
            build_model(override2, scenario="simple_conversion_plus,two_hours")
        assert check_error_or_warning(error, "Primary_carrier_in `coal` not one")

        # should fail: primary_carrier_out not one of the carriers_out
        with pytest.raises(exceptions.ModelError) as error:
            build_model(override3, scenario="simple_conversion_plus,two_hours")
        assert check_error_or_warning(error, "Primary_carrier_out `coal` not one")

    def test_incorrect_subset_time(self):
        """
        If time_subset is a list, it must have two entries (start_time, end_time)
        If time_subset is not a list, it should successfully subset on the given
        string/integer
        """

        def override(param):
            return AttrDict.from_yaml_string(
                "config.init.time_subset: {}".format(param)
            )

        # should fail: one string in list
        with pytest.raises(exceptions.ModelError):
            build_model(override_dict=override(["2005-01"]), scenario="simple_supply")

        # should fail: three strings in list
        with pytest.raises(exceptions.ModelError):
            build_model(
                override_dict=override(["2005-01-01", "2005-01-02", "2005-01-03"]),
                scenario="simple_supply",
            )

        # should pass: two string in list as slice
        model = build_model(
            override_dict=override(["2005-01-01", "2005-01-07"]),
            scenario="simple_supply",
        )
        assert all(
            model.inputs.timesteps.to_index()
            == pd.date_range("2005-01", "2005-01-07 23:00:00", freq="H")
        )

        # should fail: must be a list, not a string
        with pytest.raises(exceptions.ModelError):
            model = build_model(
                override_dict=override("2005-01"), scenario="simple_supply"
            )

        # should fail: time subset out of range of input data
        with pytest.raises(exceptions.ModelError) as error:
            build_model(
                override_dict=override(["2005-03", "2005-04"]), scenario="simple_supply"
            )

        assert check_error_or_warning(
            error,
            "subset time range ['2005-03', '2005-04'] is outside the input data time range [2005-01-01, 2005-02-01]",
        )

        # should fail: time subset out of range of input data
        with pytest.raises(exceptions.ModelError):
            build_model(
                override_dict=override(["2005-02-01", "2005-02-05"]),
                scenario="simple_supply",
            )

    def test_incorrect_date_format(self):
        """
        Test the date parser catches a different date format from file than
        user input/default (inc. if it is just one line of a file that is incorrect)
        """

        # should pass: changing datetime format from default
        override1 = {
            "config.init.time_format": "%d/%m/%Y %H:%M:%S",
            "techs.test_demand_heat.constraints.sink_equals": "file=demand_heat_diff_dateformat.csv",
            "techs.test_demand_elec.constraints.sink_equals": "file=demand_heat_diff_dateformat.csv",
        }
        model = build_model(override_dict=override1, scenario="simple_conversion")
        assert all(
            model.inputs.timesteps.to_index()
            == pd.date_range("2005-01", "2005-02-01 23:00:00", freq="H")
        )

        # should fail: wrong dateformat input for one file
        override2 = {
            "techs.test_demand_heat.constraints.sink_equals": "file=demand_heat_diff_dateformat.csv"
        }

        with pytest.raises(exceptions.ModelError):
            build_model(override_dict=override2, scenario="simple_conversion")

        # should fail: wrong dateformat input for all files
        override3 = {"config.init.time_format": "%d/%m/%Y %H:%M:%S"}

        with pytest.raises(exceptions.ModelError):
            build_model(override_dict=override3, scenario="simple_supply")

        # should fail: one value wrong in file
        override4 = {
            "techs.test_demand_heat.constraints.sink_equals": "file=demand_heat_wrong_dateformat.csv"
        }
        # check in output error that it points to: 07/01/2005 10:00:00
        with pytest.raises(exceptions.ModelError):
            build_model(override_dict=override4, scenario="simple_conversion")

    def test_inconsistent_time_indeces(self):
        """
        Test that, including after any time subsetting, the indeces of all time
        varying input data are consistent with each other
        """
        # should fail: wrong length of demand_heat csv vs demand_elec
        override1 = {
            "techs.test_demand_heat.constraints.sink_equals": "file=demand_heat_wrong_length.csv"
        }
        # check in output error that it points to: 07/01/2005 10:00:00
        with pytest.raises(exceptions.ModelError):
            build_model(override_dict=override1, scenario="simple_conversion")

        # should pass: wrong length of demand_heat csv, but time subsetting removes the difference
        build_model(override_dict=override1, scenario="simple_conversion,one_day")

    def test_single_timestep(self):
        """
        Test that warning is raised on using 1 timestep, that timestep resolution will
        be inferred to be 1 hour
        """
        override1 = {
            "config.init.time_subset": ["2005-01-01 00:00:00", "2005-01-01 00:00:00"]
        }
        # check in output error that it points to: 07/01/2005 10:00:00
        with pytest.warns(exceptions.ModelWarning) as warn_info:
            model = build_model(override_dict=override1, scenario="simple_supply")

        assert check_error_or_warning(
            warn_info,
            "Only one timestep defined. Inferring timestep resolution to be 1 hour",
        )
        assert model.inputs.timestep_resolution == [1]

    def test_empty_key_on_explode(self):
        """
        On exploding nodes (from ``'1--3'`` or ``'1,2,3'`` to
        ``['1', '2', '3']``), raise error on the resulting list being empty
        """
        list1 = calliope.preprocess.nodes.explode_nodes("1--3")
        list2 = calliope.preprocess.nodes.explode_nodes("1,2,3")

        assert list1 == list2 == ["1", "2", "3"]

    def test_key_clash_on_set_loc_key(self):
        """
        Raise error on attempted overwrite of information regarding a recently
        exploded location
        """
        override = {
            "nodes.a.techs.test_supply_elec.constraints.source_max": 10,
            "nodes.a,b.techs.test_supply_elec.constraints.source_max": 15,
        }

        with pytest.raises(KeyError):
            build_model(override_dict=override, scenario="simple_supply,one_day")

    def test_calculate_depreciation(self):
        """
        Technologies which define investment costs *must* define lifetime and
        interest rate, so that a depreciation rate can be calculated.
        If lifetime == inf and interested > 0, depreciation rate will be inf, so
        we want to avoid that too.
        """

        override1 = {"techs.test_supply_elec.costs.monetary.flow_cap": 10}
        with pytest.raises(exceptions.ModelError) as error:
            build_model(override_dict=override1, scenario="simple_supply,one_day")
        assert check_error_or_warning(
            error, "Must specify constraints.lifetime and costs.monetary.interest_rate"
        )

        override2 = {
            "techs.test_supply_elec.constraints.lifetime": 10,
            "techs.test_supply_elec.costs.monetary.flow_cap": 10,
        }
        with pytest.raises(exceptions.ModelError) as error:
            build_model(override_dict=override2, scenario="simple_supply,one_day")
        assert check_error_or_warning(
            error, "Must specify constraints.lifetime and costs.monetary.interest_rate"
        )

        override3 = {
            "techs.test_supply_elec.costs.monetary.interest_rate": 0.1,
            "techs.test_supply_elec.costs.monetary.flow_cap": 10,
        }
        with pytest.raises(exceptions.ModelError) as error:
            build_model(override_dict=override3, scenario="simple_supply,one_day")
        assert check_error_or_warning(
            error, "Must specify constraints.lifetime and costs.monetary.interest_rate"
        )

        override4 = {
            "techs.test_supply_elec.constraints.lifetime": 10,
            "techs.test_supply_elec.costs.monetary.interest_rate": 0,
            "techs.test_supply_elec.costs.monetary.flow_cap": 10,
        }
        with pytest.warns(exceptions.ModelWarning) as excinfo:
            build_model(override_dict=override4, scenario="simple_supply,one_day")
        assert check_error_or_warning(excinfo, "`monetary` interest rate of zero")

        override5 = {
            "techs.test_supply_elec.constraints.lifetime": np.inf,
            "techs.test_supply_elec.costs.monetary.interest_rate": 0,
            "techs.test_supply_elec.costs.monetary.flow_cap": 10,
        }
        with pytest.warns(exceptions.ModelWarning) as excinfo:
            build_model(override_dict=override5, scenario="simple_supply,one_day")
        assert check_error_or_warning(
            excinfo,
            "No investment monetary cost will be incurred for `test_supply_elec`",
        )

        override6 = {
            "techs.test_supply_elec.constraints.lifetime": np.inf,
            "techs.test_supply_elec.costs.monetary.interest_rate": 0.1,
            "techs.test_supply_elec.costs.monetary.flow_cap": 10,
        }
        with pytest.warns(exceptions.ModelWarning) as excinfo:
            build_model(override_dict=override6, scenario="simple_supply,one_day")
        assert check_error_or_warning(
            excinfo,
            "No investment monetary cost will be incurred for `test_supply_elec`",
        )

        override7 = {
            "techs.test_supply_elec.constraints.lifetime": 10,
            "techs.test_supply_elec.costs.monetary.interest_rate": 0.1,
            "techs.test_supply_elec.costs.monetary.flow_cap": 10,
        }
        build_model(override_dict=override7, scenario="simple_supply,one_day")

    def test_delete_interest_rate(self):
        """
        If only 'interest_rate' is given in the cost class for a technology, we
        should be able to handle deleting it without leaving an empty cost key.
        """

        override1 = {"techs.test_supply_elec.costs.monetary.interest_rate": 0.1}
        m = build_model(override_dict=override1, scenario="simple_supply,one_day")
        assert "loc_techs_cost" not in m._model_data.dims

    def test_empty_cost_class(self):
        """
        If cost is defined, but its value is not a dictionary, ensure it is
        deleted
        """
        override1 = {"techs.test_supply_elec.costs.carbon": None}
        with pytest.warns(exceptions.ModelWarning) as warn_info:
            m = build_model(
                override_dict=override1,
                scenario="simple_supply,one_day,investment_costs",
            )

        assert check_error_or_warning(
            warn_info,
            "Deleting empty cost class `carbon` for technology `test_supply_elec` at `a`.",
        )

        assert (
            "carbon" not in m._model_run.nodes["b"].techs.test_supply_elec.costs.keys()
        )
        assert "carbon" not in m._model_data.coords["costs"].values

    def test_strip_link(self):
        override = {
            "links.a, c.techs": {"test_transmission_elec": None},
            "nodes.c.techs": {"test_supply_elec": None},
        }
        m = build_model(override_dict=override, scenario="simple_supply,one_day")
        assert "c" in m._model_run.nodes["a"].links.keys()

    def test_dataframes_passed(self):
        """
        If model config specifies dataframes to be loaded in (via df=...),
        these time series must be passed as arguments in calliope.Model(...).
        """
        override = {"techs.test_demand_elec.constraints.sink_equals": "df=demand_elec"}
        with pytest.raises(exceptions.ModelError) as error:
            build_model(
                model_file="model_minimal.yaml",
                override_dict=override,
                timeseries_dataframes=None,
            )
        assert check_error_or_warning(
            error, "no timeseries passed " "as arguments in calliope.Model(...)."
        )

    def test_dataframe_keys(self):
        """
        Any timeseries specified via df=... must correspond to a key in
        timeseries_dataframes. An error should be thrown.
        """
        override = {"techs.test_demand_elec.constraints.sink_equals": "df=key_1"}
        ts_df = {"key_2": pd.DataFrame(np.arange(10))}

        with pytest.raises(exceptions.ModelError) as error:
            build_model(
                model_file="model_minimal.yaml",
                override_dict=override,
                timeseries_dataframes=ts_df,
            )
        assert check_error_or_warning(
            error, "Model attempted to load dataframe with key"
        )

    def test_invalid_dataframes_passed(self):
        """
        `timeseries_dataframes` should be dict of pandas DataFrames.
        """
        override = {"techs.test_demand_elec.constraints.sink_equals": "df=demand_elec"}

        ts_df_nodict = pd.DataFrame(np.arange(10))  # Not a dict
        ts_df_numpy_arrays = {"demand_elec": np.arange(10)}  # No pd DataFrames

        for timeseries_dataframes in [ts_df_nodict, ts_df_numpy_arrays]:
            with pytest.raises(exceptions.ModelError) as error:
                build_model(
                    model_file="model_minimal.yaml",
                    override_dict=override,
                    timeseries_dataframes=timeseries_dataframes,
                )
            assert check_error_or_warning(
                error, "`timeseries_dataframes` must be dict of pandas DataFrames."
            )


class TestChecks:
    def test_unrecognised_top_level_keys(self):
        """
        Check that the only top level keys can be 'model', 'run', 'nodes',
        'techs', 'tech_groups' (+ '_model_def_path', but that is an internal addition)
        """
        override = {"nonsensical_key": "random_string"}

        with pytest.warns(exceptions.ModelWarning) as excinfo:
            build_model(override_dict=override, scenario="simple_supply")

        assert check_error_or_warning(
            excinfo, "Unrecognised top-level configuration item: nonsensical_key"
        )

    def test_missing_config_key(self):
        """
        Check that missing 'nodes' raises an error
        """
        with pytest.raises(exceptions.ModelError) as excinfo:
            build_model()  # Not selecting any scenario means no nodes are defined

        assert check_error_or_warning(
            excinfo, "Model is missing required top-level configuration item: nodes"
        )

    @pytest.mark.parametrize("top_level_key", ["init", "build", "solve"])
    def test_unrecognised_config_keys(self, top_level_key):
        """
        Check that the only keys allowed in 'model' and 'run' are those in the
        model defaults
        """
        override = {f"config.{top_level_key}.nonsensical_key": "random_string"}

        with pytest.warns(exceptions.ModelWarning) as excinfo:
            build_model(override_dict=override, scenario="simple_supply")

        assert check_error_or_warning(
            excinfo,
            f"Unrecognised setting in `{top_level_key}` configuration: nonsensical_key",
        )

    @pytest.mark.skip(
        reason="SPORES mode will fail until the cost max group constraint can be reproduced"
    )
    def test_warn_null_number_of_spores(self):
        """
        Check that spores number is greater than 0 if spores run mode is selected
        """
        override = {"config.build.spores_number": 0}

        with pytest.warns(exceptions.ModelWarning) as warn:
            build_model(scenario="spores,simple_supply", override_dict=override)

        assert check_error_or_warning(
            warn, "spores run mode is selected, but a number of 0 spores is requested"
        )

    @pytest.mark.skip(
        reason="SPORES mode will fail until the cost max group constraint can be reproduced"
    )
    def test_non_string_score_cost_class(self):
        """
        Check that the score_cost_class for spores scoring is a string
        """
        override = {"config.build.spores_score_cost_class": 0}

        with pytest.raises(exceptions.ModelError) as excinfo:
            build_model(scenario="spores,simple_supply", override_dict=override)

        assert check_error_or_warning(
            excinfo, "`config.build.spores_score_cost_class` must be a string"
        )

    @pytest.mark.parametrize(
        "invalid_key", [("monetary"), ("emissions"), ("name"), ("anything_else_really")]
    )
    def test_unrecognised_tech_keys(self, invalid_key):
        """
        Check that no invalid keys are defined for technologies.
        """
        override1 = {"techs.test_supply_gas.{}".format(invalid_key): "random_string"}

        with pytest.warns(exceptions.ModelWarning):
            build_model(override_dict=override1, scenario="simple_supply")

    def test_model_version_mismatch(self):
        """
        Model config says config.init.calliope_version = 0.1, which is not what we
        are running, so we want a warning.
        """
        override = {"config.init.calliope_version": 0.1}

        with pytest.warns(exceptions.ModelWarning) as excinfo:
            build_model(override_dict=override, scenario="simple_supply,one_day")

        assert check_error_or_warning(
            excinfo, "Model configuration specifies calliope_version"
        )

    def test_unknown_carrier_tier(self):
        """
        User can only use 'carrier_' + ['in', 'out', 'in_2', 'out_2', 'in_3',
        'out_3', 'ratios']
        """

        override1 = AttrDict.from_yaml_string(
            """
            techs.test_supply_elec.essentials.carrier_1: power
            """
        )

        with pytest.raises(exceptions.ModelError):
            build_model(override_dict=override1, scenario="simple_supply,one_day")

        override2 = AttrDict.from_yaml_string(
            """
            techs.test_conversion_plus.essentials.carrier_out_4: power
            """
        )

        with pytest.raises(exceptions.ModelError):
            build_model(
                override_dict=override2, scenario="simple_conversion_plus,one_day"
            )

    def test_name_overlap(self):
        """
        No tech may have the same identifier as a tech group
        """
        override = AttrDict.from_yaml_string(
            """
            techs:
                supply:
                    essentials:
                        name: Supply tech
                        carrier: gas
                        parent: supply
                    constraints:
                        flow_cap_max: 10
                        source_max: .inf
            nodes:
                1.techs.supply:
                0.techs.supply:
            """
        )

        with pytest.raises(exceptions.ModelError):
            build_model(override_dict=override, scenario="one_day")

    @pytest.mark.parametrize(
        "loc_tech",
        (
            ({"nodes": ["1", "foo"]}),
            ({"techs": ["test_supply_elec", "bar"]}),
            ({"nodes": ["1", "foo"], "techs": ["test_supply_elec", "bar"]}),
        ),
    )
    @pytest.mark.xfail(reason="Planning to remove group constraints")
    def test_inexistent_group_constraint_loc_tech(self, loc_tech):
        override = {"group_constraints.mygroup": {"flow_cap_max": 100, **loc_tech}}

        with pytest.warns(exceptions.ModelWarning) as excinfo:
            m = build_model(override_dict=override, scenario="simple_supply")

        assert check_error_or_warning(
            excinfo, "Possible misspelling in group constraints:"
        )

        loc_techs = m._model_data.group_constraint_loc_techs_mygroup.values
        assert "foo:test_supply_elec" not in loc_techs
        assert "1:bar" not in loc_techs
        assert "foo:bar" not in loc_techs

    @pytest.mark.xfail(reason="Planning to remove group constraints")
    def test_inexistent_group_constraint_empty_loc_tech(self):
        override = {"group_constraints.mygroup": {"flow_cap_max": 100, "locs": ["foo"]}}

        with pytest.warns(exceptions.ModelWarning) as excinfo:
            m = build_model(override_dict=override, scenario="simple_supply")

        assert check_error_or_warning(
            excinfo, "Constraint group `mygroup` will be completely ignored"
        )

        assert m._model_run.group_constraints.mygroup.get("active", True) is False

    @pytest.mark.filterwarnings(
        "ignore:(?s).*Not building the link a,b:calliope.exceptions.ModelWarning"
    )
    def test_abstract_base_tech_group_override(self):
        """
        Abstract base technology groups can be overridden
        """

        override = AttrDict.from_yaml_string(
            """
            tech_groups:
                supply:
                    constraints:
                        lifetime: 25
            nodes:
                b.techs.test_supply_elec:
                b.techs.test_demand_elec:
            """
        )

        build_model(override_dict=override, scenario="one_day")

    def test_unspecified_parent(self):
        """
        All technologies and technology groups must specify a parent
        """

        override = AttrDict.from_yaml_string(
            """
            techs.test_supply_no_parent:
                    essentials:
                        name: Supply tech
                        carrier: gas
                    constraints:
                        flow_cap_max: 10
                        source_max: .inf
            nodes.b.techs.test_supply_no_parent:
            """
        )

        with pytest.raises(exceptions.ModelError):
            build_model(override_dict=override, scenario="simple_supply,one_day")

    def test_tech_as_parent(self):
        """
        All technologies and technology groups must specify a parent
        """

        override1 = AttrDict.from_yaml_string(
            """
            techs.test_supply_tech_parent:
                    essentials:
                        name: Supply tech
                        carrier: gas
                        parent: test_supply_elec
                    constraints:
                        flow_cap_max: 10
                        source_max: .inf
            nodes.b.techs.test_supply_tech_parent:
            """
        )

        with pytest.raises(exceptions.ModelError) as error:
            build_model(override_dict=override1, scenario="simple_supply,one_day")
        check_error_or_warning(
            error, "tech `test_supply_tech_parent` has another tech as a parent"
        )

        override2 = AttrDict.from_yaml_string(
            """
            tech_groups.test_supply_group:
                    essentials:
                        carrier: gas
                        parent: test_supply_elec
                    constraints:
                        flow_cap_max: 10
                        source_max: .inf
            techs.test_supply_tech_parent.essentials:
                        name: Supply tech
                        parent: test_supply_group
            nodes.b.techs.test_supply_tech_parent:
            """
        )

        with pytest.raises(exceptions.ModelError) as error:
            build_model(override_dict=override2, scenario="simple_supply,one_day")
        check_error_or_warning(
            error, "tech_group `test_supply_group` has a tech as a parent"
        )

    def test_missing_required_constraints(self):
        """
        A technology within an abstract base technology must define a subset of
        hardcoded constraints in order to function
        """
        # should fail: missing one of ['flow_cap_max', 'flow_cap_equals', 'flow_cap_per_unit']
        override_supply1 = AttrDict.from_yaml_string(
            """
            techs:
                demand_missing_constraint:
                    essentials:
                        parent: demand
                        carrier: electricity
                        name: demand missing constraint
                    switches:
                        sink_unit: power
            nodes.b.techs.demand_missing_constraint:
            """
        )
        with pytest.raises(exceptions.ModelError):
            build_model(
                override_dict=override_supply1, scenario="simple_supply,one_day"
            )

        # should pass: giving one of ['flow_cap_max', 'flow_cap_equals', 'flow_cap_per_unit']
        override_supply2 = AttrDict.from_yaml_string(
            """
            techs:
                supply_missing_constraint:
                    essentials:
                        parent: supply
                        carrier: electricity
                        name: supply missing constraint
                    constraints.flow_cap_max: 10
            nodes.b.techs.supply_missing_constraint:
            """
        )
        build_model(override_dict=override_supply2, scenario="simple_supply,one_day")

    def test_defining_non_allowed_constraints(self):
        """
        A technology within an abstract base technology can only define a subset
        of hardcoded constraints, anything else will not be implemented, so are
        not allowed for that technology. This includes misspellings
        """
        # should fail: storage_cap_max not allowed for supply tech
        override_supply1 = AttrDict.from_yaml_string(
            """
            techs.test_supply_elec.constraints.storage_cap_max: 10
            """
        )
        with pytest.raises(exceptions.ModelError):
            build_model(
                override_dict=override_supply1, scenario="simple_supply,one_day"
            )

    def test_defining_non_allowed_costs(self):
        """
        A technology within an abstract base technology can only define a subset
        of hardcoded costs, anything else will not be implemented, so are
        not allowed for that technology. This includes misspellings
        """
        # should fail: storage_cap_max not allowed for supply tech
        override = AttrDict.from_yaml_string(
            """
            techs.test_supply_elec.costs.monetary.storage_cap: 10
            """
        )
        with pytest.raises(exceptions.ModelError):
            build_model(override_dict=override, scenario="simple_supply,one_day")

        # should fail: flow_out not allowed for demand tech
        override = AttrDict.from_yaml_string(
            """
            techs.test_demand_elec.costs.monetary.flow_out: 10
            """
        )
        with pytest.raises(exceptions.ModelError):
            build_model(override_dict=override, scenario="simple_supply,one_day")

    def test_defining_cost_class_with_name_of_cost(self):
        """
        A cost class with the same name as one of the possible cost types was
        defined, suggesting a user mistake with indentation.
        """
        override = AttrDict.from_yaml_string(
            """
            techs.test_supply_elec.costs.storage_cap: 10
            """
        )
        with pytest.warns(exceptions.ModelWarning) as excinfo:
            build_model(override_dict=override, scenario="simple_supply,one_day")

        assert check_error_or_warning(
            excinfo, "`test_supply_elec` at `b` defines storage_cap as a cost class."
        )

    def test_exporting_unspecified_carrier(self):
        """
        User can only define an export carrier if it is defined in
        ['carrier_out', 'carrier_out_2', 'carrier_out_3']
        """

        def override_supply(param):
            return AttrDict.from_yaml_string(
                "techs.test_supply_elec.constraints.export_carrier: {}".format(param)
            )

        def override_converison_plus(param):
            return AttrDict.from_yaml_string(
                "techs.test_conversion_plus.constraints.export_carrier: {}".format(
                    param
                )
            )

        # should fail: exporting `heat` not allowed for electricity supply tech
        with pytest.raises(exceptions.ModelError):
            build_model(
                override_dict=override_supply("heat"), scenario="simple_supply,one_day"
            )

        # should fail: exporting `random` not allowed for conversion_plus tech
        with pytest.raises(exceptions.ModelError):
            build_model(
                override_dict=override_converison_plus("random"),
                scenario="simple_conversion_plus,one_day",
            )

        # should pass: exporting electricity for supply tech
        build_model(
            override_dict=override_supply("electricity"),
            scenario="simple_supply,one_day",
        )

        # should pass: exporting heat for conversion tech
        build_model(
            override_dict=override_converison_plus("heat"),
            scenario="simple_conversion_plus,one_day",
        )

    def test_tech_directly_in_nodes(self):
        """
        A tech defined directly within a location rather than within techs
        inside that location is probably an oversight.
        """
        override = {"nodes.b.test_supply_elec.costs.storage_cap": 10}

        with pytest.raises(exceptions.ModelError) as excinfo:
            build_model(override_dict=override, scenario="simple_supply,one_day")

        assert check_error_or_warning(
            excinfo, "Node `b` contains unrecognised keys ['test_supply_elec']"
        )

    def test_tech_defined_twice_in_links(self):
        """
        A technology can only be defined once for a link, even if that link is
        defined twice (i.e. `A,B` and `B,A`).
        """

        override = {
            "links.a,b.techs.test_transmission_elec": None,
            "links.b,a.techs.test_transmission_elec": None,
        }
        with pytest.raises(exceptions.ModelError) as excinfo:
            build_model(override_dict=override, scenario="simple_supply,one_day")

        assert check_error_or_warning(
            excinfo,
            "Technology test_transmission_elec defined twice on a link defined "
            "in both directions (e.g. `A,B` and `B,A`)",
        )

        override = {
            "links.a,b.techs": {
                "test_transmission_elec": None,
                "test_transmission_heat": None,
            },
            "links.b,a.techs": {
                "test_transmission_elec": None,
                "test_transmission_heat": None,
            },
        }
        with pytest.raises(exceptions.ModelError) as excinfo:
            build_model(override_dict=override, scenario="simple_supply,one_day")

        assert check_error_or_warning(
            excinfo, ["test_transmission_elec", "test_transmission_heat"]
        )

        # We do allow a link to be defined twice, so long as the same tech isn't in both
        override = {
            "techs.test_transmission_heat_2": {
                "essentials.name": "Transmission heat tech",
                "essentials.carrier": "heat",
                "essentials.parent": "transmission",
            },
            "links.a,b.techs": {"test_transmission_elec": None},
            "links.b,a.techs": {"test_transmission_heat_2": None},
        }
        build_model(override_dict=override, scenario="simple_supply,one_day")

    @pytest.mark.filterwarnings(
        "ignore:(?s).*Updated from coordinate system:calliope.exceptions.ModelWarning"
    )
    def test_incorrect_node_coordinates(self):
        """
        Either all or no nodes must have `coordinates` defined and, if all
        defined, they must be in the same coordinate system (lat/lon or x/y)
        """

        def _override(param0, param1):
            override = {}
            if param0 is not None:
                override.update({"nodes.a.coordinates": param0})
            if param1 is not None:
                override.update({"nodes.b.coordinates": param1})
            return override

        cartesian0 = {"x": 0, "y": 1}
        cartesian1 = {"x": 1, "y": 1}
        geographic0 = {"lat": 0, "lon": 1}
        geographic1 = {"lat": 1, "lon": 1}
        fictional0 = {"a": 0, "b": 1}
        fictional1 = {"a": 1, "b": 1}

        # should fail: cannot have nodes in one place and not in another
        with pytest.raises(exceptions.ModelError) as error:
            build_model(
                override_dict=_override(cartesian0, None),
                scenario="simple_supply,one_day",
            )
        check_error_or_warning(
            error, "Either all or no nodes must have `coordinates` defined"
        )

        # should fail: cannot have cartesian coordinates in one place and geographic in another
        with pytest.raises(exceptions.ModelError) as error:
            build_model(
                override_dict=_override(cartesian0, geographic1),
                scenario="simple_supply,one_day",
            )
        check_error_or_warning(error, "All nodes must use the same coordinate format")

        # should fail: cannot use a non-cartesian or non-geographic coordinate system
        with pytest.raises(exceptions.ModelError) as error:
            build_model(
                override_dict=_override(fictional0, fictional1),
                scenario="simple_supply,one_day",
            )
        check_error_or_warning(error, "Unidentified coordinate system")

        # should fail: coordinates must be given as key:value pairs
        with pytest.raises(exceptions.ModelError) as error:
            build_model(
                override_dict=_override([0, 1], [1, 1]),
                scenario="simple_supply,one_day",
            )
        check_error_or_warning(error, "Coordinates must be given in the format")

        # should pass: cartesian coordinates in both places
        build_model(
            override_dict=_override(cartesian0, cartesian1),
            scenario="simple_supply,one_day",
        )

        # should pass: geographic coordinates in both places
        build_model(
            override_dict=_override(geographic0, geographic1),
            scenario="simple_supply,one_day",
        )

    def test_one_way(self):
        """
        With one_way transmission, we remove one direction of a link from
        loc_tech_carriers_prod and the other from loc_tech_carriers_con.
        """
        override = {
            "links.X1,N1.techs.heat_pipes.switches.one_way": True,
            "links.N1,X2.techs.heat_pipes.switches.one_way": True,
            "links.N1,X3.techs.heat_pipes.switches.one_way": True,
        }
        m = calliope.examples.urban_scale(
            override_dict=override, time_subset=["2005-01-01", "2005-01-01"]
        )
        m.build()
        removed_prod_links = [
            {"nodes": "X1", "techs": "heat_pipes:N1"},
            {"nodes": "N1", "techs": "heat_pipes:X2"},
            {"nodes": "N1", "techs": "heat_pipes:X3"},
        ]
        removed_con_links = [
            {"nodes": "N1", "techs": "heat_pipes:X1"},
            {"nodes": "X2", "techs": "heat_pipes:N1"},
            {"nodes": "X3", "techs": "heat_pipes:N1"},
        ]

        for link in removed_prod_links:
            assert m.backend.variables.flow_out.loc[link].isnull().all()

        for link in removed_con_links:
            assert m.backend.variables.flow_in.loc[link].isnull().all()

    def test_carrier_ratio_for_inexistent_carrier(self):
        """
        A tech should not define a carrier ratio for a carrier it does
        not actually use.
        """
        override = AttrDict.from_yaml_string(
            """
            nodes.1.techs.test_conversion_plus.constraints.carrier_ratios:
                carrier_in:
                    some_carrier: 1.0
                carrier_out_2:
                    another_carrier: 2.0
            """
        )
        with pytest.warns(exceptions.ModelWarning) as excinfo:
            build_model(
                override_dict=override, scenario="simple_conversion_plus,one_day"
            )

        assert check_error_or_warning(
            excinfo,
            "Tech `test_conversion_plus` gives a carrier ratio for `another_carrier`, but does not actually",
        )

    def test_carrier_ratio_for_specified_carrier(self, recwarn):
        """
        The warning for not defining a carrier ratio for a carrier a tech does
        not actually use should not be triggered if the carrier is defined.
        """
        override = AttrDict.from_yaml_string(
            """
            nodes.b.techs.test_conversion_plus.constraints.carrier_ratios:
                carrier_in:
                    heat: 1.0
            """
        )

        build_model(override_dict=override, scenario="simple_conversion_plus,one_day")

        assert "Tech `test_conversion_plus` gives a carrier ratio" not in [
            str(i) for i in recwarn.list
        ]

    def test_carrier_ratio_from_file(self, recwarn):
        """
        It is possible to load a timeseries carrier_ratio from file
        """
        override = AttrDict.from_yaml_string(
            """
            nodes.b.techs.test_conversion_plus.constraints.carrier_ratios:
                carrier_out.heat: file=carrier_ratio.csv
            """
        )
        build_model(override_dict=override, scenario="simple_conversion_plus,one_day")

        assert "Cannot load data from file for configuration" not in [
            str(i) for i in recwarn.list
        ]

    @pytest.mark.filterwarnings("ignore:(?s).*Integer:calliope.exceptions.ModelWarning")
    def test_milp_constraints(self):
        """
        If `units` is defined, but not `flow_cap_per_unit`, throw an error
        """

        # should fail: no flow_cap_per_unit
        override1 = AttrDict.from_yaml_string(
            "techs.test_supply_elec.constraints.units_max: 4"
        )

        with pytest.raises(exceptions.ModelError):
            build_model(override_dict=override1, scenario="simple_supply,one_day")

        # should pass: flow_cap_per_unit given
        override2 = AttrDict.from_yaml_string(
            """
            techs.test_supply_elec.constraints:
                        units_max: 4
                        flow_cap_per_unit: 5
            """
        )

        build_model(override_dict=override2, scenario="simple_supply,one_day")

    def test_source_equals_cannot_be_inf(self):
        override = {
            "techs.test_supply_elec.constraints.source_equals": np.inf,
        }

        with pytest.raises(exceptions.ModelError) as excinfo:
            build_model(override_dict=override, scenario="simple_supply,one_day")

        assert check_error_or_warning(excinfo, "Cannot include infinite values")

    def test_override_coordinates(self):
        """
        Check that warning is raised if we are completely overhauling the
        coordinate system with an override
        """
        override = {
            "nodes": {
                "X1.coordinates": {"lat": 51.4596158, "lon": -0.1613446},
                "X2.coordinates": {"lat": 51.4652373, "lon": -0.1141548},
                "X3.coordinates": {"lat": 51.4287016, "lon": -0.1310635},
                "N1.coordinates": {"lat": 51.4450766, "lon": -0.1247183},
            },
            "links": {
                "X1,X2.techs.power_lines.distance": 10,
                "X1,X3.techs.power_lines.distance": 5,
                "X1,N1.techs.heat_pipes.distance": 3,
                "N1,X2.techs.heat_pipes.distance": 3,
                "N1,X3.techs.heat_pipes.distance": 4,
            },
        }
        with pytest.warns(exceptions.ModelWarning) as excinfo:
            calliope.examples.urban_scale(override_dict=override)

        assert check_error_or_warning(excinfo, "Updated from coordinate system")

    @pytest.mark.xfail(
        reason="Removed this check because it has to happen *after* calling `build`"
    )
    def test_clustering_and_cyclic_storage(self):
        """
        Don't allow time clustering with cyclic storage if not also using
        storage_inter_cluster
        """

        override = {
            "config.init.time_subset": ["2005-01-01", "2005-01-04"],
            "config.init.time_cluster": "cluster_days.csv",
            "config.build.cyclic_storage": True,
        }

        with pytest.raises(exceptions.ModelError) as error:
            build_model(override, scenario="simple_supply")

        assert check_error_or_warning(error, "cannot have cyclic storage")

    def test_incorrect_source_unit(self):
        """
        Only `absolute`, `per_cap`, or `per_area` is allowed under
        `source unit`.
        """

        def _override(source_unit):
            return {"techs.test_supply_elec.switches.source_unit": source_unit}

        with pytest.raises(exceptions.ModelError) as error:
            build_model(_override("power"), scenario="simple_supply")

        build_model(_override("absolute"), scenario="simple_supply")
        build_model(_override("per_cap"), scenario="simple_supply")
        build_model(_override("per_area"), scenario="simple_supply")

        assert check_error_or_warning(
            error, "`power` is an unknown source unit for `test_supply_elec`"
        )

    @pytest.mark.parametrize(
        "constraints,costs",
        (
            ({"units_max": 2, "flow_cap_per_unit": 5}, None),
            ({"units_min": 2, "flow_cap_per_unit": 5}, None),
            (None, {"purchase": 2}),
        ),
    )
    @pytest.mark.xfail(
        reason="Expected fail because now the setting of integer/binary variables is more explicit, so users should be aware without the need of a warning"
    )
    def test_milp_supply_warning(self, constraints, costs):
        override_constraints = {}
        override_costs = {}
        if constraints is not None:
            override_constraints.update(
                {"techs.test_supply_elec.constraints": constraints}
            )
        if costs is not None:
            override_costs.update({"techs.test_supply_elec.costs.monetary": costs})
        override = {**override_constraints, **override_costs}

        with pytest.warns(exceptions.ModelWarning) as warn:
            build_model(
                override_dict=override,
                scenario="simple_supply,one_day,investment_costs",
            )

        assert check_error_or_warning(
            warn,
            "Integer and / or binary decision variables are included in this model",
        )

    @pytest.mark.parametrize(
        "constraints,costs",
        (
            (
                {"units_max": 2, "storage_cap_per_unit": 5, "flow_cap_per_unit": 5},
                None,
            ),
            (
                {
                    "units_equals": 2,
                    "storage_cap_per_unit": 5,
                    "flow_cap_per_unit": 5,
                },
                None,
            ),
            (
                {"units_min": 2, "storage_cap_per_unit": 5, "flow_cap_per_unit": 5},
                None,
            ),
            (None, {"purchase": 2}),
        ),
    )
    @pytest.mark.xfail(
        reason="Expected fail because now the setting of integer/binary variables is more explicit, so users should be aware without the need of a warning"
    )
    def test_milp_storage_warning(self, constraints, costs):
        override_constraints = {}
        override_costs = {}
        if constraints is not None:
            override_constraints.update({"techs.test_storage.constraints": constraints})
        if costs is not None:
            override_costs.update({"techs.test_storage.costs.monetary": costs})
        override = {**override_constraints, **override_costs}

        with pytest.warns(exceptions.ModelWarning) as warn:
            build_model(
                override_dict=override,
                scenario="simple_storage,one_day,investment_costs",
            )

        assert check_error_or_warning(
            warn,
            "Integer and / or binary decision variables are included in this model",
        )

    @pytest.mark.parametrize(
        "override",
        [
            ({"parameters.objective_cost_weights.data": None}),
            (
                {
                    "parameters.objective_cost_weights": {
                        "data": [None, 1],
                        "index": ["monetary", "emissions"],
                    }
                }
            ),
        ],
    )
    def test_warn_on_no_weight(self, override):
        with pytest.raises(exceptions.ModelError) as excinfo:
            build_model(model_file="weighted_obj_func.yaml", override_dict=override)

        assert check_error_or_warning(
            excinfo, "Objective cost class weights must be numeric"
        )

    @pytest.mark.skip(reason="check is now taken care of in typedconfig")
    def test_storage_initial_fractional_value(self):
        """
        Check that the storage_initial value is a fraction
        """
        with pytest.raises(exceptions.ModelError) as error:
            build_model(
                {"techs.test_storage.constraints.storage_initial": 5},
                "simple_storage,two_hours,investment_costs",
            )

        assert check_error_or_warning(
            error, "storage_initial values larger than 1 are not allowed."
        )

    @pytest.mark.skip(reason="check is now taken care of in typedconfig")
    def test_storage_initial_smaller_than_discharge_depth(self):
        """
        Check that the storage_initial value is at least equalt to the storage_discharge_depth
        """
        with pytest.raises(exceptions.ModelError) as error:
            build_model(
                {"techs.test_storage.constraints.storage_initial": 0},
                "simple_storage,two_hours,investment_costs,storage_discharge_depth",
            )

        assert check_error_or_warning(
            error, "storage_initial is smaller than storage_discharge_depth."
        )

    @pytest.mark.skip(reason="check is now taken care of in typedconfig")
    def test_storage_inter_cluster_vs_storage_discharge_depth(self):
        """
        Check that the storage_inter_cluster is not used together with storage_discharge_depth
        """
        with pytest.raises(exceptions.ModelError) as error:
            override = {"config.init.time_subset": ["2005-01-01", "2005-01-04"]}
            build_model(override, "clustering,simple_storage,storage_discharge_depth")

        assert check_error_or_warning(
            error,
            "storage_discharge_depth is currently not allowed when time clustering is active.",
        )

    @pytest.mark.skip(reason="check is now taken care of in typedconfig")
    def test_warn_on_undefined_cost_classes(self):
        with pytest.warns(exceptions.ModelWarning) as warn:
            build_model(
                model_file="weighted_obj_func.yaml",
                scenario="undefined_class_objective",
            )

        assert check_error_or_warning(
            warn,
            "Cost classes `{'random_class'}` are defined in the objective options but not ",
        )


# TODO: move into model_data test for the distance getter method
@pytest.xfail(reason="moved into model_data")
class TestUtil:
    def test_vincenty(self):
        # London to Paris: about 344 km
        coords = [51.507222, -0.1275, 48.8567, 2.3508]
        distance = calliope.preprocess.util.vincenty(coords[0], coords[1])
        assert distance == pytest.approx(343834)  # in meters


class TestTime:
    @pytest.fixture
    def model_national(self, load_timeseries_from_dataframes):
        """
        Return national scale example model. If load_timeseries_from_dataframes
        is True, timeseries are read into dataframes and model is called using them.
        If not, the timeseries are read in from CSV.
        """
        if load_timeseries_from_dataframes:
            # Create dictionary with dataframes
            time_data_path = (
                calliope.examples.EXAMPLE_MODEL_DIR
                / "national_scale"
                / "timeseries_data"
            )
            timeseries_dataframes = {}
            timeseries_dataframes["csp_resource"] = pd.read_csv(
                time_data_path / "csp_resource.csv", index_col=0
            )
            timeseries_dataframes["demand_1"] = pd.read_csv(
                time_data_path / "demand-1.csv", index_col=0
            )
            timeseries_dataframes["demand_2"] = pd.read_csv(
                time_data_path / "demand-2.csv", index_col=0
            )
            # Create override dict telling calliope to load timeseries from df
            override_dict = {
                "techs.csp.constraints.source_max": "df=csp_resource",
                "nodes.region1.techs.demand_power.constraints.sink_equals": "df=demand_1:demand",
                "nodes.region2.techs.demand_power.constraints.sink_equals": "df=demand_2:demand",
            }
            return calliope.examples.national_scale(
                timeseries_dataframes=timeseries_dataframes, override_dict=override_dict
            )
        else:
            return calliope.examples.national_scale()

    @pytest.fixture
    def model_urban(self):
        return calliope.examples.urban_scale(
            override_dict={"config.init.time_subset": ["2005-01-01", "2005-01-10"]}
        )

    @pytest.mark.parametrize("load_timeseries_from_dataframes", [False, True])
    def test_timeseries_from_csv(self, model_national):
        """
        Timeseries data should be successfully loaded into national_scale example
        model. This test checks whether this happens with timeseries loaded both
        from CSV (`load_timeseries_from_dataframes`=False, called via file=...) and
        from dataframes (`load_timeseries_from_dataframes`=True, called via df=...).
        """

        model = model_national
        assert model.inputs.sink_equals.loc[("region1", "demand_power")].values[
            0
        ] == approx(25284.48)
        assert model.inputs.sink_equals.loc[("region2", "demand_power")].values[
            0
        ] == approx(2254.098)
        assert model.inputs.source_max.loc[("region1_1", "csp")].values[8] == approx(
            0.263805
        )
        assert model.inputs.source_max.loc[("region1_2", "csp")].values[8] == approx(
            0.096755
        )
        assert model.inputs.source_max.loc[("region1_3", "csp")].values[8] == approx(
            0.0
        )
