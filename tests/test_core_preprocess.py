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
            this_path, "common", "test_model", model_dict.config.init["time_data_path"]
        )
        # test as AttrDict
        calliope.Model(model_dict)

        # test as dict
        calliope.Model(model_dict.as_dict())

    @pytest.mark.filterwarnings(
        "ignore:(?s).*(links, test_link_a_b_elec) | Deactivated:calliope.exceptions.ModelWarning"
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
                    techs.test_supply_gas.flow_cap_max: 20
                two:
                    techs.test_supply_elec.flow_cap_max: 20

            nodes:
                a:
                    techs:
                        test_supply_gas:
                        test_supply_elec:
                        test_demand_elec:
            """
        )
        model = build_model(override_dict=override, scenario="scenario_1")

        assert model._model_def_dict.techs.test_supply_gas.flow_cap_max == 20
        assert model._model_def_dict.techs.test_supply_elec.flow_cap_max == 20

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
                    techs.test_supply_gas.flow_cap_max: 20
                two:
                    techs.test_supply_elec.flow_cap_max: 20
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

        assert model._model_def_dict.techs.test_supply_gas.flow_cap_max == 20
        assert model._model_def_dict.techs.test_supply_elec.flow_cap_max == 20

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
        with pytest.raises(exceptions.ModelError) as excinfo:
            build_model(override_dict=override, scenario="scenario_1")

        assert check_error_or_warning(
            excinfo, "(scenarios, scenario_1) | Unrecognised override name: techs."
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
        with pytest.raises(exceptions.ModelError) as excinfo:
            build_model(override_dict=override, scenario="scenario_1")

        assert check_error_or_warning(
            excinfo, "(scenarios, scenario_1) | Unrecognised override name: foo."
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
            build_model(override_dict=override, scenario="simple_supply,one_day")

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
                    parent: supply
                    name: test
                    source_max: .inf
                    flow_cap_max: .inf
            nodes.1.techs.test_undefined_carrier:
            """
        )
        with pytest.raises(exceptions.ModelError):
            build_model(override_dict=override, scenario="simple_supply,one_day")

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
            override_dict=override(["2005-01-01", "2005-01-01"]),
            scenario="simple_supply",
        )
        assert all(
            model.inputs.timesteps.to_index()
            == pd.date_range("2005-01", "2005-01-01 23:00:00", freq="H")
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
            "subset time range ['2005-03', '2005-04'] is outside the input data time range [2005-01-01 00:00:00, 2005-01-05 23:00:00]",
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
            "node_groups.init_nodes.techs.test_demand_elec.sink_equals": "file=demand_heat_diff_dateformat.csv",
            "nodes.a.techs.test_demand_heat.sink_equals": "file=demand_heat_diff_dateformat.csv",
            "nodes.b.techs.test_demand_heat.sink_equals": "file=demand_heat_diff_dateformat.csv",
        }
        model = build_model(override_dict=override1, scenario="simple_conversion")
        assert all(
            model.inputs.timesteps.to_index()
            == pd.date_range("2005-01", "2005-01-02 23:00:00", freq="H")
        )

        # should fail: wrong dateformat input for one file
        override2 = {
            "node_groups.init_nodes.techs.test_demand_elec.sink_equals": "file=demand_heat_diff_dateformat.csv"
        }

        with pytest.raises(exceptions.ModelError):
            build_model(override_dict=override2, scenario="simple_conversion")

        # should fail: wrong dateformat input for all files
        override3 = {"config.init.time_format": "%d/%m/%Y %H:%M:%S"}

        with pytest.raises(exceptions.ModelError):
            build_model(override_dict=override3, scenario="simple_supply")

        # should fail: one value wrong in file
        override4 = {
            "node_groups.init_nodes.techs.test_demand_elec.sink_equals": "file=demand_heat_wrong_dateformat.csv"
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
            "node_groups.init_nodes.techs.test_demand_elec.sink_equals": "file=demand_heat_wrong_length.csv"
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

    def test_dataframes_passed(self):
        """
        If model config specifies dataframes to be loaded in (via df=...),
        these time series must be passed as arguments in calliope.Model(...).
        """
        override = {"nodes.a.techs.test_demand_elec.sink_equals": "df=demand_elec"}
        with pytest.raises(exceptions.ModelError) as error:
            build_model(
                model_file="model_minimal.yaml",
                override_dict=override,
                timeseries_dataframes=None,
            )
        assert check_error_or_warning(
            error,
            "Missing timeseries dataframes passed as an argument in calliope.Model(...).",
        )

    def test_dataframe_keys(self):
        """
        Any timeseries specified via df=... must correspond to a key in
        timeseries_dataframes. An error should be thrown.
        """
        override = {"nodes.a.techs.test_demand_elec.sink_equals": "df=key_1"}
        ts_df = {"key_2": pd.DataFrame(np.arange(10))}

        with pytest.raises(KeyError) as error:
            build_model(
                model_file="model_minimal.yaml",
                override_dict=override,
                timeseries_dataframes=ts_df,
            )
        assert check_error_or_warning(
            error, "Attempted to load dataframe with undefined key"
        )

    @pytest.mark.parametrize(
        "input", [pd.DataFrame(np.arange(10)), {"demand_elec": np.arange(10)}]
    )
    def test_invalid_dataframes_passed(self, input):
        """
        `timeseries_dataframes` should be dict of pandas DataFrames.
        """
        override = {"nodes.a.techs.test_demand_elec.sink_equals": "df=demand_elec"}

        with pytest.raises(exceptions.ModelError) as error:
            build_model(
                model_file="model_minimal.yaml",
                override_dict=override,
                timeseries_dataframes=input,
            )
        assert check_error_or_warning(
            error, "`timeseries_dataframes` must be dict of pandas DataFrames."
        )


class TestChecks:
    @pytest.mark.parametrize("top_level_key", ["init", "solve"])
    def test_unrecognised_config_keys(self, top_level_key):
        """
        Check that the only keys allowed in 'model' and 'run' are those in the
        model defaults
        """
        override = {f"config.{top_level_key}.nonsensical_key": "random_string"}

        with pytest.raises(exceptions.ModelError) as excinfo:
            build_model(override_dict=override, scenario="simple_supply")
        assert check_error_or_warning(
            excinfo,
            "Additional properties are not allowed ('nonsensical_key' was unexpected)",
        )

    def test_model_version_mismatch(self):
        """
        Model config says config.init.calliope_version = 0.1, which is not what we
        are running, so we want a warning.
        """
        override = {"config.init.calliope_version": "0.1"}

        with pytest.warns(exceptions.ModelWarning) as excinfo:
            build_model(override_dict=override, scenario="simple_supply,one_day")

        assert check_error_or_warning(
            excinfo, "Model configuration specifies calliope version"
        )

    def test_unspecified_parent(self):
        """
        All technologies and technology groups must specify a parent
        """

        override = AttrDict.from_yaml_string(
            """
            techs.test_supply_no_parent:
                    name: Supply tech
                    carrier_out: gas
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
                name: Supply tech
                carrier_out: gas
                parent: test_supply_elec
                flow_cap_max: 10
                source_max: .inf
            nodes.b.techs.test_supply_tech_parent:
            """
        )

        with pytest.raises(exceptions.ModelError) as error:
            build_model(override_dict=override1, scenario="simple_supply,one_day")
        assert check_error_or_warning(
            error,
            "Errors during validation of the tech definition at node `b` dictionary.",
        )

    @pytest.mark.skip(
        reason="one_way doesn't work yet. We'll need to move this test to `model_data` once it does work."
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

    @pytest.mark.xfail(
        reason="Removed this check because it has to happen *after* calling `build`"
    )
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
                "techs.csp.source_max": "df=csp_resource",
                "nodes.region1.techs.demand_power.sink_equals": "df=demand_1:demand",
                "nodes.region2.techs.demand_power.sink_equals": "df=demand_2:demand",
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
        assert model.inputs.sink_equals.sel(
            nodes="region1", techs="demand_power"
        ).values[0] == approx(25284.48)
        assert model.inputs.sink_equals.sel(
            nodes="region2", techs="demand_power"
        ).values[0] == approx(2254.098)
        assert model.inputs.source_max.sel(nodes="region1_1", techs="csp").values[
            8
        ] == approx(0.263805)
        assert model.inputs.source_max.sel(nodes="region1_2", techs="csp").values[
            8
        ] == approx(0.096755)
        assert model.inputs.source_max.sel(nodes="region1_3", techs="csp").values[
            8
        ] == approx(0.0)
