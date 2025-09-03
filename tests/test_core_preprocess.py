import warnings

import pandas as pd
import pytest
from pydantic import ValidationError

import calliope
import calliope.exceptions as exceptions
from calliope.io import read_rich_yaml

from .common.util import build_test_model as build_model
from .common.util import check_error_or_warning


class TestModelRun:
    def test_read_dict(self, data_source_dir):
        """Test creating a model from dict/AttrDict instead of from YAML"""
        model_dir = data_source_dir.parent
        model_location = model_dir / "model.yaml"
        model_dict = calliope.io.read_rich_yaml(model_location)
        node_dict = calliope.AttrDict(
            {
                "nodes": {
                    "a": {"techs": {"test_supply_elec": {}, "test_demand_elec": {}}},
                    "b": {"techs": {"test_supply_elec": {}, "test_demand_elec": {}}},
                }
            }
        )
        model_dict.union(node_dict)
        for src in model_dict["data_tables"].values():
            src["data"] = (model_dir / src["data"]).as_posix()
        # test as AttrDict
        calliope.read_dict(model_dict)

        # test as dict
        calliope.read_dict(model_dict.as_dict())

    @pytest.fixture
    def subset_time_model(self):
        def _subset_time_model(param):
            override = read_rich_yaml(f"config.init.subset.timesteps: {param}")
            build_model(override_dict=override, scenario="simple_supply")

        return _subset_time_model

    def correct_time_subset(self, subset_time_model):
        # should pass: two string in list as slice
        model = subset_time_model(["2005-01-01", "2005-01-01"])
        assert all(
            model.inputs.timesteps.to_index()
            == pd.date_range("2005-01", "2005-01-01 23:00:00", freq="h")
        )

    @pytest.mark.parametrize(
        "time_subset", [["2005-01"], ["2005-01-01", "2005-01-02", "2005-01-03"]]
    )
    def test_incorrect_subset_time(self, subset_time_model, time_subset):
        """If time_subset is a list, it must have two entries (start_time, end_time)"""

        with pytest.raises(exceptions.ModelError) as excinfo:
            subset_time_model(time_subset)
        assert check_error_or_warning(
            excinfo,
            f"Timeseries subset must be a list of two timestamps. Received: {time_subset}",
        )

    def test_subset_time_as_string(self, subset_time_model):
        """Invalid to use a string to subset time."""
        with pytest.raises(ValidationError):
            subset_time_model("2005-01")

    @pytest.mark.parametrize(
        "time_subset", [["2005-03", "2005-04"], ["2005-02-01", "2005-02-05"]]
    )
    def test_subset_time_out_of_range(self, subset_time_model, time_subset):
        """If time_subset is out of range of the input data, raise an error."""
        # should fail: time subset out of range of input data
        with pytest.raises(exceptions.ModelError) as excinfo:
            subset_time_model(time_subset)
        assert check_error_or_warning(
            excinfo,
            f"subset time range {time_subset} is outside the input data time range",
        )

    def test_inconsistent_time_indices_passes_thanks_to_time_subsetting(self):
        override = read_rich_yaml(
            "data_tables.demand_elec.data: data_tables/demand_heat_wrong_length.csv"
        )
        # should pass: wrong length of demand_heat csv, but time subsetting removes the difference
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            build_model(override_dict=override, scenario="simple_conversion,one_day")

    def test_single_timestep(self):
        """Test that warning is raised on using 1 timestep, that timestep resolution will
        be inferred to be 1 hour
        """
        override1 = {
            "config.init.subset.timesteps": [
                "2005-01-01 00:00:00",
                "2005-01-01 00:00:00",
            ]
        }
        # check in output error that it points to: 07/01/2005 10:00:00
        with pytest.warns(exceptions.ModelWarning) as warn_info:
            model = build_model(override_dict=override1, scenario="simple_supply")

        assert check_error_or_warning(
            warn_info,
            "Only one timestep defined. Inferring timestep resolution to be 1 hour",
        )
        assert model.inputs.timestep_resolution == [1]


class TestChecks:
    @pytest.mark.parametrize(
        "top_level_key", ["init", "build", "solve", "build.operate", "solve.spores"]
    )
    def test_unrecognised_config_keys(self, top_level_key):
        """Check that no extra keys are allowed in the configuration."""
        override = {f"config.{top_level_key}.nonsensical_key": "random_string"}

        with pytest.raises(ValidationError):
            build_model(override_dict=override, scenario="simple_supply")

    def test_model_version_mismatch(self):
        """Model config says config.init.calliope_version = 0.1, which is not what we
        are running, so we want a warning.
        """
        override = {"config.init.calliope_version": "0.1"}

        with pytest.warns(exceptions.ModelWarning) as excinfo:
            build_model(override_dict=override, scenario="simple_supply,one_day")

        assert check_error_or_warning(
            excinfo, "Model configuration specifies calliope version"
        )

    @pytest.mark.skip(
        reason="one_way doesn't work yet. We'll need to move this test to `model_data` once it does work."
    )
    def test_one_way(self):
        """With one_way transmission, we remove one direction of a link from
        loc_tech_carriers_prod and the other from loc_tech_carriers_con.
        """
        override = {
            "links.X1,N1.techs.heat_pipes.switches.one_way": True,
            "links.N1,X2.techs.heat_pipes.switches.one_way": True,
            "links.N1,X3.techs.heat_pipes.switches.one_way": True,
        }
        m = calliope.examples.urban_scale(
            override_dict=override, subset={"timesteps": ["2005-01-01", "2005-01-01"]}
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
        """Don't allow time clustering with cyclic storage if not also using
        storage_inter_cluster
        """
        override = {
            "config.init.subset.timesteps": ["2005-01-01", "2005-01-04"],
            "config.init.time_cluster": "cluster_days_param",
            "config.build.cyclic_storage": True,
            "data_tables.cluster_days": {
                "data": "data_tables/cluster_days.csv",
                "rows": "datesteps",
                "add_dims": {"parameters": "cluster_days_param"},
            },
        }

        with pytest.raises(exceptions.ModelError) as error:
            build_model(override, scenario="simple_supply")

        assert check_error_or_warning(error, "cannot have cyclic storage")

    @pytest.mark.xfail(
        reason="Removed this check because it has to happen *after* calling `build`"
    )
    def test_storage_inter_cluster_vs_storage_discharge_depth(self):
        """Check that the storage_inter_cluster is not used together with storage_discharge_depth"""
        override = {"config.init.subset.timesteps": ["2005-01-01", "2005-01-04"]}
        with pytest.raises(exceptions.ModelError) as error:
            build_model(override, "clustering,simple_storage,storage_discharge_depth")

        assert check_error_or_warning(
            error,
            "storage_discharge_depth is currently not allowed when time clustering is active.",
        )
