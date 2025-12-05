import numpy as np
import pandas as pd
import pytest  # noqa: F401
import xarray as xr

from calliope import exceptions
from calliope.io import read_rich_yaml
from calliope.preprocess import time

from ..common.util import build_test_model


class TestTimeFormat:
    def test_change_date_format(self):
        """
        Test the date parser catches a different date format from file than
        user input/default (inc. if it is just one line of a file that is incorrect)
        """

        # should pass: changing datetime format from default
        override = read_rich_yaml(
            """
            config.init.datetime_format: "%d/%m/%Y %H:%M"
            data_tables:
                demand_elec.data: data_tables/demand_heat_diff_dateformat.csv
                demand_heat.data: data_tables/demand_heat_diff_dateformat.csv
        """
        )
        model = build_test_model(override_dict=override, scenario="simple_conversion")
        assert all(
            model.inputs.timesteps.to_index()
            == pd.date_range("2005-01", "2005-01-02 23:00:00", freq="h")
        )

    def test_incorrect_date_format_one(self):
        # should fail: wrong dateformat input for one file
        override = read_rich_yaml(
            "data_tables.demand_elec.data: data_tables/demand_heat_diff_dateformat.csv"
        )

        with pytest.raises(exceptions.ModelError):
            build_test_model(override_dict=override, scenario="simple_conversion")

    def test_incorrect_date_format_multi(self):
        # should fail: wrong dateformat input for all files
        override3 = {"config.init.datetime_format": "%d/%m/%Y %H:%M"}

        with pytest.raises(exceptions.ModelError):
            build_test_model(override_dict=override3, scenario="simple_supply")

    def test_incorrect_date_format_one_value_only(self):
        """All time formatted values should be checked against the configured ISO."""
        override = read_rich_yaml(
            "data_tables.demand_elec.data: data_tables/demand_heat_wrong_dateformat.csv"
        )
        with pytest.raises(
            exceptions.ModelError,
            match="Time data 02/01/2005 00:00 is not ISO8601 format",
        ):
            build_test_model(override_dict=override, scenario="simple_conversion")


class TestSubsetTime:
    @pytest.fixture(scope="class")
    def ts_index(self):
        return pd.date_range("2005-01-01", "2005-01-05", freq="h")

    @pytest.mark.parametrize(
        "time_subset",
        [
            ["2005", "2005"],
            ["2005-01", "2005-01"],
            ["2005-01-02", "2005-01-04"],
            ["2005-01-02 00", "2005-01-04 00"],
            ["2005-01-02 00:00", "2005-01-04 00:00"],
            ["2005-01-02 00:00:00", "2005-01-04 00:00:00"],
        ],
    )
    def test_check_time_subset_all_good(self, ts_index, time_subset):
        """A nicely format subset raises no errors."""
        time.check_time_subset(ts_index, time_subset)

    def test_check_time_subset_no_overlap(self, ts_index):
        """The subset must overlap at least partially with the timeseries index."""
        time_subset = ["2005-01-06", "2005-01-07"]
        with pytest.raises(exceptions.ModelError, match="subset time range"):
            time.check_time_subset(ts_index, time_subset)

    def test_check_time_subset_too_many_list_items(self, ts_index):
        """Subset must be a slice of two timestamps."""
        time_subset = ["2005-01-02", "2005-01-04", "2005-01-05"]
        with pytest.raises(
            exceptions.ModelError, match="subset must be a list of two timestamps"
        ):
            time.check_time_subset(ts_index, time_subset)

    @pytest.mark.parametrize(
        "time_subset",
        [["01/01/2005", "03/01/2005"], ["01/01/2005 00:00", "03/01/2005 00:00"]],
    )
    def test_check_time_format(self, ts_index, time_subset):
        """Subset must be in ISO format"""
        with pytest.raises(
            exceptions.ModelError, match="Timeseries subset must be in ISO format"
        ):
            time.check_time_subset(ts_index, time_subset)


class TestClustering:
    @pytest.fixture(scope="class", params=["datetime", "string"])
    def cluster_dtype(self, request):
        """The method should be able to handle datetime and string lookup data"""
        return request.param

    @pytest.fixture(scope="class")
    def dummy_model_to_cluster(self, dummy_int, cluster_dtype):
        """Create a dummy model with a parameter to cluster 3 years of data into 5 representative days."""
        ts = pd.date_range(
            "2025-01-01", "2028-01-01", freq="h", inclusive="left", name="timesteps"
        )
        da = pd.Series(dummy_int, index=ts).to_xarray()
        cluster_ts = pd.date_range(
            "2025-01-01", "2028-01-01", freq="1d", inclusive="left", name="datesteps"
        )
        if cluster_dtype == "string":
            cluster_ts = cluster_ts.strftime("%Y-%m-%d")

        random_sample = np.random.choice(
            cluster_ts.to_series().sample(5).index, len(cluster_ts)
        )
        cluster_da = pd.Series(random_sample, index=cluster_ts).to_xarray()
        ds = xr.Dataset(
            {
                "ts_data": da,
                "clustering_param": cluster_da,
                "non_ts_data": xr.DataArray(42),
            }
        )
        return ds

    @pytest.fixture(scope="class")
    def dummy_clustered_model(self, dummy_model_to_cluster):
        return time.cluster(
            dummy_model_to_cluster,
            clustering_param="clustering_param",
            time_format="ISO8601",
        )

    def test_has_clustering_lookup_arrays(self, dummy_clustered_model):
        """Check that all expected clustering lookup arrays are present, along with the original data"""
        expected = [
            "ts_data",
            "clustering_param",
            "timestep_cluster",
            "timestep_weights",
            "cluster_first_timestep",
            "lookup_cluster_last_timestep",
            "lookup_datestep_cluster",
            "lookup_datestep_last_cluster_timestep",
            "non_ts_data",
        ]
        assert not set(dummy_clustered_model.data_vars.keys()).symmetric_difference(
            expected
        )

    def test_has_clusters(self, dummy_clustered_model):
        """Check that the correct number of clusters have been created"""
        expected = pd.Index([0, 1, 2, 3, 4], name="clusters")
        pd.testing.assert_index_equal(
            dummy_clustered_model.clusters.to_index(), expected
        )

    def test_has_reduced_timeseries(self, dummy_clustered_model):
        """Check that the timesteps dimension has been reduced correctly"""
        assert len(dummy_clustered_model.timesteps) == 120  # 5 days * 24 hours

    def test_clustered_timestep_weights(self, dummy_clustered_model, cluster_dtype):
        """Check that the timestep weights have been calculated correctly based on the number of days each clustered day represents"""
        timestep_idx = dummy_clustered_model.timesteps.to_index()
        dates = (
            timestep_idx.date
            if cluster_dtype == "datetime"
            else timestep_idx.strftime("%Y-%m-%d")
        )
        expected = (
            dummy_clustered_model.clustering_param.to_series().value_counts().loc[dates]
        )
        expected.index = timestep_idx
        expected_da = (
            expected.rename_axis(index="timesteps")
            .rename("timestep_weights")
            .to_xarray()
        )
        assert dummy_clustered_model.timestep_weights.equals(expected_da)

    def test_data_preserved(self, dummy_clustered_model, dummy_int):
        """Check that input data is preserved in the clustered model"""
        assert dummy_clustered_model.non_ts_data.item() == 42
        assert (dummy_clustered_model.ts_data == dummy_int).all()

    @pytest.fixture(
        scope="class", params=["cluster_days", "cluster_days_diff_dateformat"]
    )
    def clustered_model(self, request):
        cluster_init = {
            "subset": {"timesteps": ["2005-01-01", "2005-01-04"]},
            "time_cluster": request.param,
            "override_dict": {
                "data_tables": {
                    "clustering": {
                        "data": f"data_tables/{request.param}.csv",
                        "rows": "datesteps",
                        "add_dims": {"parameters": request.param},
                    }
                }
            },
        }
        if "diff_dateformat" in request.param:
            cluster_init["override_dict"]["data_tables.demand_elec"] = {
                "data": "data_tables/demand_heat_diff_dateformat.csv"
            }
            cluster_init["datetime_format"] = "%d/%m/%Y %H:%M"
            cluster_init["date_format"] = "%d/%m/%Y"

        return build_test_model(scenario="simple_supply", **cluster_init)

    def test_cluster_numbers(self, clustered_model):
        assert (
            clustered_model.inputs.clusters.to_index()
            .symmetric_difference([0, 1])
            .empty
        )

    def test_cluster_timesteps(self, clustered_model):
        expected = pd.Series(
            {
                f"2005-01-0{day} {hour}:00:00": 0 if day == 1 else 1
                for day in [1, 3]
                for hour in range(0, 24)
            }
        )
        expected.index = pd.to_datetime(expected.index)

        pd.testing.assert_series_equal(
            clustered_model.inputs.timestep_cluster.to_series(),
            expected,
            check_names=False,
        )

    def test_cluster_datesteps(self, clustered_model):
        expected = pd.DatetimeIndex(
            ["2005-01-01", "2005-01-02", "2005-01-03", "2005-01-04", "2005-01-05"],
            name="datesteps",
        )
        pd.testing.assert_index_equal(
            clustered_model.inputs.datesteps.to_index(), expected
        )

    @pytest.mark.parametrize(
        "var",
        [
            "cluster_first_timestep",
            "lookup_cluster_last_timestep",
            "lookup_datestep_cluster",
            "lookup_datestep_last_cluster_timestep",
        ],
    )
    def test_cluster_has_lookup_arrays(self, clustered_model, var):
        assert var in clustered_model.inputs.data_vars


class TestResamplingAndCluster:
    def test_resampling_to_6h_then_clustering(self):
        model = build_test_model(
            scenario="simple_supply",
            subset={"timesteps": ["2005-01-01", "2005-01-04"]},
            resample={"timesteps": "6h"},
            time_cluster="cluster_days_param",
            override_dict={
                "data_tables.cluster_days": {
                    "data": "data_tables/cluster_days.csv",
                    "rows": "datesteps",
                    "add_dims": {"parameters": "cluster_days_param"},
                }
            },
        )

        dtindex = pd.DatetimeIndex(
            [
                "2005-01-01 00:00:00",
                "2005-01-01 06:00:00",
                "2005-01-01 12:00:00",
                "2005-01-01 18:00:00",
                "2005-01-03 00:00:00",
                "2005-01-03 06:00:00",
                "2005-01-03 12:00:00",
                "2005-01-03 18:00:00",
            ]
        )

        assert dtindex.equals(model.inputs.timesteps.to_index())


class TestResampling:
    def test_resampling_unknown_format(self):
        with pytest.raises(
            exceptions.ModelError,
            match="Unknown `timesteps` resampling frequency: unknown",
        ):
            build_test_model(
                {}, scenario="simple_supply", resample={"timesteps": "unknown"}
            )

    def test_15min_resampling_to_6h(self):
        # The data is identical for '2005-01-01' and '2005-01-03' timesteps,
        # it is only different for '2005-01-02'
        override = read_rich_yaml(
            "data_tables.demand_elec.data: data_tables/demand_elec_15mins.csv"
        )

        model = build_test_model(
            override, scenario="simple_supply", resample={"timesteps": "6h"}
        )

        dtindex = pd.DatetimeIndex(
            [
                "2005-01-01 00:00:00",
                "2005-01-01 06:00:00",
                "2005-01-01 12:00:00",
                "2005-01-01 18:00:00",
                "2005-01-02 00:00:00",
                "2005-01-02 06:00:00",
                "2005-01-02 12:00:00",
                "2005-01-02 18:00:00",
            ]
        )

        assert dtindex.equals(model.inputs.timesteps.to_index())

    def test_15min_to_2h_resampling_to_2h(self):
        """
        CSV has daily timeseries varying from 15min to 2h resolution, resample all to 2h
        """
        override = read_rich_yaml(
            "data_tables.demand_elec.data: data_tables/demand_elec_15T_to_2h.csv"
        )

        model = build_test_model(
            override, scenario="simple_supply,one_day", resample={"timesteps": "2h"}
        )

        dtindex = pd.DatetimeIndex(
            [
                "2005-01-01 00:00:00",
                "2005-01-01 02:00:00",
                "2005-01-01 04:00:00",
                "2005-01-01 06:00:00",
                "2005-01-01 08:00:00",
                "2005-01-01 10:00:00",
                "2005-01-01 12:00:00",
                "2005-01-01 14:00:00",
                "2005-01-01 16:00:00",
                "2005-01-01 18:00:00",
                "2005-01-01 20:00:00",
                "2005-01-01 22:00:00",
            ]
        )

        assert dtindex.equals(model.inputs.timesteps.to_index())

    @pytest.mark.filterwarnings(
        "ignore:(?s).*Possibly missing data on the timesteps dimension*:calliope.exceptions.ModelWarning"
    )
    def test_different_ts_resolutions_resampling_to_6h(self):
        # The data is identical for '2005-01-01' and '2005-01-03' timesteps,
        # it is only different for '2005-01-02'
        override = read_rich_yaml(
            """
            data_tables:
                demand_elec:
                    select:
                        nodes: a
                demand_elec_15m:
                    data: data_tables/demand_elec_15mins.csv
                    rows: timesteps
                    columns: nodes
                    select:
                        nodes: b
                    add_dims:
                        parameters: sink_use_equals
                        techs: test_demand_elec
            """
        )

        model = build_test_model(
            override, scenario="simple_supply", resample={"timesteps": "6h"}
        )

        dtindex = pd.DatetimeIndex(
            [
                "2005-01-01 00:00:00",
                "2005-01-01 06:00:00",
                "2005-01-01 12:00:00",
                "2005-01-01 18:00:00",
                "2005-01-02 00:00:00",
                "2005-01-02 06:00:00",
                "2005-01-02 12:00:00",
                "2005-01-02 18:00:00",
            ]
        )
        assert dtindex.equals(model.inputs.timesteps.to_index())

        # We have the same data in each row for both,
        # but with resampling, the 15minute data should be 4x larger as it is summed on resampling.
        assert (
            model.inputs.sink_use_equals.sel(nodes="a").fillna(0)
            == model.inputs.sink_use_equals.sel(nodes="b").fillna(0) / 4
        ).all()


class TestMissingTimeData:
    """Test detection of missing timeseries data."""

    def test_warn_missing_timeseries_data(self):
        """Test warning when timeseries has gaps."""
        import numpy as np
        import xarray as xr

        ds = xr.Dataset(
            {"demand": (["timesteps", "nodes"], [[1, 2], [np.nan, 4], [5, 6]])},
            coords={"timesteps": pd.date_range("2005-01-01", periods=3, freq="h")},
        )

        with pytest.warns(match="Possibly missing data on the timesteps dimension"):
            time._check_missing_data(ds, "timesteps")

    def test_no_warn_all_data_present(self):
        """Test no warning when all data present."""
        import xarray as xr

        ds = xr.Dataset(
            {"demand": (["timesteps", "nodes"], [[1, 2], [3, 4], [5, 6]])},
            coords={"timesteps": pd.date_range("2005-01-01", periods=3, freq="h")},
        )

        # Should not raise warning - use context manager to verify
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            time._check_missing_data(ds, "timesteps")


class TestDatetimeConversion:
    """Test datetime format conversion."""

    @pytest.mark.parametrize(
        ("time_format", "expected"),
        [
            ("ISO8601", pd.Timestamp("2005-01-01")),
            ("%Y-%m-%d", pd.Timestamp("2005-01-01")),
        ],
    )
    def test_datetime_index_formats(self, time_format, expected):
        """Test _datetime_index handles different formats."""
        index = pd.Index(["2005-01-01"])
        result = time._datetime_index(index, time_format)
        assert result[0] == expected


class TestAddInferredTimeParams:
    def test_single_timestep(self):
        """Test that warning is raised on using 1 timestep, that timestep resolution will
        be inferred to be 1 hour
        """
        dataset = xr.Dataset().assign_coords(
            timesteps=pd.date_range("2005-01-01", periods=1, freq="H")
        )
        # check in output error that it points to: 07/01/2005 10:00:00
        with pytest.warns(
            exceptions.ModelWarning,
            match=r"Only one timestep defined. Inferring timestep resolution to be 1 hour",
        ):
            updated_dataset = time.add_inferred_time_params(dataset)
        assert (updated_dataset.timestep_resolution == 1).all()
