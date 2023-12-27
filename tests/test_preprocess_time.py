import pandas as pd
import pytest  # noqa: F401

from .common.util import build_test_model


class TestClustering:
    @pytest.fixture
    def clustered_model(self):
        cluster_init = {
            "time_subset": ["2005-01-01", "2005-01-04"],
            "time_cluster": "cluster_days.csv",
        }
        return build_test_model(scenario="simple_supply", **cluster_init)

    def test_no_operate_mode_allowed(self, clustered_model):
        assert clustered_model._model_data.attrs["allow_operate_mode"] == 0

    def test_cluster_numbers(self, clustered_model):
        assert (
            clustered_model._model_data.clusters.to_index()
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
            clustered_model._model_data.timestep_cluster.to_series(),
            expected,
            check_names=False,
        )

    def test_cluster_datesteps(self, clustered_model):
        expected = pd.DatetimeIndex(
            ["2005-01-01", "2005-01-02", "2005-01-03", "2005-01-04", "2005-01-05"],
            name="datesteps",
        )
        pd.testing.assert_index_equal(
            clustered_model._model_data.datesteps.to_index(), expected
        )

    @pytest.mark.parametrize(
        "var",
        [
            "lookup_cluster_first_timestep",
            "lookup_cluster_last_timestep",
            "lookup_datestep_cluster",
            "lookup_datestep_last_cluster_timestep",
        ],
    )
    def test_cluster_has_lookup_arrays(self, clustered_model, var):
        assert var in clustered_model._model_data.data_vars


class TestResamplingAndCluster:
    def test_resampling_to_6h_then_clustering(self):
        model = build_test_model(
            scenario="simple_supply",
            time_subset=["2005-01-01", "2005-01-04"],
            time_resample="6H",
            time_cluster="cluster_days.csv",
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

        assert dtindex.equals(model._model_data.timesteps.to_index())
        assert model._model_data.attrs["allow_operate_mode"] == 0


class TestResampling:
    def test_15min_resampling_to_6h(self):
        # The data is identical for '2005-01-01' and '2005-01-03' timesteps,
        # it is only different for '2005-01-02'
        override = {"techs.test_demand_elec.sink_equals": "file=demand_elec_15mins.csv"}

        model = build_test_model(
            override,
            scenario="simple_supply",
            time_resample="6H",
            time_subset=["2005-01-01", "2005-01-03"],
        )
        data = model._model_data

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
                "2005-01-03 00:00:00",
                "2005-01-03 06:00:00",
                "2005-01-03 12:00:00",
                "2005-01-03 18:00:00",
            ]
        )

        assert dtindex.equals(data.timesteps.to_index())
        assert data.attrs["allow_operate_mode"] == 1

    def test_15min_to_2h_resampling_to_2h(self):
        """
        CSV has daily timeseries varying from 15min to 2h resolution, resample all to 2h
        """
        override = {
            "techs.test_demand_elec.sink_equals": "file=demand_elec_15T_to_2h.csv"
        }

        model = build_test_model(
            override, scenario="simple_supply,one_day", time_resample="2H"
        )
        data = model._model_data

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

        assert dtindex.equals(data.timesteps.to_index())
