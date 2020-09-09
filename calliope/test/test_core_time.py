import pandas as pd
import numpy as np
import pytest  # pylint: disable=unused-import

import calliope
from calliope import exceptions
from calliope.time import funcs, masks
from calliope.test.common.util import (
    build_test_model,
    check_error_or_warning,
    python36_or_higher,
)


class TestClustering:
    @pytest.fixture
    def model_national(self, scope="module"):
        return calliope.examples.national_scale(
            override_dict={
                "model.random_seed": 23,
                "model.subset_time": ["2005-01-01", "2005-03-31"],
            }
        )

    def test_kmeans_mean(self, model_national):
        data = model_national._model_data

        data_clustered = funcs.apply_clustering(
            data,
            timesteps=None,
            clustering_func="kmeans",
            how="mean",
            normalize=True,
            k=5,
        )

        assert len(data_clustered.clusters.to_pandas().unique()) == 5

    def test_kmeans_closest(self, model_national):
        data = model_national._model_data

        data_clustered = funcs.apply_clustering(
            data,
            timesteps=None,
            clustering_func="kmeans",
            how="closest",
            normalize=True,
            k=5,
        )

    def test_hierarchical_mean(self, model_national):
        data = model_national._model_data

        data_clustered = funcs.apply_clustering(
            data,
            timesteps=None,
            clustering_func="hierarchical",
            how="mean",
            normalize=True,
            k=5,
        )

        assert len(data_clustered.clusters.to_pandas().unique()) == 5

    def test_hierarchical_closest(self, model_national):
        data = model_national._model_data

        data_clustered = funcs.apply_clustering(
            data,
            timesteps=None,
            clustering_func="hierarchical",
            how="closest",
            normalize=True,
            k=5,
        )

        # FIXME

    def test_hartigans_rule(self, model_national):
        data = model_national._model_data

        with pytest.warns(exceptions.ModelWarning) as excinfo:
            funcs.apply_clustering(
                data,
                timesteps=None,
                clustering_func="kmeans",
                how="mean",
                normalize=True,
            )

        assert check_error_or_warning(excinfo, "a good number of clusters is 5")

    def test_hierarchical_no_hartigans_rule(self, model_national):
        data = model_national._model_data

        with pytest.raises(exceptions.ModelError) as excinfo:
            funcs.apply_clustering(
                data,
                timesteps=None,
                clustering_func="hierarchical",
                how="mean",
                normalize=True,
            )

        assert check_error_or_warning(
            excinfo, "Cannot undertake hierarchical clustering"
        )

    def test_15min_clustering(self):
        # The data is identical for '2005-01-01' and '2005-01-03' timesteps,
        # it is only different for '2005-01-02'
        override = {
            "techs.test_demand_elec.constraints.resource": "file=demand_elec_15mins.csv",
            "model.subset_time": None,
        }

        model = build_test_model(override, scenario="simple_supply,one_day")
        data = model._model_data

        data_clustered_kmeans = funcs.apply_clustering(
            data,
            timesteps=None,
            clustering_func="kmeans",
            how="mean",
            normalize=True,
            k=2,
        )

        data_clustered_hierarchical = funcs.apply_clustering(
            data,
            timesteps=None,
            clustering_func="hierarchical",
            how="mean",
            normalize=True,
            k=2,
        )
        assert len(data_clustered_kmeans.clusters.to_pandas().unique()) == 2
        assert len(data_clustered_hierarchical.clusters.to_pandas().unique()) == 2

        days = np.unique(
            data_clustered_kmeans.timesteps.to_index().strftime("%Y-%m-%d")
        )
        # not sure which of '2005-01-01' and '2005-01-03' it will choose to
        # label the cluster of those two days
        assert "2005-01-02" in days and ("2005-01-01" in days or "2005-01-03" in days)

        assert np.array_equal(
            data_clustered_kmeans.timestep_resolution.values,
            [0.25 for i in range(24 * 4 * 2)],
        )

    def test_15min_to_2h_clustering(self):
        # The data is identical for '2005-01-01' and '2005-01-03' timesteps,
        # it is only different for '2005-01-02'
        override = {
            "techs.test_demand_elec.constraints.resource": "file=demand_elec_15T_to_2h.csv",
            "model.subset_time": None,
        }

        model = build_test_model(override, scenario="simple_supply,one_day")
        data = model._model_data

        data_clustered_kmeans = funcs.apply_clustering(
            data,
            timesteps=None,
            clustering_func="kmeans",
            how="mean",
            normalize=True,
            k=2,
        )

        data_clustered_hierarchical = funcs.apply_clustering(
            data,
            timesteps=None,
            clustering_func="hierarchical",
            how="mean",
            normalize=True,
            k=2,
        )
        assert len(data_clustered_kmeans.clusters.to_pandas().unique()) == 2
        assert len(data_clustered_hierarchical.clusters.to_pandas().unique()) == 2

        days = np.unique(
            data_clustered_kmeans.timesteps.to_index().strftime("%Y-%m-%d")
        )
        # not sure which of '2005-01-01' and '2005-01-03' it will choose to
        # label the cluster of those two days
        assert "2005-01-02" in days and ("2005-01-01" in days or "2005-01-03" in days)

        assert np.array_equal(
            data_clustered_kmeans.timestep_resolution.values,
            [
                0.25,
                0.25,
                0.25,
                0.25,
                0.5,
                0.5,
                0.5,
                0.5,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                2,
                1,
                1,
                2,
                2,
                2,
                0.25,
                0.25,
                0.25,
                0.25,
                0.5,
                0.5,
                0.5,
                0.5,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                2,
                1,
                1,
                2,
                2,
                2,
            ],
        )

    @python36_or_higher
    def test_predefined_clusters(self):
        override = {
            "model.subset_time": ["2005-01-01", "2005-01-04"],
            "model.time": {
                "function": "apply_clustering",
                "function_options": {
                    "clustering_func": "file=clusters.csv:0",
                    "how": "mean",
                },
            },
        }

        model = build_test_model(override, scenario="simple_supply")

        assert np.array_equal(
            model._model_data.clusters.to_pandas().unique(), [0, 1, 2]
        )

        override2 = {
            **override,
            **{
                "model.time.function_options.clustering_func": "file=cluster_days.csv:1"
            },
        }

        model = build_test_model(override2, scenario="simple_supply")

        assert np.array_equal(
            model._model_data.clusters.to_pandas().unique(), [0, 1, 2]
        )

        override3 = {
            **override,
            **{
                "model.time.function_options.clustering_func": "file=cluster_days.csv:1",
                "model.time.function_options.how": "closest",
            },
        }

        model = build_test_model(override3, scenario="simple_supply")

        assert np.array_equal(model._model_data.clusters.to_pandas().unique(), [0])

    @python36_or_higher
    def test_predefined_clusters_fail(self):
        override = {
            "model.subset_time": ["2005-01-01", "2005-01-04"],
            "model.time": {
                "function": "apply_clustering",
                "function_options": {
                    "clustering_func": "file=clusters.csv:0",
                    "how": "mean",
                },
            },
        }
        # should fail - no CSV data column defined
        override1 = {
            **override,
            **{"model.time.function_options.clustering_func": "file=clusters.csv"},
        }
        with pytest.raises(exceptions.ModelError) as error:
            build_test_model(override1, scenario="simple_supply")

        assert check_error_or_warning(error, "No time clustering column given")

        # should fail - unknown CSV data column defined
        override2 = {
            **override,
            **{"model.time.function_options.clustering_func": "file=clusters.csv:1"},
        }

        with pytest.raises(KeyError) as error:
            build_test_model(override2, scenario="simple_supply")

        assert check_error_or_warning(error, "time clustering column 1 not found")

        # should fail - more than one cluster given to any one day
        override3 = {
            **override,
            **{"model.time.function_options.clustering_func": "file=clusters.csv:b"},
        }

        with pytest.raises(exceptions.ModelError) as error:
            build_test_model(override3, scenario="simple_supply")

        assert check_error_or_warning(
            error, "More than one cluster value assigned to a day in `clusters.csv:b`"
        )

        # should fail - not enough data in clusters.csv to cover subset_time
        override4 = {
            **override,
            **{
                "model.subset_time": ["2005-01-01", "2005-01-06"],
                "model.time.function_options.clustering_func": "file=cluster_days.csv:1",
            },
        }

        with pytest.raises(exceptions.ModelError) as error:
            build_test_model(override4, scenario="simple_supply")

        assert check_error_or_warning(error, "Missing cluster days")


class TestMasks:
    @pytest.fixture
    def model_national(self, scope="module"):
        return calliope.examples.national_scale(
            override_dict={"model.subset_time": ["2005-01-01", "2005-01-31"]}
        )

    @pytest.fixture
    def model_urban(self, scope="module"):
        return calliope.examples.urban_scale(
            override_dict={"model.subset_time": ["2005-01-01", "2005-01-31"]}
        )

    def test_zero(self, model_national):
        data = model_national._model_data_original.copy()
        mask = masks.zero(data, "csp", var="resource")

        dtindex = pd.DatetimeIndex(
            [
                "2005-01-01 00:00:00",
                "2005-01-01 01:00:00",
                "2005-01-01 02:00:00",
                "2005-01-01 03:00:00",
                "2005-01-01 04:00:00",
                "2005-01-01 05:00:00",
                "2005-01-01 06:00:00",
                "2005-01-01 16:00:00",
                "2005-01-01 17:00:00",
                "2005-01-01 18:00:00",
                "2005-01-01 19:00:00",
                "2005-01-01 20:00:00",
                "2005-01-01 21:00:00",
                "2005-01-01 22:00:00",
                "2005-01-01 23:00:00",
                "2005-01-02 00:00:00",
                "2005-01-02 01:00:00",
                "2005-01-02 02:00:00",
                "2005-01-02 03:00:00",
                "2005-01-02 04:00:00",
                "2005-01-02 05:00:00",
                "2005-01-02 06:00:00",
                "2005-01-02 16:00:00",
                "2005-01-02 17:00:00",
                "2005-01-02 18:00:00",
                "2005-01-02 19:00:00",
                "2005-01-02 20:00:00",
                "2005-01-02 21:00:00",
                "2005-01-02 22:00:00",
                "2005-01-02 23:00:00",
                "2005-01-03 00:00:00",
                "2005-01-03 01:00:00",
                "2005-01-03 02:00:00",
                "2005-01-03 03:00:00",
                "2005-01-03 04:00:00",
                "2005-01-03 05:00:00",
                "2005-01-03 06:00:00",
                "2005-01-03 16:00:00",
                "2005-01-03 17:00:00",
                "2005-01-03 18:00:00",
                "2005-01-03 19:00:00",
                "2005-01-03 20:00:00",
                "2005-01-03 21:00:00",
                "2005-01-03 22:00:00",
                "2005-01-03 23:00:00",
                "2005-01-04 00:00:00",
                "2005-01-04 01:00:00",
                "2005-01-04 02:00:00",
                "2005-01-04 03:00:00",
                "2005-01-04 04:00:00",
                "2005-01-04 05:00:00",
                "2005-01-04 06:00:00",
                "2005-01-04 16:00:00",
                "2005-01-04 17:00:00",
                "2005-01-04 18:00:00",
                "2005-01-04 19:00:00",
                "2005-01-04 20:00:00",
                "2005-01-04 21:00:00",
                "2005-01-04 22:00:00",
                "2005-01-04 23:00:00",
                "2005-01-05 00:00:00",
                "2005-01-05 01:00:00",
                "2005-01-05 02:00:00",
                "2005-01-05 03:00:00",
                "2005-01-05 04:00:00",
                "2005-01-05 05:00:00",
                "2005-01-05 06:00:00",
                "2005-01-05 16:00:00",
                "2005-01-05 17:00:00",
                "2005-01-05 18:00:00",
                "2005-01-05 19:00:00",
                "2005-01-05 20:00:00",
                "2005-01-05 21:00:00",
                "2005-01-05 22:00:00",
                "2005-01-05 23:00:00",
            ]
        )

        assert dtindex.equals(mask[0:75])

    def test_extreme(self, model_national):
        data = model_national._model_data_original.copy()
        mask = masks.extreme(
            data, "csp", var="resource", how="max", length="2D", n=1, padding="2H"
        )

        dtindex = pd.DatetimeIndex(
            [
                "2005-01-18 22:00:00",
                "2005-01-18 23:00:00",
                "2005-01-19 00:00:00",
                "2005-01-19 01:00:00",
                "2005-01-19 02:00:00",
                "2005-01-19 03:00:00",
                "2005-01-19 04:00:00",
                "2005-01-19 05:00:00",
                "2005-01-19 06:00:00",
                "2005-01-19 07:00:00",
                "2005-01-19 08:00:00",
                "2005-01-19 09:00:00",
                "2005-01-19 10:00:00",
                "2005-01-19 11:00:00",
                "2005-01-19 12:00:00",
                "2005-01-19 13:00:00",
                "2005-01-19 14:00:00",
                "2005-01-19 15:00:00",
                "2005-01-19 16:00:00",
                "2005-01-19 17:00:00",
                "2005-01-19 18:00:00",
                "2005-01-19 19:00:00",
                "2005-01-19 20:00:00",
                "2005-01-19 21:00:00",
                "2005-01-19 22:00:00",
                "2005-01-19 23:00:00",
                "2005-01-20 00:00:00",
                "2005-01-20 01:00:00",
                "2005-01-20 02:00:00",
                "2005-01-20 03:00:00",
                "2005-01-20 04:00:00",
                "2005-01-20 05:00:00",
                "2005-01-20 06:00:00",
                "2005-01-20 07:00:00",
                "2005-01-20 08:00:00",
                "2005-01-20 09:00:00",
                "2005-01-20 10:00:00",
                "2005-01-20 11:00:00",
                "2005-01-20 12:00:00",
                "2005-01-20 13:00:00",
                "2005-01-20 14:00:00",
                "2005-01-20 15:00:00",
                "2005-01-20 16:00:00",
                "2005-01-20 17:00:00",
                "2005-01-20 18:00:00",
                "2005-01-20 19:00:00",
                "2005-01-20 20:00:00",
                "2005-01-20 21:00:00",
                "2005-01-20 22:00:00",
                "2005-01-20 23:00:00",
                "2005-01-21 00:00:00",
                "2005-01-21 01:00:00",
            ]
        )

        assert dtindex.equals(mask)

    def test_extreme_diff_and_normalize(self, model_urban):
        data = model_urban._model_data_original.copy()
        mask = masks.extreme_diff(
            data,
            "demand_heat",
            "demand_electricity",
            var="resource",
            how="min",
            length="1D",
            n=2,
            normalize=True,
        )

        dtindex = pd.DatetimeIndex(
            [
                "2005-01-13 00:00:00",
                "2005-01-13 01:00:00",
                "2005-01-13 02:00:00",
                "2005-01-13 03:00:00",
                "2005-01-13 04:00:00",
                "2005-01-13 05:00:00",
                "2005-01-13 06:00:00",
                "2005-01-13 07:00:00",
                "2005-01-13 08:00:00",
                "2005-01-13 09:00:00",
                "2005-01-13 10:00:00",
                "2005-01-13 11:00:00",
                "2005-01-13 12:00:00",
                "2005-01-13 13:00:00",
                "2005-01-13 14:00:00",
                "2005-01-13 15:00:00",
                "2005-01-13 16:00:00",
                "2005-01-13 17:00:00",
                "2005-01-13 18:00:00",
                "2005-01-13 19:00:00",
                "2005-01-13 20:00:00",
                "2005-01-13 21:00:00",
                "2005-01-13 22:00:00",
                "2005-01-13 23:00:00",
                "2005-01-19 00:00:00",
                "2005-01-19 01:00:00",
                "2005-01-19 02:00:00",
                "2005-01-19 03:00:00",
                "2005-01-19 04:00:00",
                "2005-01-19 05:00:00",
                "2005-01-19 06:00:00",
                "2005-01-19 07:00:00",
                "2005-01-19 08:00:00",
                "2005-01-19 09:00:00",
                "2005-01-19 10:00:00",
                "2005-01-19 11:00:00",
                "2005-01-19 12:00:00",
                "2005-01-19 13:00:00",
                "2005-01-19 14:00:00",
                "2005-01-19 15:00:00",
                "2005-01-19 16:00:00",
                "2005-01-19 17:00:00",
                "2005-01-19 18:00:00",
                "2005-01-19 19:00:00",
                "2005-01-19 20:00:00",
                "2005-01-19 21:00:00",
                "2005-01-19 22:00:00",
                "2005-01-19 23:00:00",
            ]
        )

        assert dtindex.equals(mask)

    def test_extreme_week_1d(self, model_national):
        data = model_national._model_data_original.copy()
        mask = masks.extreme(
            data,
            "csp",
            var="resource",
            how="max",
            length="1D",
            n=1,
            padding="calendar_week",
        )

        found_days = list(mask.dayofyear.unique())
        days = [18, 19, 20, 21, 22, 23, 24]

        assert days == found_days

    def test_extreme_week_2d(self, model_national):
        data = model_national._model_data_original.copy()
        with pytest.raises(ValueError):
            mask = masks.extreme(
                data,
                "csp",
                var="resource",
                how="max",
                length="2D",
                n=1,
                padding="calendar_week",
            )

    def test_15min_masking_1D(self):
        # The data is identical for '2005-01-01' and '2005-01-03' timesteps,
        # it is only different for '2005-01-02'
        override = {
            "techs.test_demand_elec.constraints.resource": "file=demand_elec_15mins.csv",
            "model.subset_time": None,
        }

        model = build_test_model(override, scenario="simple_supply,one_day")
        data = model._model_data

        mask = masks.extreme(
            data, "test_demand_elec", var="resource", how="max", length="1D"
        )

        dtindex = pd.DatetimeIndex(
            [
                "2005-01-02 00:00:00",
                "2005-01-02 00:15:00",
                "2005-01-02 00:30:00",
                "2005-01-02 00:45:00",
                "2005-01-02 01:00:00",
                "2005-01-02 01:15:00",
                "2005-01-02 01:30:00",
                "2005-01-02 01:45:00",
                "2005-01-02 02:00:00",
                "2005-01-02 02:15:00",
                "2005-01-02 02:30:00",
                "2005-01-02 02:45:00",
                "2005-01-02 03:00:00",
                "2005-01-02 03:15:00",
                "2005-01-02 03:30:00",
                "2005-01-02 03:45:00",
                "2005-01-02 04:00:00",
                "2005-01-02 04:15:00",
                "2005-01-02 04:30:00",
                "2005-01-02 04:45:00",
                "2005-01-02 05:00:00",
                "2005-01-02 05:15:00",
                "2005-01-02 05:30:00",
                "2005-01-02 05:45:00",
                "2005-01-02 06:00:00",
                "2005-01-02 06:15:00",
                "2005-01-02 06:30:00",
                "2005-01-02 06:45:00",
                "2005-01-02 07:00:00",
                "2005-01-02 07:15:00",
                "2005-01-02 07:30:00",
                "2005-01-02 07:45:00",
                "2005-01-02 08:00:00",
                "2005-01-02 08:15:00",
                "2005-01-02 08:30:00",
                "2005-01-02 08:45:00",
                "2005-01-02 09:00:00",
                "2005-01-02 09:15:00",
                "2005-01-02 09:30:00",
                "2005-01-02 09:45:00",
                "2005-01-02 10:00:00",
                "2005-01-02 10:15:00",
                "2005-01-02 10:30:00",
                "2005-01-02 10:45:00",
                "2005-01-02 11:00:00",
                "2005-01-02 11:15:00",
                "2005-01-02 11:30:00",
                "2005-01-02 11:45:00",
                "2005-01-02 12:00:00",
                "2005-01-02 12:15:00",
                "2005-01-02 12:30:00",
                "2005-01-02 12:45:00",
                "2005-01-02 13:00:00",
                "2005-01-02 13:15:00",
                "2005-01-02 13:30:00",
                "2005-01-02 13:45:00",
                "2005-01-02 14:00:00",
                "2005-01-02 14:15:00",
                "2005-01-02 14:30:00",
                "2005-01-02 14:45:00",
                "2005-01-02 15:00:00",
                "2005-01-02 15:15:00",
                "2005-01-02 15:30:00",
                "2005-01-02 15:45:00",
                "2005-01-02 16:00:00",
                "2005-01-02 16:15:00",
                "2005-01-02 16:30:00",
                "2005-01-02 16:45:00",
                "2005-01-02 17:00:00",
                "2005-01-02 17:15:00",
                "2005-01-02 17:30:00",
                "2005-01-02 17:45:00",
                "2005-01-02 18:00:00",
                "2005-01-02 18:15:00",
                "2005-01-02 18:30:00",
                "2005-01-02 18:45:00",
                "2005-01-02 19:00:00",
                "2005-01-02 19:15:00",
                "2005-01-02 19:30:00",
                "2005-01-02 19:45:00",
                "2005-01-02 20:00:00",
                "2005-01-02 20:15:00",
                "2005-01-02 20:30:00",
                "2005-01-02 20:45:00",
                "2005-01-02 21:00:00",
                "2005-01-02 21:15:00",
                "2005-01-02 21:30:00",
                "2005-01-02 21:45:00",
                "2005-01-02 22:00:00",
                "2005-01-02 22:15:00",
                "2005-01-02 22:30:00",
                "2005-01-02 22:45:00",
                "2005-01-02 23:00:00",
                "2005-01-02 23:15:00",
                "2005-01-02 23:30:00",
                "2005-01-02 23:45:00",
            ]
        )

        assert dtindex.equals(mask)

    def test_15min_to_2h_masking_1D(self):
        # The data is identical for '2005-01-01' and '2005-01-03' timesteps,
        # it is only different for '2005-01-02'
        override = {
            "techs.test_demand_elec.constraints.resource": "file=demand_elec_15T_to_2h.csv",
            "model.subset_time": None,
        }

        model = build_test_model(override, scenario="simple_supply,one_day")
        data = model._model_data

        mask = masks.extreme(
            data, "test_demand_elec", var="resource", how="max", length="1D"
        )

        dtindex = pd.DatetimeIndex(
            [
                "2005-01-02 00:00:00",
                "2005-01-02 00:15:00",
                "2005-01-02 00:30:00",
                "2005-01-02 00:45:00",
                "2005-01-02 01:00:00",
                "2005-01-02 01:30:00",
                "2005-01-02 02:00:00",
                "2005-01-02 02:30:00",
                "2005-01-02 03:00:00",
                "2005-01-02 04:00:00",
                "2005-01-02 05:00:00",
                "2005-01-02 06:00:00",
                "2005-01-02 08:00:00",
                "2005-01-02 10:00:00",
                "2005-01-02 12:00:00",
                "2005-01-02 14:00:00",
                "2005-01-02 16:00:00",
                "2005-01-02 17:00:00",
                "2005-01-02 18:00:00",
                "2005-01-02 20:00:00",
                "2005-01-02 22:00:00",
            ]
        )

        assert dtindex.equals(mask)


class TestResampling:
    def test_15min_masking_1D_resampling_to_2h(self):
        # The data is identical for '2005-01-01' and '2005-01-03' timesteps,
        # it is only different for '2005-01-02'
        override = {
            "techs.test_demand_elec.constraints.resource": "file=demand_elec_15mins.csv",
            "model.subset_time": None,
            "model.time": {
                "masks": [
                    {
                        "function": "extreme",
                        "options": {"tech": "test_demand_elec", "how": "max"},
                    }
                ],
                "function": "resample",
                "function_options": {"resolution": "2H"},
            },
        }

        model = build_test_model(override, scenario="simple_supply,one_day")
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
                "2005-01-02 00:00:00",
                "2005-01-02 00:15:00",
                "2005-01-02 00:30:00",
                "2005-01-02 00:45:00",
                "2005-01-02 01:00:00",
                "2005-01-02 01:15:00",
                "2005-01-02 01:30:00",
                "2005-01-02 01:45:00",
                "2005-01-02 02:00:00",
                "2005-01-02 02:15:00",
                "2005-01-02 02:30:00",
                "2005-01-02 02:45:00",
                "2005-01-02 03:00:00",
                "2005-01-02 03:15:00",
                "2005-01-02 03:30:00",
                "2005-01-02 03:45:00",
                "2005-01-02 04:00:00",
                "2005-01-02 04:15:00",
                "2005-01-02 04:30:00",
                "2005-01-02 04:45:00",
                "2005-01-02 05:00:00",
                "2005-01-02 05:15:00",
                "2005-01-02 05:30:00",
                "2005-01-02 05:45:00",
                "2005-01-02 06:00:00",
                "2005-01-02 06:15:00",
                "2005-01-02 06:30:00",
                "2005-01-02 06:45:00",
                "2005-01-02 07:00:00",
                "2005-01-02 07:15:00",
                "2005-01-02 07:30:00",
                "2005-01-02 07:45:00",
                "2005-01-02 08:00:00",
                "2005-01-02 08:15:00",
                "2005-01-02 08:30:00",
                "2005-01-02 08:45:00",
                "2005-01-02 09:00:00",
                "2005-01-02 09:15:00",
                "2005-01-02 09:30:00",
                "2005-01-02 09:45:00",
                "2005-01-02 10:00:00",
                "2005-01-02 10:15:00",
                "2005-01-02 10:30:00",
                "2005-01-02 10:45:00",
                "2005-01-02 11:00:00",
                "2005-01-02 11:15:00",
                "2005-01-02 11:30:00",
                "2005-01-02 11:45:00",
                "2005-01-02 12:00:00",
                "2005-01-02 12:15:00",
                "2005-01-02 12:30:00",
                "2005-01-02 12:45:00",
                "2005-01-02 13:00:00",
                "2005-01-02 13:15:00",
                "2005-01-02 13:30:00",
                "2005-01-02 13:45:00",
                "2005-01-02 14:00:00",
                "2005-01-02 14:15:00",
                "2005-01-02 14:30:00",
                "2005-01-02 14:45:00",
                "2005-01-02 15:00:00",
                "2005-01-02 15:15:00",
                "2005-01-02 15:30:00",
                "2005-01-02 15:45:00",
                "2005-01-02 16:00:00",
                "2005-01-02 16:15:00",
                "2005-01-02 16:30:00",
                "2005-01-02 16:45:00",
                "2005-01-02 17:00:00",
                "2005-01-02 17:15:00",
                "2005-01-02 17:30:00",
                "2005-01-02 17:45:00",
                "2005-01-02 18:00:00",
                "2005-01-02 18:15:00",
                "2005-01-02 18:30:00",
                "2005-01-02 18:45:00",
                "2005-01-02 19:00:00",
                "2005-01-02 19:15:00",
                "2005-01-02 19:30:00",
                "2005-01-02 19:45:00",
                "2005-01-02 20:00:00",
                "2005-01-02 20:15:00",
                "2005-01-02 20:30:00",
                "2005-01-02 20:45:00",
                "2005-01-02 21:00:00",
                "2005-01-02 21:15:00",
                "2005-01-02 21:30:00",
                "2005-01-02 21:45:00",
                "2005-01-02 22:00:00",
                "2005-01-02 22:15:00",
                "2005-01-02 22:30:00",
                "2005-01-02 22:45:00",
                "2005-01-02 23:00:00",
                "2005-01-02 23:15:00",
                "2005-01-02 23:30:00",
                "2005-01-02 23:45:00",
                "2005-01-03 00:00:00",
                "2005-01-03 02:00:00",
                "2005-01-03 04:00:00",
                "2005-01-03 06:00:00",
                "2005-01-03 08:00:00",
                "2005-01-03 10:00:00",
                "2005-01-03 12:00:00",
                "2005-01-03 14:00:00",
                "2005-01-03 16:00:00",
                "2005-01-03 18:00:00",
                "2005-01-03 20:00:00",
                "2005-01-03 22:00:00",
            ]
        )

        assert dtindex.equals(data.timesteps.to_index())

    def test_15min_resampling_to_6h(self):
        # The data is identical for '2005-01-01' and '2005-01-03' timesteps,
        # it is only different for '2005-01-02'
        override = {
            "techs.test_demand_elec.constraints.resource": "file=demand_elec_15mins.csv",
            "model.subset_time": None,
            "model.time": {
                "function": "resample",
                "function_options": {"resolution": "6H"},
            },
        }

        model = build_test_model(override, scenario="simple_supply,one_day")
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

    def test_15min_to_2h_resampling_to_2h(self):
        """
        CSV has daily timeseries varying from 15min to 2h resolution, resample all to 2h
        """
        override = {
            "techs.test_demand_elec.constraints.resource": "file=demand_elec_15T_to_2h.csv",
            "model.subset_time": None,
            "model.time": {
                "function": "resample",
                "function_options": {"resolution": "2H"},
            },
        }

        model = build_test_model(override, scenario="simple_supply,one_day")
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
                "2005-01-02 00:00:00",
                "2005-01-02 02:00:00",
                "2005-01-02 04:00:00",
                "2005-01-02 06:00:00",
                "2005-01-02 08:00:00",
                "2005-01-02 10:00:00",
                "2005-01-02 12:00:00",
                "2005-01-02 14:00:00",
                "2005-01-02 16:00:00",
                "2005-01-02 18:00:00",
                "2005-01-02 20:00:00",
                "2005-01-02 22:00:00",
                "2005-01-03 00:00:00",
                "2005-01-03 02:00:00",
                "2005-01-03 04:00:00",
                "2005-01-03 06:00:00",
                "2005-01-03 08:00:00",
                "2005-01-03 10:00:00",
                "2005-01-03 12:00:00",
                "2005-01-03 14:00:00",
                "2005-01-03 16:00:00",
                "2005-01-03 18:00:00",
                "2005-01-03 20:00:00",
                "2005-01-03 22:00:00",
            ]
        )

        assert dtindex.equals(data.timesteps.to_index())


class TestFuncs:
    @pytest.fixture
    def model_national(self, scope="module"):
        return calliope.examples.national_scale(
            override_dict={"model.subset_time": "2005-01"}
        )

    def test_drop_invalid_timesteps(self, model_national):
        data = model_national._model_data_original.copy()
        timesteps = ["XXX2005-01-01 23:00"]

        with pytest.raises(exceptions.ModelError):
            funcs.drop(data, timesteps)

    def test_drop(self, model_national):
        data = model_national._model_data_original.copy()
        timesteps = ["2005-01-01 23:00", "2005-01-01 22:00"]

        data_dropped = funcs.drop(data, timesteps)

        assert len(data_dropped.timesteps) == 742

        result_timesteps = list(data_dropped.coords["timesteps"].values)

        assert "2005-01-01 21:00" not in result_timesteps
        assert "2005-01-01 22:00" not in result_timesteps


class TestLoadTimeseries:
    def test_invalid_csv_columns(self):
        override = {
            "locations": {
                "2.techs": {"test_supply_elec": None, "test_demand_elec": None},
                "3.techs": {"test_supply_elec": None, "test_demand_elec": None},
            },
            "links": {
                "0,1": {"exists": False},
                "2,3.techs": {"test_transmission_elec": None},
            },
        }
        with pytest.raises(exceptions.ModelError) as excinfo:
            build_test_model(override_dict=override, scenario="one_day")

        assert check_error_or_warning(
            excinfo,
            [
                "column `2` not found in dataframe `demand_elec.csv`, but was requested by loc::tech `2::test_demand_elec`.",
                "column `3` not found in dataframe `demand_elec.csv`, but was requested by loc::tech `3::test_demand_elec`.",
            ],
        )
