import pandas as pd
import pytest  # pylint: disable=unused-import

import calliope
from calliope import exceptions
from calliope.core.time import clustering, funcs, masks


class TestClustering:
    @pytest.fixture
    def model_national(self, scope='module'):
        return calliope.examples.national_scale(
            override_dict={
                'model.random_seed': 23,
                'model.subset_time': ['2005-01-01', '2005-03-31']
            }
        )

        # FIXME

    def test_kmeans_mean(self, model_national):
        data = model_national._model_data

        data_clustered = funcs.apply_clustering(
            data,
            timesteps=None,
            clustering_func='get_clusters_kmeans',
            how='mean',
            normalize=True,
            k=5
        )

        # FIXME

    def test_kmeans_closest(self, model_national):
        data = model_national._model_data

        data_clustered = funcs.apply_clustering(
            data,
            timesteps=None,
            clustering_func='get_clusters_kmeans',
            how='closest',
            normalize=True,
            k=5
        )

    def test_hierarchical_mean(self, model_national):
        data = model_national._model_data

        data_clustered = funcs.apply_clustering(
            data,
            timesteps=None,
            clustering_func='get_clusters_kmeans',
            how='closest',
            normalize=True,
            k=5
        )

        # FIXME

    def test_hierarchical_closest(self, model_national):
        data = model_national._model_data

        data_clustered = funcs.apply_clustering(
            data,
            timesteps=None,
            clustering_func='get_clusters_kmeans',
            how='closest',
            normalize=True,
            k=5
        )

        # FIXME

    def test_hartigans_rule(self, model_national):
        data = model_national._model_data

        with pytest.warns(exceptions.ModelWarning) as excinfo:
            funcs.apply_clustering(
                data,
                timesteps=None,
                clustering_func='get_clusters_kmeans',
                how='mean',
                normalize=True
            )

        all_warnings = ','.join(str(excinfo.list[i]) for i in range(len(excinfo.list)))

        assert '5 is a good number of clusters' in all_warnings


class TestMasks:
    @pytest.fixture
    def model_national(self, scope='module'):
        return calliope.examples.national_scale(
            override_dict={'model.subset_time': ['2005-01-01', '2005-01-31']}
        )

    @pytest.fixture
    def model_urban(self, scope='module'):
        return calliope.examples.urban_scale(
            override_dict={'model.subset_time': ['2005-01-01', '2005-01-31']}
        )

    def test_zero(self, model_national):
        data = model_national._model_data_original.copy()
        mask = masks.zero(data, 'csp', var='resource')

        dtindex = pd.DatetimeIndex([
            '2005-01-01 00:00:00', '2005-01-01 01:00:00',
            '2005-01-01 02:00:00', '2005-01-01 03:00:00',
            '2005-01-01 04:00:00', '2005-01-01 05:00:00',
            '2005-01-01 06:00:00', '2005-01-01 16:00:00',
            '2005-01-01 17:00:00', '2005-01-01 18:00:00',
            '2005-01-01 19:00:00', '2005-01-01 20:00:00',
            '2005-01-01 21:00:00', '2005-01-01 22:00:00',
            '2005-01-01 23:00:00', '2005-01-02 00:00:00',
            '2005-01-02 01:00:00', '2005-01-02 02:00:00',
            '2005-01-02 03:00:00', '2005-01-02 04:00:00',
            '2005-01-02 05:00:00', '2005-01-02 06:00:00',
            '2005-01-02 16:00:00', '2005-01-02 17:00:00',
            '2005-01-02 18:00:00', '2005-01-02 19:00:00',
            '2005-01-02 20:00:00', '2005-01-02 21:00:00',
            '2005-01-02 22:00:00', '2005-01-02 23:00:00',
            '2005-01-03 00:00:00', '2005-01-03 01:00:00',
            '2005-01-03 02:00:00', '2005-01-03 03:00:00',
            '2005-01-03 04:00:00', '2005-01-03 05:00:00',
            '2005-01-03 06:00:00', '2005-01-03 16:00:00',
            '2005-01-03 17:00:00', '2005-01-03 18:00:00',
            '2005-01-03 19:00:00', '2005-01-03 20:00:00',
            '2005-01-03 21:00:00', '2005-01-03 22:00:00',
            '2005-01-03 23:00:00', '2005-01-04 00:00:00',
            '2005-01-04 01:00:00', '2005-01-04 02:00:00',
            '2005-01-04 03:00:00', '2005-01-04 04:00:00',
            '2005-01-04 05:00:00', '2005-01-04 06:00:00',
            '2005-01-04 16:00:00', '2005-01-04 17:00:00',
            '2005-01-04 18:00:00', '2005-01-04 19:00:00',
            '2005-01-04 20:00:00', '2005-01-04 21:00:00',
            '2005-01-04 22:00:00', '2005-01-04 23:00:00',
            '2005-01-05 00:00:00', '2005-01-05 01:00:00',
            '2005-01-05 02:00:00', '2005-01-05 03:00:00',
            '2005-01-05 04:00:00', '2005-01-05 05:00:00',
            '2005-01-05 06:00:00', '2005-01-05 16:00:00',
            '2005-01-05 17:00:00', '2005-01-05 18:00:00',
            '2005-01-05 19:00:00', '2005-01-05 20:00:00',
            '2005-01-05 21:00:00', '2005-01-05 22:00:00',
            '2005-01-05 23:00:00'])

        assert dtindex.equals(mask[0:75])

    def test_extreme(self, model_national):
        data = model_national._model_data_original.copy()
        mask = masks.extreme(
            data, 'csp', var='resource', how='max',
            length='2D', n=1, padding='2H'
        )

        dtindex = pd.DatetimeIndex([
            '2005-01-18 22:00:00', '2005-01-18 23:00:00',
            '2005-01-19 00:00:00', '2005-01-19 01:00:00',
            '2005-01-19 02:00:00', '2005-01-19 03:00:00',
            '2005-01-19 04:00:00', '2005-01-19 05:00:00',
            '2005-01-19 06:00:00', '2005-01-19 07:00:00',
            '2005-01-19 08:00:00', '2005-01-19 09:00:00',
            '2005-01-19 10:00:00', '2005-01-19 11:00:00',
            '2005-01-19 12:00:00', '2005-01-19 13:00:00',
            '2005-01-19 14:00:00', '2005-01-19 15:00:00',
            '2005-01-19 16:00:00', '2005-01-19 17:00:00',
            '2005-01-19 18:00:00', '2005-01-19 19:00:00',
            '2005-01-19 20:00:00', '2005-01-19 21:00:00',
            '2005-01-19 22:00:00', '2005-01-19 23:00:00',
            '2005-01-20 00:00:00', '2005-01-20 01:00:00',
            '2005-01-20 02:00:00', '2005-01-20 03:00:00',
            '2005-01-20 04:00:00', '2005-01-20 05:00:00',
            '2005-01-20 06:00:00', '2005-01-20 07:00:00',
            '2005-01-20 08:00:00', '2005-01-20 09:00:00',
            '2005-01-20 10:00:00', '2005-01-20 11:00:00',
            '2005-01-20 12:00:00', '2005-01-20 13:00:00',
            '2005-01-20 14:00:00', '2005-01-20 15:00:00',
            '2005-01-20 16:00:00', '2005-01-20 17:00:00',
            '2005-01-20 18:00:00', '2005-01-20 19:00:00',
            '2005-01-20 20:00:00', '2005-01-20 21:00:00',
            '2005-01-20 22:00:00', '2005-01-20 23:00:00',
            '2005-01-21 00:00:00', '2005-01-21 01:00:00'])

        assert dtindex.equals(mask)

    def test_extreme_diff_and_normalize(self, model_urban):
        data = model_urban._model_data_original.copy()
        mask = masks.extreme_diff(
            data, 'demand_heat', 'demand_electricity',
            var='resource', how='min',
            length='1D', n=2, normalize=True
        )

        dtindex = pd.DatetimeIndex([
            '2005-01-13 00:00:00', '2005-01-13 01:00:00',
            '2005-01-13 02:00:00', '2005-01-13 03:00:00',
            '2005-01-13 04:00:00', '2005-01-13 05:00:00',
            '2005-01-13 06:00:00', '2005-01-13 07:00:00',
            '2005-01-13 08:00:00', '2005-01-13 09:00:00',
            '2005-01-13 10:00:00', '2005-01-13 11:00:00',
            '2005-01-13 12:00:00', '2005-01-13 13:00:00',
            '2005-01-13 14:00:00', '2005-01-13 15:00:00',
            '2005-01-13 16:00:00', '2005-01-13 17:00:00',
            '2005-01-13 18:00:00', '2005-01-13 19:00:00',
            '2005-01-13 20:00:00', '2005-01-13 21:00:00',
            '2005-01-13 22:00:00', '2005-01-13 23:00:00',
            '2005-01-19 00:00:00', '2005-01-19 01:00:00',
            '2005-01-19 02:00:00', '2005-01-19 03:00:00',
            '2005-01-19 04:00:00', '2005-01-19 05:00:00',
            '2005-01-19 06:00:00', '2005-01-19 07:00:00',
            '2005-01-19 08:00:00', '2005-01-19 09:00:00',
            '2005-01-19 10:00:00', '2005-01-19 11:00:00',
            '2005-01-19 12:00:00', '2005-01-19 13:00:00',
            '2005-01-19 14:00:00', '2005-01-19 15:00:00',
            '2005-01-19 16:00:00', '2005-01-19 17:00:00',
            '2005-01-19 18:00:00', '2005-01-19 19:00:00',
            '2005-01-19 20:00:00', '2005-01-19 21:00:00',
            '2005-01-19 22:00:00', '2005-01-19 23:00:00'])

        assert dtindex.equals(mask)

    def test_extreme_week_1d(self, model_national):
        data = model_national._model_data_original.copy()
        mask = masks.extreme(
            data, 'csp', var='resource', how='max',
            length='1D', n=1, padding='calendar_week'
        )

        found_days = list(mask.dayofyear.unique())
        days = [18, 19, 20, 21, 22, 23, 24]

        assert days == found_days

    def test_extreme_week_2d(self, model_national):
        data = model_national._model_data_original.copy()
        with pytest.raises(ValueError):
            mask = masks.extreme(
                data, 'csp', var='resource', how='max',
                length='2D', n=1, padding='calendar_week'
            )


class TestFuncs:
    @pytest.fixture
    def model_national(self, scope='module'):
        return calliope.examples.national_scale(
            override_dict={
                'model.subset_time': '2005-01'
            }
        )

    def test_drop_invalid_timesteps(self, model_national):
        data = model_national._model_data_original.copy()
        timesteps = ['XXX2005-01-01 23:00']

        with pytest.raises(exceptions.ModelError):
            funcs.drop(data, timesteps)

    def test_drop(self, model_national):
        data = model_national._model_data_original.copy()
        timesteps = ['2005-01-01 23:00', '2005-01-01 22:00']

        data_dropped = funcs.drop(data, timesteps)

        assert len(data_dropped.timesteps) == 742

        result_timesteps = list(data_dropped.coords['timesteps'].values)

        assert '2005-01-01 21:00' not in result_timesteps
        assert '2005-01-01 22:00' not in result_timesteps
