import numpy as np
import pytest  # pylint: disable=unused-import

import calliope
from calliope.core.preprocess import time


class TestTime:
    @pytest.fixture
    def model(self):
        return calliope.examples.urban_scale(
            override_dict={'model.subset_time': ['2005-01-01', '2005-01-10']}
        )

    def test_add_max_demand_timesteps(self, model):
        data = model._model_data_original.copy()
        data = time.add_max_demand_timesteps(data)

        assert (
            data['max_demand_timesteps'].loc[dict(carriers='heat')].values ==
            np.datetime64('2005-01-05T07:00:00')
        )

        assert (
            data['max_demand_timesteps'].loc[dict(carriers='electricity')].values ==
            np.datetime64('2005-01-10T09:00:00')
        )
