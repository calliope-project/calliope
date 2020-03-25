import pytest
import os

import pandas as pd
import numpy as np

import calliope
import calliope.exceptions as exceptions
from calliope.core.attrdict import AttrDict
from calliope.core.preprocess import time

from calliope.test.common.util import build_test_model as build_model
from calliope.test.common.util import \
    constraint_sets, defaults, check_error_or_warning


class TestTimeSeriesFromDataFrames:
    def __init__(self):
        self.model_dir = '../example_models/national_scale/'
        self.timeseries_dir = os.path.join(self.model_dir, 'timeseries_data')
    
    def build_national_scale_example_model(self, override_dict=None,
                                           timeseries_dataframes=None):
        model = calliope.Model(os.path.join(self.model_dir, 'model.yaml'),
                               override_dict=override_dict,
                               timeseries_dataframes=timeseries_dataframes)
        return model

    def get_timeseries_dataframes(self):
        csp_resource = pd.read_csv(os.path.join(self.timeseries_dir,
                                                'csp_resource.csv'),
                                   index_col=0)
        demand_1 = pd.read_csv(os.path.join(self.timeseries_dir, 'demand-1.csv'),
                               index_col=0)
        demand_2 = pd.read_csv(os.path.join(self.timeseries_dir, 'demand-2.csv'),
                               index_col=0)
        timeseries_dataframes = {'csp_resource': csp_resource,
                                 'demand_1': demand_1,
                                 'demand_2': demand_2}
        return timeseries_dataframes

    def test_warning_timeseries_path_dataframes():
        """
        Calliope should give a warning when all timeseries are loaded via
        dataframes but a timeseries path is still specified.
        """

        #### TODO: Make test
        pass

    def test_dataframes_passed(self):
        """
        If model config specifies dataframes to be loaded in (via df=...),
        these time series must be passed as arguments in calliope.Model(...).
        """
        
        override_dict = {
            'techs.csp.constraints.resource': 'df=csp_resource',
            'locations.region1.techs.demand_power.constraints.resource': 'df=demand_1:demand',
            'locations.region2.techs.demand_power.constraints.resource': 'df=demand_2:demand'
        }
        with pytest.raises(exceptions.ModelError) as error:
            model = self.build_national_scale_example_model(override_dict=override_dict)
        assert check_error_or_warning(error, 'no timeseries passed '
                                      'as arguments in calliope.Model(...).')

    def test_no_dataframes_if_read_csv(self):
        """
        If model config specifies dataframes to be read from csv (via file=...),
        no time series should be passed as arguments in calliope.Model(...).
        """
        
        timeseries_dataframes = self.get_timeseries_dataframes()
        with pytest.raises(exceptions.ModelError) as error:
            model = self.build_national_scale_example_model(
                timeseries_dataframes=timeseries_dataframes
            )
            
        assert check_error_or_warning(
            error, 'Either load all timeseries from `timeseries_dataframes` and df=..., '
            'or set `timeseries_dataframes=None` and load load all from CSV files'
        )
        
        



if __name__ == '__main__':
    test = TestAH()
    test.dev_test()
