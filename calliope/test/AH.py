import pytest
import os

import pandas as pd
import numpy as np

import calliope
import calliope.exceptions as exceptions
from calliope.core.attrdict import AttrDict
from calliope.preprocess import time

from calliope.test.common.util import build_test_model as build_model
from calliope.test.common.util import constraint_sets, defaults, check_error_or_warning


class TestAH:
    def dev_test(self):

        model_dir = "../example_models/national_scale/"
        timeseries_dir = os.path.join(model_dir, "timeseries_data")

        csp_resource = pd.read_csv(
            os.path.join(timeseries_dir, "csp_resource.csv"), index_col=0
        )
        demand_1 = pd.read_csv(
            os.path.join(timeseries_dir, "demand-1.csv"), index_col=0
        )
        demand_2 = pd.read_csv(
            os.path.join(timeseries_dir, "demand-2.csv"), index_col=0
        )

        timeseries_dataframes = {
            "csp_resource": csp_resource,
            "demand_1": demand_1,
            "demand_2": demand_2,
        }
        timeseries_dataframes_mixed = {"demand_1": demand_1, "demand_2": demand_2}

        override_dict_csv = {
            "techs.csp.constraints.resource": "file=csp_resource.csv",
            "locations.region1.techs.demand_power.constraints.resource": (
                "file=demand-1.csv:demand"
            ),
            "locations.region2.techs.demand_power.constraints.resource": (
                "file=demand-2.csv:demand"
            ),
        }

        override_dict_df = {
            "techs.csp.constraints.resource": "df=csp_resource",
            "locations.region1.techs.demand_power.constraints.resource": (
                "df=demand_1:demand"
            ),
            "locations.region2.techs.demand_power.constraints.resource": (
                "df=demand_2:demand"
            ),
        }

        override_dict_mixed = {
            "techs.csp.constraints.resource": "file=csp_resource.csv",
            "locations.region1.techs.demand_power.constraints.resource": (
                "df=demand_1:demand"
            ),
            "locations.region2.techs.demand_power.constraints.resource": (
                "df=demand_2:demand"
            ),
        }

        # import sys
        # if sys.argv[1] == 'csv':
        #     override_dict = override_dict_csv
        #     timeseries_dataframes = None
        # elif sys.argv[1] == 'df':
        #     override_dict = override_dict_df
        #     timeseries_dataframes = timeseries_dataframes
        # else:
        #     raise NotImplementedError

        mode = "mixed"

        if mode == "csv":
            override_dict = override_dict_csv
            timeseries_dataframes_arg = None
        if mode == "df":
            override_dict = override_dict_df
            timeseries_dataframes_arg = timeseries_dataframes
        if mode == "mixed":
            override_dict = override_dict_mixed
            timeseries_dataframes_arg = timeseries_dataframes_mixed

        # with pytest.raises(exceptions.ModelError) as error:
        #     model = calliope.Model(
        #         os.path.join(model_dir, 'model.yaml'),
        #         override_dict=override_dict,
        #         timeseries_dataframes=timeseries_dataframes)

        model = calliope.Model(
            os.path.join(model_dir, "model.yaml"),
            override_dict=override_dict,
            timeseries_dataframes=timeseries_dataframes_arg,
        )


if __name__ == "__main__":
    test = TestAH()
    test.dev_test()
