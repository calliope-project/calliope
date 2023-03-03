import pytest  # noqa: F401
import calliope
import logging
import datetime
import os
import tempfile

import xarray as xr

from calliope.core.util import dataset

from calliope.core.util.tools import memoize, memoize_instancemethod

from calliope.core.util.logging import log_time
from calliope.core.util.generate_runs import generate_runs
from calliope.test.common.util import (
    python36_or_higher,
    check_error_or_warning,
)

_MODEL_NATIONAL = os.path.join(
    os.path.dirname(__file__), "..", "example_models", "national_scale", "model.yaml"
)

_MODEL_URBAN = os.path.join(
    os.path.dirname(__file__), "..", "example_models", "urban_scale", "model.yaml"
)


class TestDataset:
    @pytest.fixture()
    def example_dataarray(self):
        return xr.DataArray(
            [
                [[[0], [1], [2]], [[3], [4], [5]], [[6], [7], [8]]],
                [[[0], [1], [2]], [[3], [4], [5]], [[6], [7], [8]]],
            ],
            dims=("timesteps", "nodes", "techs", "costs"),
            coords={
                "timesteps": ["foo", "bar"],
                "nodes": ["a", "b", "c"],
                "techs": ["foo", "bar", "baz"],
                "costs": ["foo"],
            },
        )

    @pytest.fixture()
    def example_one_dim_dataarray(self):
        return xr.DataArray(
            [0, 1, 2], dims=("timesteps"), coords={"timesteps": ["foo", "bar", "baz"]}
        )

    @pytest.fixture()
    def example_dataset(self, example_dataarray):
        return xr.Dataset(
            {"foo": example_dataarray, "bar": example_dataarray.squeeze()}
        )

    def test_reorganise_dataset_dimensions(self, example_dataset):
        reorganised_dataset = dataset.reorganise_xarray_dimensions(example_dataset)
        dataset_dims = [i for i in reorganised_dataset.dims.keys()]
        assert dataset_dims == ["costs", "nodes", "techs", "timesteps"]

    def test_reorganise_dataarray_dimensions(self, example_dataarray):
        reorganised_dataset = dataset.reorganise_xarray_dimensions(example_dataarray)
        assert reorganised_dataset.dims == ("costs", "nodes", "techs", "timesteps")

    def test_fail_reorganise_dimensions(self):
        with pytest.raises(TypeError) as excinfo:
            dataset.reorganise_xarray_dimensions(
                ["timesteps", "nodes", "techs", "costs"]
            )
        assert check_error_or_warning(
            excinfo, "Must provide either xarray Dataset or DataArray to be reorganised"
        )


class TestMemoization:
    @memoize_instancemethod
    def instance_method(self, a, b):
        return a + b

    def test_memoize_one_arg(self):
        @memoize
        def test(a):
            return a + 1

        assert test(1) == 2
        assert test(1) == 2

    def test_memoize_two_args(self):
        @memoize
        def test(a, b):
            return a + b

        assert test(1, 2) == 3
        assert test(1, 2) == 3

    def test_memoize_instancemethod(self):
        assert self.instance_method(1, 2) == 3
        assert self.instance_method(1, 2) == 3


class TestLogging:
    def test_set_log_verbosity(self):
        calliope.set_log_verbosity("CRITICAL", include_solver_output=True)

        assert logging.getLogger("calliope").getEffectiveLevel() == 50
        assert logging.getLogger("py.warnings").getEffectiveLevel() == 50
        assert (
            logging.getLogger("calliope.backend.pyomo.model").getEffectiveLevel() == 10
        )

        calliope.set_log_verbosity("CRITICAL", include_solver_output=False)

        assert logging.getLogger("calliope").getEffectiveLevel() == 50
        assert logging.getLogger("py.warnings").getEffectiveLevel() == 50
        assert (
            logging.getLogger("calliope.backend.pyomo.model").getEffectiveLevel() == 50
        )

        calliope.set_log_verbosity()

        assert logging.getLogger("calliope").getEffectiveLevel() == 20
        assert logging.getLogger("py.warnings").getEffectiveLevel() == 20
        assert (
            logging.getLogger("calliope.backend.pyomo.model").getEffectiveLevel() == 10
        )

    def test_timing_log(self):
        timings = {"model_creation": datetime.datetime.now()}
        logger = logging.getLogger("calliope.testlogger")

        # TODO: capture logging output and check that comment is in string
        log_time(logger, timings, "test", comment="test_comment", level="info")
        assert isinstance(timings["test"], datetime.datetime)

        log_time(logger, timings, "test2", comment=None, level="info")
        assert isinstance(timings["test2"], datetime.datetime)

        # TODO: capture logging output and check that time_since_run_start is in the string
        log_time(
            logger,
            timings,
            "test",
            comment=None,
            level="info",
            time_since_run_start=True,
        )


class TestGenerateRuns:
    @python36_or_higher
    def test_generate_runs_scenarios(self):
        runs = generate_runs(
            _MODEL_NATIONAL, scenarios="time_resampling;profiling;time_clustering"
        )
        assert len(runs) == 3
        assert runs[0].endswith(
            "--scenario time_resampling --save_netcdf out_1_time_resampling.nc"
        )

    @python36_or_higher
    def test_generate_runs_scenarios_none_with_scenarios(self):
        runs = generate_runs(_MODEL_NATIONAL, scenarios=None)
        assert len(runs) == 2
        assert runs[0].endswith(
            "--scenario cold_fusion_with_production_share --save_netcdf out_1_cold_fusion_with_production_share.nc"
        )

    @python36_or_higher
    def test_generate_runs_scenarios_none_with_overrides(self):
        runs = generate_runs(
            _MODEL_URBAN,
            scenarios=None,
        )
        assert len(runs) == 4
        assert runs[0].endswith("--scenario milp --save_netcdf out_1_milp.nc")


class TestPandasExport:
    @pytest.fixture(scope="module")
    def model(self):
        return calliope.examples.national_scale()

    @pytest.mark.parametrize(
        "variable_name",
        sorted(
            [i for i in calliope.examples.national_scale()._model_data.data_vars.keys()]
        ),
    )
    def test_data_variables_can_be_exported_to_pandas(self, model, variable_name):
        model.inputs[variable_name].to_dataframe()
