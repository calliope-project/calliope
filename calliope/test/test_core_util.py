import pytest  # pylint: disable=unused-import
import calliope
import logging
import datetime
import os
import tempfile

import xarray as xr
import pandas as pd

from calliope.core.util import dataset, observed_dict

from calliope.core.util.tools import memoize, memoize_instancemethod

from calliope import exceptions
from calliope.core.util.logging import log_time
from calliope.core.util.generate_runs import generate_runs
from calliope.test.common.util import (
    python36_or_higher,
    check_error_or_warning,
    build_test_model,
)

_MODEL_NATIONAL = os.path.join(
    os.path.dirname(__file__), "..", "example_models", "national_scale", "model.yaml"
)

_MODEL_URBAN = os.path.join(
    os.path.dirname(__file__), "..", "example_models", "urban_scale", "model.yaml"
)


class TestDataset:
    @pytest.fixture()
    def loc_techs(self):
        loc_techs = [
            "region1-3::csp",
            "region1::demand_power",
            "region1-1::csp",
            "region1-2::csp",
            "region2::demand_power",
            "region1-1::demand_power",
        ]
        return loc_techs

    @pytest.fixture()
    def example_dataarray(self):
        return xr.DataArray(
            [[[0], [1], [2]], [[3], [4], [5]]],
            dims=("timesteps", "loc_techs_bar", "costs"),
            coords={
                "timesteps": ["foo", "bar"],
                "loc_techs_bar": ["1::foo", "2::bar", "3::baz"],
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

    def test_get_loc_techs_tech(self, loc_techs):
        loc_techs = dataset.get_loc_techs(loc_techs, tech="csp")
        assert loc_techs == ["region1-3::csp", "region1-1::csp", "region1-2::csp"]

    def test_get_loc_techs_loc(self, loc_techs):
        loc_techs = dataset.get_loc_techs(loc_techs, loc="region1-1")
        assert loc_techs == ["region1-1::csp", "region1-1::demand_power"]

    def test_get_loc_techs_loc_and_tech(self, loc_techs):
        loc_techs = dataset.get_loc_techs(loc_techs, tech="demand_power", loc="region1")
        assert loc_techs == ["region1::demand_power"]

    def test_split_loc_tech_to_dataarray(self, example_dataarray):
        formatted_array = dataset.split_loc_techs(example_dataarray)
        assert isinstance(formatted_array, xr.DataArray)
        assert formatted_array.dims == ("costs", "locs", "techs", "timesteps")

    def test_split_loc_tech_to_series(self, example_dataarray):
        formatted_series = dataset.split_loc_techs(
            example_dataarray, return_as="Series"
        )
        assert isinstance(formatted_series, pd.Series)
        assert formatted_series.index.names == ["costs", "locs", "techs", "timesteps"]

    def test_split_loc_tech_to_multiindex_dataarray(self, example_dataarray):
        formatted_array = dataset.split_loc_techs(
            example_dataarray, return_as="MultiIndex DataArray"
        )
        assert isinstance(formatted_array, xr.DataArray)
        assert formatted_array.dims == ("timesteps", "loc_techs_bar", "costs")
        assert isinstance(formatted_array.loc_techs_bar.to_index(), pd.MultiIndex)

    def test_split_loc_tech_too_many_loc_tech_dims(self, example_dataarray):
        _array = example_dataarray.rename({"costs": "loc_techs_2"})
        with pytest.raises(exceptions.ModelError) as excinfo:
            dataset.split_loc_techs(_array)
        assert check_error_or_warning(
            excinfo, "Cannot split loc_techs or loc_tech_carriers dimension"
        )

    def test_split_loc_tech_one_dim_to_dataarray(self, example_one_dim_dataarray):
        formatted_array = dataset.split_loc_techs(example_one_dim_dataarray)
        assert isinstance(formatted_array, xr.DataArray)
        assert formatted_array.dims == ("timesteps",)

    def test_split_loc_tech_one_dim_to_series(self, example_one_dim_dataarray):
        formatted_series = dataset.split_loc_techs(
            example_one_dim_dataarray, return_as="Series"
        )
        assert isinstance(formatted_series, pd.Series)
        assert formatted_series.index.names == ["timesteps"]

    def test_split_loc_tech_one_dim_to_multiindex_dataarray(
        self, example_one_dim_dataarray
    ):
        formatted_array = dataset.split_loc_techs(
            example_one_dim_dataarray, return_as="MultiIndex DataArray"
        )
        assert isinstance(formatted_array, xr.DataArray)
        assert formatted_array.dims == ("timesteps",)

    def test_split_loc_tech_unknown_output(
        self, example_dataarray, example_one_dim_dataarray
    ):
        for array in [example_dataarray, example_one_dim_dataarray]:
            with pytest.raises(ValueError) as excinfo:
                dataset.split_loc_techs(array, return_as="foo")
            assert check_error_or_warning(
                excinfo,
                "`return_as` must be `DataArray`, `Series`, or `MultiIndex DataArray`",
            )

    def test_reorganise_dataset_dimensions(self, example_dataset):
        reorganised_dataset = dataset.reorganise_xarray_dimensions(example_dataset)
        dataset_dims = [i for i in reorganised_dataset.dims.keys()]
        assert dataset_dims == ["costs", "loc_techs_bar", "timesteps"]

    def test_reorganise_dataarray_dimensions(self, example_dataarray):
        reorganised_dataset = dataset.reorganise_xarray_dimensions(example_dataarray)
        assert reorganised_dataset.dims == ("costs", "loc_techs_bar", "timesteps")

    def test_fail_reorganise_dimensions(self):
        with pytest.raises(TypeError) as excinfo:
            dataset.reorganise_xarray_dimensions(
                ["timesteps", "loc_techs_bar", "costs"]
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
            "--scenario time_resampling --save_netcdf out_1_time_resampling.nc --save_plots plots_1_time_resampling.html"
        )

    @python36_or_higher
    def test_generate_runs_scenarios_none_with_scenarios(self):
        runs = generate_runs(_MODEL_NATIONAL, scenarios=None)
        assert len(runs) == 2
        assert runs[0].endswith(
            "--scenario cold_fusion_with_production_share --save_netcdf out_1_cold_fusion_with_production_share.nc --save_plots plots_1_cold_fusion_with_production_share.html"
        )

    @python36_or_higher
    def test_generate_runs_scenarios_none_with_overrides(self):
        runs = generate_runs(
            _MODEL_URBAN,
            scenarios=None,
        )
        assert len(runs) == 4
        assert runs[0].endswith(
            "--scenario milp --save_netcdf out_1_milp.nc --save_plots plots_1_milp.html"
        )


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
        model.get_formatted_array(variable_name).to_dataframe()


class TestObservedDict:
    def as_yaml(self, _dict, strip=False):
        if strip is True:
            _dict = {
                k: v
                for k, v in _dict.items()
                if (not isinstance(v, dict) and v is not None)
                or (isinstance(v, dict) and len(v.keys()) > 0)
            }
        return calliope.AttrDict.to_yaml(calliope.AttrDict(_dict))

    @pytest.fixture(scope="module")
    def model(self):
        return calliope.examples.national_scale()

    @pytest.fixture(scope="module")
    def observer(self):
        return xr.Dataset()

    @pytest.fixture(scope="module")
    def observed_from_dict(self, observer):
        initial_dict = {"foo": "bar", "foobar": {"baz": "fob"}}
        return observed_dict.UpdateObserverDict(
            initial_dict=initial_dict, name="test", observer=observer
        )

    @pytest.fixture(scope="module")
    def observed_from_string(self, observer):
        initial_dict = {"foo": "bar", "foobar": {"baz": "fob"}}
        return observed_dict.UpdateObserverDict(
            initial_yaml_string=self.as_yaml(initial_dict),
            name="test_2",
            observer=observer,
        )

    def test_initialise_observer(
        self, observer, observed_from_dict, observed_from_string
    ):
        assert "test" in observer.attrs.keys()
        assert "test_2" in observer.attrs.keys()
        assert observer.attrs["test"] == self.as_yaml(
            {"foo": "bar", "foobar": {"baz": "fob"}}
        )
        observer.attrs["test"] == observer.attrs["test_2"]

    def test_value_error_on_initialising(self):
        initial_dict = {"foo": "bar", "foobar": {"baz": "fob"}}
        initial_string = self.as_yaml(initial_dict)
        with pytest.raises(ValueError) as error:
            observed_dict.UpdateObserverDict(name="test_2", observer=xr.Dataset())
        assert check_error_or_warning(
            error,
            "must supply one, and only one, of initial_dict or initial_yaml_string",
        )
        with pytest.raises(ValueError) as error:
            observed_dict.UpdateObserverDict(
                initial_yaml_string=initial_string,
                initial_dict=initial_dict,
                name="test_2",
                observer=xr.Dataset(),
            )
        assert check_error_or_warning(
            error,
            "must supply one, and only one, of initial_dict or initial_yaml_string",
        )

    @pytest.mark.parametrize(
        "key1,key2,value,result",
        (
            ("foo", None, 1, {"foo": 1, "foobar": {"baz": "fob"}}),
            ("foobar", "baz", 2, {"foo": 1, "foobar": {"baz": 2}}),
            (
                "foo",
                None,
                {"baz": "fob"},
                {"foo": {"baz": "fob"}, "foobar": {"baz": 2}},
            ),
            ("foo", "baz", 3, {"foo": {"baz": 3}, "foobar": {"baz": 2}}),
            ("foo", None, {}, {"foobar": {"baz": 2}}),
            ("foo", None, 5, {"foo": 5, "foobar": {"baz": 2}}),
            ("foo", None, None, {"foobar": {"baz": 2}}),
        ),
    )
    def test_set_item_observer(
        self, observed_from_dict, observer, key1, key2, value, result
    ):
        if key2 is None:
            observed_from_dict[key1] = value
        else:
            observed_from_dict[key1][key2] = value

        assert observer.attrs["test"] == self.as_yaml(result)

    def test_update_observer(self, observed_from_dict, observer):
        observed_from_dict.update({"baz": 4})
        assert observer.attrs["test"] == self.as_yaml({"foobar": {"baz": 2}, "baz": 4})

    def test_reinstate_observer(self, observed_from_dict, observer):
        observer.attrs["test"] = "{}"
        assert observer.attrs["test"] == "{}"
        observed_from_dict["foo"] = 5
        assert observer.attrs["test"] == self.as_yaml(
            {"foo": 5, "foobar": {"baz": 2}, "baz": 4}
        )

    def test_model_config(self, model):
        assert hasattr(model, "model_config")
        assert "model_config" in model._model_data.attrs.keys()
        assert model._model_data.attrs["model_config"] == self.as_yaml(
            model.model_config, strip=True
        )

        model.model_config["name"] = "new name"
        assert model.model_config["name"] == "new name"
        assert model._model_data.attrs["model_config"] == self.as_yaml(
            model.model_config, strip=True
        )

    def test_run_config(self, model):
        assert hasattr(model, "run_config")
        assert "run_config" in model._model_data.attrs.keys()
        assert model._model_data.attrs["run_config"] == self.as_yaml(
            model.run_config, strip=True
        )

        model.run_config["solver"] = "cplex"
        assert model.run_config["solver"] == "cplex"
        assert model._model_data.attrs["run_config"] == self.as_yaml(
            model.run_config, strip=True
        )

    def test_load_from_netcdf(self, model):

        with tempfile.TemporaryDirectory() as tempdir:
            out_path = os.path.join(tempdir, "model.nc")
            model.to_netcdf(out_path)

            model_from_disk = calliope.read_netcdf(out_path)
            assert hasattr(model_from_disk, "run_config")
            assert "run_config" in model_from_disk._model_data.attrs.keys()
            assert hasattr(model_from_disk, "model_config")
            assert "model_config" in model_from_disk._model_data.attrs.keys()
