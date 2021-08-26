from calliope.core.attrdict import AttrDict
import pytest  # noqa: F401
import calliope
import logging
import datetime
import os
import tempfile

import xarray as xr
import numpy as np

from calliope.core.util import dataset, observed_dict, checks

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
        model.inputs[variable_name].to_dataframe()


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


class TestChecks:
    @pytest.fixture
    def model_data(self):
        foo = [[1, 0], [0, 2], [0, 1]]
        bar = [1, 0, 0]
        x = ["a", "b", "c"]
        y = ["A", "B"]
        run_config_dict = AttrDict({"run.option1.suboption1": 1, "run.option2": 2})
        model_config_dict = AttrDict(
            {"model.option1.suboption1": True, "model.option2": False}
        )

        ds = xr.Dataset(
            data_vars={"foo": (["x", "y"], foo), "bar": (["x"], bar)},
            coords={"x": x, "y": y},
            attrs={
                "run_config": run_config_dict.to_yaml(),
                "model_config": model_config_dict.to_yaml(),
            },
        )
        return ds

    @pytest.fixture
    def check_dict(self):
        return {
            "checkname1": {
                "fail_where": ["foo=2", "and", "bar=1"],
                "error": "expected_pass",
            },
            "checkname2": {
                "fail_where": ["foo=2", "and", "bar=0"],
                "error": "expected_error",
            },
            "checkname3": {
                "fail_where": ["foo", "and", "bar"],
                "fail_if_any": {
                    "lhs": ["foo", "multiply", "bar"],
                    "operator": "ge",
                    "rhs": [1],
                },
                "warning": "expected_warning",
            },
            "checkname4": {
                "fail_where": ["foo", "and", "bar"],
                "fail_if_any": {
                    "lhs": ["foo", "multiply", "bar"],
                    "operator": "gt",
                    "rhs": [1],
                },
                "warning": "expected_pass",
            },
        }

    @pytest.mark.parametrize(
        ("method", "result"),
        [
            ("multiply", [[1, 0], [0, 0], [0, 0]]),
            ("add", [[2, 1], [0, 2], [0, 1]]),
            ("subtract", [[0, -1], [0, 2], [0, 1]]),
        ],
    )
    def test_simple_equations(self, model_data, method, result):
        parsed = checks._parse_vars(model_data, ["foo", method, "bar"], "name")
        assert np.array_equal(parsed, result)

    @pytest.mark.parametrize(
        ("method", "result"),
        [("multiply", [[1, 0], [0, 2], [0, 1]]), ("add", [[2, 1], [1, 3], [1, 2]])],
    )
    @pytest.mark.parametrize("static_val", [1, 1.0, "run.option1.suboption1"])
    def test_static_vals(self, model_data, method, result, static_val):
        parsed = checks._parse_vars(model_data, ["foo", method, static_val], "name")
        assert np.array_equal(parsed, result)

    def test_longer_equation(self, model_data):
        var_list = ["foo", "multiply", 2, "add", "bar"]
        parsed = checks._parse_vars(model_data, var_list, "name")
        assert np.array_equal(parsed, [[3, 1], [0, 4], [0, 2]])

    @pytest.mark.parametrize("var", ["foo", "bar"])
    def test_single_array(self, model_data, var):
        parsed = checks._parse_vars(model_data, [var], "name")
        assert np.array_equal(parsed, model_data[var])

    @pytest.mark.parametrize("var", [1, 2, 1e3])
    def test_single_val(self, model_data, var):
        parsed = checks._parse_vars(model_data, [var], "name")
        assert parsed == var

    def test_too_many_methods(self, model_data):
        with pytest.raises(
            AssertionError,
            match="Too many numpy operators compared to variables to check in tabular data check name",
        ):
            checks._parse_vars(model_data, ["foo", "multiply", "divide"], "name")

    def test_unknown_var(self, model_data):
        with pytest.raises(
            ValueError, match="Unable to parse variable baz in tabular data check name"
        ):
            checks._parse_vars(model_data, ["foo", "multiply", "baz"], "name")

    def test_check_yaml_config(self, model_data, check_dict):
        with tempfile.TemporaryDirectory() as tempdir:
            checklist_path = os.path.join(tempdir, "checks.yaml")
            AttrDict(check_dict).to_yaml(checklist_path)

            warnings, errors = checks.check_tabular_data(model_data, checklist_path)
            assert "expected_error" in errors
            assert "expected_pass" not in errors
            assert "expected_warning" not in errors

            assert "expected_warning" in warnings
            assert "expected_pass" not in warnings
            assert "expected_error" not in warnings
