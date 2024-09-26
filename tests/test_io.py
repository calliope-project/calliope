import os
import tempfile

import pytest  # noqa: F401
import xarray as xr

import calliope
import calliope.io
from calliope import exceptions

from .common.util import check_error_or_warning


class TestIO:
    @pytest.fixture(scope="module")
    def vars_to_add_attrs(self):
        return ["source_use_max", "flow_cap"]

    @pytest.fixture(scope="module")
    def model(self, vars_to_add_attrs):
        model = calliope.examples.national_scale()
        attrs = {
            "foo_true": True,
            "foo_false": False,
            "foo_none": None,
            "foo_dict": {"foo": {"a": 1}},
            "foo_attrdict": calliope.AttrDict({"foo": {"a": 1}}),
            "foo_set": set(["foo", "bar"]),
            "foo_set_1_item": set(["foo"]),
            "foo_list": ["foo", "bar"],
            "foo_list_1_item": ["foo"],
        }
        model._model_data = model._model_data.assign_attrs(**attrs)
        model.build()
        model.solve()

        for var in vars_to_add_attrs:
            model._model_data[var] = model._model_data[var].assign_attrs(**attrs)

        return model

    @pytest.fixture(scope="module")
    def model_file(self, tmpdir_factory, model):
        out_path = tmpdir_factory.mktemp("data").join("model.nc")
        model.to_netcdf(out_path)
        return out_path

    @pytest.fixture(scope="module")
    def model_from_file(self, model_file):
        return calliope.read_netcdf(model_file)

    @pytest.fixture(scope="module")
    def model_from_file_no_processing(self, model_file):
        return xr.open_dataset(model_file)

    @pytest.fixture(scope="module")
    def model_csv_dir(self, tmpdir_factory, model):
        out_path = tmpdir_factory.mktemp("data")
        model.to_csv(os.path.join(out_path, "csvs"))
        return os.path.join(out_path, "csvs")

    def test_save_netcdf(self, model_file):
        assert os.path.isfile(model_file)

    @pytest.mark.parametrize(
        ("attr", "expected_type", "expected_val"),
        [
            ("foo_true", bool, True),
            ("foo_false", bool, False),
            ("foo_none", type(None), None),
            ("foo_dict", dict, {"foo": {"a": 1}}),
            ("foo_attrdict", calliope.AttrDict, calliope.AttrDict({"foo": {"a": 1}})),
            ("foo_set", set, set(["foo", "bar"])),
            ("foo_set_1_item", set, set(["foo"])),
            ("foo_list", list, ["foo", "bar"]),
            ("foo_list_1_item", list, ["foo"]),
        ],
    )
    @pytest.mark.parametrize("model_name", ["model", "model_from_file"])
    def test_serialised_attrs(
        self, request, attr, expected_type, expected_val, model_name, vars_to_add_attrs
    ):
        model = request.getfixturevalue(model_name)
        var_attrs = [model._model_data[var].attrs for var in vars_to_add_attrs]
        for attrs in [model._model_data.attrs, *var_attrs]:
            assert isinstance(attrs[attr], expected_type)
            if expected_val is None:
                assert attrs[attr] is None
            else:
                assert attrs[attr] == expected_val

    @pytest.mark.parametrize(
        "serialised_list",
        ["serialised_bools", "serialised_nones", "serialised_dicts", "serialised_sets"],
    )
    @pytest.mark.parametrize("model_name", ["model", "model_from_file"])
    def test_serialised_list_popped(self, request, serialised_list, model_name):
        model = request.getfixturevalue(model_name)
        assert serialised_list not in model._model_data.attrs.keys()

    @pytest.mark.parametrize(
        ("serialisation_list_name", "list_elements"),
        [
            ("serialised_bools", ["foo_true", "foo_false"]),
            ("serialised_nones", ["foo_none", "scenario"]),
            (
                "serialised_dicts",
                ["foo_dict", "foo_attrdict", "defaults", "config", "applied_math"],
            ),
            ("serialised_sets", ["foo_set", "foo_set_1_item"]),
            ("serialised_single_element_list", ["foo_list_1_item", "foo_set_1_item"]),
        ],
    )
    def test_serialisation_lists(
        self, model_from_file_no_processing, serialisation_list_name, list_elements
    ):
        serialisation_list = calliope.io._pop_serialised_list(
            model_from_file_no_processing.attrs, serialisation_list_name
        )
        assert not set(serialisation_list).symmetric_difference(list_elements)

    @pytest.mark.parametrize(
        "attrs", [{"foo": [1]}, {"foo": [None]}, {"foo": [1, "bar"]}, {"foo": set([1])}]
    )
    def test_non_strings_in_serialised_lists(self, attrs):
        with pytest.raises(TypeError) as excinfo:
            calliope.io._serialise(attrs)
        assert check_error_or_warning(
            excinfo,
            f"Cannot serialise a sequence of values stored in a model attribute unless all values are strings, found: {attrs['foo']}",
        )

    def test_save_csv_dir_mustnt_exist(self, model):
        with tempfile.TemporaryDirectory() as tempdir:
            out_path = os.path.join(tempdir)
            with pytest.raises(FileExistsError):
                model.to_csv(out_path)

    def test_save_csv_dir_can_exist_if_overwrite_true(self, model, model_csv_dir):
        model.to_csv(model_csv_dir, allow_overwrite=True)

    @pytest.mark.parametrize(
        "file_name",
        sorted(
            [
                f"inputs_{i}.csv"
                for i in calliope.examples.national_scale().inputs.data_vars.keys()
            ]
        ),
    )
    def test_save_csv(self, model_csv_dir, file_name):
        assert os.path.isfile(os.path.join(model_csv_dir, file_name))

    def test_csv_contents(self, model_csv_dir):
        with open(
            os.path.join(model_csv_dir, "inputs_flow_cap_max_systemwide.csv")
        ) as f:
            assert "demand_power" not in f.read()

    def test_save_csv_no_dropna(self, model):
        with tempfile.TemporaryDirectory() as tempdir:
            out_path = os.path.join(tempdir, "out_dir")
            model.to_csv(out_path, dropna=False)

            with open(
                os.path.join(out_path, "inputs_flow_cap_max_systemwide.csv")
            ) as f:
                assert "demand_power" in f.read()

    @pytest.mark.xfail(reason="Not reimplemented the 'check feasibility' objective")
    def test_save_csv_not_optimal(self):
        model = calliope.examples.national_scale(
            scenario="check_feasibility",
            override_dict={"config.build.cyclic_storage": False},
        )

        model.build()
        model.solve()

        with tempfile.TemporaryDirectory() as tempdir:
            out_path = os.path.join(tempdir, "out_dir")
            with pytest.warns(exceptions.ModelWarning):
                model.to_csv(out_path, dropna=False)

    @pytest.mark.parametrize("attr", ["config"])
    def test_dicts_as_model_attrs_and_property(self, model_from_file, attr):
        assert attr in model_from_file._model_data.attrs.keys()
        assert hasattr(model_from_file, attr)

    def test_defaults_as_model_attrs_not_property(self, model_from_file):
        assert "defaults" in model_from_file._model_data.attrs.keys()
        assert not hasattr(model_from_file, "defaults")

    @pytest.mark.parametrize("attr", ["results", "inputs"])
    def test_filtered_dataset_as_property(self, model_from_file, attr):
        assert hasattr(model_from_file, attr)

    def test_save_read_solve_save_netcdf(self, model, tmpdir_factory):
        out_path = tmpdir_factory.mktemp("model_dir").join("model.nc")
        model.to_netcdf(out_path)
        model_from_disk = calliope.read_netcdf(out_path)

        # Simulate a re-run via the backend
        model_from_disk.build()
        model_from_disk.solve(force=True)

        with tempfile.TemporaryDirectory() as tempdir:
            out_path = os.path.join(tempdir, "model.nc")
            model_from_disk.to_netcdf(out_path)
            assert os.path.isfile(out_path)

    def test_save_lp(self, model):
        with tempfile.TemporaryDirectory() as tempdir:
            out_path = os.path.join(tempdir, "model.lp")
            model.backend.to_lp(out_path)

            with open(out_path) as f:
                assert "variables(flow_cap)" in f.read()

    @pytest.mark.skip(
        reason="SPORES mode will fail until the cost max group constraint can be reproduced"
    )
    def test_save_per_spore(self):
        with tempfile.TemporaryDirectory() as tempdir:
            os.mkdir(os.path.join(tempdir, "output"))
            model = calliope.examples.national_scale(scenario="spores")
            model.build()
            model.solve(
                spores_save_per_spore=True,
                save_per_spore_path=os.path.join(tempdir, "output/spore_{}.nc"),
            )

            for i in ["0", "1", "2", "3"]:
                assert os.path.isfile(os.path.join(tempdir, "output", f"spore_{i}.nc"))
            assert not os.path.isfile(os.path.join(tempdir, "output.nc"))
