import os
import tempfile

import pytest  # noqa: F401
import xarray as xr

import calliope
from calliope import exceptions


class TestIO:
    @pytest.fixture(scope="module")
    def model(self):
        model = calliope.examples.national_scale()
        model._model_data = model._model_data.assign_attrs(
            foo_true=True,
            foo_false=False,
            foo_none=None,
            foo_dict={"foo": {"a": 1}},
            foo_attrdict=calliope.AttrDict({"foo": {"a": 1}}),
        )
        model.run()
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

    def test_save_netcdf(self, model_file):
        assert os.path.isfile(model_file)

    @pytest.mark.parametrize(
        ["attr", "expected_type", "expected_val"],
        [
            ("foo_true", bool, True),
            ("foo_false", bool, False),
            ("foo_none", type(None), None),
            ("foo_dict", dict, {"foo": {"a": 1}}),
            ("foo_attrdict", calliope.AttrDict, calliope.AttrDict({"foo": {"a": 1}})),
        ],
    )
    @pytest.mark.parametrize("model_name", ["model", "model_from_file"])
    def test_serialised_attrs(
        self, request, attr, expected_type, expected_val, model_name
    ):
        model = request.getfixturevalue(model_name)
        # Ensure that boolean attrs have not changed

        assert isinstance(model._model_data.attrs[attr], expected_type)
        if expected_val is None:
            assert model._model_data.attrs[attr] is None
        else:
            assert model._model_data.attrs[attr] == expected_val

    @pytest.mark.parametrize(
        "serialised_list", ["serialised_bools", "serialised_nones", "serialised_dicts"]
    )
    @pytest.mark.parametrize("model_name", ["model", "model_from_file"])
    def test_serialised_list_popped(self, request, serialised_list, model_name):
        model = request.getfixturevalue(model_name)
        assert serialised_list not in model._model_data.attrs.keys()

    @pytest.mark.parametrize(
        ["serialised_list", "list_elements"],
        [
            ("serialised_bools", ["foo_true", "foo_false"]),
            ("serialised_nones", ["foo_none", "scenario"]),
            (
                "serialised_dicts",
                [
                    "foo_dict",
                    "foo_attrdict",
                    "defaults",
                    "subsets",
                    "model_config",
                    "run_config",
                ],
            ),
        ],
    )
    def test_serialised_list(
        self, model_from_file_no_processing, serialised_list, list_elements
    ):
        assert not set(
            model_from_file_no_processing.attrs[serialised_list]
        ).symmetric_difference(list_elements)

    def test_save_csv_dir_mustnt_exist(self, model):
        with tempfile.TemporaryDirectory() as tempdir:
            out_path = os.path.join(tempdir)
            with pytest.raises(FileExistsError):
                model.to_csv(out_path)

    @pytest.mark.parametrize(
        "file_name",
        sorted(
            [
                "inputs_{}.csv".format(i)
                for i in calliope.examples.national_scale().inputs.data_vars.keys()
            ]
        ),
    )
    def test_save_csv(self, model, file_name):
        with tempfile.TemporaryDirectory() as tempdir:
            out_path = os.path.join(tempdir, "out_dir")
            model.to_csv(out_path)
            assert os.path.isfile(os.path.join(out_path, file_name))

            with open(
                os.path.join(out_path, "inputs_energy_cap_max_systemwide.csv"), "r"
            ) as f:
                assert "demand_power" not in f.read()

    def test_save_csv_no_dropna(self, model):
        with tempfile.TemporaryDirectory() as tempdir:
            out_path = os.path.join(tempdir, "out_dir")
            model.to_csv(out_path, dropna=False)

            with open(
                os.path.join(out_path, "inputs_energy_cap_max_systemwide.csv"), "r"
            ) as f:
                assert "demand_power" in f.read()

    def test_save_csv_not_optimal(self):
        model = calliope.examples.national_scale(
            scenario="check_feasibility", override_dict={"run.cyclic_storage": False}
        )

        model.run()

        with tempfile.TemporaryDirectory() as tempdir:
            out_path = os.path.join(tempdir, "out_dir")
            with pytest.warns(exceptions.ModelWarning):
                model.to_csv(out_path, dropna=False)

    @pytest.mark.parametrize("attr", ["run_config", "model_config", "subsets"])
    def test_dicts_as_model_attrs_and_property(self, model_from_file, attr):
        assert attr in model_from_file._model_data.attrs.keys()
        assert hasattr(model_from_file, attr)

    def test_defaults_as_model_attrs_not_property(self, model_from_file):
        assert "defaults" in model_from_file._model_data.attrs.keys()
        assert not hasattr(model_from_file, "defaults")

    @pytest.mark.parametrize("attr", ["results", "inputs"])
    def test_filtered_dataset_as_property(self, model_from_file, attr):
        assert hasattr(model_from_file, attr)

    def test_save_read_solve_save_netcdf(self, model):
        with tempfile.TemporaryDirectory() as tempdir:
            out_path = os.path.join(tempdir, "model.nc")
            model.to_netcdf(out_path)
            model_from_disk = calliope.read_netcdf(out_path)

        # Ensure _model_run doesn't exist to simulate a re-run
        # via the backend
        delattr(model_from_disk, "_model_run")
        model_from_disk.run(force_rerun=True)
        assert not hasattr(model_from_disk, "_model_run")

        with tempfile.TemporaryDirectory() as tempdir:
            out_path = os.path.join(tempdir, "model.nc")
            model_from_disk.to_netcdf(out_path)
            assert os.path.isfile(out_path)

    def test_save_lp(self, model):
        with tempfile.TemporaryDirectory() as tempdir:
            out_path = os.path.join(tempdir, "model.lp")
            model.to_lp(out_path)

            with open(out_path, "r") as f:
                assert "energy_cap(region1_ccgt)" in f.read()

    @pytest.mark.skip(
        reason="SPORES mode will fail until the cost max group constraint can be reproduced"
    )
    def test_save_per_spore(self):
        with tempfile.TemporaryDirectory() as tempdir:
            os.mkdir(os.path.join(tempdir, "output"))
            model = calliope.examples.national_scale(
                scenario="spores",
                override_dict={
                    "run.spores_options.save_per_spore": True,
                    "run.spores_options.save_per_spore_path": os.path.join(
                        tempdir, "output/spore_{}.nc"
                    ),
                },
            )
            model.run()

            for i in ["0", "1", "2", "3"]:
                assert os.path.isfile(os.path.join(tempdir, "output", f"spore_{i}.nc"))
            assert not os.path.isfile(os.path.join(tempdir, "output.nc"))
