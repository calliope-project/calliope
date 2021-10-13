import os
import tempfile

import pytest  # pylint: disable=unused-import

import calliope
from calliope import exceptions
from calliope.test.common.util import check_error_or_warning


class TestIO:
    @pytest.fixture(scope="module")
    def model(self):
        model = calliope.examples.national_scale()
        model.run()
        return model

    def test_save_netcdf(self, model):
        bool_attrs = [
            k for k, v in model._model_data.attrs.items() if isinstance(v, bool)
        ]

        with tempfile.TemporaryDirectory() as tempdir:
            out_path = os.path.join(tempdir, "model.nc")
            model.to_netcdf(out_path)
            assert os.path.isfile(out_path)

        # Ensure that boolean attrs have not changed
        for k in bool_attrs:
            assert isinstance(model._model_data.attrs[k], bool)

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

    def test_solve_save_read_netcdf(self, model):
        with tempfile.TemporaryDirectory() as tempdir:
            out_path = os.path.join(tempdir, "model.nc")
            model.to_netcdf(out_path)
            assert os.path.isfile(out_path)

            model_from_disk = calliope.read_netcdf(out_path)
            # FIXME test for some data in model_from_disk

    def test_rerun_save_read_netcdf_with_mixed_dtype(self, model):
        new_model = model.backend.rerun()
        assert model._model_data.force_resource.dtype.kind == "f"
        assert new_model._model_data.force_resource.dtype.kind == "O"
        with tempfile.TemporaryDirectory() as tempdir:
            out_path = os.path.join(tempdir, "model.nc")
            with pytest.warns(exceptions.ModelWarning) as warning:
                new_model.to_netcdf(out_path)

            check_error_or_warning(
                warning, "'force_resource' contains mixed data types"
            )
            assert new_model._model_data.force_resource.dtype.kind == "O"
            model_from_disk = calliope.read_netcdf(out_path)
            assert model_from_disk._model_data.force_resource.dtype.kind == "f"

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

    def test_netcdf_roundtrip_modelrun_to_yaml(self, model):
        with tempfile.TemporaryDirectory() as tempdir:
            out_path_nc = os.path.join(tempdir, "model.nc")
            out_path_yaml = os.path.join(tempdir, "modelrun.yaml")
            model.to_netcdf(out_path_nc)
            model_from_disk = calliope.read_netcdf(out_path_nc)

            model_from_disk.save_commented_model_yaml(out_path_yaml)
            assert os.path.isfile(out_path_yaml)

    def test_netcdf_to_yaml_raises_if_attrs_missing(self, model):
        with tempfile.TemporaryDirectory() as tempdir:
            out_path_yaml = os.path.join(tempdir, "modelrun.yaml")
            model._model_run = {}
            with pytest.raises(KeyError) as error:
                model.save_commented_model_yaml(out_path_yaml)

        assert check_error_or_warning(
            error, "This model does not have the fully built model attached"
        )

    def test_save_lp(self, model):
        with tempfile.TemporaryDirectory() as tempdir:
            out_path = os.path.join(tempdir, "model.lp")
            model.to_lp(out_path)

            with open(out_path, "r") as f:
                assert "\nmin \nobj:\n+1 cost(monetary__region1_1__csp_)" in f.read()

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
