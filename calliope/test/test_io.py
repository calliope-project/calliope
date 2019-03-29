import os
import tempfile

import pytest  # pylint: disable=unused-import

import calliope
from calliope import exceptions


class TestIO:
    @pytest.fixture(scope='module')
    def model(self):
        model = calliope.examples.national_scale()
        return model

    def test_save_netcdf(self, model):
        bool_attrs = [
            k for k, v in model._model_data.attrs.items()
            if isinstance(v, bool)
        ]

        with tempfile.TemporaryDirectory() as tempdir:
            out_path = os.path.join(tempdir, 'model.nc')
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

    @pytest.mark.parametrize("file_name", sorted([
        'inputs_{}.csv'.format(i)
        for i in calliope.examples.national_scale().inputs.data_vars.keys()
    ]))
    def test_save_csv(self, model, file_name):
        with tempfile.TemporaryDirectory() as tempdir:
            out_path = os.path.join(tempdir, 'out_dir')
            model.to_csv(out_path)
            assert os.path.isfile(os.path.join(out_path, file_name))

            with open(os.path.join(out_path, 'inputs_energy_cap_max_systemwide.csv'), 'r') as f:
                assert 'demand_power' not in f.read()

    def test_save_csv_no_dropna(self, model):
        with tempfile.TemporaryDirectory() as tempdir:
            out_path = os.path.join(tempdir, 'out_dir')
            model.to_csv(out_path, dropna=False)

            with open(os.path.join(out_path, 'inputs_energy_cap_max_systemwide.csv'), 'r') as f:
                assert 'demand_power' in f.read()

    def test_save_csv_not_optimal(self):
        # Not checking for content of warnings here, since
        # check_feasibility-related warnings are tested for in
        # test_example_models
        with pytest.warns(exceptions.ModelWarning):
            model = calliope.examples.national_scale(
                scenario='check_feasibility',
                override_dict={'run.cyclic_storage': False}
            )

        model.run()

        with tempfile.TemporaryDirectory() as tempdir:
            out_path = os.path.join(tempdir, 'out_dir')
            with pytest.warns(exceptions.ModelWarning):
                model.to_csv(out_path, dropna=False)

    def test_solve_save_read_netcdf(self, model):
        model.run()

        with tempfile.TemporaryDirectory() as tempdir:
            out_path = os.path.join(tempdir, 'model.nc')
            model.to_netcdf(out_path)
            assert os.path.isfile(out_path)

            model_from_disk = calliope.read_netcdf(out_path)
            # FIXME test for some data in model_from_disk

    def test_save_lp(self, model):
        with tempfile.TemporaryDirectory() as tempdir:
            out_path = os.path.join(tempdir, 'model.lp')
            model.to_lp(out_path)

            with open(out_path, 'r') as f:
                assert '\nmin \nobj:\n+1 cost(monetary_region1_1__csp)' in f.read()
