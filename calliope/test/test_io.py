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

    def test_save_csv(self, model):
        with tempfile.TemporaryDirectory() as tempdir:
            out_path = os.path.join(tempdir, 'out_dir')
            model.to_csv(out_path)
            csv_files = [
                'inputs_resource_eff.csv', 'inputs_reserve_margin.csv', 'inputs_storage_loss.csv',
                'inputs_names.csv', 'inputs_loc_coordinates.csv', 'inputs_lookup_loc_carriers.csv',
                'inputs_lookup_loc_techs_area.csv', 'inputs_energy_con.csv', 'inputs_resource.csv',
                'inputs_storage_cap_max.csv', 'inputs_energy_cap_max_systemwide.csv', 'inputs_cost_om_con.csv',
                'inputs_charge_rate.csv', 'inputs_lifetime.csv', 'inputs_max_demand_timesteps.csv',
                'inputs_parasitic_eff.csv', 'inputs_timestep_resolution.csv', 'inputs_energy_eff.csv',
                'inputs_energy_cap_max.csv', 'inputs_timestep_weights.csv', 'inputs_cost_storage_cap.csv',
                'inputs_cost_resource_cap.csv', 'inputs_cost_depreciation_rate.csv', 'inputs_energy_prod.csv',
                'inputs_cost_resource_area.csv', 'inputs_cost_om_prod.csv', 'inputs_lookup_loc_techs.csv',
                'inputs_cost_energy_cap.csv', 'inputs_colors.csv', 'inputs_resource_unit.csv', 'inputs_inheritance.csv',
                'inputs_energy_ramping.csv', 'inputs_resource_area_max.csv', 'inputs_force_resource.csv'
            ]
            for f in csv_files:
                assert os.path.isfile(os.path.join(out_path, f))

            with open(os.path.join(out_path, 'inputs_energy_cap_max_systemwide.csv'), 'r') as f:
                assert 'demand_power' not in f.read()

    def test_save_csv_no_dropna(self, model):
        with tempfile.TemporaryDirectory() as tempdir:
            out_path = os.path.join(tempdir, 'out_dir')
            model.to_csv(out_path, dropna=False)

            with open(os.path.join(out_path, 'inputs_energy_cap_max_systemwide.csv'), 'r') as f:
                assert 'demand_power' in f.read()

    def test_save_csv_not_optimal(self):
        override_file = os.path.join(
            calliope.examples._PATHS['national_scale'],
            'overrides.yaml'
        )

        # Not checking for content of warnings here, since
        # check_feasibility-related warnings are tested for in
        # test_example_models
        with pytest.warns(exceptions.ModelWarning):
            model = calliope.examples.national_scale(
                override_file=override_file + ':check_feasibility'
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
