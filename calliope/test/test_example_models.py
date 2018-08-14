import os
import shutil

import pytest
from pytest import approx
import pandas as pd

import calliope
from calliope.test.common.util import check_error_or_warning


class TestModelPreproccesing:
    def test_preprocess_national_scale(self):
        calliope.examples.national_scale()

    def test_preprocess_time_clustering(self):
        calliope.examples.time_clustering()

    def test_preprocess_time_resampling(self):
        calliope.examples.time_resampling()

    def test_preprocess_urban_scale(self):
        calliope.examples.urban_scale()

    def test_preprocess_milp(self):
        calliope.examples.milp()

    def test_preprocess_operate(self):
        calliope.examples.operate()

    def test_preprocess_time_masking(self):
        calliope.examples.time_masking()


class TestNationalScaleExampleModelSenseChecks:
    def example_tester(self, solver='glpk', solver_io=None):
        override = {
            'model.subset_time': '2005-01-01',
            'run.solver': solver,
        }

        if solver_io:
            override['run.solver_io'] = solver_io

        model = calliope.examples.national_scale(override_dict=override)
        model.run()

        assert model.results.storage_cap.to_pandas()['region1-1::csp'] == approx(45129.950)
        assert model.results.storage_cap.to_pandas()['region2::battery'] == approx(6675.173)

        assert model.results.energy_cap.to_pandas()['region1-1::csp'] == approx(4626.588)
        assert model.results.energy_cap.to_pandas()['region2::battery'] == approx(1000)
        assert model.results.energy_cap.to_pandas()['region1::ccgt'] == approx(30000)

        assert float(model.results.cost.sum()) == approx(38997.3544)

        assert float(
            model.results.systemwide_levelised_cost.loc[dict(carriers='power')].to_pandas().T['battery']
        ) == approx(0.063543, abs=0.000001)
        assert float(
            model.results.systemwide_capacity_factor.loc[dict(carriers='power')].to_pandas().T['battery']
        ) == approx(0.2642256, abs=0.000001)

    def test_nationalscale_example_results_glpk(self):
        self.example_tester()

    def test_nationalscale_example_results_gurobi(self):
        try:
            import gurobipy
            self.example_tester(solver='gurobi', solver_io='python')
        except ImportError:
            pytest.skip('Gurobi not installed')

    def test_nationalscale_example_results_cplex(self):
        # Check for existence of the `cplex` command
        if shutil.which('cplex'):
            self.example_tester(solver='cplex')
        else:
            pytest.skip('CPLEX not installed')

    def test_nationalscale_example_results_cbc(self):
        # Check for existence of the `cbc` command
        if shutil.which('cbc'):
            self.example_tester(solver='cbc')
        else:
            pytest.skip('CBC not installed')


class TestNationalScaleExampleModelInfeasibility:
    def example_tester(self):
        override_file = os.path.join(
            calliope.examples._PATHS['national_scale'],
            'overrides.yaml'
        )
        with pytest.warns(calliope.exceptions.ModelWarning) as excinfo:
            model = calliope.examples.national_scale(override_file=override_file + ':check_feasibility')

        expected_warnings = [
            'Objective function argument `cost_class` given but not used by objective function `check_feasibility`',
            'Objective function argument `sense` given but not used by objective function `check_feasibility`'
        ]

        assert check_error_or_warning(excinfo, expected_warnings)

        model.run()

        assert model.results.attrs['termination_condition'] == 'other'

        assert 'systemwide_levelised_cost' not in model.results.data_vars
        assert 'systemwide_capacity_factor' not in model.results.data_vars

    def test_nationalscale_example_results_glpk(self):
        self.example_tester()


class TestNationalScaleExampleModelOperate:
    def example_tester(self):
        override_file = os.path.join(
            calliope.examples._PATHS['national_scale'],
            'overrides.yaml'
        )
        with pytest.warns(calliope.exceptions.ModelWarning) as excinfo:
            model = calliope.examples.national_scale(
                override_dict={'model.subset_time': ['2005-01-01', '2005-01-03']},
                override_file=override_file + ':operate')
            model.run()

        expected_warnings = [
            'Energy capacity constraint removed from region1::demand_power as force_resource is applied',
            'Energy capacity constraint removed from region2::demand_power as force_resource is applied',
            'Resource capacity constraint defined and set to infinity for all supply_plus techs'
        ]

        assert check_error_or_warning(excinfo, expected_warnings)
        assert all(model.results.timesteps == pd.date_range('2005-01', '2005-01-03 23:00:00', freq='H'))

    def test_nationalscale_example_results_glpk(self):
        self.example_tester()


class TestNationalScaleResampledExampleModelSenseChecks:
    def example_tester(self, solver='glpk', solver_io=None):
        override = {
            'model.subset_time': '2005-01-01',
            'run.solver': solver,
        }

        if solver_io:
            override['run.solver_io'] = solver_io

        model = calliope.examples.time_resampling(override_dict=override)
        model.run()

        assert model.results.storage_cap.to_pandas()['region1-1::csp'] == approx(23563.444)
        assert model.results.storage_cap.to_pandas()['region2::battery'] == approx(6315.78947)

        assert model.results.energy_cap.to_pandas()['region1-1::csp'] == approx(1440.8377)
        assert model.results.energy_cap.to_pandas()['region2::battery'] == approx(1000)
        assert model.results.energy_cap.to_pandas()['region1::ccgt'] == approx(30000)

        assert float(model.results.cost.sum()) == approx(37344.221869)

        assert float(
            model.results.systemwide_levelised_cost.loc[dict(carriers='power')].to_pandas().T['battery']
        ) == approx(0.063543, abs=0.000001)
        assert float(
            model.results.systemwide_capacity_factor.loc[dict(carriers='power')].to_pandas().T['battery']
        ) == approx(0.25, abs=0.000001)

    def test_nationalscale_resampled_example_results_glpk(self):
        self.example_tester()

    def test_nationalscale_resampled_example_results_cbc(self):
        # Check for existence of the `cbc` command
        if shutil.which('cbc'):
            self.example_tester(solver='cbc')
        else:
            pytest.skip('CBC not installed')


class TestNationalScaleClusteredExampleModelSenseChecks:
    def model_runner(self, solver='glpk', solver_io=None,
                     how='closest', storage_inter_cluster=False, cyclic=False):
        override = {
            'model.time.function_options': {
                'how': how, 'storage_inter_cluster': storage_inter_cluster
            },
            'run.solver': solver,
            'run.cyclic_storage': cyclic
        }

        if solver_io:
            override['run.solver_io'] = solver_io

        model = calliope.examples.time_clustering(override_dict=override)
        model.run()

        return model

    def example_tester_closest(self, solver='glpk', solver_io=None):
        model = self.model_runner(solver=solver, solver_io=solver_io, how='closest')
        # Full 1-hourly model run: 22312488.670967
        assert float(model.results.cost.sum()) == approx(51711873.203096)

        # Full 1-hourly model run: 0.296973
        assert float(
            model.results.systemwide_levelised_cost.loc[dict(carriers='power')].to_pandas().T['battery']
        ) == approx(0.111456, abs=0.000001)

        # Full 1-hourly model run: 0.064362
        assert float(
            model.results.systemwide_capacity_factor.loc[dict(carriers='power')].to_pandas().T['battery']
        ) == approx(0.074809, abs=0.000001)

    def example_tester_mean(self, solver='glpk', solver_io=None):
        model = self.model_runner(solver=solver, solver_io=solver_io, how='mean')
        # Full 1-hourly model run: 22312488.670967
        assert float(model.results.cost.sum()) == approx(45110415.5627)

        # Full 1-hourly model run: 0.296973
        assert float(
            model.results.systemwide_levelised_cost.loc[dict(carriers='power')].to_pandas().T['battery']
        ) == approx(0.126099, abs=0.000001)

        # Full 1-hourly model run: 0.064362
        assert float(
            model.results.systemwide_capacity_factor.loc[dict(carriers='power')].to_pandas().T['battery']
        ) == approx(0.047596, abs=0.000001)

    def example_tester_storage_inter_cluster(self):
        model = self.model_runner(storage_inter_cluster=True)

        # Full 1-hourly model run: 22312488.670967
        assert float(model.results.cost.sum()) == approx(33353390.222036)

        # Full 1-hourly model run: 0.296973
        assert float(
            model.results.systemwide_levelised_cost.loc[dict(carriers='power')].to_pandas().T['battery']
        ) == approx(0.115866, abs=0.000001)

        # Full 1-hourly model run: 0.064362
        assert float(
            model.results.systemwide_capacity_factor.loc[dict(carriers='power')].to_pandas().T['battery']
        ) == approx(0.074167, abs=0.000001)

    def example_tester_storage_inter_cluster_cyclic(self):
        model = self.model_runner(storage_inter_cluster=True, cyclic=True)
        # Full 1-hourly model run: 22312488.670967
        assert float(model.results.cost.sum()) == approx(18838244.087694)

        # Full 1-hourly model run: 0.296973
        assert float(
            model.results.systemwide_levelised_cost.loc[dict(carriers='power')].to_pandas().T['battery']
        ) == approx(0.133111, abs=0.000001)

        # Full 1-hourly model run: 0.064362
        assert float(
            model.results.systemwide_capacity_factor.loc[dict(carriers='power')].to_pandas().T['battery']
        ) == approx(0.071411, abs=0.000001)

    def test_nationalscale_clustered_example_closest_results_glpk(self):
        self.example_tester_closest()

    def test_nationalscale_clustered_example_closest_results_cbc(self):
        # Check for existence of the `cbc` command
        if shutil.which('cbc'):
            self.example_tester_closest(solver='cbc')
        else:
            pytest.skip('CBC not installed')

    def test_nationalscale_clustered_example_mean_results_glpk(self):
        self.example_tester_mean()

    def test_nationalscale_clustered_example_mean_results_cbc(self):
        # Check for existence of the `cbc` command
        if shutil.which('cbc'):
            self.example_tester_mean(solver='cbc')
        else:
            pytest.skip('CBC not installed')

    def test_nationalscale_clustered_example_storage_inter_cluster(self):
        self.example_tester_storage_inter_cluster()


class TestUrbanScaleExampleModelSenseChecks:
    def test_urbanscale_example_results(self):
        model = calliope.examples.urban_scale(
            override_dict={'model.subset_time': '2005-01-01'}
        )
        model.run()

        assert model.results.energy_cap.to_pandas()['X1::chp'] == approx(272.227204)
        assert model.results.energy_cap.to_pandas()['X2::heat_pipes:N1'] == approx(197.908691)

        assert model.results.carrier_prod.sum('timesteps').to_pandas()['X3::boiler::heat'] == approx(474.720800)
        assert float(model.results.carrier_export.sum()) == approx(0)

        assert float(model.results.cost.sum()) == approx(528.8643154)

    def test_milp_example_results(self):
        model = calliope.examples.milp(
            override_dict={'model.subset_time': '2005-01-01'}
        )
        model.run()

        assert model.results.energy_cap.to_pandas()['X1::chp'] == 300
        assert model.results.energy_cap.to_pandas()['X2::heat_pipes:N1'] == approx(188.363137)

        assert model.results.carrier_prod.sum('timesteps').to_pandas()['X1::supply_gas::gas'] == approx(12363.173036)
        assert float(model.results.carrier_export.sum()) == approx(0)

        assert model.results.purchased.to_pandas()['X2::boiler'] == 1
        assert model.results.units.to_pandas()['X1::chp'] == 1

        assert float(model.results.operating_units.sum()) == 24

        assert float(model.results.cost.sum()) == approx(540.780779)

    def test_operate_example_results(self):
        model = calliope.examples.operate(
            override_dict={'model.subset_time': ['2005-07-01', '2005-07-04']}
        )
        with pytest.warns(calliope.exceptions.ModelWarning) as excinfo:
            model.run()

        expected_warnings = [
            'Energy capacity constraint removed',
            'Resource capacity constraint defined and set to infinity for all supply_plus techs',
            'Resource capacity constraint removed'
        ]

        assert check_error_or_warning(excinfo, expected_warnings)

        assert all(model.results.timesteps == pd.date_range('2005-07', '2005-07-04 23:00:00', freq='H'))
