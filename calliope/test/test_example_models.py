import shutil

import pytest
from pytest import approx

import calliope

import pandas as pd


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


def nationalscale_example_tester(solver='glpk', solver_io=None):
    override = {
        'model.subset_time': '2005-01-01',
        'model.solver': solver,
    }

    if solver_io:
        override['model.solver_io'] = solver_io

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


class TestNationalScaleExampleModelSenseChecks:
    def test_nationalscale_example_results_glpk(self):
        nationalscale_example_tester()

    def test_nationalscale_example_results_gurobi(self):
        try:
            import gurobipy
            nationalscale_example_tester(solver='gurobi', solver_io='python')
        except ImportError:
            pytest.skip('Gurobi not installed')

    def test_nationalscale_example_results_cplex(self):
        # Check for existence of the `cplex` command
        if shutil.which('cplex'):
            nationalscale_example_tester(solver='cplex')
        else:
            pytest.skip('CPLEX not installed')

    def test_nationalscale_example_results_cbc(self):
        # Check for existence of the `cbc` command
        if shutil.which('cbc'):
            nationalscale_example_tester(solver='cbc')
        else:
            pytest.skip('CBC not installed')


def nationalscale_resampled_example_tester(solver='glpk', solver_io=None):
    override = {
        'model.subset_time': '2005-01-01',
        'model.solver': solver,
    }

    if solver_io:
        override['model.solver_io'] = solver_io

    model = calliope.examples.time_resampling(override_dict=override)
    model.run()

    assert model.results.storage_cap.to_pandas()['region1-1::csp'] == approx(23563.444)
    assert model.results.storage_cap.to_pandas()['region2::battery'] == approx(6315.78947)

    assert model.results.energy_cap.to_pandas()['region1-1::csp'] == approx(1440.8377)
    assert model.results.energy_cap.to_pandas()['region2::battery'] == approx(1000)
    assert model.results.energy_cap.to_pandas()['region1::ccgt'] == approx(30000)

    assert float(model.results.cost.sum()) == approx(37344.221869)


class TestNationalScaleResampledExampleModelSenseChecks:
    def test_nationalscale_resampled_example_results_glpk(self):
        nationalscale_resampled_example_tester()

    def test_nationalscale_resampled_example_results_cbc(self):
        # Check for existence of the `cbc` command
        if shutil.which('cbc'):
            nationalscale_resampled_example_tester(solver='cbc')
        else:
            pytest.skip('CBC not installed')


def nationalscale_clustered_example_tester(solver='glpk', solver_io=None):
    override = {
        'model.subset_time': '2005-01-01',
        'model.solver': solver,
    }

    if solver_io:
        override['model.solver_io'] = solver_io

    model = calliope.examples.time_resampling(override_dict=override)
    model.run()

    # assert model.results.storage_cap.to_pandas()['region1-1::csp'] == approx(23563.444)
    # assert model.results.storage_cap.to_pandas()['region2::battery'] == approx(6315.78947)

    # assert model.results.energy_cap.to_pandas()['region1-1::csp'] == approx(1440.8377)
    # assert model.results.energy_cap.to_pandas()['region2::battery'] == approx(1000)
    # assert model.results.energy_cap.to_pandas()['region1::ccgt'] == approx(30000)

    # assert float(model.results.cost.sum()) == approx(37344.221869)


class TestNationalScaleClusteredExampleModelSenseChecks:
    def test_nationalscale_clustered_example_results_glpk(self):
        nationalscale_clustered_example_tester()

    def test_nationalscale_clustered_example_results_cbc(self):
        # Check for existence of the `cbc` command
        if shutil.which('cbc'):
            nationalscale_clustered_example_tester(solver='cbc')
        else:
            pytest.skip('CBC not installed')


class TestUrbanScaleExampleModelSenseChecks:
    def test_urbanscale_example_results(self):
        model = calliope.examples.urban_scale(
            override_dict={'model.subset_time': '2005-01-01'}
        )
        model.run()

        assert model.results.energy_cap.to_pandas()['X1::chp'] == approx(272.227204)
        assert model.results.energy_cap.to_pandas()['X2::heat_pipes:N1'] == approx(197.908691)

        assert model.results.carrier_prod.sum('timesteps').to_pandas()['X3::boiler::heat'] == approx(462.189531)
        assert float(model.results.carrier_export.sum()) == approx(0)

        assert float(model.results.cost.sum()) == approx(510.858739)

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

        assert float(model.results.cost.sum()) == approx(522.829998)

    def test_operate_example_results(self):

        model = calliope.examples.operate(
            override_dict={'model.subset_time': ['2005-07-01', '2005-07-04']}
        )
        with pytest.warns(calliope.exceptions.ModelWarning) as excinfo:
            model.run()

        all_warnings = ','.join(str(excinfo.list[i]) for i in range(len(excinfo.list)))

        expected_warnings = [
            'Energy capacity constraint removed',
            'Resource capacity constraint defined and set to infinity for all supply_plus techs',
            'Resource capacity constraint removed'
        ]

        assert all(warning in all_warnings for warning in expected_warnings)
        assert all(model.results.timesteps == pd.date_range('2005-07', '2005-07-04 23:00:00', freq='H'))
