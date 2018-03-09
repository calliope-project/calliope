import shutil

import pytest
from pytest import approx

import calliope


class TestNationalScaleExampleModel:
    def test_preprocess_national_scale(self):
        model = calliope.examples.national_scale()

    def test_preprocess_time_clustering(self):
        model = calliope.examples.time_clustering()

    def test_preprocess_time_resampling(self):
        model = calliope.examples.time_resampling()


class TestUrbanScaleExampleModel:
    def test_preprocess_urban_scale(self):
        model = calliope.examples.urban_scale()

    def test_preprocess_milp(self):
        model = calliope.examples.milp()


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


class TestUrbanScaleExampleModelSenseChecks:
    def test_urbanscale_example_results(self):
        model = calliope.examples.urban_scale(
            override_dict={'model.subset_time': '2005-01-01'}
        )
        model.run()

    def test_preprocess_milp(self):
        model = calliope.examples.milp(
            override_dict={'model.subset_time': '2005-01-01'}
        )
        model.run()
