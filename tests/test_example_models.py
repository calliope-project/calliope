import shutil
from pathlib import Path

import pandas as pd
import pytest

import calliope
from calliope.io import read_rich_yaml

from .common.util import check_error_or_warning

approx = pytest.approx


class TestModelPreprocessing:
    def test_preprocess_national_scale(self):
        calliope.examples.national_scale()

    @pytest.mark.time_intensive
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

    def test_preprocess_operate_milp(self):
        calliope.examples.operate_milp()


class TestNationalScaleExampleModelSenseChecks:
    @pytest.fixture(scope="class")
    def nat_model_from_data_tables(self):
        df = pd.read_csv(
            calliope.examples._EXAMPLE_MODEL_DIR
            / "national_scale"
            / "data_tables"
            / "time_varying_params.csv",
            index_col=0,
            header=[0, 1, 2, 3],
        )
        model = calliope.read_yaml(
            Path(__file__).parent
            / "common"
            / "national_scale_from_data_tables"
            / "model.yaml",
            data_table_dfs={"time_varying_df": df},
            subset={"timesteps": ["2005-01-01", "2005-01-01"]},
        )
        model.build()
        return model

    @pytest.fixture(scope="class")
    def nat_model(self):
        model = calliope.examples.national_scale(
            subset={"timesteps": ["2005-01-01", "2005-01-01"]}
        )
        model.build()
        return model

    @pytest.fixture(params=["nat_model", "nat_model_from_data_tables"])
    def example_tester(self, request):
        def _example_tester(solver="cbc", solver_io=None):
            model = request.getfixturevalue(request.param)

            solve_kwargs = {"solver": solver}
            solve_kwargs["solver_io"] = solver_io

            model.solve(force=True, **solve_kwargs)

            assert model.results.storage_cap.sel(
                nodes="region1_1", techs="csp"
            ) == approx(45129.950)
            assert model.results.storage_cap.sel(
                nodes="region2", techs="battery"
            ) == approx(6675.173)

            assert model.results.flow_cap.sel(nodes="region1_1", techs="csp") == approx(
                4626.588
            )
            assert model.results.flow_cap.sel(
                nodes="region2", techs="battery"
            ) == approx(1000)
            assert model.results.flow_cap.sel(nodes="region1", techs="ccgt") == approx(
                30000
            )

            assert float(model.results.cost.sum()) == approx(38988.7442)

            assert float(
                model.results.systemwide_levelised_cost.sel(
                    carriers="power", techs="battery"
                ).item()
            ) == approx(0.063543, abs=0.000001)
            assert float(
                model.results.systemwide_capacity_factor.sel(
                    carriers="power", techs="battery"
                ).item()
            ) == approx(0.2642256, abs=0.000001)

            model.results.total_levelised_cost.item() == approx(0.05456, abs=1e-5)

        return _example_tester

    def test_nationalscale_example_results_cbc(self, example_tester):
        example_tester()

    @pytest.mark.needs_gurobi_license
    def test_nationalscale_example_results_gurobi(self, example_tester):
        pytest.importorskip("gurobipy")
        example_tester(solver="gurobi", solver_io="python")

    def test_nationalscale_example_results_cplex(self, example_tester):
        if shutil.which("cplex"):
            example_tester(solver="cplex")
        else:
            pytest.skip("CPLEX not installed")

    def test_nationalscale_example_results_glpk(self, example_tester):
        if shutil.which("glpsol"):
            example_tester(solver="glpk")
        else:
            pytest.skip("GLPK not installed")

    def test_fails_gracefully_without_timeseries(self):
        override = {"data_tables": {"_REPLACE_": {}}}
        with pytest.raises(calliope.exceptions.ModelError) as excinfo:
            calliope.examples.national_scale(override_dict=override)

        assert check_error_or_warning(
            excinfo,
            "Must define at least one timeseries data input in a Calliope model.",
        )


class TestNationalScaleExampleModelOperate:
    def example_tester(self):
        model = calliope.examples.national_scale(
            subset={"timesteps": ["2005-01-01", "2005-01-03"]}, scenario="operate"
        )
        model.build()
        model.solve()

        assert all(
            model.results.timesteps
            == pd.date_range("2005-01", "2005-01-03 23:00:00", freq="h")
        )

    def test_nationalscale_example_results_cbc(self):
        self.example_tester()


class TestNationalScaleResampledExampleModelSenseChecks:
    def example_tester(self, solver="cbc", solver_io=None):
        solve_kwargs = {"solver": solver}
        if solver_io:
            solve_kwargs["solver_io"] = solver_io

        model = calliope.examples.time_resampling(
            subset={"timesteps": ["2005-01-01", "2005-01-01"]}
        )
        model.build()
        model.solve(**solve_kwargs)

        assert model.results.storage_cap.sel(nodes="region1_1", techs="csp") == approx(
            23563.444
        )
        assert model.results.storage_cap.sel(
            nodes="region2", techs="battery"
        ) == approx(6315.78947)

        assert model.results.flow_cap.sel(nodes="region1_1", techs="csp") == approx(
            1440.8377
        )
        assert model.results.flow_cap.sel(nodes="region2", techs="battery") == approx(
            1000
        )
        assert model.results.flow_cap.sel(nodes="region1", techs="ccgt") == approx(
            30000
        )

        assert float(model.results.cost.sum()) == approx(37344.221869)

        assert float(
            model.results.systemwide_levelised_cost.loc[
                {"carriers": "power", "techs": "battery"}
            ].item()
        ) == approx(0.063543, abs=0.000001)
        assert float(
            model.results.systemwide_capacity_factor.loc[
                {"carriers": "power", "techs": "battery"}
            ].item()
        ) == approx(0.25, abs=0.000001)

    def test_nationalscale_example_results_cbc(self):
        self.example_tester()

    def test_nationalscale_resampled_example_results_glpk(self):
        if shutil.which("glpsol"):
            self.example_tester(solver="glpk")
        else:
            pytest.skip("GLPK not installed")


class TestUrbanScaleExampleModelSenseChecks:
    def example_tester(self, source_unit, solver="cbc", solver_io=None):
        data_tables = f"data_tables.pv_resource.select.scaler: {source_unit}"
        unit_override = {
            "techs.pv.source_unit": source_unit,
            **read_rich_yaml(data_tables),
        }

        model = calliope.examples.urban_scale(
            override_dict=unit_override,
            subset={"timesteps": ["2005-07-01", "2005-07-01"]},
        )

        solve_kwargs = {"solver": solver}
        if solver_io:
            solve_kwargs["solver_io"] = solver_io

        model.build()
        model.solve(**solve_kwargs)

        assert model.results.flow_cap.sel(
            nodes="X1", techs="chp", carriers="electricity"
        ) == approx(250.090112)

        # GLPK isn't able to get the same answer both times, so we have to account for that here
        if source_unit == "per_cap" and solver == "glpk":
            heat_pipe_approx = 183.45825
        else:
            heat_pipe_approx = 182.19260

        assert model.results.flow_cap.sel(
            nodes="X2", techs="N1_to_X2", carriers="heat"
        ) == approx(heat_pipe_approx)

        assert model.results.flow_out.sum("timesteps").sel(
            carriers="heat", nodes="X3", techs="boiler"
        ) == approx(0.18720)
        assert model.results.area_use.sel(nodes="X2", techs="pv") == approx(830.064659)

        assert float(model.results.flow_export.sum()) == approx(122.7156)

        # GLPK doesn't agree with commercial solvers, so we have to account for that here
        cost_sum = 430.097399 if solver == "glpk" else 430.082290
        assert float(model.results.cost.sum()) == approx(cost_sum)

    def test_urban_example_results_area(self):
        self.example_tester("per_area")

    @pytest.mark.needs_gurobi_license
    def test_urban_example_results_area_gurobi(self):
        pytest.importorskip("gurobipy")
        self.example_tester("per_area", solver="gurobi", solver_io="python")

    def test_urban_example_results_cap(self):
        self.example_tester("per_cap")

    @pytest.mark.needs_gurobi_license
    def test_urban_example_results_cap_gurobi(self):
        pytest.importorskip("gurobipy")
        self.example_tester("per_cap", solver="gurobi", solver_io="python")

    def test_milp_example_results(self):
        model = calliope.examples.milp(
            subset={"timesteps": ["2005-01-01", "2005-01-01"]}
        )
        model.build()
        model.solve(solver_options={"mipgap": 0.001})

        assert (
            model.results.flow_cap.sel(nodes="X1", techs="chp", carriers="electricity")
            == 300
        )
        assert model.results.flow_cap.sel(
            nodes="X2", techs="N1_to_X2", carriers="heat"
        ) == approx(188.363137)

        assert model.results.flow_out.sum("timesteps").sel(
            carriers="gas", nodes="X1", techs="supply_gas"
        ) == approx(12363.173036)
        assert float(model.results.flow_export.sum()) == approx(0)

        assert model.results.purchased_units.sel(nodes="X2", techs="boiler") == 1
        assert model.results.purchased_units.sel(nodes="X1", techs="chp") == 1

        assert float(model.results.operating_units.sum()) == 24

        assert float(model.results.cost.sum()) == approx(540.780779)

    @pytest.mark.time_intensive
    def test_operate_example_results(self):
        model = calliope.examples.operate(
            subset={"timesteps": ["2005-07-01", "2005-07-04"]}
        )

        model.build()
        model.solve()

        # TODO: introduce some of these warnings ?
        _ = [
            "Flow capacity constraint removed",
            "Source capacity constraint defined and set to infinity for all supply_plus techs",
            "Storage cannot be cyclic in operate run mode, setting `run.cyclic_storage` to False for this run",
        ]

        assert all(
            model.results.timesteps
            == pd.date_range("2005-07", "2005-07-04 23:00:00", freq="h")
        )

    @pytest.mark.time_intensive
    def test_operate_milp_results(self):
        """Ensure mixing operate and MILP math works adequately."""
        model = calliope.examples.operate_milp(
            subset={"timesteps": ["2005-01-01", "2005-01-01"]}
        )

        model.build()
        model.solve()

        assert model.results.flow_out.sum("timesteps").sel(
            carriers="heat", techs="chp", nodes="X1"
        ) == approx(4234.869616)
        assert model.results.flow_export.sum() == approx(309.77935138)
        assert model.results.cost.sum() == approx(8471.15341812)
