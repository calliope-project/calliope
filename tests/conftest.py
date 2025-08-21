from itertools import chain, combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from calliope.backend import latex_backend_model, pyomo_backend_model
from calliope.schemas import config_schema, math_schema

from .common.util import build_test_model as build_model

ALL_DIMS = {"nodes", "techs", "carriers", "costs", "timesteps"}

# Set the global numpy random seed to avoid occasional (random!) test failures when random sampling is used in the core code.
np.random.seed(0)


@pytest.fixture(scope="session")
def dummy_int() -> int:
    """Dummy integer value that will never be confused by a model value/default."""
    return 0xDEADBEEF


@pytest.fixture(scope="session")
def minimal_test_model_path():
    return (Path(__file__).parent / "common" / "test_model" / "model.yaml").as_posix()


@pytest.fixture(
    scope="session",
    params=set(
        chain.from_iterable(combinations(ALL_DIMS, i) for i in range(1, len(ALL_DIMS)))
    ),
)
def foreach(request):
    return request.param


@pytest.fixture(scope="session")
def default_config():
    return config_schema.CalliopeConfig()


@pytest.fixture(scope="session")
def data_source_dir():
    return Path(__file__).parent / "common" / "test_model" / "data_tables"


@pytest.fixture(scope="session")
def simple_supply():
    m = build_model({}, "simple_supply,two_hours,investment_costs")
    m.build()
    m.solve()
    return m


@pytest.fixture
def simple_supply_build_func():
    m = build_model({}, "simple_supply,two_hours,investment_costs")
    m.build()
    return m


@pytest.fixture(scope="session")
def supply_milp():
    m = build_model({}, "supply_milp,two_hours,investment_costs")
    m.build()
    m.solve()
    return m


@pytest.fixture(scope="session")
def storage_milp():
    m = build_model({}, "storage_milp,two_hours,investment_costs")
    m.build()
    m.solve()
    return m


@pytest.fixture(scope="session")
def conversion_plus_milp():
    m = build_model({}, "conversion_plus_milp,two_hours,investment_costs")
    m.build()
    m.solve()
    return m


@pytest.fixture(scope="session")
def supply_and_supply_plus_milp():
    m = build_model({}, "supply_and_supply_plus_milp,two_hours,investment_costs")
    m.build()
    m.solve()
    return m


@pytest.fixture(scope="session")
def simple_supply_and_supply_plus():
    m = build_model({}, "simple_supply_and_supply_plus,two_hours,investment_costs")
    m.build()
    m.solve()
    return m


@pytest.fixture(scope="session")
def simple_supply_spores_ready():
    m = build_model(
        {}, "var_costs,simple_supply_spores_ready,two_hours,investment_costs"
    )
    m.build()
    m.solve()
    return m


@pytest.fixture(scope="session")
def simple_storage():
    m = build_model({}, "simple_storage,two_hours,investment_costs")
    m.build()
    m.solve()
    return m


@pytest.fixture(scope="session")
def simple_conversion():
    m = build_model({}, "simple_conversion,two_hours,investment_costs")
    m.build()
    m.solve()
    return m


@pytest.fixture(scope="session")
def supply_export():
    m = build_model({}, "supply_export,two_hours,investment_costs")
    m.build()
    m.solve()
    return m


@pytest.fixture(scope="session")
def supply_purchase():
    m = build_model({}, "supply_purchase,two_hours,investment_costs")
    m.build()
    m.solve()
    return m


@pytest.fixture(scope="session")
def conversion_plus_purchase():
    m = build_model({}, "conversion_plus_purchase,two_hours,investment_costs")
    m.build()
    m.solve()
    return m


@pytest.fixture(scope="session")
def storage_purchase():
    m = build_model({}, "storage_purchase,two_hours,investment_costs")
    m.build()
    m.solve()
    return m


@pytest.fixture(scope="session")
def simple_conversion_plus():
    m = build_model({}, "simple_conversion_plus,two_hours,investment_costs")
    m.build()
    m.solve()
    return m


@pytest.fixture(scope="module")
def dummy_model_data():
    coords = {
        dim: (
            pd.to_datetime(
                [
                    "2000-01-01 00:00",
                    "2000-01-01 01:00",
                    "2000-01-01 02:00",
                    "2000-01-01 03:00",
                ]
            )
            if dim == "timesteps"
            else ["foo", "bar"]
            if dim != "techs"
            else ["foobar", "foobaz", "barfoo", "bazfoo"]
        )
        for dim in ALL_DIMS
    }
    carrier_dims = ("nodes", "techs", "carriers")
    node_tech_dims = ("nodes", "techs")
    carrier_in = xr.DataArray(
        [
            [[True, True], [True, True], [True, True], [True, True]],
            [[True, True], [False, False], [True, False], [True, False]],
        ],
        dims=carrier_dims,
        coords={k: v for k, v in coords.items() if k in carrier_dims},
    )
    carrier_out = xr.DataArray(
        [
            [[True, True], [True, True], [False, True], [True, True]],
            [[True, True], [True, True], [True, True], [True, True]],
        ],
        dims=carrier_dims,
        coords={k: v for k, v in coords.items() if k in carrier_dims},
    )
    node_tech = xr.DataArray(
        [[False, True, True, True], [True, True, False, False]],
        dims=node_tech_dims,
        coords={k: v for k, v in coords.items() if k in node_tech_dims},
    )

    model_data = xr.Dataset(
        coords=coords,
        data_vars={
            "definition_matrix": node_tech & (carrier_in | carrier_out),
            "carrier_in": carrier_in,
            "carrier_out": carrier_out,
            "with_inf": (
                ["nodes", "techs"],
                [[1.0, np.nan, 1.0, 3], [np.inf, 2.0, True, np.nan]],
            ),
            "only_techs": (["techs"], [np.nan, 1, 2, 3]),
            "no_dims": (2),
            "all_inf": (["nodes", "techs"], np.ones((2, 4)) * np.inf),
            "all_nan": (["nodes", "techs"], np.ones((2, 4)) * np.nan),
            "all_false": (["nodes", "techs"], np.zeros((2, 4)).astype(bool)),
            "all_ones": (["nodes", "techs"], np.ones((2, 4))),
            "all_true": (["nodes", "techs"], np.ones((2, 4)).astype(bool)),
            "all_true_carriers": (["carriers", "techs"], np.ones((2, 4)).astype(bool)),
            "nodes_true": (["nodes"], [True, True]),
            "nodes_false": (["nodes"], [False, False]),
            "with_inf_as_bool": (
                ["nodes", "techs"],
                [[True, False, True, True], [False, True, True, False]],
            ),
            "with_inf_as_bool_and_subset_on_bar_in_nodes": (
                ["nodes", "techs"],
                [[False, False, False, False], [False, True, True, False]],
            ),
            "with_inf_as_bool_or_subset_on_bar_in_nodes": (
                ["nodes", "techs"],
                [[True, False, True, True], [True, True, True, True]],
            ),
            "only_techs_as_bool": (["techs"], [False, True, True, True]),
            "with_inf_and_only_techs_as_bool": (
                ["nodes", "techs"],
                [[False, False, True, True], [False, True, True, False]],
            ),
            "with_inf_or_only_techs_as_bool": (
                ["nodes", "techs"],
                [[True, True, True, True], [False, True, True, True]],
            ),
            "base_tech": (
                ["techs"],
                ["supply", "transmission", "demand", "conversion"],
            ),
            "lookup_techs": (["techs"], ["foobar", np.nan, "foobaz", np.nan]),
            "lookup_techs_no_match": (["techs"], ["foo", np.nan, "bar", np.nan]),
            "lookup_multi_dim_nodes": (
                ["nodes", "techs"],
                [["bar", np.nan, "bar", np.nan], ["foo", np.nan, np.nan, np.nan]],
            ),
            "lookup_multi_dim_techs": (
                ["nodes", "techs"],
                [
                    ["foobar", np.nan, "foobaz", np.nan],
                    ["bazfoo", np.nan, np.nan, np.nan],
                ],
            ),
            "timeseries_data": (["timesteps"], [1, 1, 1, 1]),
            "timeseries_nodes_data": (["nodes", "timesteps"], np.ones((2, 4))),
        },
        attrs={"scenarios": ["foo"]},
    )
    # xarray forces np.nan to strings if all other values are strings.
    for k in ["lookup_multi_dim_nodes", "lookup_multi_dim_techs", "lookup_techs"]:
        model_data[k] = model_data[k].where(model_data[k] != "nan")

    # This value is set on the parameter directly to ensure it finds its way through to the LaTex math.
    model_data.no_dims.attrs["default"] = 0

    return model_data


@pytest.fixture(scope="module")
def dummy_model_math(dummy_model_data):
    defaults = {
        "all_inf": np.inf,
        "all_nan": np.nan,
        "with_inf": 100,
        "only_techs": 5,
        "no_dims": 0,
    }
    dtype_translator = {
        "f": "float",
        "i": "integer",
        "U": "string",
        "b": "bool",
        "O": "string",
        "M": "datetime",
    }
    dtypes = {k: v.kind for k, v in dummy_model_data.dtypes.items()}
    params = {
        k: {"dtype": dtype_translator[v], "default": defaults.get(k, np.nan)}
        for k, v in dtypes.items()
        if v in ["f", "i"]
    }
    lookups = {
        k: {"dtype": dtype_translator[v]}
        for k, v in dtypes.items()
        if v in ["U", "b", "O"]
    }
    dim_dtypes = {k: v.kind for k, v in dummy_model_data.coords.dtypes.items()}
    dims = {
        k: {"dtype": dtype_translator[v], "iterator": k.removesuffix("s")}
        for k, v in dim_dtypes.items()
    }
    return math_schema.CalliopeBuildMath.model_validate(
        {"parameters": params, "lookups": lookups, "dimensions": dims}
    )


def populate_backend_model(backend):
    backend._load_inputs()
    backend.add_variable(
        "multi_dim_var",
        {
            "foreach": ["nodes", "techs"],
            "where": "with_inf",
            "bounds": {"min": -np.inf, "max": np.inf},
            "domain": "real",
        },
    )
    backend.add_variable(
        "no_dim_var", {"bounds": {"min": -1, "max": 1}, "domain": "real"}
    )
    backend.add_global_expression(
        "multi_dim_expr",
        {
            "foreach": ["nodes", "techs"],
            "where": "all_true",
            "equations": [{"expression": "multi_dim_var * all_ones"}],
            "active": True,
        },
    )
    backend.add_constraint(
        "no_dim_constr",
        {
            "foreach": [],
            "equations": [
                {
                    "expression": "sum(multi_dim_expr, over=[nodes, techs]) + no_dim_var <= 2"
                }
            ],
            "active": True,
        },
    )
    return backend


@pytest.fixture(scope="module")
def dummy_pyomo_backend_model(dummy_model_data, dummy_model_math, default_config):
    backend = pyomo_backend_model.PyomoBackendModel(
        dummy_model_data, dummy_model_math, default_config.build
    )
    return populate_backend_model(backend)


@pytest.fixture(scope="module")
def dummy_latex_backend_model(dummy_model_data, dummy_model_math, default_config):
    backend = latex_backend_model.LatexBackendModel(
        dummy_model_data, dummy_model_math, default_config.build
    )
    return populate_backend_model(backend)


@pytest.fixture(scope="class")
def valid_latex_backend(dummy_model_data, dummy_model_math, default_config):
    backend = latex_backend_model.LatexBackendModel(
        dummy_model_data, dummy_model_math, default_config.build, include="valid"
    )
    return populate_backend_model(backend)
