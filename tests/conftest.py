import importlib
from itertools import chain, combinations

import numpy as np
import pytest
import xarray as xr
from calliope.attrdict import AttrDict
from calliope.backend import latex_backend_model, pyomo_backend_model
from calliope.util.schema import CONFIG_SCHEMA, MODEL_SCHEMA, extract_from_schema

from .common.util import build_test_model as build_model

ALL_DIMS = {"nodes", "techs", "carriers", "costs", "timesteps"}
CONFIG_DIR = importlib.resources.files("calliope") / "config"


@pytest.fixture(
    scope="session",
    params=set(
        chain.from_iterable(combinations(ALL_DIMS, i) for i in range(1, len(ALL_DIMS)))
    ),
)
def foreach(request):
    return request.param


@pytest.fixture(scope="session")
def config_defaults():
    return AttrDict(extract_from_schema(CONFIG_SCHEMA, "default"))


@pytest.fixture(scope="session")
def model_defaults():
    return AttrDict(extract_from_schema(MODEL_SCHEMA, "default"))


@pytest.fixture(scope="session")
def simple_supply():
    m = build_model({}, "simple_supply,two_hours,investment_costs")
    m.build()
    m.solve()
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
def dummy_model_data(config_defaults, model_defaults):
    coords = {
        dim: ["foo", "bar"]
        if dim != "techs"
        else ["foobar", "foobaz", "barfoo", "bazfoo"]
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
            "all_inf": (["nodes", "techs"], np.ones((2, 4)) * np.inf, {"is_result": 1}),
            "all_nan": (["nodes", "techs"], np.ones((2, 4)) * np.nan),
            "all_false": (["nodes", "techs"], np.zeros((2, 4)).astype(bool)),
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
            "parent": (["techs"], ["supply", "transmission", "demand", "conversion"]),
            "nodes_inheritance": (["nodes"], ["foo,bar", "boo"]),
            "nodes_inheritance_boo_bool": (["nodes"], [False, True]),
            "techs_inheritance": (["techs"], ["foo,bar", np.nan, "baz", "boo"]),
            "techs_inheritance_boo_bool": (["techs"], [False, False, False, True]),
            "multi_inheritance_boo_bool": (
                ["nodes", "techs"],
                [[False, False, False, False], [False, False, False, True]],
            ),
            "primary_carrier_out": (
                ["carriers", "techs"],
                [[1.0, np.nan, 1.0, np.nan], [np.nan, 1.0, np.nan, np.nan]],
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
        },
        attrs={"scenarios": ["foo"]},
    )
    # xarray forces np.nan to strings if all other values are strings.
    for k in ["lookup_multi_dim_nodes", "lookup_multi_dim_techs", "lookup_techs"]:
        model_data[k] = model_data[k].where(model_data[k] != "nan")

    for param in model_data.data_vars.values():
        param.attrs["is_result"] = 0

    config_defaults.update(
        AttrDict(
            {
                "build": {
                    "foo": True,
                    "FOO": "baz",
                    "foo1": np.inf,
                    "bar": {"foobar": "baz"},
                    "a_b": 0,
                    "b_a": [1, 2],
                }
            }
        )
    )
    model_data.attrs["config"] = config_defaults

    model_data.attrs["defaults"] = AttrDict(
        {
            "all_inf": np.inf,
            "all_nan": np.nan,
            "with_inf": 100,
            "only_techs": 5,
            **model_defaults,
        }
    )
    model_data.attrs["math"] = AttrDict(
        {"constraints": {}, "variables": {}, "global_expressions": {}, "objectives": {}}
    )
    return model_data


def populate_backend_model(backend):
    backend.add_variable(
        "multi_dim_var",
        {
            "foreach": ["nodes", "techs"],
            "where": "with_inf",
            "bounds": {"min": -np.inf, "max": np.inf},
        },
    )
    backend.add_variable("no_dim_var", {"bounds": {"min": -1, "max": 1}})
    backend.add_global_expression(
        "multi_dim_expr",
        {
            "foreach": ["nodes", "techs"],
            "where": "all_true",
            "equations": [{"expression": "multi_dim_var * all_true"}],
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
        },
    )
    return backend


@pytest.fixture(scope="module")
def dummy_pyomo_backend_model(dummy_model_data):
    backend = pyomo_backend_model.PyomoBackendModel(dummy_model_data)
    return populate_backend_model(backend)


@pytest.fixture(scope="module")
def dummy_latex_backend_model(dummy_model_data):
    backend = latex_backend_model.LatexBackendModel(dummy_model_data)
    return populate_backend_model(backend)


@pytest.fixture(scope="class")
def valid_latex_backend(dummy_model_data):
    backend = latex_backend_model.LatexBackendModel(dummy_model_data, include="valid")
    return populate_backend_model(backend)
