import pytest
from itertools import chain, combinations

import xarray as xr
import numpy as np

from calliope.test.common.util import build_test_model as build_model
from calliope import AttrDict


ALL_DIMS = {"nodes", "techs", "carriers", "costs", "timesteps", "carrier_tiers"}


@pytest.fixture(
    scope="session",
    params=set(
        chain.from_iterable(combinations(ALL_DIMS, i) for i in range(1, len(ALL_DIMS)))
    ),
)
def foreach(request):
    return request.param


@pytest.fixture(scope="session")
def simple_supply():
    m = build_model({}, "simple_supply,two_hours,investment_costs")
    m.run()
    return m


@pytest.fixture(scope="class")
def simple_supply_new_build():
    m = build_model({}, "simple_supply,two_hours,investment_costs")
    m.build()
    m.solve()
    return m


@pytest.fixture(scope="session")
def supply_milp():
    m = build_model({}, "supply_milp,two_hours,investment_costs")
    m.run()
    return m


@pytest.fixture(scope="session")
def storage_milp():
    m = build_model({}, "storage_milp,two_hours,investment_costs")
    m.run()
    return m


@pytest.fixture(scope="session")
def conversion_plus_milp():
    m = build_model({}, "conversion_plus_milp,two_hours,investment_costs")
    m.run()
    return m


@pytest.fixture(scope="session")
def supply_and_supply_plus_milp():
    m = build_model({}, "supply_and_supply_plus_milp,two_hours,investment_costs")
    m.run()
    return m


@pytest.fixture(scope="session")
def simple_supply_and_supply_plus():
    m = build_model({}, "simple_supply_and_supply_plus,two_hours,investment_costs")
    m.run()
    return m


@pytest.fixture(scope="session")
def simple_storage():
    m = build_model({}, "simple_storage,two_hours,investment_costs")
    m.run()
    return m


@pytest.fixture(scope="session")
def simple_conversion():
    m = build_model({}, "simple_conversion,two_hours,investment_costs")
    m.run()
    return m


@pytest.fixture(scope="session")
def supply_export():
    m = build_model({}, "supply_export,two_hours,investment_costs")
    m.run()
    return m


@pytest.fixture(scope="session")
def supply_purchase():
    m = build_model({}, "supply_purchase,two_hours,investment_costs")
    m.run()
    return m


@pytest.fixture(scope="session")
def conversion_plus_purchase():
    m = build_model({}, "conversion_plus_purchase,two_hours,investment_costs")
    m.run()
    return m


@pytest.fixture(scope="session")
def storage_purchase():
    m = build_model({}, "storage_purchase,two_hours,investment_costs")
    m.run()
    return m


@pytest.fixture(scope="session")
def simple_conversion_plus():
    m = build_model({}, "simple_conversion_plus,two_hours,investment_costs")
    m.run()
    return m


@pytest.fixture(scope="module")
def dummy_model_data():
    model_data = xr.Dataset(
        coords={
            dim: ["foo", "bar"]
            if dim != "techs"
            else ["foobar", "foobaz", "barfoo", "bazfoo"]
            for dim in ALL_DIMS
        },
        data_vars={
            "node_tech": (
                ["nodes", "techs"],
                np.random.choice(a=[np.nan, True], p=[0.05, 0.95], size=(2, 4)),
            ),
            "carrier": (
                ["carrier_tiers", "carriers", "techs"],
                np.random.choice(a=[np.nan, True], p=[0.05, 0.95], size=(2, 2, 4)),
            ),
            "with_inf": (
                ["nodes", "techs"],
                [[1.0, np.nan, 1.0, 3], [np.inf, 2.0, True, np.nan]],
            ),
            "only_techs": (["techs"], [np.nan, 1, 2, 3]),
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
            "inheritance": (
                ["nodes", "techs"],
                [
                    ["foo.bar", "boo", "baz", "boo"],
                    ["bar", "ar", "baz.boo", "foo.boo"],
                ],
            ),
            "boo_inheritance_bool": (
                ["nodes", "techs"],
                [[False, True, False, True], [False, False, True, True]],
            ),
            "primary_carrier_out": (
                ["carriers", "techs"],
                [[1.0, np.nan, 1.0, np.nan], [np.nan, 1.0, np.nan, np.nan]],
            ),
            "lookup_techs": (
                ["techs"],
                ["foobar", np.nan, "foobaz", np.nan],
            ),
            "link_remote_nodes": (
                ["nodes", "techs"],
                [["bar", np.nan, "bar", np.nan], ["foo", np.nan, np.nan, np.nan]],
            ),
            "link_remote_techs": (
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
    for k in ["link_remote_nodes", "link_remote_techs", "lookup_techs"]:
        model_data[k] = model_data[k].where(model_data[k] != "nan")

    model_data.attrs["run_config"] = AttrDict(
        {"foo": True, "bar": {"foobar": "baz"}, "foobar": {"baz": {"foo": np.inf}}}
    )
    model_data.attrs["model_config"] = AttrDict({"a_b": 0, "b_a": [1, 2]})

    model_data.attrs["defaults"] = AttrDict(
        {"all_inf": np.inf, "all_nan": np.nan, "with_inf": 100, "only_techs": 5}
    )
    return model_data
