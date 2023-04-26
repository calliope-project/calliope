import numpy as np
import pytest
import xarray as xr

from calliope import exceptions
from calliope.backend import helper_functions
from calliope.test.common.util import check_error_or_warning


@pytest.fixture(scope="module")
def expression(dummy_model_data):
    return helper_functions.ParsingHelperFuncs(
        "expression", model_data=dummy_model_data
    )


@pytest.fixture(scope="module")
def where(dummy_model_data):
    return helper_functions.ParsingHelperFuncs("where", model_data=dummy_model_data)


@pytest.mark.parametrize(
    ["func_class_str", "func"], [("where", "roll"), ("expression", "inheritance")]
)
def test_func_not_available(request, func_class_str, func):
    func_class = request.getfixturevalue(func_class_str)
    with pytest.raises(exceptions.BackendError) as excinfo:
        getattr(func_class, func)()
    assert check_error_or_warning(
        excinfo,
        f"Helper function `{func}` cannot be used in math `{func_class_str}` strings",
    )


def test_inheritance(dummy_model_data, where):
    boo_bool = where.inheritance("boo")
    assert boo_bool.equals(dummy_model_data.boo_inheritance_bool)


def test_any_not_exists(where):
    summed = where.any("foo", over="techs")
    assert summed.equals(xr.DataArray(False))


@pytest.mark.parametrize(
    ["var", "over", "expected"],
    [("with_inf", "techs", "nodes_true"), ("all_nan", "techs", "nodes_false")],
)
def test_any_exists(where, dummy_model_data, var, over, expected):
    summed = where.any(var, over=over)
    assert summed.equals(dummy_model_data[expected])


@pytest.mark.parametrize("over", ["techs", ["techs"]])
def test_sum_one_dim(expression, dummy_model_data, over):
    summed_array = expression.sum(dummy_model_data.only_techs, over=over)
    assert not summed_array.shape
    assert summed_array == 6


@pytest.mark.parametrize("over", ["techs", ["techs"]])
def test_sum_one_of_two_dims(expression, dummy_model_data, over):
    summed_array = expression.sum(dummy_model_data.with_inf, over=over)
    assert summed_array.shape == (2,)
    assert np.array_equal(summed_array, [5, np.inf])


def test_expression_sum_two_dims(expression, dummy_model_data):
    summed_array = expression.sum(
        dummy_model_data.with_inf_as_bool, over=["nodes", "techs"]
    )
    assert not summed_array.shape
    assert summed_array == 5


def test_reduce_carrier_dim(expression, dummy_model_data):
    reduced = expression.reduce_carrier_dim(dummy_model_data.all_true_carriers, "foo")

    assert dummy_model_data.carrier.sel(carrier_tiers="foo").sum() == reduced.sum()
    assert not set(reduced.dims).symmetric_difference(["techs"])


def test_reduce_primary_carrier_dim(expression, dummy_model_data):
    reduced = expression.reduce_primary_carrier_dim(
        dummy_model_data.all_true_carriers, "out"
    )

    assert reduced.sum() == 3
    assert reduced.max() == 1
    assert not set(reduced.dims).symmetric_difference(["techs"])


def test_reduce_primary_carrier_dim_not_in_model(expression, dummy_model_data):
    with pytest.raises(AttributeError):
        expression.reduce_primary_carrier_dim(dummy_model_data.all_true_carriers, "foo")


@pytest.mark.parametrize(
    ["lookup", "expected"],
    [
        (
            {"techs": "lookup_techs"},
            [[1.0, np.nan, np.nan, np.nan], [np.inf, np.nan, 2.0, np.nan]],
        ),
        (
            {"nodes": "link_remote_nodes", "techs": "link_remote_techs"},
            [[np.inf, np.nan, 2, np.nan], [3, np.nan, np.nan, np.nan]],
        ),
    ],
)
def test_select_from_lookup_arrays(expression, dummy_model_data, lookup, expected):
    new_array = expression.select_from_lookup_arrays(
        dummy_model_data.with_inf, **{k: dummy_model_data[v] for k, v in lookup.items()}
    )
    assert np.array_equal(
        new_array.transpose(*dummy_model_data.with_inf.dims), expected, equal_nan=True
    )
    for dim in dummy_model_data.with_inf.dims:
        assert new_array[dim].equals(dummy_model_data[dim])


def test_select_from_lookup_arrays_fail_dim_not_in_component(
    expression, dummy_model_data
):
    with pytest.raises(exceptions.BackendError) as excinfo:
        expression.select_from_lookup_arrays(
            dummy_model_data.nodes_true,
            techs=dummy_model_data.lookup_techs,
        )
    assert check_error_or_warning(
        excinfo,
        "Cannot select items from `nodes_true` on the dimensions {'techs'} since the array is not indexed over the dimensions {'techs'}",
    )


def test_select_from_lookup_arrays_fail_dim_slicer_mismatch(
    expression, dummy_model_data
):
    with pytest.raises(exceptions.BackendError) as excinfo:
        expression.select_from_lookup_arrays(
            dummy_model_data.with_inf,
            techs=dummy_model_data.lookup_techs,
            nodes=dummy_model_data.link_remote_nodes,
        )

    assert check_error_or_warning(
        excinfo,
        ["lookup arrays used to select items from `with_inf", "'techs'", "'nodes'"],
    )


@pytest.mark.parametrize(["idx", "expected"], [(0, "foo"), (1, "bar"), (-1, "bar")])
def test_get_val_at_index(expression, dummy_model_data, idx, expected):
    assert expression.get_val_at_index(timesteps=idx) == expected


@pytest.mark.parametrize("mapping", [{}, {"foo": 1, "bar": 1}])
def test_get_val_at_index_not_one_mapping(expression, dummy_model_data, mapping):
    with pytest.raises(ValueError) as excinfo:
        expression.get_val_at_index(**mapping)
    assert check_error_or_warning(
        excinfo, "Supply one (and only one) dimension:index mapping"
    )


@pytest.mark.parametrize(["to_roll", "expected"], [(1, 3), (1.0, 3), (-3.0, 3)])
def test_roll(expression, dummy_model_data, to_roll, expected):
    rolled = expression.roll(dummy_model_data.with_inf, techs=to_roll)
    assert rolled.sel(nodes="foo", techs="foobar") == expected
