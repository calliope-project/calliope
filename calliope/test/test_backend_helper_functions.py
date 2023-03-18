import pytest

import numpy as np
import xarray as xr

from calliope.backend import helper_functions
from calliope import exceptions
from calliope.test.common.util import check_error_or_warning


def test_inheritance(dummy_model_data):
    inheritance = helper_functions.inheritance(dummy_model_data)
    boo_bool = inheritance("boo")
    assert boo_bool.equals(dummy_model_data.boo_inheritance_bool)


def test_imask_sum_not_exists(dummy_model_data):
    imask_sum = helper_functions.imask_sum(dummy_model_data)
    summed = imask_sum("foo", over="techs")
    assert summed.equals(xr.DataArray(False))


@pytest.mark.parametrize(
    ["var", "over", "expected"],
    [("with_inf", "techs", "nodes_true"), ("all_nan", "techs", "nodes_false")],
)
def test_imask_sum_exists(dummy_model_data, var, over, expected):
    imask_sum = helper_functions.imask_sum(dummy_model_data)
    summed = imask_sum(var, over=over)
    assert summed.equals(dummy_model_data[expected])


@pytest.mark.parametrize("over", ["techs", ["techs"]])
def test_expression_sum_one_dim(dummy_model_data, over):
    expression_sum = helper_functions.expression_sum()
    summed_array = expression_sum(dummy_model_data.only_techs, over=over)
    assert not summed_array.shape
    assert summed_array == 6


@pytest.mark.parametrize("over", ["techs", ["techs"]])
def test_expression_sum_one_of_two_dims(dummy_model_data, over):
    expression_sum = helper_functions.expression_sum()
    summed_array = expression_sum(dummy_model_data.with_inf, over=over)
    assert summed_array.shape == (2,)
    assert np.array_equal(summed_array, [5, np.inf])


def test_expression_sum_two_dims(dummy_model_data):
    expression_sum = helper_functions.expression_sum()
    summed_array = expression_sum(
        dummy_model_data.with_inf_as_bool, over=["nodes", "techs"]
    )
    assert not summed_array.shape
    assert summed_array == 5


def test_squeeze_carriers(dummy_model_data):
    squeeze_carriers = helper_functions.squeeze_carriers(dummy_model_data)
    squeezed = squeeze_carriers(dummy_model_data.all_true_carriers, "foo")

    assert dummy_model_data.carrier.sel(carrier_tiers="foo").sum() == squeezed.sum()
    assert not set(squeezed.dims).symmetric_difference(["techs"])


def test_squeeze_primary_carriers(dummy_model_data):
    squeeze_carriers = helper_functions.squeeze_primary_carriers(dummy_model_data)
    squeezed = squeeze_carriers(dummy_model_data.all_true_carriers, "out")

    assert squeezed.sum() == 3
    assert squeezed.max() == 1
    assert not set(squeezed.dims).symmetric_difference(["techs"])


def test_squeeze_primary_carriers_not_in_model(dummy_model_data):
    squeeze_carriers = helper_functions.squeeze_primary_carriers(dummy_model_data)
    with pytest.raises(AttributeError):
        squeeze_carriers(dummy_model_data.all_true_carriers, "foo")


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
def test_select_from_lookup_table(dummy_model_data, lookup, expected):
    select_from_lookup_table = helper_functions.select_from_lookup_table(
        dummy_model_data
    )

    new_array = select_from_lookup_table(
        dummy_model_data.with_inf, **{k: dummy_model_data[v] for k, v in lookup.items()}
    )
    assert np.array_equal(
        new_array.transpose(*dummy_model_data.with_inf.dims), expected, equal_nan=True
    )


def test_select_from_lookup_table_fail_dim_not_in_component(dummy_model_data):
    select_from_lookup_table = helper_functions.select_from_lookup_table(
        dummy_model_data
    )
    with pytest.raises(exceptions.BackendError) as excinfo:
        select_from_lookup_table(
            dummy_model_data.nodes_true,
            techs=dummy_model_data.lookup_techs,
        )
    assert check_error_or_warning(
        excinfo,
        "Cannot select items from `nodes_true` on the dimensions {'techs'} since the array is not indexed over the dimensions {'techs'}",
    )


def test_select_from_lookup_table_fail_dim_slicer_mismatch(dummy_model_data):
    select_from_lookup_table = helper_functions.select_from_lookup_table(
        dummy_model_data
    )
    with pytest.raises(exceptions.BackendError) as excinfo:
        select_from_lookup_table(
            dummy_model_data.with_inf,
            techs=dummy_model_data.lookup_techs,
            nodes=dummy_model_data.link_remote_nodes,
        )

    assert check_error_or_warning(
        excinfo,
        ["lookup tables used to select items from `with_inf", "'techs'", "'nodes'"],
    )


@pytest.mark.parametrize(["ix", "expected"], [(0, "foo"), (1, "bar"), (-1, "bar")])
def test_get_val_at_index(dummy_model_data, ix, expected):
    get_val_at_index = helper_functions.get_val_at_index(dummy_model_data)
    assert get_val_at_index(dim="timesteps", idx=ix) == expected


@pytest.mark.parametrize(["to_roll", "expected"], [(1, 3), (1.0, 3), (-3.0, 3)])
def test_roll(dummy_model_data, to_roll, expected):
    roll = helper_functions.roll()
    rolled = roll(dummy_model_data.with_inf, techs=to_roll)
    assert rolled.sel(nodes="foo", techs="foobar") == expected
