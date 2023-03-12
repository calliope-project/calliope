import pytest

import numpy as np
import xarray as xr

from calliope.backend import helper_functions


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


def test_get_connected_link(dummy_model_data):
    get_connected_link = helper_functions.get_connected_link(dummy_model_data)
    connected_link = get_connected_link(dummy_model_data.with_inf)
    assert np.array_equal(
        connected_link, [[np.inf, np.nan, 2, 3], [3, 2, 1, np.nan]], equal_nan=True
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
