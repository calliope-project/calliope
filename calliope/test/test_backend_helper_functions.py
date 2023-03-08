import pytest

import numpy as np

from calliope.backend import helper_functions


@pytest.mark.parametrize("over", ["techs", ["techs"]])
def test_backend_sum_one_dim(dummy_model_data, over):
    backend_sum = helper_functions.backend_sum()
    summed_array = backend_sum(dummy_model_data.only_techs, over=over)
    assert not summed_array.shape
    assert summed_array == 6


@pytest.mark.parametrize("over", ["techs", ["techs"]])
def test_backend_sum_one_of_two_dims(dummy_model_data, over):
    backend_sum = helper_functions.backend_sum()
    summed_array = backend_sum(dummy_model_data.with_inf, over=over)
    assert summed_array.shape == (2,)
    assert np.array_equal(summed_array, [5, np.inf])


def test_backend_sum_two_dims(dummy_model_data):
    backend_sum = helper_functions.backend_sum()
    summed_array = backend_sum(
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
    squeezed = squeeze_carriers(dummy_model_data.all_true_carriers, ["out"])

    assert squeezed.sum() == 3
    assert squeezed.max() == 1
    assert not set(squeezed.dims).symmetric_difference(["techs"])


def test_squeeze_primary_carriers_missing_brackets(dummy_model_data):
    squeeze_carriers = helper_functions.squeeze_primary_carriers(dummy_model_data)
    with pytest.raises(AttributeError):
        squeeze_carriers(dummy_model_data.all_true_carriers, "out")


def test_get_connected_link(dummy_model_data):
    get_connected_link = helper_functions.get_connected_link(dummy_model_data)
    connected_link = get_connected_link(dummy_model_data.with_inf)
    assert np.array_equal(
        connected_link, [[np.inf, np.nan, 2, 3], [3, 2, 1, np.nan]], equal_nan=True
    )


@pytest.mark.parametrize(["ix", "expected"], [(0, "foo"), (1, "bar"), (-1, "bar")])
def test_get_timestep(dummy_model_data, ix, expected):
    get_timestep = helper_functions.get_timestep(dummy_model_data)
    assert get_timestep(ix) == expected


@pytest.mark.parametrize(["to_roll", "expected"], [(1, 3), (1.0, 3), (-3.0, 3)])
def test_roll(dummy_model_data, to_roll, expected):
    roll = helper_functions.roll()
    rolled = roll(dummy_model_data.with_inf, techs=to_roll)
    assert rolled.sel(nodes="foo", techs="foobar") == expected
