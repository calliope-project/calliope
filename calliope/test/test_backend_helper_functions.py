import pytest

import numpy as np
import xarray as xr

from calliope.backend import helper_functions


class TestFuncs:
    def test_inheritance(self, dummy_model_data):
        inheritance = helper_functions.inheritance(dummy_model_data)
        boo_bool = inheritance("boo")
        assert boo_bool.equals(dummy_model_data.boo_inheritance_bool)

    def test_imask_sum_not_exists(self, dummy_model_data):
        imask_sum = helper_functions.imask_sum(dummy_model_data)
        summed = imask_sum("foo", over="techs")
        assert summed.equals(xr.DataArray(False))

    @pytest.mark.parametrize(
        ["var", "over", "expected"],
        [("with_inf", "techs", "nodes_true"), ("all_nan", "techs", "nodes_false")],
    )
    def test_imask_sum_exists(self, dummy_model_data, var, over, expected):
        imask_sum = helper_functions.imask_sum(dummy_model_data)
        summed = imask_sum(var, over=over)
        assert summed.equals(dummy_model_data[expected])

    @pytest.mark.parametrize("over", ["techs", ["techs"]])
    def test_expression_sum_one_dim(self, dummy_model_data, over):
        expression_sum = helper_functions.expression_sum()
        summed_array = expression_sum(dummy_model_data.only_techs, over=over)
        assert not summed_array.shape
        assert summed_array == 6

    @pytest.mark.parametrize("over", ["techs", ["techs"]])
    def test_expression_sum_one_of_two_dims(self, dummy_model_data, over):
        expression_sum = helper_functions.expression_sum()
        summed_array = expression_sum(dummy_model_data.with_inf, over=over)
        assert summed_array.shape == (2,)
        assert np.array_equal(summed_array, [5, np.inf])

    def test_expression_sum_two_dims(self, dummy_model_data):
        expression_sum = helper_functions.expression_sum()
        summed_array = expression_sum(
            dummy_model_data.with_inf_as_bool, over=["nodes", "techs"]
        )
        assert not summed_array.shape
        assert summed_array == 5

    def test_squeeze_carriers(self, dummy_model_data):
        squeeze_carriers = helper_functions.squeeze_carriers(dummy_model_data)
        squeezed = squeeze_carriers(dummy_model_data.all_true_carriers, "foo")

        assert dummy_model_data.carrier.sel(carrier_tiers="foo").sum() == squeezed.sum()
        assert not set(squeezed.dims).symmetric_difference(["techs"])

    def test_squeeze_primary_carriers(self, dummy_model_data):
        squeeze_carriers = helper_functions.squeeze_primary_carriers(dummy_model_data)
        squeezed = squeeze_carriers(dummy_model_data.all_true_carriers, "out")

        assert squeezed.sum() == 3
        assert squeezed.max() == 1
        assert not set(squeezed.dims).symmetric_difference(["techs"])

    def test_squeeze_primary_carriers_not_in_model(self, dummy_model_data):
        squeeze_carriers = helper_functions.squeeze_primary_carriers(dummy_model_data)
        with pytest.raises(AttributeError):
            squeeze_carriers(dummy_model_data.all_true_carriers, "foo")

    def test_get_connected_link(self, dummy_model_data):
        get_connected_link = helper_functions.get_connected_link(dummy_model_data)
        connected_link = get_connected_link(dummy_model_data.with_inf)
        assert np.array_equal(
            connected_link, [[np.inf, np.nan, 2, 3], [3, 2, 1, np.nan]], equal_nan=True
        )

    @pytest.mark.parametrize(["ix", "expected"], [(0, "foo"), (1, "bar"), (-1, "bar")])
    def test_get_val_at_index(self, dummy_model_data, ix, expected):
        get_val_at_index = helper_functions.get_val_at_index(dummy_model_data)
        assert get_val_at_index(dim="timesteps", idx=ix) == expected

    @pytest.mark.parametrize(["to_roll", "expected"], [(1, 3), (1.0, 3), (-3.0, 3)])
    def test_roll(self, dummy_model_data, to_roll, expected):
        roll = helper_functions.roll()
        rolled = roll(dummy_model_data.with_inf, techs=to_roll)
        assert rolled.sel(nodes="foo", techs="foobar") == expected


class TestAsLatex:
    def test_inheritance(self, dummy_model_data):
        inheritance = helper_functions.inheritance(dummy_model_data, as_latex=True)
        assert inheritance("boo") == r"\text{tech_group=boo}"

    def test_imask_sum_not_exists(self, dummy_model_data):
        imask_sum = helper_functions.imask_sum(dummy_model_data, as_latex=True)
        summed_string = imask_sum("foo", over="techs")
        assert summed_string == r"\sum\limits_{\text{tech} \in \text{techs}} (foo)"

    @pytest.mark.parametrize(
        ["over", "expected_substring"],
        [
            ("techs", r"\text{tech} \in \text{techs}"),
            (["techs"], r"\substack{\text{tech} \in \text{techs}}"),
            (
                ["nodes", "techs"],
                r"\substack{\text{node} \in \text{nodes} \\ \text{tech} \in \text{techs}}",
            ),
        ],
    )
    @pytest.mark.parametrize(
        "func", [helper_functions.imask_sum, helper_functions.expression_sum]
    )
    def test_sum(self, dummy_model_data, over, expected_substring, func):
        imask_sum = func(model_data=dummy_model_data, as_latex=True)
        summed_string = imask_sum(r"\textit{with_inf}_\text{node,tech}", over=over)
        assert (
            summed_string
            == rf"\sum\limits_{{{expected_substring}}} (\textit{{with_inf}}_\text{{node,tech}})"
        )

    def test_squeeze_carriers(self, dummy_model_data):
        squeeze_carriers = helper_functions.squeeze_carriers(
            dummy_model_data, as_latex=True
        )
        squeezed_string = squeeze_carriers("foo", "out")
        assert (
            squeezed_string
            == r"\sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (foo)"
        )

    def test_squeeze_primary_carriers(self, dummy_model_data):
        squeeze_carriers = helper_functions.squeeze_primary_carriers(
            dummy_model_data, as_latex=True
        )
        squeezed_string = squeeze_carriers("foo", "out")
        assert (
            squeezed_string == r"\sum\limits_{\text{carrier=primary_carrier_out}} (foo)"
        )

    def test_get_connected_link(self, dummy_model_data):
        get_connected_link = helper_functions.get_connected_link(
            dummy_model_data, as_latex=True
        )
        connected_link_string = get_connected_link(r"\textit{node}_\text{node,tech}")
        assert (
            connected_link_string
            == r"\textit{node}_\text{node=remote_node,tech=remote_tech}"
        )

    def test_get_val_at_index(self, dummy_model_data):
        get_val_at_index = helper_functions.get_val_at_index(
            dummy_model_data, as_latex=True
        )
        outstring = get_val_at_index(dim="timesteps", idx=1)
        assert outstring == "timesteps[1]"

    @pytest.mark.parametrize(
        ["instring", "expected_substring"],
        [
            (r"\textit{foo}_\text{foo}", r"foo+1"),
            (r"\textit{foo}_\text{bar}", r"bar"),
            (r"\textit{foo}_\text{foo,bar}", r"foo+1,bar"),
            (r"\textit{foo}_\text{foo}", r"foo+1"),
            (r"\textit{foo}_\text{foobar,bar_foo,foo}", r"foobar,bar_foo,foo+1"),
            (r"\textit{foo}_\text{foo=bar,foo}", r"foo=bar,foo+1"),
            (r"\textit{foo}_\text{baz,foo,bar,foo}", r"baz,foo+1,bar,foo+1"),
        ],
    )
    def test_roll(self, instring, expected_substring):
        roll = helper_functions.roll(as_latex=True)
        rolled_string = roll(instring, foo=-1)
        assert rolled_string == rf"\textit{{foo}}_\text{{{expected_substring}}}"
