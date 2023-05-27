import numpy as np
import pytest
import xarray as xr

from calliope import exceptions
from calliope.backend import helper_functions
from calliope.test.common.util import check_error_or_warning


class TestFuncs:
    def test_inheritance(self, dummy_model_data):
        inheritance = helper_functions.inheritance(dummy_model_data)
        boo_bool = inheritance("boo")
        assert boo_bool.equals(dummy_model_data.boo_inheritance_bool)

    def test_where_sum_not_exists(self, dummy_model_data):
        where_sum = helper_functions.where_sum(dummy_model_data)
        summed = where_sum("foo", over="techs")
        assert summed.equals(xr.DataArray(False))

    @pytest.mark.parametrize(
        ["var", "over", "expected"],
        [("with_inf", "techs", "nodes_true"), ("all_nan", "techs", "nodes_false")],
    )
    def test_where_sum_exists(self, dummy_model_data, var, over, expected):
        where_sum = helper_functions.where_sum(dummy_model_data)
        summed = where_sum(var, over=over)
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
    def test_select_from_lookup_arrays(self, dummy_model_data, lookup, expected):
        select_from_lookup_arrays = helper_functions.select_from_lookup_arrays(
            dummy_model_data
        )

        new_array = select_from_lookup_arrays(
            dummy_model_data.with_inf,
            **{k: dummy_model_data[v] for k, v in lookup.items()},
        )
        assert np.array_equal(
            new_array.transpose(*dummy_model_data.with_inf.dims),
            expected,
            equal_nan=True,
        )
        for dim in dummy_model_data.with_inf.dims:
            assert new_array[dim].equals(dummy_model_data[dim])

    def test_select_from_lookup_arrays_fail_dim_not_in_component(
        self, dummy_model_data
    ):
        select_from_lookup_arrays = helper_functions.select_from_lookup_arrays(
            dummy_model_data
        )
        with pytest.raises(exceptions.BackendError) as excinfo:
            select_from_lookup_arrays(
                dummy_model_data.nodes_true,
                techs=dummy_model_data.lookup_techs,
            )
        assert check_error_or_warning(
            excinfo,
            "Cannot select items from `nodes_true` on the dimensions {'techs'} since the array is not indexed over the dimensions {'techs'}",
        )

    def test_select_from_lookup_arrays_fail_dim_slicer_mismatch(self, dummy_model_data):
        select_from_lookup_arrays = helper_functions.select_from_lookup_arrays(
            dummy_model_data
        )
        with pytest.raises(exceptions.BackendError) as excinfo:
            select_from_lookup_arrays(
                dummy_model_data.with_inf,
                techs=dummy_model_data.lookup_techs,
                nodes=dummy_model_data.link_remote_nodes,
            )

        assert check_error_or_warning(
            excinfo,
            ["lookup arrays used to select items from `with_inf", "'techs'", "'nodes'"],
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

    def test_where_sum_not_exists(self, dummy_model_data):
        where_sum = helper_functions.where_sum(dummy_model_data, as_latex=True)
        summed_string = where_sum("foo", over="techs")
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
        "func", [helper_functions.where_sum, helper_functions.expression_sum]
    )
    def test_sum(self, dummy_model_data, over, expected_substring, func):
        where_sum = func(model_data=dummy_model_data, as_latex=True)
        summed_string = where_sum(r"\textit{with_inf}_\text{node,tech}", over=over)
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

    def test_select_from_lookup_arrays(self, dummy_model_data):
        select_from_lookup_arrays = helper_functions.select_from_lookup_arrays(
            dummy_model_data, as_latex=True
        )
        select_from_lookup_arrays_string = select_from_lookup_arrays(
            r"\textit{node}_\text{node,tech}",
            nodes="new_nodes",
            techs=r"\textit{new_techs}_{node,tech}",
        )
        assert (
            select_from_lookup_arrays_string
            == r"\textit{node}_\text{node=new_nodes[node],tech=\textit{new_techs}_{node,tech}[tech]}"
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
