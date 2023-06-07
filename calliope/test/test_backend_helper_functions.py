import numpy as np
import pytest
import xarray as xr

from calliope import exceptions
from calliope.backend import helper_functions
from calliope.test.common.util import check_error_or_warning


@pytest.fixture(scope="module")
def expression():
    return helper_functions._registry["expression"]


@pytest.fixture(scope="module")
def where():
    return helper_functions._registry["where"]


@pytest.fixture(scope="class")
def where_inheritance(where, parsing_kwargs):
    return where["inheritance"](**parsing_kwargs)


@pytest.fixture(scope="class")
def where_any(where, parsing_kwargs):
    return where["any"](**parsing_kwargs)


@pytest.fixture(scope="class")
def expression_sum(expression, parsing_kwargs):
    return expression["sum"](**parsing_kwargs)


@pytest.fixture(scope="class")
def expression_reduce_carrier_dim(expression, parsing_kwargs):
    return expression["reduce_carrier_dim"](**parsing_kwargs)


@pytest.fixture(scope="class")
def expression_reduce_primary_carrier_dim(expression, parsing_kwargs):
    return expression["reduce_primary_carrier_dim"](**parsing_kwargs)


@pytest.fixture(scope="class")
def expression_select_from_lookup_arrays(expression, parsing_kwargs):
    return expression["select_from_lookup_arrays"](**parsing_kwargs)


@pytest.fixture(scope="class")
def expression_get_val_at_index(expression, parsing_kwargs):
    return expression["get_val_at_index"](**parsing_kwargs)


@pytest.fixture(scope="class")
def expression_roll(expression, parsing_kwargs):
    return expression["roll"](**parsing_kwargs)


class TestAsArray:
    @pytest.fixture(scope="class")
    def parsing_kwargs(self, dummy_model_data):
        return {"model_data": dummy_model_data}

    @pytest.mark.parametrize(
        ["string_type", "func_name"], [("where", "inheritance"), ("expression", "sum")]
    )
    def test_duplicate_name_exception(self, string_type, func_name):
        with pytest.raises(ValueError) as excinfo:

            class MyNewFunc(helper_functions.ParsingHelperFunction):
                NAME = func_name
                ALLOWED_IN = [string_type]

                def __call__(self):
                    return None

        assert check_error_or_warning(
            excinfo,
            f"`{string_type}` string helper function `{func_name}` already exists",
        )

    @pytest.mark.parametrize(
        ["string_types", "func_name"],
        [(["where"], "sum"), (["expression", "where"], "my_new_func")],
    )
    def test_new_func(self, string_types, func_name):
        class MyNewFunc(helper_functions.ParsingHelperFunction):
            NAME = func_name
            ALLOWED_IN = string_types

            def __call__(self):
                return None

        assert all(func_name in helper_functions._registry[i] for i in string_types)

    def test_inheritance(self, where_inheritance, dummy_model_data):
        boo_bool = where_inheritance("boo")
        assert boo_bool.equals(dummy_model_data.boo_inheritance_bool)

    def test_any_not_exists(self, where_any):
        summed = where_any("foo", over="techs")
        assert summed.equals(xr.DataArray(False))

    @pytest.mark.parametrize(
        ["var", "over", "expected"],
        [("with_inf", "techs", "nodes_true"), ("all_nan", "techs", "nodes_false")],
    )
    def test_any_exists(self, where_any, dummy_model_data, var, over, expected):
        summed = where_any(var, over=over)
        assert summed.equals(dummy_model_data[expected])

    @pytest.mark.parametrize("over", ["techs", ["techs"]])
    def test_sum_one_dim(self, expression_sum, dummy_model_data, over):
        summed_array = expression_sum(dummy_model_data.only_techs, over=over)
        assert not summed_array.shape
        assert summed_array == 6

    @pytest.mark.parametrize("over", ["techs", ["techs"]])
    def test_sum_one_of_two_dims(self, expression_sum, dummy_model_data, over):
        summed_array = expression_sum(dummy_model_data.with_inf, over=over)
        assert summed_array.shape == (2,)
        assert np.array_equal(summed_array, [5, np.inf])

    def test_expression_sum_two_dims(self, expression_sum, dummy_model_data):
        summed_array = expression_sum(
            dummy_model_data.with_inf_as_bool, over=["nodes", "techs"]
        )
        assert not summed_array.shape
        assert summed_array == 5

    def test_reduce_carrier_dim(self, expression_reduce_carrier_dim, dummy_model_data):
        reduced = expression_reduce_carrier_dim(
            dummy_model_data.all_true_carriers, "foo"
        )

        assert dummy_model_data.carrier.sel(carrier_tiers="foo").sum() == reduced.sum()
        assert not set(reduced.dims).symmetric_difference(["techs"])

    def test_reduce_primary_carrier_dim(
        self, expression_reduce_primary_carrier_dim, dummy_model_data
    ):
        reduced = expression_reduce_primary_carrier_dim(
            dummy_model_data.all_true_carriers, "out"
        )

        assert reduced.sum() == 3
        assert reduced.max() == 1
        assert not set(reduced.dims).symmetric_difference(["techs"])

    def test_reduce_primary_carrier_dim_not_in_model(
        self, expression_reduce_primary_carrier_dim, dummy_model_data
    ):
        with pytest.raises(AttributeError):
            expression_reduce_primary_carrier_dim(
                dummy_model_data.all_true_carriers, "foo"
            )

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
    def test_select_from_lookup_arrays(
        self, expression_select_from_lookup_arrays, dummy_model_data, lookup, expected
    ):
        new_array = expression_select_from_lookup_arrays(
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
        self, expression_select_from_lookup_arrays, dummy_model_data
    ):
        with pytest.raises(exceptions.BackendError) as excinfo:
            expression_select_from_lookup_arrays(
                dummy_model_data.nodes_true,
                techs=dummy_model_data.lookup_techs,
            )
        assert check_error_or_warning(
            excinfo,
            "Cannot select items from `nodes_true` on the dimensions {'techs'} since "
            "the array is not indexed over the dimensions {'techs'}",
        )

    def test_select_from_lookup_arrays_fail_dim_slicer_mismatch(
        self, expression_select_from_lookup_arrays, dummy_model_data
    ):
        with pytest.raises(exceptions.BackendError) as excinfo:
            expression_select_from_lookup_arrays(
                dummy_model_data.with_inf,
                techs=dummy_model_data.lookup_techs,
                nodes=dummy_model_data.link_remote_nodes,
            )
        assert check_error_or_warning(
            excinfo,
            ["lookup arrays used to select items from `with_inf", "'techs'", "'nodes'"],
        )

    @pytest.mark.parametrize(["idx", "expected"], [(0, "foo"), (1, "bar"), (-1, "bar")])
    def test_get_val_at_index(self, expression_get_val_at_index, idx, expected):
        assert expression_get_val_at_index(timesteps=idx) == expected

    @pytest.mark.parametrize("mapping", [{}, {"foo": 1, "bar": 1}])
    def test_get_val_at_index_not_one_mapping(
        self, expression_get_val_at_index, mapping
    ):
        with pytest.raises(ValueError) as excinfo:
            expression_get_val_at_index(**mapping)
        assert check_error_or_warning(
            excinfo, "Supply one (and only one) dimension:index mapping"
        )

    @pytest.mark.parametrize(["to_roll", "expected"], [(1, 3), (1.0, 3), (-3.0, 3)])
    def test_roll(self, expression_roll, dummy_model_data, to_roll, expected):
        rolled = expression_roll(dummy_model_data.with_inf, techs=to_roll)
        assert rolled.sel(nodes="foo", techs="foobar") == expected


class TestAsLatex:
    @pytest.fixture(scope="class")
    def parsing_kwargs(self, dummy_model_data):
        return {"model_data": dummy_model_data, "as_latex": True}

    def test_inheritance(self, where_inheritance):
        assert where_inheritance("boo") == r"\text{tech_group=boo}"

    def test_any_not_exists(self, where_any):
        summed_string = where_any("foo", over="techs")
        assert summed_string == r"\bigvee\limits_{\text{tech} \in \text{techs}} (foo)"

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
        ["func_string", "latex_func"],
        [("where_any", r"\bigvee"), ("expression_sum", r"\sum")],
    )
    def test_sum(self, request, over, expected_substring, func_string, latex_func):
        func = request.getfixturevalue(func_string)
        summed_string = func(r"\textit{with_inf}_\text{node,tech}", over=over)
        assert (
            summed_string
            == rf"{latex_func}\limits_{{{expected_substring}}} (\textit{{with_inf}}_\text{{node,tech}})"
        )

    def test_squeeze_carriers(self, expression_reduce_carrier_dim):
        reduced_string = expression_reduce_carrier_dim("foo", "out")
        assert (
            reduced_string
            == r"\sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (foo)"
        )

    def test_squeeze_primary_carriers(self, expression_reduce_primary_carrier_dim):
        reduced_string = expression_reduce_primary_carrier_dim("foo", "out")
        assert (
            reduced_string == r"\sum\limits_{\text{carrier=primary_carrier_out}} (foo)"
        )

    def test_select_from_lookup_arrays(self, expression_select_from_lookup_arrays):
        select_from_lookup_arrays_string = expression_select_from_lookup_arrays(
            r"\textit{node}_\text{node,tech}",
            nodes="new_nodes",
            techs=r"\textit{new_techs}_{node,tech}",
        )
        assert (
            select_from_lookup_arrays_string
            == r"\textit{node}_\text{node=new_nodes[node],tech=\textit{new_techs}_{node,tech}[tech]}"
        )

    def test_get_val_at_index(self, expression_get_val_at_index):
        outstring = expression_get_val_at_index(timesteps="1")
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
    def test_roll(self, expression_roll, instring, expected_substring):
        rolled_string = expression_roll(instring, foo="-1")
        assert rolled_string == rf"\textit{{foo}}_\text{{{expected_substring}}}"
