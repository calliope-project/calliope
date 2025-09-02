# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).

"""Functions that can be used to process data in math `where` and `expression` strings.

`NAME` is the function name to use in the math strings.
"""

import functools
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
import pandas as pd
import xarray as xr

from calliope.exceptions import BackendError
from calliope.schemas.math_schema import CalliopeBuildMath
from calliope.util import DTYPE_OPTIONS

if TYPE_CHECKING:
    from calliope.backend.backend_model import BackendModel
_registry: dict[
    Literal["where", "expression"], dict[str, type["ParsingHelperFunction"]]
] = {"where": {}, "expression": {}}


class ParsingHelperFunction(ABC):
    """Abstract base class for helper function parsing."""

    def __init__(
        self,
        return_type: Literal["array", "math_string"],
        *,
        equation_name: str,
        input_data: xr.Dataset,
        math: "CalliopeBuildMath",
        backend_interface: type["BackendModel"] | None = None,
        **kwargs,
    ) -> None:
        """Abstract helper function class, which all helper functions must subclass.

        The abstract properties and methods defined here must be defined by all helper functions.

        """
        self._equation_name = equation_name
        self._input_data = input_data
        self._backend_interface = backend_interface
        self._math = math
        self._return_type = return_type

    @property
    @abstractmethod
    def ALLOWED_IN(self) -> list[Literal["where", "expression"]]:
        """List of parseable math strings that this function can be accessed from."""

    @property
    @abstractmethod
    def NAME(self) -> str:
        """Helper function name that is used in the math expression/where string."""

    @property
    def ignore_where(self) -> bool:
        """If True, `where` arrays will not be applied to the incoming data variables (valid for expression helpers)."""
        return False

    @abstractmethod
    def as_math_string(self, *args, **kwargs) -> str:
        """Method to update LaTeX math strings to include the action applied by the helper function.

        This method is called when the class is initialised with ``return_type=math_string``.
        """

    @abstractmethod
    def as_array(self, *args, **kwargs) -> xr.DataArray:
        """Method to apply the helper function to provide an n-dimensional array output.

        This method is called when the class is initialised with ``return_type=array``.
        """

    def __call__(self, *args, **kwargs) -> Any:
        """When a helper function is accessed by evaluating a parsing string, this method is called.

        The value of `return_type` on initialisation of the class defines whether this
        method returns either:
        - a string (``return_type=math_string``)
        - :meth:xr.DataArray (``return_type=array``)
        """
        if self._return_type == "math_string":
            return self.as_math_string(*args, **kwargs)
        elif self._return_type == "array":
            return self.as_array(*args, **kwargs)

    def __init_subclass__(cls):
        """Override subclass definition.

        1. Do not allow new helper functions to have a name that is already defined (be it a built-in function or a custom function).
        2. Wrap helper function __call__ in a check for the function being allowed in specific parsing string types.
        """
        super().__init_subclass__()
        for allowed in cls.ALLOWED_IN:
            if cls.NAME in _registry[allowed].keys():
                raise ValueError(
                    f"`{allowed}` string helper function `{cls.NAME}` already exists"
                )
        for allowed in cls.ALLOWED_IN:
            _registry[allowed][cls.NAME] = cls

    @staticmethod
    def _update_iterator(
        instring: str,
        iterator_converter: dict[str, str],
        method: Literal["add", "replace"],
    ) -> str:
        r"""Utility function for generating latex strings in multiple helper functions.

        Find an iterator in the iterator substring of the component string
        (anything wrapped in `_text{}`). Other parts of the iterator substring can be anything
        except curly braces, e.g. the standalone `foo` will be found here and acted upon:
        `\\textit{my_param}_\text{bar,foo,foo=bar,foo+1}`

        Args:
            instring (str): String in which the iterator substring can be found.
            iterator_converter (dict[str, str]):
                key: the iterator to search for.
                val: The new string to **append** to the iterator name (if method = add) or **replace**.
            method (Literal[add, replace]): Whether to add to the iterator or replace it entirely
        Returns:
            str: `instring`, but with `iterator` replaced with `iterator + new_string`
        """

        def __replace_in_iterator(matched):
            iterator_list = matched.group(2).split(",")
            new_iterator_list = []
            for it in iterator_list:
                if it in iterator_converter:
                    it = (
                        it + iterator_converter[it]
                        if method == "add"
                        else iterator_converter[it]
                    )
                new_iterator_list.append(it)

            return matched.group(1) + ",".join(new_iterator_list) + matched.group(3)

        return re.sub(r"(_\\text{)([^{}]*?)(})", __replace_in_iterator, instring)

    def _get_dims_from_iterators(self, instring: str) -> list[str]:
        """For a given math string describing a math component, extract the iterators and return the dimensions (a.k.a., sets) that they are members of.

        Args:
            instring (str): string describing a math component.

        Returns:
            list[str]: List of dimensions over which the math component is iterating.

        Note:
            the iterator -> dimension conversion is very simple (appends an `s`) so it won't work with math components that have already have `_update_iterator` applied.
        """

        def __extract_dims(matched) -> str:
            iterators = matched.group(2)
            # Split on `,`, add 's' back in to singular iterators to refer to dimension names,
            # then rejoin `,` as we must return a string.
            return ",".join(
                [
                    dim_name
                    for i in iterators.split(",")
                    for dim_name, dim_math in self._math.dimensions.root.items()
                    if dim_math.iterator == i
                ]
            )

        dims = re.sub(r"^.*(_\\text{)([^{}]*?)(})", __extract_dims, instring)
        return dims.split(",")

    def _instr(self, dim: str) -> str:
        """Utility function for generating latex strings in multiple helper functions.

        Args:
            dim (str): Dimension suffixed with a "s" (e.g., "techs")

        Returns:
            str: LaTeX string for iterator in a set (e.g., "tech in techs")
        """
        iterator = self._dim_iterator(dim)
        return rf"\text{{{iterator}}} \in \text{{{dim}}}"

    def _listify(self, vals: list[str] | str) -> list[str]:
        """Force a string to a list of length one if not already provided as a list.

        Args:
            vals (list[str] | str): Values (or single value) to force to a list.

        Returns:
            list[str]: Input forced to a list.
        """
        if not isinstance(vals, list):
            vals = [vals]
        return vals

    def _dim_iterator(self, dim: str) -> str:
        return self._math.dimensions[dim].iterator


class WhereAny(ParsingHelperFunction):
    """Apply `any` over a dimension in `where` string."""

    # Class name doesn't match NAME to avoid a clash with typing.Any
    #:
    NAME = "any"
    #:
    ALLOWED_IN = ["where"]

    def as_math_string(self, array: str, *, over: str | list[str]) -> str:  # noqa: D102, override
        if isinstance(over, str):
            overstring = self._instr(over)
        else:
            foreach_string = r" \\ ".join(self._instr(i) for i in over)
            overstring = rf"\substack{{{foreach_string}}}"
        # Using bigvee for "collective-or"
        return rf"\bigvee\limits_{{{overstring}}} ({array})"

    def as_array(self, input_component: str, *, over: str | list[str]) -> xr.DataArray:
        """Reduce the boolean where array of a model input by applying `any` over some dimension(s).

        If the component exists in the model, returns a boolean array with dimensions reduced
        by applying a boolean OR operation along the dimensions given in `over`.
        If the component does not exist, returns a dimensionless False array.

        Args:
            input_component (str): Reference to a model input.
            over (str | list[str]): dimension(s) over which to apply `any`.

        Returns:
            xr.DataArray: resulting array.
        """
        if input_component in self._input_data.data_vars:
            component_da = self._input_data[input_component]
            bool_component_da = (
                component_da.notnull()
                & (component_da != np.inf)
                & (component_da != -np.inf)
            )
        elif (
            self._backend_interface is not None
            and input_component in self._backend_interface._dataset
        ):
            bool_component_da = self._backend_interface._dataset[
                input_component
            ].notnull()
        else:
            bool_component_da = xr.DataArray(False)
        over = self._listify(over)
        available_dims = set(bool_component_da.dims).intersection(over)

        return bool_component_da.any(dim=available_dims, keep_attrs=True)


class Defined(ParsingHelperFunction):
    """Find all items of one dimension that are defined in an item of another dimension."""

    #:
    NAME = "defined"
    #:
    ALLOWED_IN = ["where"]

    def as_math_string(self, *, within: str, how: Literal["all", "any"], **dims) -> str:  # noqa: D102, override
        substrings = []
        for name, vals in dims.items():
            substrings.append(self._latex_substring(how, name, vals, within))
        if len(substrings) == 1:
            return substrings[0]
        else:
            return rf"\bigwedge({', '.join(substrings)})"

    def as_array(
        self, *, within: str, how: Literal["all", "any"], **dims: str
    ) -> xr.DataArray:
        """Find whether members of a model dimension are defined inside another.

        For instance, whether a node defines a specific tech (or group of techs).
        Or, whether a tech defines a specific carrier.

        Args:
            within (str): the model dimension to check.
            how (Literal[all, any]): Whether to return True for `any` match of nested members or for `all` nested members.

        Kwargs:
            dims (dict[str, str]):
                **key**: dimension whose members will be searched for as being defined under the primary dimension (`within`).
                Must be one of the core model dimensions: [nodes, techs, carriers]
                **value**: subset of the dimension members to find.
                Transmission techs can be called using the base tech name (e.g., `ac_transmission`) and all link techs will be collected (e.g., [`ac_transmission:region1`, `ac_transmission:region2`]).


        Returns:
            xr.DataArray:
                For each member of `within`, True if any/all member(s) in `dims` is nested within that member.

        Examples:
            Check for any of a list of techs being defined at nodes.
            Assuming a YAML definition of:

            ```yaml
            nodes:
              node1:
                techs:
                  tech1:
                  tech3:
              node2:
                techs:
                  tech2:
                  tech3:
            ```
            Then:
            ```
                >>> defined(techs=[tech1, tech2], within=nodes, how=any)
                [out] <xarray.DataArray (nodes: 2)>
                      array([ True, False])
                      Coordinates:
                      * nodes    (nodes) <U5 'node1' 'node2'

                >>> defined(techs=[tech1, tech2], within=nodes, how=all)
                [out] <xarray.DataArray (nodes: 2)>
                      array([ False, False])
                      Coordinates:
                      * nodes    (nodes) <U5 'node1' 'node2'
            ```
        """
        dim_names = list(dims.keys())
        dims_with_list_vals = {
            dim: self._listify(vals) if dim == "techs" else self._listify(vals)
            for dim, vals in dims.items()
        }
        definition_matrix = self._input_data.definition_matrix
        dim_within_da = definition_matrix.any(self._dims_to_remove(dim_names, within))
        within_da = getattr(dim_within_da.sel(**dims_with_list_vals), how)(dim_names)

        return within_da

    def _dims_to_remove(self, dim_names: list[str], within: str) -> set:
        """From the definition matrix, get the dimensions that have not been defined.

        This includes dimensions not defined as keys of `dims` or as the value of `within`.

        Args:
            dim_names (list[str]): Keys of `dims`.
            within (str): dimension whose members are being checked.

        Raises:
            ValueError: Can only define dimensions that exist in model.definition_matrix.

        Returns:
            set: Undefined dimensions to remove from the definition matrix.
        """
        definition_matrix = self._input_data.definition_matrix
        missing_dims = set([*dim_names, within]).difference(definition_matrix.dims)
        if missing_dims:
            raise ValueError(
                f"Unexpected model dimension referenced in `{self.NAME}` helper function. "
                "Only dimensions given by `model.inputs.definition_matrix` can be used. "
                f"Received: {missing_dims}"
            )
        return set(definition_matrix.dims).difference([*dim_names, within])

    def _latex_substring(
        self, how: Literal["all", "any"], dim: str, vals: str | list[str], within: str
    ) -> str:
        if how == "all":
            # Using wedge for "collective-and"
            tex_how = "wedge"
        elif how == "any":
            # Using vee for "collective-or"
            tex_how = "vee"
        vals = self._listify(vals)
        within_iterator = self._dim_iterator(within)
        dim_iterator = self._dim_iterator(dim)
        selection = rf"\text{{{dim_iterator}}} \in \text{{[{','.join(vals)}]}}"

        return rf"\big{tex_how}\limits_{{\substack{{{selection}}}}}\text{{{dim_iterator} defined in {within_iterator}}}"


class Sum(ParsingHelperFunction):
    """Apply a summation over dimension(s) in math expressions."""

    NAME = "sum"
    #:
    ALLOWED_IN = ["expression"]

    def as_math_string(self, array: str, *, over: str | list[str]) -> str:  # noqa: D102, override
        if isinstance(over, str):
            overstring = self._instr(over)
        else:
            foreach_string = r" \\ ".join(self._instr(i) for i in over)
            overstring = rf"\substack{{{foreach_string}}}"
        return rf"\sum\limits_{{{overstring}}} ({array})"

    def as_array(self, array: xr.DataArray, *, over: str | list[str]) -> xr.DataArray:
        """Sum an expression array over the given dimension(s).

        Args:
            array (xr.DataArray): expression array
            over (str | list[str]): dimension(s) over which to apply `sum`.

        Returns:
            xr.DataArray:
                Array with dimensions reduced by applying a summation over the dimensions given in `over`.
                NaNs are ignored (xarray.DataArray.sum arg: `skipna: True`) and if all values along the dimension(s) are NaN,
                the summation will lead to a NaN (xarray.DataArray.sum arg: `min_count=1`).
        """
        filtered_over = set(self._listify(over)).intersection(array.dims)
        return array.sum(filtered_over, min_count=1, skipna=True)


class ReduceCarrierDim(ParsingHelperFunction):
    """Sum over the carrier dimension in math components."""

    #:
    NAME = "reduce_carrier_dim"
    #:
    ALLOWED_IN = ["expression"]

    def as_math_string(self, array: str, flow_direction: Literal["in", "out"]) -> str:  # noqa: D102, override
        return rf"\sum\limits_{{\text{{carrier}} \in \text{{carrier_{flow_direction}}}}} ({array})"

    def as_array(
        self, array: xr.DataArray, flow_direction: Literal["in", "out"]
    ) -> xr.DataArray:
        """Reduce expression array data by selecting the carrier that corresponds to the given carrier tier and then dropping the `carriers` dimension.

        Args:
            array (xr.DataArray): Expression array.
            flow_direction (Literal["in", "out"]): Flow direction in which to check for the existence of carrier(s) for technologies defined in `array`.

        Returns:
            xr.DataArray: `array` reduced by the `carriers` dimension.
        """
        sum_helper = Sum(
            return_type=self._return_type,
            equation_name=self._equation_name,
            input_data=self._input_data,
            math=self._math,
        )

        return sum_helper(
            array.where(self._input_data[f"carrier_{flow_direction}"]), over="carriers"
        )


class SelectFromLookupArrays(ParsingHelperFunction):
    """N-dimensional indexing functionality."""

    #:
    NAME = "select_from_lookup_arrays"
    #:
    ALLOWED_IN = ["expression"]

    def as_math_string(self, array: str, **lookup_arrays: str) -> str:  # noqa: D102, override
        new_strings = {
            (iterator := self._dim_iterator(dim)): rf"={array}[{iterator}]"
            for dim, array in lookup_arrays.items()
        }
        array = self._update_iterator(array, new_strings, "add")
        return array

    def as_array(
        self, array: xr.DataArray, **lookup_arrays: xr.DataArray
    ) -> xr.DataArray:
        """Apply vectorised indexing on an arbitrary number of an input array's dimensions.

        Args:
            array (xr.DataArray): Array on which to apply vectorised indexing.

        Kwargs:
            lookup_arrays (dict[str, xr.DataArray]):
                key: dimension on which to apply vectorised indexing
                value: array whose values are either NaN or values from the dimension given in the key.

        Raises:
            BackendError: `array` must be indexed over the dimensions given in the `lookup_arrays` dict keys.
            BackendError: All `lookup_arrays` must be indexed over all the dimensions given in the `lookup_arrays` dict keys.

        Returns:
            xr.DataArray:
                `array` with rearranged values (coordinates remain unchanged).
                Any NaN index coordinates in the lookup arrays will be NaN in the returned array.

        Examples:
            >>> coords = {"foo": ["A", "B", "C"]}
            >>> array = xr.DataArray([1, 2, 3], coords=coords)
            >>> lookup_array = xr.DataArray(np.array(["B", "A", np.nan], dtype="O"), coords=coords, name="bar")
            >>> model_data = xr.Dataset({"bar": lookup_array})
            >>> select_from_lookup_arrays = SelectFromLookupArrays(model_data=model_data)
            >>> select_from_lookup_arrays(array, foo=lookup_array)
            <xarray.DataArray 'bar' (foo: 3)>
            array([ 2.,  1., nan])
            Coordinates:
            * foo      (foo) object 'A' 'B' 'C'

        The lookup array assigns the value at "B" to "A" and vice versa.
        "C" is masked since the lookup array value is NaN.
        """
        # Inspired by https://github.com/pydata/xarray/issues/1553#issuecomment-748491929
        # Reindex does not presently support vectorized lookups: https://github.com/pydata/xarray/issues/1553
        # Sel does (e.g. https://github.com/pydata/xarray/issues/4630) but can't handle missing keys

        dims = set(lookup_arrays.keys())
        missing_dims_in_component = dims.difference(array.dims)
        missing_dims_in_lookup_tables = any(
            dim not in lookup.dims for dim in dims for lookup in lookup_arrays.values()
        )
        if missing_dims_in_component:
            raise BackendError(
                f"Cannot select items from `{array.name}` on the dimensions {dims} since the array is not indexed over the dimensions {missing_dims_in_component}"
            )
        if missing_dims_in_lookup_tables:
            raise BackendError(
                f"All lookup arrays used to select items from `{array.name}` must be indexed over the dimensions {dims}"
            )

        dim = "dim_0"
        ixs = {}
        masks = []

        # Turn string lookup values to numeric ones.
        # We stack the dimensions to handle multidimensional lookups
        for index_dim, index in lookup_arrays.items():
            stacked_lookup = self._input_data[index.name].stack({dim: dims})
            ix = array.indexes[index_dim].get_indexer(stacked_lookup)
            if (ix == -1).all():
                received_lookup = self._input_data[index.name].to_series().dropna()
                raise IndexError(
                    f"Trying to select items on the dimension {index_dim} from the {index.name} lookup array, but no matches found. Received: {received_lookup}"
                )
            ixs[index_dim] = xr.DataArray(
                np.fmax(0, ix), coords={dim: stacked_lookup[dim]}
            )
            masks.append(ix >= 0)

        # Create a mask to nullify any lookup values that are not given (i.e., are np.nan in the lookup array)
        mask = functools.reduce(lambda x, y: x & y, masks)

        result = array[ixs]

        if not mask.all():
            result[{dim: ~mask}] = np.nan
        unstacked_result = result.drop_vars(dims).unstack(dim)
        return unstacked_result


class GetValAtIndex(ParsingHelperFunction):
    """Getter functionality for obtaining values at specific integer indices."""

    #:
    NAME = "get_val_at_index"
    #:
    ALLOWED_IN = ["expression", "where"]

    def as_math_string(self, **dim_idx_mapping: str) -> str:  # noqa: D102, override
        dim, idx = self._mapping_to_dim_idx(**dim_idx_mapping)
        return f"{dim}[{idx}]"

    def as_array(self, **dim_idx_mapping: int) -> xr.DataArray:
        """Get value of a model dimension at a given integer index.

        This function is primarily useful for timeseries data.

        Args:
            **dim_idx_mapping (int): kwargs with
                key (str): Model dimension in which to extract value.
                value (int): Integer index of the value to extract (assuming zero-indexing).

        Returns:
            xr.DataArray: Dimensionless array containing one value.

        Examples:
            >>> coords = {"timesteps": ["2000-01-01 00:00", "2000-01-01 01:00", "2000-01-01 02:00"]}
            >>> model_data = xr.Dataset(coords=coords)
            >>> get_val_at_index = GetValAtIndex(model_data=model_data)
            >>> get_val_at_index(model_data)(timesteps=0)
            <xarray.DataArray 'timesteps' ()>
            array('2000-01-01 00:00', dtype='<U16')
            Coordinates:
                timesteps  <U16 '2000-01-01 00:00'
            >>> get_val_at_index(model_data)(timesteps=-1)
            <xarray.DataArray 'timesteps' ()>
            array('2000-01-01 00:00', dtype='<U16')
            Coordinates:
                timesteps  <U16 '2000-01-01 02:00'
        """
        dim, idx = self._mapping_to_dim_idx(**dim_idx_mapping)
        return self._input_data.coords[dim][int(idx)]

    # For as_array
    @overload
    @staticmethod
    def _mapping_to_dim_idx(**dim_idx_mapping: int) -> tuple[str, int]: ...

    # For as_math_string
    @overload
    @staticmethod
    def _mapping_to_dim_idx(**dim_idx_mapping: str) -> tuple[str, str]: ...

    @staticmethod
    def _mapping_to_dim_idx(**dim_idx_mapping) -> tuple[str, str | int]:
        if len(dim_idx_mapping) != 1:
            raise ValueError("Supply one (and only one) dimension:index mapping")
        return next(iter(dim_idx_mapping.items()))


class Roll(ParsingHelperFunction):
    """Roll (a.k.a. shift) items along ordered dimensions."""

    #:
    NAME = "roll"
    #:
    ALLOWED_IN = ["expression"]

    @property
    def ignore_where(self) -> bool:
        """Whether or not to ignore `where` functionality."""
        return True

    def as_math_string(self, array: str, **roll_kwargs: str) -> str:  # noqa: D102, override
        new_strings = {
            self._dim_iterator(k): f"{-1 * int(v):+d}" for k, v in roll_kwargs.items()
        }
        component = self._update_iterator(array, new_strings, "add")
        return component

    def as_array(self, array: xr.DataArray, **roll_kwargs: int) -> xr.DataArray:
        """Roll (a.k.a., shift) the array along the given dimension(s) by the given number of places.

        Rolling keeps the array index labels in the same position, but moves the data by the given number of places.

        Args:
            array (xr.DataArray): Array on which to roll data.
            **roll_kwargs (int): kwargs with the following
                key (str): name of dimension on which to roll.
                value (int): number of places to roll data.

        Returns:
            xr.DataArray: `array` with rolled data.

        Examples:
            >>> array = xr.DataArray([1, 2, 3], coords={"foo": ["A", "B", "C"]})
            >>> model_data = xr.Dataset({"bar": array})
            >>> roll = Roll()
            >>> roll("bar", foo=1)
            <xarray.DataArray 'bar' (foo: 3)>
            array([3, 1, 2])
            Coordinates:
            * foo      (foo) <U1 'A' 'B' 'C'
        """
        roll_kwargs_int: Mapping = {k: int(v) for k, v in roll_kwargs.items()}
        return array.roll(roll_kwargs_int)


class DefaultIfEmpty(ParsingHelperFunction):
    """Fill empty (NaN) items in arrays."""

    #:
    NAME = "default_if_empty"
    #:
    ALLOWED_IN = ["expression"]

    def as_math_string(self, var: str, default: float | int) -> str:  # noqa: D102, override
        return rf"({var}\vee{{}}{default})"

    def as_array(self, var: xr.DataArray, default: float | int) -> xr.DataArray:
        """Get an array with filled NaNs if present in the model, or a single default value if not.

        This function is useful for avoiding excessive sub expressions where it is a choice between an expression or a single numeric value.

        Args:
            var (xr.DataArray): array of backend expression objects or an un-indexed "string" object array with the var name (if not present in the model).
            default (float | int): Numeric value with which to fill / replace `var`.

        Returns:
            xr.DataArray:
                If var is an array of backend expression objects, NaNs will be filled with `default`.
                If var is an unindexed array with a single string value, an unindexed array with the default value.

        Examples:
            ```
            >>> default_if_empty(flow_cap, 0)
            [out] <xarray.DataArray (techs: 2)>
                  array(['ac_transmission:node1', 'ac_transmission:node2'], dtype=object)
            >>> default_if_empty(flow_export, 0)
            [out] <xarray.DataArray ()>
                  array(0, dtype=np.int)
            ```
        """
        if var.attrs.get("obj_type", "") == "string":
            return xr.DataArray(default)
        else:
            return var.fillna(default)


class Where(ParsingHelperFunction):
    """Apply `where` array _within_ an expression string."""

    #:
    NAME = "where"
    #:
    ALLOWED_IN = ["expression"]

    def as_math_string(self, array: str, condition: str) -> str:  # noqa: D102, override
        return rf"({array} \text{{if }} {condition} == True)"

    def as_array(self, array: xr.DataArray, condition: xr.DataArray) -> xr.DataArray:
        """Apply a `where` condition to a math array within an expression string.

        Args:
            array (xr.DataArray): Math component array.
            condition (xr.DataArray):
                Boolean where array.
                If not `bool` type, NaNs and 0 will be assumed as False and all other values will be assumed as True.

        Returns:
            xr.DataArray:
                Returns the input array with the condition applied,
                including having been broadcast across any new dimensions provided by the condition.

        Examples:
            One common use-case is to introduce a new dimension to the variable which represents subsets of one of the main model dimensions.
            In this case, each member of `cap_node_groups` is a subset of `nodes` and we want to sum `flow_cap` over each of those subsets and set a maximum value.

            input:
            ```yaml
            data_definitions:
              node_grouping:
                data: True
                index: [[group_1, region1], [group_1, region1_1], [group_2, region1_2], [group_2, region1_3], [group_3, region2]]
                dims: [cap_node_groups, nodes]
              node_group_max:
                data: [1, 2, 3]
                index: [group_1, group_2, group_3]
                dims: cap_node_groups
            ```

            math:
            ```yaml
            constraints:
                my_new_constraint:
                    foreach: [techs, cap_node_groups]
                    equations:
                        - expression: sum(where(flow_cap, node_grouping), over=nodes) <= node_group_max
            ```
        """
        if self._backend_interface is not None:
            condition = self._input_data[condition.name]

        return array.where(condition.fillna(False).astype(bool))


class GroupSum(ParsingHelperFunction):
    """Apply a summation over an array grouping."""

    #:
    NAME = "group_sum"
    #:
    ALLOWED_IN = ["expression"]

    ignore_where = True

    def as_math_string(self, array: str, groupby: str, group_dim: str) -> str:  # noqa: D102, override
        group_dim_singular = self._dim_iterator(group_dim)
        sum_lim_string = rf"\text{{ if }} {groupby} = \text{{{group_dim_singular}}}"
        over = [self._instr(i) for i in self._get_dims_from_iterators(groupby)]
        foreach_string = r" \\ ".join([*over, sum_lim_string])
        overstring = rf"\substack{{{foreach_string}}}"
        return rf"\sum\limits_{{{overstring}}} ({array})"

    def as_array(
        self, array: xr.DataArray, groupby: xr.DataArray, group_dim: str
    ) -> xr.DataArray:
        """Sum an array over the given groupings.

        Args:
            array (xr.DataArray): expression array
            groupby (xr.DataArray): Array with which to group the array.
            group_dim (str): Name of dimension that the `groupby` values are members of.
                This will become a new dimension over which the array is indexed once grouping is complete.

        Returns:
            xr.DataArray:
                Array with dimension(s) aggregated over the `groupby`.

        Note:
            - The array is returned with all dimensions over which `groupby` is indexed replaced by a new dimension named by `group_dim`.
            - To groupby datetime periods (weeks, months, dates, etc.), consider using `group_datetime` for convenience, as you do not need to define a separate `groupby` array.

        Examples:
            To get the sum over an ad-hoc combination of techs at nodes, e.g. to limit their overall outflow in any given timestep, you would do the following:

            1. Define an array linking node-tech combinations with a group:
            ```yaml
            data_definitions:
              # You may prefer to define this in a CSV file or when referring to the techs within the `nodes` model definition.
              power_plant_groups:
                data: [low_emission_plant, low_emission_plant, high_emission_plant, high_emission_plant]
                index: [
                  [tech_1, node_1],
                  [tech_2, node_1],
                  [tech_1, node_2],
                  [tech_2, node_2],
                ]
                dims: [techs, nodes]
            ```
            2. Define a set of outflow limits:
            ```yaml
            data_definitions:
              emission_limits:
                data: [20, 10]
                index: [low_emission_plant, high_emission_plant]
                dims: [emission_groups]
            ```
            3. Define the math to link the two, using `group_sum`:
            ```yaml
            constraints:
              node_tech_emission_group_max:
                foreach: [emission_groups, carriers, timesteps]
                where: emission_limits
                equations:
                  - expression: group_sum(flow_out, power_plant_groups, emission_groups) <= emission_limits
            ```
        """
        # We can't apply typical xarray rolling window functionality
        grouped: dict[str | int, xr.DataArray] = {}

        if self._backend_interface is not None:
            groupby = self._input_data[groupby.name]

        grouping_dims = groupby.dims
        groups = array.stack(_stacked=grouping_dims).groupby(
            groupby.stack(_stacked=grouping_dims)
        )
        for group_name, array_subset in groups:
            grouped[group_name] = array_subset.sum("_stacked", min_count=1, skipna=True)

        array = xr.concat(
            grouped.values(), dim=pd.Index(grouped.keys(), name=group_dim)
        )
        return array


class GroupDatetime(ParsingHelperFunction):
    """Apply a summation over a datetime group on a datetime dimension in math expressions."""

    NAME = "group_datetime"
    #:
    ALLOWED_IN = ["expression"]

    ignore_where = True

    def as_math_string(self, array: str, over: str, group: str) -> str:  # noqa: D102, override
        overstring = self._instr(over)
        foreach_string = rf"{overstring} \text{{ if }} \text{{{group}}}(\text{{{self._dim_iterator(over)}}}) = \text{{{self._dim_iterator(group)}}}"
        overstring = rf"\substack{{{foreach_string}}}"

        return rf"\sum\limits_{{{overstring}}} ({array})"

    def as_array(self, array: xr.DataArray, over: str, group: str) -> xr.DataArray:
        """Sum an expression array over the given dimension(s).

        Args:
            array (xr.DataArray): expression array
            over (str): dimension name over which to group
            group (str): datetime grouper.
                Any xarray/pandas datetime grouper options
                datetime grouper options include 'date', 'dayofweek', 'month', etc.


        Returns:
            xr.DataArray:
                Array with datetime dimension aggregated over the grouper.

        Note:
            - The array is returned with the `over` dimension replaced by the name of the grouper.
              So, if you select to resample to monthly, the returned array will include the `month` dimension.
            - the `date`/`time` groupers will return the date/time as a string in ISO8601 format (e.g. "2025-01-01"/"01:00:00").
              All other groupers will return integer values (e.g. month 1, 2, 3, etc.).

        Examples:
            One common use-case is to allow demand to be met at any point on a given date.
            For such a demand tech, the daily demand should be indexed over `date`, e.g.:

            sink_use_equals_daily.csv
            ```
            date,sink_use_equals_daily
            2000-01-01,10
            2000-01-02,15
            ...
            ```

            Then, to set the daily flow into the demand tech to those values:
            ```yaml
            constraints:
                daily_demand:
                    foreach: [nodes, techs, carriers, date]
                    where: sink_use_equals_daily
                    equations:
                        - expression: "group_datetime(flow_in, timesteps, date) == sink_use_equals_daily"
            ```

            Similarly, a monthly maximum resource to a supply technology might be used, to simulate e.g. biofuel feedstock availability:

            source_use_max_monthly.csv
            ```
            month,source_use_max_monthly
            1,10
            2,15
            ...
            ```

            Then, to set the daily flow into the demand tech to those values:
            ```yaml
            constraints:
                daily_demand:
                    foreach: [nodes, techs, carriers, month]
                    where: source_use_max_monthly
                    equations:
                        - expression: "group_datetime(flow_in, timesteps, month) <= source_use_max_monthly"
            ```
        """
        dtype = DTYPE_OPTIONS[self._math.dimensions[group].dtype]
        group_sum_helper = GroupSum(
            return_type=self._return_type,
            equation_name=self._equation_name,
            input_data=self._input_data,
            math=self._math,
        )
        array = group_sum_helper(
            array, getattr(array[over].dt, group).astype(dtype), group
        )

        return array


class SumNextN(ParsingHelperFunction):
    """Sum the next N items in an array.

    Works best for ordered arrays (datetime, integer) and is equivalent to a summation over a rolling window.
    """

    #:
    NAME = "sum_next_n"
    #:
    ALLOWED_IN = ["expression"]

    def as_math_string(self, array: str, over: str, N: int) -> str:  # noqa: D102, override
        over_singular = rf"\text{{{self._dim_iterator(over)}}}"
        new_iterator = over[0]
        updated_iterator_array = self._update_iterator(
            array, {self._dim_iterator(over): new_iterator}, "replace"
        )

        return rf"\sum\limits_{{\text{{{new_iterator}}}={over_singular}}}^{{{over_singular}+{N}}} ({updated_iterator_array})"

    def as_array(self, array: xr.DataArray, over: str, N: int) -> xr.DataArray:
        """Sum values from current up to N from current on the dimension `over`.

        Works best for ordered arrays (datetime, integer).


        Args:
            array (xr.DataArray): Math component array.
            over (str): Dimension over which to sum
            N (int): number of items beyond the current value to sum from

        Returns:
            xr.DataArray:
                Returns the input array with the condition applied,
                including having been broadcast across any new dimensions provided by the condition.

        Note:
            - The rolling window does not wrap around to the start of the set when reaching the end.
              That is, if you have N = 4 then for a dimension of length T, at T - 1 it will sum over dimension positions (T - 1, T), not (T - 1, T, 0, 1).
            - You will find that this over-constrains the model unless you limit the constraint (using the `where` string) to only apply over `len(over) - N`.
              This is linked to the abovementioned lack of wrapping.
              E.g. `where: timesteps<=get_val_at_index(timesteps=-24)` if N == 24.
            - This function is based on an integer number of steps from the current step.
              For datetime dimensions like `timesteps`, you will (a) need to be using a regular time frequency (e.g. hourly) and (b) update `N` to reflect the resolution of your time dimension
              (N = 4 in if resample.timesteps=`1h` -> N = 2 if resample.timesteps=`2h`).

        Examples:
            One common use-case is to collate N timesteps beyond a given timestep to apply a constraint to it
            (e.g., demand must be less than X in the next 24 hours):

            For such a demand tech, the portion of its demand that is flexible should be separated from `sink_use_equals` to e.g.,
            a `sink_use_flexible` timeseries parameter which we will use in the DSR constraint:

            ```yaml
            constraints:
                4hr_demand_side_response:
                    foreach: ["nodes", "techs", "carriers", "timesteps"]
                    where: "carrier_in AND sink_use_flexible AND timesteps<=get_val_at_index(timesteps=-24)"
                    equations:
                        - expression: sum_next_n(flow_in, timesteps, 4) == sum_next_n(sink_use_flexible, timesteps, 4)"
            ```
        """
        # We cannot use the xarray rolling window method as it doesn't like operating on Python objects, which our optimisation problem components are.
        results: list[xr.DataArray] = []
        for i in range(len(self._input_data.coords[over])):
            results.append(
                array.isel(**{over: slice(i, i + int(N))}).sum(over, min_count=1)
            )
        final_array = xr.concat(
            results, dim=self._input_data.coords[over]
        ).broadcast_like(array)
        return final_array
