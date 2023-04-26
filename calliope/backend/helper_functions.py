# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).

"""
helper_functions.py
~~~~~~~~~~~~~

Functions that can be used to process data in math `where` and `expression` strings.
"""

from abc import ABC, abstractmethod
from typing import Literal, Mapping, Union

import pandas as pd
import xarray as xr

from calliope.exceptions import BackendError

_registry: dict[
    Literal["where", "expression"], dict[str, type["ParsingHelperFunction"]]
] = {"where": {}, "expression": {}}


class ParsingHelperFunction(ABC):
    """
    Abstract helper function class, which all helper functions must subclass.
    The abstract properties and methods defined here must be defined by all helper functions.
    """

    def __init__(
        self,
        **kwargs,
    ) -> None:
        self._kwargs = kwargs

    @property
    @abstractmethod
    def ALLOWED_IN(self) -> list[Literal["where", "expression"]]:
        """List of parseable math strings that this function can be accessed from."""

    @property
    @abstractmethod
    def NAME(self) -> str:
        """Helper function name that is used in the math expression/where string."""

    @abstractmethod
    def __call__(self, *args, **kwargs) -> xr.DataArray:
        """Primary function of the helper class, which will be called by the helper when called by evaluating a parsing string."""

    def __init_subclass__(cls):
        """
        Override subclass definition in two ways:
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


class Inheritance(ParsingHelperFunction):
    #:
    ALLOWED_IN = ["where"]
    #:
    NAME = "inheritance"

    def __call__(self, tech_group: str) -> xr.DataArray:
        """
        Find all technologies which inherit from a particular technology group.
        The technology group can be an abstract base group (e.g., `supply`, `storage`) or a user-defined technology group which itself inherits from one of the abstract base groups.

        Args:
            model_data (xr.Dataset): Calliope model data
        """
        inheritance_lists = (
            self._kwargs["model_data"].inheritance.to_series().str.split(".")
        )
        return inheritance_lists.apply(lambda x: tech_group in x).to_xarray()


class Any(ParsingHelperFunction):
    #:
    NAME = "any"
    #:
    ALLOWED_IN = ["where"]

    def __call__(self, parameter: str, *, over: Union[str, list[str]]) -> xr.DataArray:
        """
        Reduce the boolean where array of a model parameter by applying `any` over some dimension(s).

        Args:
            parameter (str): Reference to a model input parameter
            over (Union[str, list[str]]): dimension(s) over which to apply `any`.

        Returns:
            xr.DataArray:
                If the parameter exists in the model, returns a boolean array with dimensions reduced by applying a boolean OR operation along the dimensions given in `over`.
                If the parameter does not exist, returns a dimensionless False array.
        """
        if parameter in self._kwargs["model_data"].data_vars:
            parameter_da = self._kwargs["model_data"][parameter]
            with pd.option_context("mode.use_inf_as_na", True):
                bool_parameter_da = (
                    parameter_da.where(pd.notnull(parameter_da))  # type: ignore
                    .notnull()
                    .any(dim=over, keep_attrs=True)
                )
        else:
            bool_parameter_da = xr.DataArray(False)
        return bool_parameter_da


class Sum(ParsingHelperFunction):
    #:
    NAME = "sum"
    #:
    ALLOWED_IN = ["expression"]

    def __call__(
        self, array: xr.DataArray, *, over: Union[str, list[str]]
    ) -> xr.DataArray:
        """
        Sum an expression array over the given dimension(s).

        Args:
            array (xr.DataArray): expression array
            over (Union[str, list[str]]): dimension(s) over which to apply `sum`.

        Returns:
            xr.DataArray:
                Array with dimensions reduced by applying a summation over the dimensions given in `over`.
                NaNs are ignored (xarray.DataArray.sum arg: `skipna: True`) and if all values along the dimension(s) are NaN, the summation will lead to a NaN (xarray.DataArray.sum arg: `min_count=1`).
        """
        to_return = array.sum(over, min_count=1, skipna=True)

        return to_return


class ReduceCarrierDim(ParsingHelperFunction):
    #:
    NAME = "reduce_carrier_dim"
    #:
    ALLOWED_IN = ["expression"]

    def __call__(
        self,
        array: xr.DataArray,
        carrier_tier: Literal["in", "out", "in_2", "out_2", "in_3", "out_3"],
    ) -> xr.DataArray:
        """Reduce expression array data by selecting the carrier that corresponds to the given carrier tier and then dropping the `carriers` dimension.

        Args:
            array (xr.DataArray): Expression array.
            carrier_tier (Literal["in", "out", "in_2", "out_2", "in_3", "out_3"]): Carrier tier on which to slice the model `exists` array to find the carrier that exists for the elements of `array`.

        Returns:
            xr.DataArray: `array` reduced by the `carriers` dimension.
        """
        return Sum(**self._kwargs)(
            array.where(
                self._kwargs["model_data"]
                .carrier.sel(carrier_tiers=carrier_tier)
                .notnull()
            ),
            over="carriers",
        )


class ReducePrimaryCarrierDim(ParsingHelperFunction):
    #:
    NAME = "reduce_primary_carrier_dim"
    #:
    ALLOWED_IN = ["expression"]

    def __call__(
        self, array: xr.DataArray, carrier_tier: Literal["in", "out"]
    ) -> xr.DataArray:
        """Reduce expression array data by selecting the carrier that corresponds to the primary carrier and then dropping the `carriers` dimension.
        This function is only valid for `conversion_plus` technologies, so should only be included in a math component if the `where` string includes `inheritance(conversion_plus)` or an equivalent expression.

        Args:
            array (xr.DataArray): Expression array.
            carrier_tier (Literal["in", "out"]): Carrier tier to select one of the `primary_carrier_in`/`primary_carrier_out` arrays, in which the primary carrier for the technology is defined.


        Returns:
            xr.DataArray: `array` reduced by the `carriers` dimension.
        """
        return Sum(**self._kwargs)(
            array.where(
                getattr(
                    self._kwargs["model_data"], f"primary_carrier_{carrier_tier}"
                ).notnull()
            ),
            over="carriers",
        )


class SelectFromLookupArrays(ParsingHelperFunction):
    #:
    NAME = "select_from_lookup_arrays"
    #:
    ALLOWED_IN = ["expression"]

    def __call__(
        self, array: xr.DataArray, **lookup_arrays: xr.DataArray
    ) -> xr.DataArray:
        """
        Apply vectorised indexing on an arbitrary number of an input array's dimensions.

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

        stacked_and_dense_lookup_arrays = {
            # Although we have the lookup array, its values are backend objects,
            # so we grab the same array from the unadulterated model data.
            # FIXME: do not add lookup tables as backend objects.
            dim_name: self._kwargs["model_data"][lookup.name]
            # Stacking ensures that the dimensions on `component` are not reordered on calling `.sel()`.
            .stack(idx=list(dims))
            # Cannot select on NaNs, so we drop them all.
            .dropna("idx")
            for dim_name, lookup in lookup_arrays.items()
        }
        sliced_component = array.sel(stacked_and_dense_lookup_arrays)

        return (
            sliced_component.drop_vars(dims)
            .unstack("idx")
            .reindex_like(array, copy=False)
        )


class GetValAtIndex(ParsingHelperFunction):
    #:
    NAME = "get_val_at_index"
    #:
    ALLOWED_IN = ["expression", "where"]

    def __call__(self, **dim_idx_mapping: int) -> xr.DataArray:
        """Get value of a model dimension at a given integer index.
        This function is primarily useful for timeseries data

        Keyword Args:
            key (str): Model dimension in which to extract value.
            value (int): Integer index of the value to extract (assuming zero-indexing).

        Raises:
            ValueError: Exactly one dimension:index mapping is expected.

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
        if len(dim_idx_mapping) != 1:
            raise ValueError("Supply one (and only one) dimension:index mapping")
        dim, idx = next(iter(dim_idx_mapping.items()))
        return self._kwargs["model_data"].coords[dim][int(idx)]


class Roll(ParsingHelperFunction):
    #:
    NAME = "roll"
    #:
    ALLOWED_IN = ["expression"]

    def __call__(self, array: xr.DataArray, **roll_kwargs: int) -> xr.DataArray:
        """
        Roll (a.k.a., shift) the array along the given dimension(s) by the given number of places.
        Rolling keeps the array index labels in the same position, but moves the data by the given number of places.

        Args:
            array (xr.DataArray): Array on which to roll data.
        Keyword Args:
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
