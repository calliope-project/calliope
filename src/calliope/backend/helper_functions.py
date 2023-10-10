# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).

"""
helper_functions.py
~~~~~~~~~~~~~~~~~~~

Functions that can be used to process data in math `where` and `expression` strings.
"""
import functools
import re
from abc import ABC, abstractmethod
from typing import Any, Literal, Mapping, Union, overload

import numpy as np
import xarray as xr

from calliope.exceptions import BackendError

_registry: dict[
    Literal["where", "expression"], dict[str, type["ParsingHelperFunction"]]
] = {"where": {}, "expression": {}}


class ParsingHelperFunction(ABC):
    def __init__(
        self,
        as_latex: bool = False,
        **kwargs,
    ) -> None:
        """Abstract helper function class, which all helper functions must subclass.

        The abstract properties and methods defined here must be defined by all helper functions.

        Args:
            as_latex (bool, optional):
                If True, will return a LaTeX math string on calling the class.
                Defaults to False.
        """
        self._kwargs = kwargs
        self._as_latex = as_latex

    @property
    @abstractmethod
    def ALLOWED_IN(self) -> list[Literal["where", "expression"]]:
        "List of parseable math strings that this function can be accessed from."

    @property
    @abstractmethod
    def NAME(self) -> str:
        "Helper function name that is used in the math expression/where string."

    @abstractmethod
    def as_latex(self, *args, **kwargs) -> str:
        """Method to update LaTeX math strings to include the action applied by the helper function.

        This method is called when the class is initialised with ``as_latex=True``.
        """

    @abstractmethod
    def as_array(self, *args, **kwargs) -> xr.DataArray:
        """Method to apply the helper function to provide an n-dimensional array output.

        This method is called when the class is initialised with ``as_latex=False``.
        """

    def __call__(self, *args, **kwargs) -> Any:
        """
        When a helper function is accessed by evaluating a parsing string, this method is called.
        The value of `as_latex` on initialisation of the class defines whether this method returns a string (``as_latex=True``) or :meth:xr.DataArray (``as_latex=False``)
        """
        if self._as_latex:
            return self.as_latex(*args, **kwargs)
        else:
            return self.as_array(*args, **kwargs)

    def __init_subclass__(cls):
        """Override subclass definition in two ways:

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
    def _add_to_iterator(instring: str, iterator_converter: dict[str, str]) -> str:
        """Utility function for generating latex strings in multiple helper functions.

        Find an iterator in the iterator substring of the component string
        (anything wrapped in `_text{}`). Other parts of the iterator substring can be anything
        except curly braces, e.g. the standalone `foo` will be found here and acted upon:
        `\\textit{my_param}_\text{bar,foo,foo=bar,foo+1}`

        Args:
            instring (str): String in which the iterator substring can be found.
            iterator_converter (dict[str, str]):
                key: the iterator to search for.
                val: The new string to **append** to the iterator name.

        Returns:
            str: `instring`, but with `iterator` replaced with `iterator + new_string`
        """

        def _replace_in_iterator(matched):
            iterator_list = matched.group(2).split(",")
            new_iterator_list = []
            for it in iterator_list:
                if it in iterator_converter:
                    it += iterator_converter[it]
                new_iterator_list.append(it)

            return matched.group(1) + ",".join(new_iterator_list) + matched.group(3)

        return re.sub(r"(_\\text{)([^{}]*?)(})", _replace_in_iterator, instring)

    @staticmethod
    def _instr(dim: str) -> str:
        """Utility function for generating latex strings in multiple helper functions.

        Args:
            dim (str): Dimension suffixed with a "s" (e.g., "techs")

        Returns:
            str: LaTeX string for iterator in a set (e.g., "tech in techs")
        """
        dim_singular = dim.removesuffix("s")
        return rf"\text{{{dim_singular}}} \in \text{{{dim}}}"

    def _listify(
        self, vals: Union[list[str], str], expand_link_techs: bool = False
    ) -> list[str]:
        if not isinstance(vals, list):
            vals = [vals]
        if expand_link_techs:
            vals = self._expand_link_techs(vals)
        return vals

    def _expand_link_techs(self, vals: list[str]) -> list[str]:
        to_remove = []
        to_add = []
        for val in vals:
            link_techs = self._kwargs["model_data"].techs.str.startswith(val + ":")
            if link_techs.any():
                to_add.extend(list(link_techs[link_techs].techs.data))
                to_remove.append(val)
        for i in to_remove:
            vals.remove(i)
        vals.extend(to_add)
        return vals


class Inheritance(ParsingHelperFunction):
    #:
    ALLOWED_IN = ["where"]
    #:
    NAME = "inheritance"

    def as_latex(self, tech_group: str) -> str:
        return rf"\text{{tech_group={tech_group}}}"

    def as_array(self, tech_group: str) -> xr.DataArray:
        """Find all technologies which inherit from a particular technology group.

        The technology group can be an abstract base group (e.g., `supply`, `storage`) or a user-defined technology group which itself inherits from one of the abstract base groups.

        Args:
            model_data (xr.Dataset): Calliope model data
        """
        inheritance_lists = (
            self._kwargs["model_data"].inheritance.to_series().str.split(".")
        )
        return inheritance_lists.apply(lambda x: tech_group in x).to_xarray()


class WhereAny(ParsingHelperFunction):
    # Class name doesn't match NAME to avoid a clash with typing.Any
    #:
    NAME = "any"
    #:
    ALLOWED_IN = ["where"]

    def as_latex(self, array: str, *, over: Union[str, list[str]]) -> str:
        if isinstance(over, str):
            overstring = self._instr(over)
        else:
            foreach_string = r" \\ ".join(self._instr(i) for i in over)
            overstring = rf"\substack{{{foreach_string}}}"
        # Using bigvee for "collective-or"
        return rf"\bigvee\limits_{{{overstring}}} ({array})"

    def as_array(self, parameter: str, *, over: Union[str, list[str]]) -> xr.DataArray:
        """Reduce the boolean where array of a model parameter by applying `any` over some dimension(s).

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
            bool_parameter_da = (
                parameter_da.notnull()
                & (parameter_da != np.inf)
                & (parameter_da != -np.inf)
            )
        elif parameter in self._kwargs.get("backend_dataset", xr.Dataset()).data_vars:
            bool_parameter_da = self._kwargs["backend_dataset"][parameter].notnull()
        else:
            bool_parameter_da = xr.DataArray(False)
        over = self._listify(over)
        available_dims = set(bool_parameter_da.dims).intersection(over)

        return bool_parameter_da.any(dim=available_dims, keep_attrs=True)


class Defined(ParsingHelperFunction):
    #:
    NAME = "defined"
    #:
    ALLOWED_IN = ["where"]

    def as_latex(self, *, within: str, how: Literal["all", "any"], **dims) -> str:
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
                Must be one of the core model dimensions: [nodes, techs, carriers, carrier_tiers]
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
            dim: self._listify(vals, expand_link_techs=True)
            for dim, vals in dims.items()
        }
        definition_matrix = self._kwargs["model_data"].definition_matrix
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
        definition_matrix = self._kwargs["model_data"].definition_matrix
        missing_dims = set([*dim_names, within]).difference(definition_matrix.dims)
        if missing_dims:
            raise ValueError(
                f"Unexpected model dimension referenced in `{self.NAME}` helper function. "
                "Only dimensions given by `model.inputs.definition_matrix` can be used. "
                f"Received: {missing_dims}"
            )
        return set(definition_matrix.dims).difference([*dim_names, within])

    def _latex_substring(
        self,
        how: Literal["all", "any"],
        dim: str,
        vals: Union[str, list[str]],
        within: str,
    ) -> str:
        if how == "all":
            # Using wedge for "collective-and"
            tex_how = "wedge"
        elif how == "any":
            # Using vee for "collective-or"
            tex_how = "vee"

        vals = self._listify(vals, expand_link_techs=True)
        within_singular = within.removesuffix("s")
        dim_singular = dim.removesuffix("s")
        selection = rf"\text{{{dim_singular}}} \in \text{{[{','.join(vals)}]}}"

        return rf"\big{tex_how}\limits_{{\substack{{{selection}}}}}\text{{{dim_singular} defined in {within_singular}}}"


class Sum(ParsingHelperFunction):
    #:
    NAME = "sum"
    #:
    ALLOWED_IN = ["expression"]

    def as_latex(self, array: str, *, over: Union[str, list[str]]) -> str:
        if isinstance(over, str):
            overstring = self._instr(over)
        else:
            foreach_string = r" \\ ".join(self._instr(i) for i in over)
            overstring = rf"\substack{{{foreach_string}}}"
        return rf"\sum\limits_{{{overstring}}} ({array})"

    def as_array(
        self, array: xr.DataArray, *, over: Union[str, list[str]]
    ) -> xr.DataArray:
        """Sum an expression array over the given dimension(s).

        Args:
            array (xr.DataArray): expression array
            over (Union[str, list[str]]): dimension(s) over which to apply `sum`.

        Returns:
            xr.DataArray:
                Array with dimensions reduced by applying a summation over the dimensions given in `over`.
                NaNs are ignored (xarray.DataArray.sum arg: `skipna: True`) and if all values along the dimension(s) are NaN,
                the summation will lead to a NaN (xarray.DataArray.sum arg: `min_count=1`).
        """
        return array.sum(over, min_count=1, skipna=True)


class ReduceCarrierDim(ParsingHelperFunction):
    #:
    NAME = "reduce_carrier_dim"
    #:
    ALLOWED_IN = ["expression"]

    def as_latex(
        self,
        array: str,
        carrier_tier: Literal["in", "out", "in_2", "out_2", "in_3", "out_3"],
    ) -> str:
        return rf"\sum\limits_{{\text{{carrier}} \in \text{{carrier_tier({carrier_tier})}}}} ({array})"

    def as_array(
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
        return Sum(as_latex=self._as_latex, **self._kwargs)(
            array.where(
                self._kwargs["model_data"].definition_matrix.sel(
                    carrier_tiers=carrier_tier
                )
            ),
            over="carriers",
        )


class ReducePrimaryCarrierDim(ParsingHelperFunction):
    #:
    NAME = "reduce_primary_carrier_dim"
    #:
    ALLOWED_IN = ["expression"]

    def as_latex(self, array: str, carrier_tier: Literal["in", "out"]) -> str:
        return rf"\sum\limits_{{\text{{carrier=primary_carrier_{carrier_tier}}}}} ({array})"

    def as_array(
        self, array: xr.DataArray, carrier_tier: Literal["in", "out"]
    ) -> xr.DataArray:
        """Reduce expression array data by selecting the carrier that corresponds to the primary carrier and then dropping the `carriers` dimension.

        This function is only valid for `conversion_plus` technologies,
        so should only be included in a math component if the `where` string includes `inheritance(conversion_plus)` or an equivalent expression.

        Args:
            array (xr.DataArray): Expression array.
            carrier_tier (Literal["in", "out"]): Carrier tier to select one of the `primary_carrier_in`/`primary_carrier_out` arrays, in which the primary carrier for the technology is defined.


        Returns:
            xr.DataArray: `array` reduced by the `carriers` dimension.
        """
        return Sum(as_latex=self._as_latex, **self._kwargs)(
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

    def as_latex(self, array: str, **lookup_arrays: str) -> str:
        new_strings = {
            (iterator := dim.removesuffix("s")): rf"={array}[{iterator}]"
            for dim, array in lookup_arrays.items()
        }
        array = self._add_to_iterator(array, new_strings)
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
            stacked_lookup = self._kwargs["model_data"][index.name].stack({dim: dims})
            ix = array.indexes[index_dim].get_indexer(stacked_lookup)
            if (ix == -1).all():
                received_lookup = (
                    self._kwargs["model_data"][index.name].to_series().dropna()
                )
                raise IndexError(
                    f"Trying to select items on the dimension {index_dim} from the {index.name} lookup array, but no matches found. Received: {received_lookup}"
                )
            ixs[index_dim] = xr.DataArray(
                np.fmax(0, ix),
                coords={dim: stacked_lookup[dim]},
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
    #:
    NAME = "get_val_at_index"
    #:
    ALLOWED_IN = ["expression", "where"]

    def as_latex(self, **dim_idx_mapping: str) -> str:
        dim, idx = self._mapping_to_dim_idx(**dim_idx_mapping)
        return f"{dim}[{idx}]"

    def as_array(self, **dim_idx_mapping: int) -> xr.DataArray:
        """Get value of a model dimension at a given integer index.

        This function is primarily useful for timeseries data.

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
        dim, idx = self._mapping_to_dim_idx(**dim_idx_mapping)
        return self._kwargs["model_data"].coords[dim][int(idx)]

    @overload
    @staticmethod
    def _mapping_to_dim_idx(**dim_idx_mapping: int) -> tuple[str, int]:
        "used in as_array"

    @overload
    @staticmethod
    def _mapping_to_dim_idx(**dim_idx_mapping: str) -> tuple[str, str]:
        "used in as_latex"

    @staticmethod
    def _mapping_to_dim_idx(**dim_idx_mapping) -> tuple[str, Union[str, int]]:
        if len(dim_idx_mapping) != 1:
            raise ValueError("Supply one (and only one) dimension:index mapping")
        return next(iter(dim_idx_mapping.items()))


class Roll(ParsingHelperFunction):
    #:
    NAME = "roll"
    #:
    ALLOWED_IN = ["expression"]

    def as_latex(self, array: str, **roll_kwargs: str) -> str:
        new_strings = {
            k.removesuffix("s"): f"{-1 * int(v):+d}" for k, v in roll_kwargs.items()
        }
        component = self._add_to_iterator(array, new_strings)
        return component

    def as_array(self, array: xr.DataArray, **roll_kwargs: int) -> xr.DataArray:
        """Roll (a.k.a., shift) the array along the given dimension(s) by the given number of places.
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


class GetTransmissionTechs(ParsingHelperFunction):
    #:
    NAME = "get_transmission_techs"
    #:
    ALLOWED_IN = ["expression"]

    def as_latex(self, vals: Union[str, list[str]]) -> str:
        expanded_vals = self._listify(vals, expand_link_techs=True)
        return f"techs=[{','.join(expanded_vals)}]"

    def as_array(self, vals: Union[str, list[str]]) -> xr.DataArray:
        """Get an array of model techs marking which are linked to a base transmission tech(s).

        This function is useful for slicing an array's `techs` dimension for all transmission techs linked to the same base tech(s).

        Args:
            base_tech_name: Transmission tech base name or list of names, i.e., without remote node attached to the tech (`ac_transmission` instead of `ac_transmission:region1`).

        Returns:
            xr.DataArray: Coordinate `techs` array containing only members linked to the provided base transmission tech(s).

        Examples:
            ```yaml
            links:
                node1,node2:
                    techs:
                        ac_transmission:
                node1,node3:
                    techs:
                        free_transmission:
            ```
            Then:
            ```
            >>> get_transmission_techs(ac_transmission)
            [out] <xarray.DataArray (techs: 2)>
                  array(['ac_transmission:node1', 'ac_transmission:node2'], dtype=object)
            >>> get_transmission_techs(free_transmission)
            [out] <xarray.DataArray (techs: 2)>
                  array(['free_transmission:node1', 'free_transmission:node3'], dtype=object)
            >>> get_transmission_techs([free_transmission, ac_transmission])
            [out] <xarray.DataArray (techs: 2)>
                  array(['free_transmission:node1', 'free_transmission:node3', 'ac_transmission:node1', 'ac_transmission:node2'], dtype=object)
            ```
        """
        return xr.DataArray(self._listify(vals, expand_link_techs=True), dims=["techs"])
