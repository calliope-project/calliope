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
import xarray as xr

from calliope.exceptions import BackendError

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
        backend_interface: type["BackendModel"] | None = None,
        **kwargs,
    ) -> None:
        """Abstract helper function class, which all helper functions must subclass.

        The abstract properties and methods defined here must be defined by all helper functions.

        """
        self._equation_name = equation_name
        self._input_data = input_data
        self._backend_interface = backend_interface
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
    def _add_to_iterator(instring: str, iterator_converter: dict[str, str]) -> str:
        r"""Utility function for generating latex strings in multiple helper functions.

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


class Inheritance(ParsingHelperFunction):
    """Find all nodes / techs that inherit from a template."""

    #:
    ALLOWED_IN = ["where"]
    #:
    NAME = "inheritance"

    def as_math_string(  # noqa: D102, override
        self, nodes: str | None = None, techs: str | None = None
    ) -> str:
        strings = []
        if nodes is not None:
            strings.append(f"nodes={nodes}")
        if techs is not None:
            strings.append(f"techs={techs}")
        return rf"\text{{inherits({','.join(strings)})}}"

    def as_array(
        self, *, nodes: str | None = None, techs: str | None = None
    ) -> xr.DataArray:
        """Find all technologies and/or nodes which inherit from a particular template.

        The group items being referenced must be defined by the user in `templates`.

        Args:
            nodes (str | None, optional): group name to search for inheritance of on the `nodes` dimension. Default is None.
            techs (str | None, optional): group name to search for inheritance of on the `techs` dimension. Default is None.

        Returns:
            xr.Dataset: Boolean array where values are True where the group is inherited, False otherwise. Array dimensions will equal the number of non-None inputs.

        Examples:
            With:
            ```yaml
            templates:
              foo:
                available_area: 1
              bar:
                flow_cap_max: 1
              baz:
                template: bar
                flow_out_eff: 0.5
            nodes:
              node_1:
                template: foo
                techs: {tech_1, tech_2}
              node_2:
                techs: {tech_1, tech_2}
            techs:
              tech_1:
                ...
                template: bar
              tech_2:
                ...
                template: baz
            ```

            >>> inheritance(nodes=foo)
            <xarray.DataArray (nodes: 2)>
            array([True, False])
            Coordinates:
            * nodes      (nodes) <U1 'node_1' 'node_2'

            >>> inheritance(techs=bar)  # tech_2 inherits `bar` via `baz`.
            <xarray.DataArray (techs: 2)>
            array([True, True])
            Coordinates:
            * techs      (techs) <U1 'tech_1' 'tech_2'

            >>> inheritance(techs=baz)
            <xarray.DataArray (techs: 2)>
            array([False, True])
            Coordinates:
            * techs      (techs) <U1 'tech_1' 'tech_2'

            >>> inheritance(nodes=foo, techs=baz)
            <xarray.DataArray (nodes: 2, techs: 2)>
            array([[False, False],
                   [True, False]])
            Coordinates:
            * nodes      (nodes) <U1 'node_1' 'node_2'
            * techs      (techs) <U1 'tech_1' 'tech_2'

        """
        inherits_nodes = xr.DataArray(True)
        inherits_techs = xr.DataArray(True)
        if nodes is not None:
            inherits_nodes = self._input_data.get(
                "nodes_inheritance", xr.DataArray("")
            ).str.contains(f"{nodes}(?:,|$)")
        if techs is not None:
            inherits_techs = self._input_data.get(
                "techs_inheritance", xr.DataArray("")
            ).str.contains(f"{techs}(?:,|$)")
        return inherits_nodes & inherits_techs


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

    def as_array(self, parameter: str, *, over: str | list[str]) -> xr.DataArray:
        """Reduce the boolean where array of a model parameter by applying `any` over some dimension(s).

        Args:
            parameter (str): Reference to a model input parameter
            over (str | list[str]): dimension(s) over which to apply `any`.

        Returns:
            xr.DataArray:
                If the parameter exists in the model, returns a boolean array with dimensions reduced by applying a boolean OR operation along the dimensions given in `over`.
                If the parameter does not exist, returns a dimensionless False array.
        """
        if parameter in self._input_data.data_vars:
            parameter_da = self._input_data[parameter]
            bool_parameter_da = (
                parameter_da.notnull()
                & (parameter_da != np.inf)
                & (parameter_da != -np.inf)
            )
        elif (
            self._backend_interface is not None
            and parameter in self._backend_interface._dataset
        ):
            bool_parameter_da = self._backend_interface._dataset[parameter].notnull()
        else:
            bool_parameter_da = xr.DataArray(False)
        over = self._listify(over)
        available_dims = set(bool_parameter_da.dims).intersection(over)

        return bool_parameter_da.any(dim=available_dims, keep_attrs=True)


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
        within_singular = within.removesuffix("s")
        dim_singular = dim.removesuffix("s")
        selection = rf"\text{{{dim_singular}}} \in \text{{[{','.join(vals)}]}}"

        return rf"\big{tex_how}\limits_{{\substack{{{selection}}}}}\text{{{dim_singular} defined in {within_singular}}}"


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
            k.removesuffix("s"): f"{-1 * int(v):+d}" for k, v in roll_kwargs.items()
        }
        component = self._add_to_iterator(array, new_strings)
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
            parameters:
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
