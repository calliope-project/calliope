# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Model data processing functionality."""

import functools
import itertools
import logging
from abc import ABC
from collections.abc import Hashable, Iterable, Mapping
from typing import Literal, TypeVar, overload

import pandas as pd
import xarray as xr
from geographiclib import geodesic

from calliope import exceptions
from calliope.preprocess import data_tables, time
from calliope.schemas import config_schema, math_schema, runtime_attrs_schema
from calliope.schemas.dimension_data_schema import (
    CalliopeNode,
    CalliopeNodes,
    CalliopeTech,
    CalliopeTechs,
    CalliopeTransmissionTech,
    IndexedData,
)
from calliope.schemas.general import CalliopeBaseModel, CalliopeDictModel
from calliope.schemas.model_def_schema import CalliopeModelDef
from calliope.util import DATETIME_DTYPE, DTYPE_OPTIONS
from calliope.util.tools import listify

LOGGER = logging.getLogger(__name__)

DATA_T = float | int | bool | str | None | list[float | int | bool | str | None]
DEF_T = TypeVar("DEF_T", bound=CalliopeBaseModel | CalliopeDictModel)


class ModelDTypeUpdater(ABC):
    """Model data production class."""

    config: config_schema.Init
    math: math_schema.CalliopeBuildMath

    def _update_dtypes(
        self, ds: Mapping[Hashable, "xr.DataArray"], id_: str = ""
    ) -> Mapping[Hashable, "xr.DataArray"]:
        """Update data types of coordinates or data variables in the dataset.

        Args:
            ds (xr.Dataset): Dataset to update.
            math (math_schema.CalliopeBuildMath): Model math definition.
            id_ (str, optional): ID of the dataset being updated, for logging purposes. Defaults to an empty string.

        Raises:
            exceptions.ModelError: If there is a mismatch between the provided variable and its definition in the model math.

        Returns:
            xr.Dataset: `ds` with data types updated.
        """
        prefix = f"{id_} | " if id_ else ""
        for var_name, var_data in ds.items():
            try:
                math_def = self.math.find(
                    var_name, subset=["lookups", "parameters", "dimensions"]
                )
            except KeyError:
                LOGGER.info(
                    f"{prefix}input data `{var_name}` not defined in model math; "
                    "it will not be available in the optimisation problem."
                )
                continue

            dtype_str = math_def.dtype  # type: ignore
            dtype = DTYPE_OPTIONS[dtype_str]
            LOGGER.debug(
                f"{prefix}{math_def._group} | Updating values of `{var_name}` to {dtype_str} type"
            )
            match dtype_str:
                case "string":
                    updated_var = var_data.astype(dtype).where(var_data.notnull())
                case "datetime":
                    updated_var = time._datetime_index(
                        var_data.to_series(), self.config.datetime_format
                    ).to_xarray()
                case "date":
                    updated_var = (
                        time._datetime_index(
                            var_data.to_series(), self.config.date_format
                        )
                        .to_xarray()
                        .assign_attrs(var_data.attrs)
                    )
                case "bool":
                    updated_var = var_data.fillna(False).astype(dtype)
                case _:
                    updated_var = var_data.astype(dtype)

            ds[var_name] = updated_var
        return ds


class ModelDataBuilder(ModelDTypeUpdater):
    """Model data builder class."""

    def __init__(
        self,
        init_config: config_schema.Init,
        model_definition: CalliopeModelDef,
        math: math_schema.CalliopeBuildMath,
        tables: Iterable[data_tables.DataTable] | None = None,
    ):
        """Take a Calliope model definition dictionary and convert it into an xarray Dataset, ready for constraint generation.

        This includes resampling/clustering timeseries data as necessary.

        Args:
            init_config (config_schema.Init): Model initialisation configuration (i.e., `config`).
            model_definition (CalliopeModelDef): Definition of model input data.
            math (math_schema.CalliopeBuildMath): Math to apply to the model.
            tables (Iterable[data_tables.DataTable], None): Loaded data tables. Defaults to None.
        """
        self.config = init_config
        self.tech_data_from_tables = CalliopeTechs()
        self.math = math
        self.model_definition = model_definition
        self.dataset = xr.Dataset()
        if tables:
            self.init_from_data_tables(tables)

    def build(self):
        """Build dataset from model definition."""
        self.add_node_tech_data()
        self.add_top_level_data_definitions()

    @overload
    @staticmethod
    def _update_def(
        to_update: CalliopeBaseModel,
        update_def: CalliopeBaseModel | CalliopeDictModel,
        key: str | None = None,
        overwrite: bool = False,
    ) -> CalliopeBaseModel: ...

    @overload
    @staticmethod
    def _update_def(
        to_update: CalliopeDictModel,
        update_def: CalliopeDictModel,
        key: None = None,
        overwrite: bool = False,
    ) -> CalliopeDictModel: ...

    @staticmethod
    def _update_def(
        to_update: CalliopeBaseModel | CalliopeDictModel,
        update_def: CalliopeBaseModel | CalliopeDictModel,
        key: str | None = None,
        overwrite: bool = False,
    ) -> CalliopeBaseModel | CalliopeDictModel:
        """Update a pydantic model using another pydantic model.

        If a key is given, the update will ignore keys already present in the `key` sub-model of `to_update`, allowing for a form of precedence in the inheritance tree.

        Args:
            to_update (CalliopeBaseModel | CalliopeDictModel):
                Pydantic model to update.
            update_def (CalliopeBaseModel | CalliopeDictModel):
                pydantic model to update from.
            key (str | None, optional):
                If given, update the sub-model corresponding to the key. Defaults to None.
            overwrite (bool, optional):
                If True, values in `update_def` will overwrite those already in `to_update`.
                Defaults to False.

        Returns:
            CalliopeBaseModel | CalliopeDictModel:
                The updated pydantic model, of the same class as the input `to_update`.
        """
        kwargs = {"exclude_unset": True}
        if key is not None and isinstance(to_update, CalliopeDictModel):
            msg = "Cannot specify a key for updating a CalliopeDictModel, as it has no predefined field names."
            raise ValueError(msg)
        if key is not None and key not in to_update.__class__.model_fields:
            raise KeyError(
                f"Key '{key}' not found in model fields of {to_update.__class__.__name__}"
            )
        # Update `update_def` with the contents of `to_update` if we do not allow overwriting.
        if overwrite:
            pre_update = {}
        elif key is not None:
            pre_update = getattr(to_update, key).model_dump(**kwargs)
        else:
            pre_update = to_update.model_dump(**kwargs)
        update_dict = update_def.update(pre_update).model_dump(**kwargs)

        if key is not None:
            updated = to_update.update({key: update_dict})
        else:
            updated = to_update.update(update_dict)
        return updated

    def init_from_data_tables(self, data_tables: Iterable[data_tables.DataTable]):
        """Initialise the model definition and dataset using data loaded from file / in-memory objects.

        A basic skeleton of the dictionary format model definition is created from the data tables,
        namely technology and technology-at-node lists (without parameter definitions).

        Args:
            data_tables (list[data_tables.DataTable]): Pre-loaded data tables.
        """
        for data_table in data_tables:
            tech_def, base_tech_data = data_table.tech_def()
            self.model_definition = self._update_def(
                self.model_definition, tech_def, key="techs"
            )
            self.tech_data_from_tables = self._update_def(
                self.tech_data_from_tables, base_tech_data, overwrite=True
            )

        techs_incl_inheritance = self._inherit_defs("techs")
        for data_table in data_tables:
            node_def = data_table.node_def(techs_incl_inheritance)
            self.model_definition = self._update_def(
                self.model_definition, node_def, key="nodes"
            )
            for param, param_config in self.math.lookups.root.items():
                lookup_dim = param_config.pivot_values_to_dim
                if lookup_dim is not None:
                    lookup_def = data_table.lookup_def_from_param(param, lookup_dim)
                    self.tech_data_from_tables = self._update_def(
                        self.tech_data_from_tables, lookup_def, overwrite=True
                    )
                    data_table.drop(param)

        # Pre-populate the dataset with model nodes and techs
        self.dataset = self.dataset.assign_coords(
            nodes=list(self.model_definition.nodes.root.keys()),
            techs=list(self.model_definition.techs.root.keys()),
        )
        for data_table in data_tables:
            self._add_to_dataset(
                data_table.dataset, f"(data_tables, {data_table.name})"
            )

    def add_node_tech_data(self):
        """For each node, extract technology definitions and other input data and convert them to arrays.

        The node definition will be updated with each defined tech (which will also be updated according to its inheritance tree).

        Node and tech definitions will be validated against the model definition schema here.
        """
        active_node_def = self._inherit_defs("nodes")
        links_at_nodes = self._links_to_node_format(active_node_def)

        node_tech_data = []
        for node_name, node_data in active_node_def.root.items():
            techs_this_node = node_data.techs
            node_ref_vars = self._get_relevant_node_refs(techs_this_node, node_name)

            techs_this_node_incl_inheritance = self._inherit_defs(
                "techs", techs_this_node, nodes=node_name
            )
            self._raise_error_on_transmission_tech_def(
                techs_this_node_incl_inheritance, node_name
            )
            if node_name in links_at_nodes.root:
                techs_this_node_incl_inheritance = self._update_def(
                    techs_this_node_incl_inheritance,
                    links_at_nodes[node_name].techs,
                    overwrite=True,
                )

            tech_ds = self._definition_to_ds(techs_this_node_incl_inheritance, "techs")

            tech_ds.coords["nodes"] = node_name
            for ref_var in node_ref_vars:
                tech_ds[ref_var] = tech_ds[ref_var].expand_dims("nodes")
            for ref_var in ["carrier_in", "carrier_out"]:
                if ref_var in tech_ds.data_vars:
                    tech_ds[ref_var] = tech_ds[ref_var].expand_dims("nodes")
            if not tech_ds.nodes.shape:
                tech_ds["nodes"] = tech_ds["nodes"].expand_dims("nodes")

            node_tech_data.append(tech_ds)

        node_tech_ds = xr.combine_nested(
            node_tech_data,
            concat_dim="nodes",
            data_vars="minimal",
            combine_attrs="no_conflicts",
            coords="minimal",
        )

        node_ds = self._definition_to_ds(active_node_def, "nodes")
        ds = xr.merge([node_tech_ds, node_ds])
        self._add_to_dataset(ds, "YAML definition")

    def add_top_level_data_definitions(self):
        """Process any input data defined in the `data_definitions` key.

        Raises:
            KeyError: Cannot provide the same name for an indexed input as those defined already at the tech/node level.

        """
        for name, data in self.model_definition.data_definitions.root.items():
            if name in self.dataset.data_vars:
                exceptions.warn(
                    f"(Model inputs, {name}) | "
                    "Model input data with this name has already been defined in a data table or at a node/tech level. "
                    f"Non-NaN data defined here will override existing data for it."
                )
            input_dict = self._prepare_input_data(name, data)
            input_da = self._input_data_to_array(name, input_dict)
            self._log_input_data_updates(name, input_da)
            input_ds = input_da.to_dataset()

            if "techs" in input_da.dims and "nodes" in input_da.dims:
                valid_node_techs = (
                    input_da.to_series().dropna().groupby(["nodes", "techs"]).first()
                )
                exceptions.warn(
                    f"(Model inputs, {name}) | This input data will only take effect if you have already defined"
                    f" the following combinations of techs at nodes in your model definition: {valid_node_techs.index.values}"
                )

            self._add_to_dataset(input_ds, f"(Model inputs, {name})")

    def _get_relevant_node_refs(self, techs_def: CalliopeTechs, node: str) -> list[str]:
        """Get all references to input data made in technologies at nodes.

        This defines those arrays in the dataset that *must* be indexed over `nodes` as well as `techs`.

        If timeseries files/dataframes are referenced in a tech at a node, the node name is added as the column name in-place.
        Techs *must* define these timeseries references explicitly at nodes to access different data columns at different nodes.

        Args:
            techs_def (CalliopeTechs): Dictionary of technologies defined at a node.
            node (str): Name of the node.

        Returns:
            list[str]: List of input data at this node that must be indexed over the node dimension.
        """
        refs = set()

        for tech_name, tech_data in techs_def.root.items():
            if tech_data is None:
                continue
            elif not tech_data.active:
                self._deactivate_item(techs=tech_name, nodes=node)
            else:
                if tech_data.base_tech is not None:
                    raise exceptions.ModelError(
                        f"(nodes, {node}), (techs, {tech_name}) | Defining a technology `base_tech` at a node is not supported; "
                        "limit yourself to defining this lookup within `techs` or `templates`"
                    )
                refs.update(tech_data.model_fields_set)

        return list(refs)

    def _input_data_to_array(self, name: str, input_data: IndexedData) -> xr.DataArray:
        """Take a validated input data dictionary and convert it to an xarray DataArray.

        Args:
            name (str): Name of the parameter being converted.
            input_data (ValidatedInput): validated dictionary. I.e., keys/values follow an expected structure.

        Returns:
            xr.DataArray: Array representation of the parameter.
        """
        if input_data.dims:
            input_data_series = pd.Series(
                data=input_data.data, index=[tuple(idx) for idx in input_data.index]
            )
            input_data_series.index = pd.MultiIndex.from_tuples(
                input_data_series.index, names=input_data.dims
            )
            input_data_da = input_data_series.to_xarray()
        else:
            input_data_da = xr.DataArray(input_data.data)
        input_data_da = input_data_da.rename(name)
        return input_data_da

    @overload
    def _definition_to_ds(
        self, definition: CalliopeNodes, dim_name: Literal["nodes"]
    ) -> xr.Dataset: ...

    @overload
    def _definition_to_ds(
        self, definition: CalliopeTechs, dim_name: Literal["techs"]
    ) -> xr.Dataset: ...
    def _definition_to_ds(
        self,
        definition: CalliopeTechs | CalliopeNodes,
        dim_name: Literal["techs", "nodes"],
    ) -> xr.Dataset:
        """Convert nodes/techs definition with their input data definitions into an xarray dataset.

        Node/tech name will be injected into each input's `index` and `dims` lists so that the resulting arrays include those dimensions.

        Args:
            definition (CalliopeTechs | CalliopeNodes): Dictionary of `techs` or `nodes` definitions, including any input data definitions nested within them.
                This should already include inherited parameters from the base definitions.
            dim_name (Literal[nodes, techs]): Dimension name of the dictionary items.

        Returns:
            xr.Dataset: Dataset with arrays indexed over (at least) the input `dim_name`.
        """
        exclude = {"techs"} if dim_name == "nodes" else None
        input_data_ds = xr.Dataset()
        for idx_name, idx_inputs in definition.root.items():
            if idx_inputs is None:
                continue
            input_data_das: list[xr.DataArray] = []
            for name, input_data in idx_inputs.model_dump(
                exclude=exclude, exclude_defaults=True
            ).items():
                validated_data = self._prepare_input_data(name, input_data)
                validated_data = validated_data.update(
                    {
                        "index": [[idx_name] + idx for idx in validated_data.index],
                        "dims": [dim_name, *validated_data.dims],
                    }
                )
                input_data_das.append(self._input_data_to_array(name, validated_data))
            input_data_ds = xr.merge(
                [input_data_ds, xr.combine_by_coords(input_data_das)]
            )

        return input_data_ds

    def _prepare_input_data(
        self, name: str, raw_input_data: dict | list[str] | DATA_T
    ) -> IndexedData:
        """Convert a range of input data definitions into the `ValidatedInput` format.

        Args:
            name (str): input data name (used only in error messages).
            raw_input_data (dict | list[str] | DATA_T): unformatted input data.

        Raises:
            ValueError: If the input data is unindexed (i.e., no `dims`/`index`) and is
                not a lookup array (see LOOKUP_PARAMS), it cannot define a list of data.

        Returns:
            IndexedData: validated input data dictionary.
        """
        if isinstance(raw_input_data, IndexedData):
            data_def = raw_input_data
        elif isinstance(raw_input_data, dict):
            data_def = IndexedData.model_validate(raw_input_data)
            broadcast_input_data = self.config.broadcast_input_data
            if not broadcast_input_data and len(listify(data_def.data)) != len(
                data_def.index
            ):
                raise exceptions.ModelError(
                    f"{name} | Length mismatch between data ({data_def.data}) and index ({data_def.index}) in input definition. "
                    "Check lengths of arrays or set `config.broadcast_input_data` to True "
                    "to allow single data entries to be broadcast across all parameter index items."
                )
        elif (
            name in self.math.lookups.root
            and self.math.lookups[name].pivot_values_to_dim is not None
            and raw_input_data is not None
        ):
            dims = self.math.lookups[name].pivot_values_to_dim
            data_def = IndexedData.model_validate(
                {"data": True, "index": raw_input_data, "dims": dims}
            )
        else:
            if isinstance(raw_input_data, list):
                raise ValueError(
                    f"{name} | Cannot pass un-indexed list input data. Received: {raw_input_data}."
                )
            data_def = IndexedData.model_validate({"data": raw_input_data})
        return data_def

    @overload
    def _inherit_defs(
        self,
        dim_name: Literal["techs"],
        dim_def: CalliopeTechs | None = None,
        **connected_dims: str,
    ) -> CalliopeTechs: ...

    @overload
    def _inherit_defs(
        self,
        dim_name: Literal["nodes"],
        dim_def: CalliopeNodes | None = None,
        **connected_dims: str,
    ) -> CalliopeNodes: ...

    def _inherit_defs(
        self,
        dim_name: Literal["nodes", "techs"],
        dim_def: CalliopeTechs | CalliopeNodes | None = None,
        **connected_dims: str,
    ) -> CalliopeTechs | CalliopeNodes:
        """For a set of node/tech definitions, climb the inheritance tree to build a final definition dictionary.

        For `techs` at `nodes`, they inherit the technology definition from `techs`.

        Base definitions will take precedence over inherited ones and more recent inherited definitions will take precedence over older ones.

        If a `tech`/`node` has the `active` parameter set to `False` (including if it inherits this parameter), it will not make it into the output dictionary.

        Args:
            dim_name (Literal[nodes, techs]): Name of dimension we're working with.
            dim_def (CalliopeTechs | CalliopeNodes | None, optional):
                Base definition to work from.
                If not defined, `dim_name` will be used to access the base definition from the model definition.
                Defaults to None.

        Keyword Args:
            connected_dims (str):
                Any dimension index items connected to the one for which we're tracing inheritance.
                E.g., if looking at technologies at a node `A`, we would be using `dim_name=techs` and `connected_dims={nodes=A}`
        Raises:
            KeyError: Cannot define a `tech` at a `node` if it isn't already defined under the `techs` top-level key.

        Returns:
            CalliopeTechs | CalliopeNodes: Dictionary containing all active tech/node definitions with inherited parameters.
        """
        if connected_dims:
            debug_message_prefix = (
                ", ".join([f"({k}, {v})" for k, v in connected_dims.items()]) + ", "
            )
        else:
            debug_message_prefix = ""

        updated_defs = CalliopeTechs() if dim_name == "techs" else CalliopeNodes()
        if dim_def is None:
            dim_def = self.model_definition[dim_name]

        for item_name, item_def in dim_def.root.items():
            if item_def is None:
                item_def = CalliopeTech() if dim_name == "techs" else CalliopeNode()
            if dim_name == "techs":
                base_def = self.model_definition["techs"]
                if item_name not in base_def.root:
                    raise KeyError(
                        f"{debug_message_prefix}({dim_name}, {item_name}) | Reference to item not defined in base {dim_name}"
                    )

                item_base_def = self._update_def(
                    base_def[item_name], item_def, overwrite=True
                )

                if item_name in self.tech_data_from_tables.root:
                    item_base_def = self._update_def(
                        self.tech_data_from_tables[item_name],
                        item_base_def,
                        overwrite=True,
                    )
            else:
                item_base_def = item_def

            if not item_base_def.active:
                LOGGER.debug(
                    f"{debug_message_prefix}({dim_name}, {item_name}) | Deactivated."
                )
                self._deactivate_item(**{dim_name: item_name, **connected_dims})
                continue

            updated_defs = updated_defs.update(
                {item_name: item_base_def.model_dump(exclude_unset=True)}
            )

        return updated_defs

    def _deactivate_item(self, **item_ref):
        for dim_name, item_name in item_ref.items():
            if item_name not in self.dataset.coords.get(dim_name, xr.DataArray()):
                return None
        if len(item_ref) == 1:
            self.dataset = self.dataset.drop_sel(**item_ref)
        else:
            if "carrier_in" in self.dataset:
                self.dataset["carrier_in"].loc[item_ref] = False
            if "carrier_out" in self.dataset:
                self.dataset["carrier_out"].loc[item_ref] = False

    def _links_to_node_format(self, active_node_def: CalliopeNodes) -> CalliopeNodes:
        """Process `transmission` techs into links by assigning them to the nodes defined by their `link_from` and `link_to` keys.

        Args:
            active_node_def (CalliopeNodes):
                Definition of nodes that are active in this model.
                If a transmission tech references a non-active / undefined node, a link will not be generated.

        Returns:
            CalliopeNodes: Node definition with transmission techs distributed to nodes (of the form {node_name: {tech_name: {...}, ...}}).
        """
        active_techs = self._inherit_defs("techs")
        link_tech_def = CalliopeNodes()

        for link_name, link_data in active_techs.root.items():
            if not isinstance(link_data, CalliopeTransmissionTech):
                continue
            node_from, node_to = link_data.link_from, link_data.link_to
            nodes_exists = all(
                node in active_node_def.root
                or node in self.dataset.coords.get("nodes", xr.DataArray())
                for node in [node_from, node_to]
            )

            if not nodes_exists:
                LOGGER.debug(
                    f"(links, {link_name}) | Deactivated due to missing/deactivated `link_from` or `link_to` node."
                )
                self._deactivate_item(techs=link_name)
                continue

            exclude_from = {"link_from", "link_to", "one_way"}
            exclude_to = {"link_from", "link_to", "one_way"}
            if link_data.one_way:
                exclude_from.update(["carrier_out"])
                exclude_to.update(["carrier_in"])
            node_from_data = link_data.model_dump(
                exclude=exclude_from, exclude_unset=True
            )
            node_to_data = link_data.model_dump(exclude=exclude_to, exclude_unset=True)
            link_tech_def = link_tech_def.update(
                {
                    node_from: {"techs": {link_name: node_from_data}},
                    node_to: {"techs": {link_name: node_to_data}},
                }
            )
        if not link_tech_def.root:
            LOGGER.debug("links | No links between nodes defined.")

        return link_tech_def

    def _add_to_dataset(self, to_add: xr.Dataset, id_: str):
        """Add new data to the central class dataset.

        Before being added, dimension and parameters types will be handled.

        Args:
            to_add (xr.Dataset): Dataset to merge into the central dataset.
            id_ (str): ID of dataset being added, to use in log messages
        """
        to_add_update_dim_dtype = to_add.assign_coords(
            self._update_dtypes(to_add.coords, id_)
        )
        self.dataset = xr.merge(
            [to_add_update_dim_dtype, self.dataset],
            combine_attrs="no_conflicts",
            compat="override",
        ).fillna(self.dataset)

    def _log_input_data_updates(self, name: str, input_data_da: xr.DataArray):
        """Logger for model input data updates.

        Checks array coordinates to see if:
            1. any are new compared to the base model dimensions.
            2. any are adding new elements to an existing base model dimension.

        Args:
            name (str): name of input being added to the model.
            input_data_da (xr.DataArray): array of input data.
        """
        for coord_name, coord_data in input_data_da.coords.items():
            if coord_name not in self.dataset.coords:
                LOGGER.debug(
                    f"(Model inputs, {name}) | Adding a new dimension to the model: {coord_name}"
                )
            else:
                new_coord_data = coord_data[
                    ~coord_data.isin(self.dataset.coords[coord_name])
                ]
                if new_coord_data.size > 0:
                    LOGGER.debug(
                        f"(Model inputs, {name}) | Adding a new value to the "
                        f"`{coord_name}` model coordinate: {new_coord_data.values}"
                    )

    def _raise_error_on_transmission_tech_def(
        self, tech_def: CalliopeTechs, node_name: str
    ):
        """Do not allow any transmission techs to be defined in the node-level tech dict.

        Args:
            tech_def (CalliopeTechs): Tech definition (after full inheritance) at a node.
            node_name (str): Node name.

        Raises:
            exceptions.ModelError: Raise if any defined techs have the `transmission` base_tech.
        """
        transmission_techs = list(
            filter(
                lambda k: tech_def[k].base_tech == "transmission", tech_def.root.keys()
            )
        )

        if transmission_techs:
            raise exceptions.ModelError(
                f"(nodes, {node_name}) | Transmission techs cannot be directly defined at nodes; "
                f"they will be automatically assigned to nodes based on `link_to` and `link_from` for: {transmission_techs}."
            )


class ModelDataCleaner(ModelDTypeUpdater):
    """Model data cleaning class."""

    # Output of: sns.color_palette('cubehelix', 10).as_hex()
    _DEFAULT_PALETTE = [
        "#19122b",
        "#17344c",
        "#185b48",
        "#3c7632",
        "#7e7a36",
        "#bc7967",
        "#d486af",
        "#caa9e7",
        "#c2d2f3",
        "#d6f0ef",
    ]

    def __init__(
        self,
        init_config: config_schema.Init,
        dataset: xr.Dataset,
        math: math_schema.CalliopeBuildMath,
        runtime: runtime_attrs_schema.CalliopeRuntime,
    ):
        """Take a Calliope model definition dictionary and convert it into an xarray Dataset, ready for constraint generation.

        This includes resampling/clustering timeseries data as necessary.

        Args:
            init_config (config_schema.Init): Model initialisation configuration (i.e., `config`).
            dataset (xr.Dataset): Dataset containing model input data.
            math (math_schema.CalliopeInputMath): Math schema to apply to the model.
            runtime (runtime_attrs_schema.CalliopeRuntime): Runtime attributes of the model.
        """
        self.config = init_config
        self.math = math
        self.dataset = dataset.copy()
        self.runtime = runtime

    def clean(self):
        """Clean built dataset."""
        # If input dataset is empty, stop here.
        self.clean_data_from_undefined_members()
        self.add_colors()
        self.add_link_distances()
        self.update_and_resample_dimensions()
        self.dataset = self.dataset.assign_coords(
            self._update_dtypes(self.dataset.coords)
        )
        self.dataset = self._update_dtypes(self.dataset)
        self.runtime = self.runtime.update({"instantiated": True})

    def clean_data_from_undefined_members(self):
        """Generate the `definition_matrix` array and remove undefined members.

        Members stripped:
        - Any dimension items that are NaN in all arrays.
        - Any arrays that are NaN in all index positions.
        """
        ds = self._update_dtypes(self.dataset)
        def_matrix = ds.carrier_in | ds.carrier_out
        # NaNing values where they are irrelevant requires definition_matrix to be boolean
        for var_name, var_data in ds.data_vars.items():
            non_dims = set(def_matrix.dims).difference(var_data.dims)
            var_updated = var_data.where(def_matrix.any(non_dims))
            ds[var_name] = (
                var_updated
                if var_data.dtype.kind != "b"
                else var_updated.fillna(False).astype(bool)
            )
        # dropping index values where they are irrelevant requires definition_matrix to be NaN where False
        self.dataset = self._drop_undefined(ds, def_matrix)

    def add_colors(self):
        """If technology colours have not been provided / only partially provided, generate a sequence of colors to fill the gap.

        This is a convenience function for downstream plotting.
        Since we have removed core plotting components from Calliope, it is not a strictly necessary preprocessing step.
        """
        techs = self.dataset.techs
        color_array = self.dataset.get("color")
        default_palette_cycler = itertools.cycle(range(len(self._DEFAULT_PALETTE)))
        new_color_array = xr.DataArray(
            [self._DEFAULT_PALETTE[next(default_palette_cycler)] for tech in techs],
            coords={"techs": techs},
        )
        if color_array is None:
            LOGGER.debug("Building technology color array from default palette.")
            self.dataset["color"] = new_color_array
        elif color_array.isnull().any():
            LOGGER.debug(
                "Filling missing technology color array values from default palette."
            )
            self.dataset["color"] = self.dataset["color"].fillna(new_color_array)

    def add_link_distances(self):
        """If latitude/longitude are provided but distances between nodes have not been computed, compute them now.

        The schema will have already handled the fact that if one of lat/lon is provided, the other must also be provided.
        """
        # If no distance was given, we calculate it from coordinates
        if (
            "latitude" in self.dataset.data_vars
            and "longitude" in self.dataset.data_vars
            and (self.dataset.base_tech == "transmission").any()
        ):
            distances = {}
            for tech in self.dataset.techs:
                if self.dataset.base_tech.sel(techs=tech).item() != "transmission":
                    continue
                tech_def = self.dataset.definition_matrix.sel(techs=tech).any(
                    "carriers"
                )
                node1, node2 = tech_def.where(tech_def).dropna("nodes").nodes.values
                distances[tech.item()] = self._get_distance(node1, node2)
            distance_array = pd.Series(distances).rename_axis(index="techs").to_xarray()
            if self.config.distance_unit == "km":
                distance_array /= 1000
        else:
            LOGGER.debug(
                "Link distances will not be computed automatically since lat/lon coordinates are not defined."
            )
            return None

        if "distance" not in self.dataset.data_vars:
            self.dataset["distance"] = distance_array
            LOGGER.debug(
                "Link distance matrix automatically computed from lat/lon coordinates."
            )
        else:
            self.dataset["distance"] = self.dataset["distance"].fillna(distance_array)
            LOGGER.debug(
                "Any missing link distances automatically computed from lat/lon coordinates."
            )

    def update_and_resample_dimensions(self):
        """If resampling/clustering is requested in the initialisation config, apply it here."""
        if not any(
            dim.dtype.kind == DATETIME_DTYPE for dim in self.dataset.coords.values()
        ):
            raise exceptions.ModelError(
                "Must define at least one timeseries data input in a Calliope model."
            )
        runtime_updater = {}
        if self.config.subset != self.runtime.subset:
            self._subset_dims()
            runtime_updater["subset"] = self.config.subset.model_dump()
        if self.config.resample != self.runtime.resample:
            self._resample_dims()
            runtime_updater["resample"] = self.config.resample.model_dump()

        if not self.runtime.instantiated:
            self.dataset = time.add_inferred_time_params(self.dataset)

        if self.runtime.time_cluster is None and self.config.time_cluster is not None:
            self.dataset = time.cluster(
                self.dataset, self.config.time_cluster, self.config.datetime_format
            )
            runtime_updater["time_cluster"] = self.config.time_cluster
        elif self.config.time_cluster != self.runtime.time_cluster:
            raise exceptions.ModelError(
                "Cannot change time clustering configuration at this stage."
            )
        self.runtime = self.runtime.update(runtime_updater)

    @staticmethod
    def _drop_undefined(ds: xr.Dataset, def_matrix: xr.DataArray) -> xr.Dataset:
        """Drop undefined members from a dataset.

        Members dropped:
        - Any dimension items that are NaN in all arrays.
        - Any arrays that are NaN in all index positions.

        Args:
            ds (xr.Dataset): Dataset to drop undefined members from.
            def_matrix (xr.DataArray): Definition matrix to use to identify undefined members.

        Returns:
            xr.Dataset: Input `ds` with undefined members dropped.
        """
        ds["definition_matrix"] = def_matrix.where(def_matrix)
        for dim in def_matrix.dims:
            orig_dim_vals = set(ds.coords[dim].data)
            ds = ds.dropna(dim, how="all", subset=["definition_matrix"])
            deleted_dim_vals = orig_dim_vals.difference(set(ds.coords[dim].data))
            if deleted_dim_vals:
                LOGGER.debug(
                    f"Deleting {dim} values as they are not defined anywhere in the model: {deleted_dim_vals}"
                )

        # The boolean version of definition_matrix is what we keep
        ds["definition_matrix"] = def_matrix

        vars_to_delete = [
            var_name for var_name, var in ds.data_vars.items() if var.isnull().all()
        ]
        if vars_to_delete:
            LOGGER.debug(f"Deleting empty input data: {vars_to_delete}")
        return ds.drop_vars(vars_to_delete)

    @functools.lru_cache(maxsize=1000)
    def _get_distance(self, node1: str, node2: str) -> float:
        """Get and cache the distance between two nodes.

        Args:
            node1 (str): The first node.
            node2 (str): The second node.

        Returns:
            float: The geodesic distance between the two nodes.
        """
        geod = geodesic.Geodesic.WGS84
        return geod.Inverse(
            self.dataset.latitude.sel(nodes=node1).item(),
            self.dataset.longitude.sel(nodes=node1).item(),
            self.dataset.latitude.sel(nodes=node2).item(),
            self.dataset.longitude.sel(nodes=node2).item(),
        )["s12"]

    def _subset_dims(self):
        """Subset all timeseries dimensions according to an input slice of start and end times.

        Args:
            ds (xr.Dataset): Dataset containing timeseries data to subset.

        Returns:
            xr.Dataset: Input `ds` with subset timeseries coordinates.
        """
        selectors = {}

        for dim_name, subset in self.config.subset.root.items():
            if subset is None:
                continue
            elif dim_name not in self.dataset.coords:
                LOGGER.debug(f"Skipping subsetting for undefined dimension: {dim_name}")
                continue
            is_ordered = self.math.dimensions[dim_name].ordered
            dim_vals = self.dataset.coords[dim_name]

            if dim_vals.dtype.kind == DATETIME_DTYPE:
                time.check_time_subset(dim_vals.to_index(), subset)

            if is_ordered:
                selectors[dim_name] = slice(*subset)
            else:
                selectors[dim_name] = subset

        subset_dataset = self.dataset.sel(**selectors)

        # Drop any transmission links that are now hanging (i.e., only connected to one node)
        hanging_links = (
            subset_dataset.definition_matrix.sel(
                techs=subset_dataset.base_tech == "transmission"
            )
            .sum("nodes")
            .where(lambda x: x == 1, drop=True)
            .techs
        )
        subset_dataset = subset_dataset.drop_sel(techs=hanging_links)

        self.dataset = self._drop_undefined(
            subset_dataset, subset_dataset.definition_matrix
        )

    def _resample_dims(self):
        ds = self.dataset
        for dim_name, resampler in self.config.resample.root.items():
            if resampler is None:
                continue
            elif dim_name not in ds.coords:
                LOGGER.debug(f"Skipping resampling for undefined dimension: {dim_name}")
                continue
            if ds.coords[dim_name].dtype.kind != DATETIME_DTYPE:
                raise exceptions.ModelError(
                    f"Cannot resample a non-datetime dimension, received `{dim_name}`"
                )
            ds = time.resample(ds, self.math, dim_name, resampler)
        self.dataset = ds
