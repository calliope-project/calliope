# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Functions to read and save model results and configuration."""

import importlib.resources
import logging
import os
from copy import deepcopy
from io import StringIO
from pathlib import Path
from typing import Literal

# We import netCDF4 before xarray to mitigate a numpy warning:
# https://github.com/pydata/xarray/issues/7259
import netCDF4  # noqa: F401
import numpy as np
import pandas as pd
import ruamel.yaml as ruamel_yaml
import xarray as xr

from calliope.attrdict import AttrDict
from calliope.util.tools import listify, relative_path

logger = logging.getLogger(__name__)

CONFIG_DIR = importlib.resources.files("calliope") / "config"
YAML_INDENT = 2
YAML_BLOCK_SEQUENCE_INDENT = 0


def read_netcdf(
    path: str | Path,
) -> dict[Literal["inputs", "results", "attrs"], xr.Dataset]:
    """Read model_data from NetCDF file."""
    datasets: dict[Literal["inputs", "results", "attrs"], xr.Dataset] = {}
    for group in ["inputs", "results", "attrs"]:
        try:
            with xr.open_dataset(path, group=group) as model_data:
                model_data.load()
        except OSError:
            datasets[group] = xr.Dataset()
            continue
        _deserialise(model_data.attrs)
        for var in model_data.data_vars.values():
            _deserialise(var.attrs)
        datasets[group] = model_data
    return datasets


def _pop_serialised_list(
    attrs: dict, serialised_items: str | list
) -> list | np.ndarray:
    """Pop a list of serialised attributes from the attribute dictionary."""
    serialised_ = attrs.pop(serialised_items, [])
    return listify(serialised_)


def _serialise(attrs: dict) -> None:
    """Convert troublesome datatypes to nicer ones in xarray attribute dictionaries.

    This will tackle dictionaries (to string), booleans (to int), None (to string), and sets (to list).

    We also make note of any single element lists as they will be stored as simple strings in netcdf format,
    but we want them to be returned as lists on loading the data from file.

    Args:
        attrs (dict):
            Attribute dictionary from an xarray Dataset/DataArray.
            Changes will be made in-place, so be sure to supply a copy of your dictionary if you want access to its original state.
    """
    # Keep track of single element lists, to listify them again when loading from file.

    attrs["serialised_single_element_list"] = [
        k for k, v in attrs.items() if isinstance(v, list) and len(v) == 1
    ]

    # Convert dict attrs to yaml strings
    dict_attrs = [k for k, v in attrs.items() if isinstance(v, dict)]
    attrs["serialised_dicts"] = dict_attrs
    for attr in dict_attrs:
        attrs[attr] = to_yaml(attrs[attr])

    # Convert boolean attrs to ints
    bool_attrs = [k for k, v in attrs.items() if isinstance(v, bool)]
    attrs["serialised_bools"] = bool_attrs
    for attr in bool_attrs:
        attrs[attr] = int(attrs[attr])

    # Convert None attrs to 'None'
    none_attrs = [k for k, v in attrs.items() if v is None]
    attrs["serialised_nones"] = none_attrs
    for attr in none_attrs:
        attrs[attr] = "None"

    # Convert set attrs to lists
    set_attrs = [k for k, v in attrs.items() if isinstance(v, set)]
    for attr in set_attrs:
        attrs[attr] = list(attrs[attr])

    list_attrs = [k for k, v in attrs.items() if isinstance(v, list)]
    for attr in list_attrs:
        if any(not isinstance(i, str) for i in attrs[attr]):
            raise TypeError(
                f"Cannot serialise a sequence of values stored in a model attribute unless all values are strings, found: {attrs[attr]}"
            )
    else:
        attrs["serialised_sets"] = set_attrs
        # Also keep track of single element sets
        attrs["serialised_single_element_list"].extend(
            [k for k in set_attrs if len(attrs[k]) == 1]
        )


def _deserialise(attrs: dict) -> None:
    """Convert troublesome datatypes in xarray attribute dictionaries from their stored data type to the data types expected by Calliope.

    This will tackle dictionaries (from string), booleans (from int), None (from string), and sets (from list).

    Args:
        attrs (dict):
            Attribute dictionary from an xarray Dataset/DataArray.
            Changes will be made in-place, so be sure to supply a copy of your dictionary if you want access to its original state.
    """
    for attr in _pop_serialised_list(attrs, "serialised_dicts"):
        attrs[attr] = read_rich_yaml(attrs[attr])
    for attr in _pop_serialised_list(attrs, "serialised_bools"):
        attrs[attr] = bool(attrs[attr])
    for attr in _pop_serialised_list(attrs, "serialised_nones"):
        attrs[attr] = None
    for attr in _pop_serialised_list(attrs, "serialised_single_element_list"):
        attrs[attr] = listify(attrs[attr])
    for attr in _pop_serialised_list(attrs, "serialised_sets"):
        attrs[attr] = set(attrs[attr])


def save_netcdf(
    model_data: xr.Dataset,
    group_name: str,
    mode: Literal["a", "w"],
    path: str | Path,
    **kwargs,
):
    """Save the model to a netCDF file."""
    original_model_data_attrs = deepcopy(model_data.attrs)
    for key, value in kwargs.items():
        model_data.attrs[key] = value

    _serialise(model_data.attrs)
    for var in model_data.data_vars.values():
        _serialise(var.attrs)
    for var in model_data.coords.values():
        _serialise(var.attrs)

    encoding: dict[str, dict] = {
        k: (
            {"zlib": False, "_FillValue": None}
            if v.dtype.kind in ["U", "O"]
            else {"zlib": False, "_FillValue": None, "dtype": "int8"}
            if v.dtype.kind == "b"
            else {"zlib": True, "complevel": 4}
        )
        for k, v in model_data.data_vars.items()
    }

    try:
        model_data.to_netcdf(
            path,
            mode=mode,
            format="netCDF4",
            engine="netcdf4",
            encoding=encoding,
            group=group_name,
        )
        model_data.close()
    finally:  # Revert model_data.attrs back
        model_data.attrs = original_model_data_attrs
        for var in model_data.data_vars.values():
            _deserialise(var.attrs)
        for var in model_data.coords.values():
            _deserialise(var.attrs)


def save_csv(
    model_data: xr.Dataset,
    group_name: Literal["inputs", "results"],
    path: str | Path,
    dropna: bool = True,
    allow_overwrite: bool = False,
):
    """Save results to CSV.

    One file per dataset array will be generated, with the filename matching the array name.

    If termination condition was not optimal, filters inputs only, and warns that results will not be saved.

    Args:
        model_data (xr.Dataset): Calliope model data.
        group_name (Literal["inputs", "results"]): which of input or output data the `model_data` represents.
        path (str | Path): Directory to which the CSV files will be saved
        dropna (bool, optional):
            If True, drop all NaN values in the data series before saving to file.
            Defaults to True.
        allow_overwrite (bool, optional):
            If True, allow the option to overwrite the directory contents if it already exists.
            This will overwrite CSV files one at a time, so if the dataset has different arrays to the previous saved models, you will get a mix of old and new files.
            Defaults to False.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=allow_overwrite)

    for var_name, var in model_data.items():
        out_path = path / f"{group_name}_{var_name}.csv"
        if not var.shape:
            series = pd.Series(var.item())
            keep_index = False
        else:
            series = var.to_series()
            keep_index = True
        if dropna:
            series = series.dropna()
        series.to_csv(out_path, header=True, index=keep_index)


def load_config(filename: str):
    """Load model configuration from a file."""
    with importlib.resources.as_file(CONFIG_DIR / filename) as f:
        loaded = read_rich_yaml(f)
    return loaded


def read_rich_yaml(yaml: str | Path, allow_override: bool = False) -> AttrDict:
    """Returns an AttrDict initialised from the given YAML file or string.

    Uses calliope's "flavour" for YAML files.

    Args:
        yaml (str | Path): YAML file path or string.
        allow_override (bool, optional): Allow overrides for already defined keys. Defaults to False.

    Raises:
        ValueError: Import solving requested for non-file input YAML.
    """
    if isinstance(yaml, str) and not os.path.exists(yaml):
        yaml_path = None
        yaml_text = yaml
    else:
        yaml_path = Path(yaml)
        yaml_text = yaml_path.read_text(encoding="utf-8")

    yaml_dict = AttrDict(_yaml_load(yaml_text))
    yaml_dict = _resolve_yaml_imports(
        yaml_dict, base_path=yaml_path, allow_override=allow_override
    )
    return yaml_dict


def _yaml_load(src: str):
    """Load YAML from a file object or path with useful parser errors."""
    yaml = ruamel_yaml.YAML(typ="safe")
    src_name = "<yaml string>"
    try:
        result = yaml.load(src)
        if not isinstance(result, dict):
            raise ValueError(f"Could not parse {src_name} as YAML")
        return result
    except ruamel_yaml.YAMLError:
        logger.error(f"Parser error when reading YAML from {src_name}.")
        raise


def _resolve_yaml_imports(
    loaded: AttrDict, base_path: str | Path | None, allow_override: bool
) -> AttrDict:
    loaded_dict = loaded
    imports = loaded_dict.get_key("import", None)
    if imports:
        if not isinstance(imports, list):
            raise ValueError("`import` must be a list.")
        if base_path is None:
            raise ValueError("Imports are not possible for non-file yaml inputs.")

        for k in imports:
            path = relative_path(base_path, k)
            imported = read_rich_yaml(path)
            # loaded is added to imported (i.e. it takes precedence)
            imported.union(loaded_dict, allow_override=allow_override)
            loaded_dict = imported
        # 'import' key itself is no longer needed
        loaded_dict.del_key("import")

    return loaded_dict


def to_yaml(data: AttrDict | dict, path: None | str | Path = None) -> str:
    """Conversion to YAML.

    Saves the AttrDict to the ``path`` as a YAML file or returns a YAML string
    if ``path`` is None.
    """
    result = AttrDict(data).copy()
    # Prepare YAML parsing settings
    yaml_ = ruamel_yaml.YAML()
    yaml_.indent = YAML_INDENT
    yaml_.block_seq_indent = YAML_BLOCK_SEQUENCE_INDENT
    # Keep dictionary order
    yaml_.sort_base_mapping_type_on_output = False  # type: ignore[assignment]

    # Numpy objects should be converted to regular Python objects,
    # so that they are properly displayed in the resulting YAML output
    for k in result.keys_nested():
        # Convert numpy numbers to regular python ones
        v = result.get_key(k)
        if isinstance(v, np.floating):
            result.set_key(k, float(v))
        elif isinstance(v, np.integer):
            result.set_key(k, int(v))
        elif isinstance(v, np.bool_):
            result.set_key(k, bool(v))
        # Lists are turned into seqs so that they are formatted nicely
        elif isinstance(v, list):
            result.set_key(k, yaml_.seq(v))
        elif isinstance(v, Path):
            result.set_key(k, str(v))

    result = result.as_dict()

    if path is not None:
        with open(path, "w") as f:
            yaml_.dump(result, f)

    stream = StringIO()
    yaml_.dump(result, stream)
    return stream.getvalue()
