"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

model.py
~~~~~~~~

Implements the core Model class.

"""

from io import StringIO
import logging
import warnings

import numpy as np
import xarray as xr
import ruamel.yaml as ruamel_yaml

from calliope.postprocess import results as postprocess_results
from calliope.postprocess import plotting
from calliope.core import io
from calliope.preprocess import (
    model_run_from_yaml,
    model_run_from_dict,
    build_model_data,
    apply_time_clustering,
    final_timedimension_processing,
)
from calliope.core.attrdict import AttrDict
from calliope.core.util.logging import log_time
from calliope.core.util.dataset import split_loc_techs
from calliope.core.util.tools import apply_to_dict
from calliope.core.util.observed_dict import UpdateObserverDict
from calliope import exceptions
from calliope.backend.run import run as run_backend

logger = logging.getLogger(__name__)


def read_netcdf(path):
    """
    Return a Model object reconstructed from model data in a NetCDF file.

    """
    model_data = io.read_netcdf(path)
    return Model(config=None, model_data=model_data)


class Model(object):
    """
    A Calliope Model.

    """

    def __init__(self, config, model_data=None, *args, **kwargs):
        """
        Returns a new Model from either the path to a YAML model
        configuration file or a dict fully specifying the model.

        Parameters
        ----------
        config : str or dict or AttrDict
            If str, must be the path to a model configuration file.
            If dict or AttrDict, must fully specify the model.
        model_data : Dataset, optional
            Create a Model instance from a fully built model_data Dataset.
            This is only used if `config` is explicitly set to None
            and is primarily used to re-create a Model instance from
            a model previously saved to a NetCDF file.

        """
        self._timings = {}
        # try to set logging output format assuming python interactive. Will
        # use CLI logging format if model called from CLI
        log_time(logger, self._timings, "model_creation", comment="Model: initialising")
        if isinstance(config, str):
            model_run, debug_data = model_run_from_yaml(config, *args, **kwargs)
            self._init_from_model_run(model_run, debug_data)
        elif isinstance(config, dict):
            model_run, debug_data = model_run_from_dict(config, *args, **kwargs)
            self._init_from_model_run(model_run, debug_data)
        elif model_data is not None and config is None:
            self._init_from_model_data(model_data)
        else:
            # expected input is a string pointing to a YAML file of the run
            # configuration or a dict/AttrDict in which the run and model
            # configurations are defined
            raise ValueError(
                "Input configuration must either be a string or a dictionary."
            )
        self._check_future_deprecation_warnings()

        self.plot = plotting.ModelPlotMethods(self)

    def _init_from_model_run(self, model_run, debug_data):
        self._model_run = model_run
        self._debug_data = debug_data
        log_time(
            logger,
            self._timings,
            "model_run_creation",
            comment="Model: preprocessing stage 1 (model_run)",
        )

        self._model_data_original = build_model_data(model_run)
        log_time(
            logger,
            self._timings,
            "model_data_original_creation",
            comment="Model: preprocessing stage 2 (model_data)",
        )

        random_seed = self._model_run.get_key("model.random_seed", None)
        if random_seed:
            np.random.seed(seed=random_seed)

        # After setting the random seed, time clustering can take place
        time_config = model_run.model.get("time", None)
        if not time_config:
            _model_data = self._model_data_original
        else:
            _model_data = apply_time_clustering(self._model_data_original, model_run)
            log_time(
                logger,
                self._timings,
                "model_data_clustered",
                comment="Model: time resampling/clustering complete",
            )
        self._model_data = final_timedimension_processing(_model_data)
        log_time(
            logger,
            self._timings,
            "model_data_creation",
            comment="Model: preprocessing complete",
        )

        # Ensure model and run attributes of _model_data update themselves
        for var in self._model_data.data_vars:
            self._model_data[var].attrs["is_result"] = 0
        self.inputs = self._model_data.filter_by_attrs(is_result=0)

        model_config = {
            k: v for k, v in model_run.get("model", {}).items() if k != "file_allowed"
        }
        self.model_config = UpdateObserverDict(
            initial_dict=model_config, name="model_config", observer=self._model_data
        )
        self.run_config = UpdateObserverDict(
            initial_dict=model_run.get("run", {}),
            name="run_config",
            observer=self._model_data,
        )

    def _init_from_model_data(self, model_data):
        if "_model_run" in model_data.attrs:
            self._model_run = AttrDict.from_yaml_string(model_data.attrs["_model_run"])
            del model_data.attrs["_model_run"]

        if "_debug_data" in model_data.attrs:
            self._debug_data = AttrDict.from_yaml_string(
                model_data.attrs["_debug_data"]
            )
            del model_data.attrs["_debug_data"]

        self._model_data = model_data
        self._add_model_data_methods()

        log_time(
            logger,
            self._timings,
            "model_data_loaded",
            comment="Model: loaded model_data",
        )

    def _add_model_data_methods(self):
        self.inputs = self._model_data.filter_by_attrs(is_result=0)
        self.results = self._model_data.filter_by_attrs(is_result=1)
        self.model_config = UpdateObserverDict(
            initial_yaml_string=self._model_data.attrs.get("model_config", "{}"),
            name="model_config",
            observer=self._model_data,
        )
        self.run_config = UpdateObserverDict(
            initial_yaml_string=self._model_data.attrs.get("run_config", "{}"),
            name="run_config",
            observer=self._model_data,
        )

    def save_commented_model_yaml(self, path):
        """
        Save a fully built and commented version of the model to a YAML file
        at the given ``path``. Comments in the file indicate where values
        were overridden. This is Calliope's internal representation of
        a model directly before the model_data xarray.Dataset is built,
        and can be useful for debugging possible issues in the model
        formulation.

        """
        if not self._model_run or not self._debug_data:
            raise KeyError(
                "This model does not have the fully built model attached, "
                "so `save_commented_model_yaml` is not available. Likely "
                "reason is that the model was built with a verion of Calliope "
                "prior to 0.6.5."
            )

        yaml = ruamel_yaml.YAML()

        model_run_debug = self._model_run.copy()
        try:
            del model_run_debug["timeseries_data"]  # Can't be serialised!
        except KeyError:
            # Possible that timeseries_data is already gone if the model
            # was read from a NetCDF file
            pass

        # Turn sets in model_run into lists for YAML serialization
        for k, v in model_run_debug.sets.items():
            model_run_debug.sets[k] = list(v)

        debug_comments = self._debug_data["comments"]

        stream = StringIO()
        yaml.dump(model_run_debug.as_dict(), stream=stream)
        debug_yaml = yaml.load(stream.getvalue())

        for k in debug_comments.model_run.keys_nested():
            v = debug_comments.model_run.get_key(k)
            if v:
                keys = k.split(".")
                apply_to_dict(
                    debug_yaml, keys[:-1], "yaml_add_eol_comment", (v, keys[-1])
                )

        dumper = ruamel_yaml.dumper.RoundTripDumper
        dumper.ignore_aliases = lambda self, data: True

        with open(path, "w") as f:
            ruamel_yaml.dump(
                debug_yaml, stream=f, Dumper=dumper, default_flow_style=False
            )

    def run(self, force_rerun=False, **kwargs):
        """
        Run the model. If ``force_rerun`` is True, any existing results
        will be overwritten.

        Additional kwargs are passed to the backend.

        """
        # Check that results exist and are non-empty
        if hasattr(self, "results") and self.results.data_vars and not force_rerun:
            raise exceptions.ModelError(
                "This model object already has results. "
                "Use model.run(force_rerun=True) to force"
                "the results to be overwritten with a new run."
            )

        if (
            self.run_config["mode"] == "operate"
            and not self._model_data.attrs["allow_operate_mode"]
        ):
            raise exceptions.ModelError(
                "Unable to run this model in operational mode, probably because "
                "there exist non-uniform timesteps (e.g. from time masking)"
            )

        results, self._backend_model, self._backend_model_opt, interface = run_backend(
            self._model_data, self._timings, **kwargs
        )

        # Add additional post-processed result variables to results
        if results.attrs.get("termination_condition", None) in ["optimal", "feasible"]:
            results = postprocess_results.postprocess_model_results(
                results, self._model_data, self._timings
            )
        self._model_data.attrs.update(results.attrs)
        self._model_data = xr.merge(
            [results, self._model_data], compat="override", combine_attrs="no_conflicts"
        )
        self._add_model_data_methods()

        self.backend = interface(self)

    def get_formatted_array(self, var, index_format="index"):
        """
        Return an xr.DataArray with locs, techs, and carriers as
        separate dimensions.

        Parameters
        ----------
        var : str
            Decision variable for which to return a DataArray.
        index_format : str, default = 'index'
            'index' to return the `loc_tech(_carrier)` dimensions as individual
            indexes, 'multiindex' to return them as a MultiIndex. The latter
            has the benefit of having a smaller memory footprint, but you cannot
            undertake dimension specific operations (e.g. formatted_array.sum('locs'))
        """

        if var not in self._model_data.data_vars:
            raise KeyError("Variable {} not in Model data".format(var))

        if index_format not in ["index", "multiindex"]:
            raise ValueError(
                "Argument 'index_format' must be one of 'index' or 'multiindex'"
            )
        elif index_format == "index":
            return_as = "DataArray"
        elif index_format == "multiindex":
            return_as = "MultiIndex DataArray"

        return split_loc_techs(self._model_data[var], return_as=return_as)

    def to_netcdf(self, path):
        """
        Save complete model data (inputs and, if available, results)
        to a NetCDF file at the given ``path``.

        """
        io.save_netcdf(self._model_data, path, model=self)

    def to_csv(self, path, dropna=True):
        """
        Save complete model data (inputs and, if available, results)
        as a set of CSV files to the given ``path``.

        Parameters
        ----------
        dropna : bool, optional
            If True (default), NaN values are dropped when saving,
            resulting in significantly smaller CSV files.

        """
        io.save_csv(self._model_data, path, dropna)

    def to_lp(self, path):
        """
        Save built model to LP format at the given ``path``. If the backend
        model has not been built yet, it is built prior to saving.
        """
        io.save_lp(self, path)

    def info(self):
        info_strings = []
        model_name = self.model_config.get("name", "None")
        info_strings.append("Model name:   {}".format(model_name))
        msize = "{locs} locations, {techs} technologies, {times} timesteps".format(
            locs=len(self._model_data.coords.get("locs", [])),
            techs=(
                len(self._model_data.coords.get("techs_non_transmission", []))
                + len(self._model_data.coords.get("techs_transmission_names", []))
            ),
            times=len(self._model_data.coords.get("timesteps", [])),
        )
        info_strings.append("Model size:   {}".format(msize))
        return "\n".join(info_strings)

    def _check_future_deprecation_warnings(self):
        """
        Method for all FutureWarnings and DeprecationWarnings. Comment above each
        warning should specify Calliope version in which it was added, and the
        version in which it should be updated/removed.
        """

        # Warning that group_share constraints will removed in 0.7.0 #
        # Added in 0.6.4-dev, to be removed in v0.7.0-dev
        if any("group_share_" in i for i in self._model_data.data_vars.keys()):
            warnings.warn(
                "`group_share` constraints will be removed in v0.7.0 -- "
                "use the new model-wide constraints instead.",
                FutureWarning,
            )

        # Warning that charge rate will be removed in 0.7.0
        # Added in 0.6.4-dev, to be removed in 0.7.0-dev
        # Rename charge rate to energy_cap_per_storage_cap_max
        if self._model_data is not None and "charge_rate" in self._model_data:
            warnings.warn(
                "`charge_rate` is renamed to `energy_cap_per_storage_cap_max` "
                "and will be removed in v0.7.0.",
                FutureWarning,
            )
