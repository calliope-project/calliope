# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).

"""
model.py
~~~~~~~~

Implements the core Model class.

"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Callable, Literal, Optional, TypeVar, Union

import xarray

import calliope
from calliope import exceptions
from calliope.backend import backends, latex_backend, parsing
from calliope.core import io
from calliope.core.attrdict import AttrDict
from calliope.core.util.logging import log_time
from calliope.core.util.tools import copy_docstring, relative_path
from calliope.postprocess import results as postprocess_results
from calliope.preprocess import model_run_from_dict, model_run_from_yaml
from calliope.preprocess.model_data import ModelDataFactory

logger = logging.getLogger(__name__)

T = TypeVar(
    "T", bound=Union[backends.PyomoBackendModel, latex_backend.LatexBackendModel]
)


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

    _BACKENDS: dict[str, Callable] = {
        "pyomo": backends.PyomoBackendModel,
    }

    def __init__(
        self,
        config: Optional[Union[str, dict]],
        model_data: Optional[xarray.Dataset] = None,
        debug: bool = False,
        *args,
        **kwargs,
    ):
        """
        Returns a new Model from either the path to a YAML model
        configuration file or a dict fully specifying the model.

        Args:
            config (Optional[Union[str, dict]]):
                If str, must be the path to a model configuration file.
                If dict or AttrDict, must fully specify the model.
            model_data (Optional[xarray.Dataset], optional):
                Create a Model instance from a fully built model_data Dataset.
                This is only used if `config` is explicitly set to None and is primarily used to re-create a Model instance from a model previously saved to a NetCDF file.
                Defaults to None.
            debug (bool, optional):
                If True, additional debug data will be included in the built model.
                Defaults to False.
        Keyword Args:
            timeseries_dataframes (dict[str, pd.DataFrame]):
                If supplying `config` as a dictionary, in-memory timeseries data can be referred to using `df=...`.
                The referenced data must be supplied here as a dicitionary of dataframes.
            scenario (str):
                Comma delimited string of pre-defined `scenarios` to apply to the model,
            override_dict (dict):
                Additional overrides to apply to `config`.
                These will be applied *after* applying any defined `scenario` overrides.
        Raises:
            ValueError: `config` must be provided (as one of `str`, `int`, `None`).
        """
        self._timings: dict = {}
        self.defaults: AttrDict
        self.model_config: AttrDict
        self.run_config: AttrDict
        self.math: AttrDict
        self._config_path: Optional[str]
        self.math_documentation = latex_backend.MathDocumentation(self._build)

        # try to set logging output format assuming python interactive. Will
        # use CLI logging format if model called from CLI
        log_time(logger, self._timings, "model_creation", comment="Model: initialising")
        if isinstance(config, str):
            self._config_path = config
            model_run, debug_data = model_run_from_yaml(config, *args, **kwargs)
            self._init_from_model_run(model_run, debug_data, debug)
        elif isinstance(config, dict):
            self._config_path = None
            model_run, debug_data = model_run_from_dict(config, *args, **kwargs)
            self._init_from_model_run(model_run, debug_data, debug)
        elif model_data is not None and config is None:
            self._init_from_model_data(model_data)
        else:
            # expected input is a string pointing to a YAML file of the run
            # configuration or a dict/AttrDict in which the run and model
            # configurations are defined
            raise ValueError(
                "Input configuration must either be a string or a dictionary."
            )

    def _init_from_model_run(
        self, model_run: calliope.AttrDict, debug_data: calliope.AttrDict, debug: bool
    ) -> None:
        """Initialise the model using a `model_run` dictionary, which may have been loaded from YAML.

        Args:
            model_run (calliope.AttrDict): Preprocessed model configuration.
            debug_data (calliope.AttrDict): Additional data from processing the input configuration.
            debug (bool): If True, `debug_data` will be attached to the Model object as the attribute `calliope.Model._debug_data`.
        """
        self._model_run = model_run
        log_time(
            logger,
            self._timings,
            "model_run_creation",
            comment="Model: preprocessing stage 1 (model_run)",
        )

        model_data_factory = ModelDataFactory(model_run)
        (
            model_data_pre_clustering,
            model_data,
            data_pre_time,
            stripped_keys,
        ) = model_data_factory()

        self._model_data_pre_clustering = model_data_pre_clustering
        self._model_data = model_data
        if debug:
            self._debug_data = debug_data
            self._model_data_pre_time = data_pre_time
            self._model_data_stripped_keys = stripped_keys
        log_time(
            logger,
            self._timings,
            "model_data_original_creation",
            comment="Model: preprocessing stage 2 (model_data)",
        )

        # Ensure model and run attributes of _model_data update themselves
        model_config = {
            k: v for k, v in model_run.get("model", {}).items() if k != "file_allowed"
        }

        self._add_observed_dict("model_config", model_config)
        self._add_observed_dict("run_config", model_run["run"])
        self._add_observed_dict("defaults", self._generate_default_dict())

        math = self._add_math(model_config["custom_math"])
        self._add_observed_dict("math", math)

        self.inputs = self._model_data.filter_by_attrs(is_result=0)
        log_time(
            logger,
            self._timings,
            "model_data_creation",
            comment="Model: preprocessing complete",
        )

    def _init_from_model_data(self, model_data: xarray.Dataset) -> None:
        """
        Initialise the model using a pre-built xarray dataset.
        This must be a Calliope-compatible dataset, usually a dataset from another Calliope model.

        Args:
            model_data (xarray.Dataset):
                Model dataset with input parameters as arrays and configuration stored in the dataset attributes dictionary.
        """
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
        """
        1. Filter model dataset to produce views on the input/results data
        2. Add top-level configuration dictionaries simultaneously to the model data attributes and as attributes of this class.

        """
        self.inputs = self._model_data.filter_by_attrs(is_result=0)
        self.results = self._model_data.filter_by_attrs(is_result=1)
        self._add_observed_dict("model_config")
        self._add_observed_dict("run_config")
        self._add_observed_dict("math")

        self.inputs = self._model_data.filter_by_attrs(is_result=0)
        results = self._model_data.filter_by_attrs(is_result=1)
        if len(results.data_vars) > 0:
            self.results = results
        log_time(
            logger,
            self._timings,
            "model_data_loaded",
            comment="Model: loaded model_data",
        )

    def _add_observed_dict(self, name: str, dict_to_add: Optional[dict] = None) -> None:
        """
        Add the same dictionary as property of model object and an attribute of the model xarray dataset.

        Args:
            name (str):
                Name of dictionary which will be set as the model property name and (if necessary) the dataset attribute name.
            dict_to_add (Optional[dict], optional):
                If given, set as both the model property and the dataset attribute, otherwise set an existing dataset attribute as a model property of the same name. Defaults to None.

        Raises:
            exceptions.ModelError: If `dict_to_add` is not given, it must be an attribute of model data.
            TypeError: `dict_to_add` must be a dictionary.
        """
        if dict_to_add is None:
            try:
                dict_to_add = self._model_data.attrs[name]
            except KeyError:
                raise exceptions.ModelError(
                    f"Expected the model property `{name}` to be a dictionary attribute of the model dataset. If you are loading the model from a NetCDF file, ensure it is a valid Calliope model."
                )
        if not isinstance(dict_to_add, dict):
            raise TypeError(
                f"Attempted to add dictionary property `{name}` to model, but received argument of type `{type(dict_to_add).__name__}`"
            )
        else:
            dict_to_add = AttrDict(dict_to_add)
        self._model_data.attrs[name] = dict_to_add
        setattr(self, name, dict_to_add)

    def _add_math(self, custom_math: list) -> AttrDict:
        """
        Load the base math and optionally override with custom math from a list of references to custom math files.

        Args:
            custom_math (list):
                List of references to files containting custom mathematical formulations that will be merged with the base formulation.

        Raises:
            exceptions.ModelError:
                Referenced internal custom math files or user-defined custom math files must exist.

        Returns:
            AttrDict: Dictionary of math (constraints, variables, objectives, and global expressions).
        """
        math_dir = Path(calliope.__file__).parent / "math"
        base_math = AttrDict.from_yaml(math_dir / "base.yaml")

        file_errors = []

        for filename in custom_math:
            if not f"{filename}".endswith((".yaml", ".yml")):
                yaml_filepath = math_dir / f"{filename}.yaml"
            else:
                yaml_filepath = Path(relative_path(self._config_path, filename))

            if not yaml_filepath.is_file():
                file_errors.append(filename)
                continue
            else:
                override_dict = AttrDict.from_yaml(yaml_filepath)

            base_math.union(override_dict, allow_override=True)
        if file_errors:
            raise exceptions.ModelError(
                f"Attempted to load custom math that does not exist: {file_errors}"
            )
        return base_math

    def _generate_default_dict(self) -> AttrDict:
        """Process input parameter default YAML configuration file into a dictionary of
        defaults that match parameter names in the processed model dataset
        (e.g., costs are prepended with `cost_`).

        Returns:
            AttrDict: Flat dictionary of `parameter_name`:`parameter_default` pairs.
        """
        raw_defaults = AttrDict.from_yaml(
            Path(calliope.__file__).parent / "config" / "defaults.yaml"
        )
        default_tech_dict = raw_defaults.techs.default_tech
        default_cost_dict = {
            "cost_{}".format(k): v
            for k, v in default_tech_dict.costs.default_cost.items()
        }
        default_node_dict = {
            "available_area": raw_defaults.nodes.default_node.available_area
        }

        return AttrDict(
            {
                **default_tech_dict.constraints.as_dict(),
                **default_tech_dict.switches.as_dict(),
                **default_cost_dict,
                **default_node_dict,
            }
        )

    def _add_run_mode_custom_math(self) -> None:
        """If not given in the custom_math list, override model math with run mode math"""
        run_mode = self.run_config["mode"]
        # FIXME: available modes should not be hardcoded here.
        # They should come from a YAML schema.
        not_run_mode = {"plan", "operate", "spores"}.difference([run_mode])
        run_mode_mismatch = not_run_mode.intersection(self.model_config["custom_math"])
        if run_mode_mismatch:
            exceptions.warn(
                f"Running in {run_mode} mode, but run mode(s) {run_mode_mismatch} custom "
                "math being loaded from file via the model configuration"
            )

        if run_mode != "plan" and run_mode not in self.model_config["custom_math"]:
            filepath = Path(calliope.__file__).parent / "math" / f"{run_mode}.yaml"
            self.math.union(AttrDict.from_yaml(filepath), allow_override=True)

    def build(
        self, force: bool = False, backend_interface: Literal["pyomo"] = "pyomo"
    ) -> None:
        """Build description of the optimisation problem in the chosen backend interface.

        Args:
            force (bool, optional):
                If ``force`` is True, any existing results will be overwritten.
                Defaults to False.
            backend_interface (Literal["pyomo"], optional):
                Backend interface in which to build the problem. Defaults to "pyomo".
        """

        if hasattr(self, "backend") and not force:
            raise exceptions.ModelError(
                "This model object already has a built optimisation problem. Use model.build(force=True) "
                "to force the existing optimisation problem to be overwritten with a new one."
            )
        backend = self._BACKENDS[backend_interface]()
        self.backend = self._build(backend)

    def _build(self, backend: T) -> T:
        backend.add_all_parameters(self._model_data, self.run_config)
        log_time(
            logger,
            self._timings,
            "backend_parameters_generated",
            comment="Model: Generated optimisation problem parameters",
        )
        self._add_run_mode_custom_math()
        # The order of adding components matters!
        # 1. Variables, 2. Global Expressions, 3. Constraints, 4. Objectives
        for components in [
            "variables",
            "global_expressions",
            "constraints",
            "objectives",
        ]:
            component = components.removesuffix("s")
            if components in ["variables", "expressions"]:
                backend.valid_math_element_names.update(self.math[components].keys())
            for name, dict_ in self.math[components].items():
                if dict_.get("active", True):
                    getattr(backend, f"add_{component}")(self._model_data, name, dict_)
                else:
                    logger.debug(
                        f"({component}, {name}): Component deactivated and therefore not built."
                    )
            log_time(
                logger,
                self._timings,
                f"backend_{components}_generated",
                comment=f"Model: Generated optimisation problem {components}",
            )
        return backend

    @copy_docstring(backends.BackendModel.verbose_strings)
    def verbose_strings(self) -> None:
        if not hasattr(self, "backend"):
            raise NotImplementedError(
                "Call `build()` to generate an optimisation problem before calling this function."
            )
        self.backend.verbose_strings()

    def solve(self, force: bool = False, warmstart: bool = False) -> None:
        """
        Run the built optimisation problem.

        Args:
            force (bool, optional):
                If ``force`` is True, any existing results will be overwritten.
                Defaults to False.
            warmstart (bool, optional):
                If True and the optimisation problem has already been run in this session
                (i.e., `force` is not True), the next optimisation will be run with
                decision variables initially set to their previously optimal values.
                If the optimisation problem is similar to the previous run, this can
                decrease the solution time.
                Warmstart will not work with some solvers (e.g., CBC, GLPK).
                Defaults to False.

        Raises:
            exceptions.ModelError: Optimisation problem must already be built.
            exceptions.ModelError: Cannot run the model if there are already results loaded, unless `force` is True.
            exceptions.ModelError: Some preprocessing steps will stop a run mode of "operate" from being possible.
        """
        run_mode = self.run_config["mode"]

        # Check that results exist and are non-empty
        if not hasattr(self, "backend"):
            raise exceptions.ModelError(
                "You must build the optimisation problem (`.build()`) "
                "before you can run it."
            )

        if hasattr(self, "results"):
            if self.results.data_vars and not force:
                raise exceptions.ModelError(
                    "This model object already has results. "
                    "Use model.solve(force=True) to force"
                    "the results to be overwritten with a new run."
                )
            else:
                to_drop = self.results.data_vars
        else:
            to_drop = []

        if run_mode == "operate" and not self._model_data.attrs["allow_operate_mode"]:
            raise exceptions.ModelError(
                "Unable to run this model in operational mode, probably because "
                "there exist non-uniform timesteps (e.g. from time masking)"
            )

        log_time(
            logger,
            self._timings,
            "solve_start",
            comment=f"Backend: starting model solve in {run_mode} mode",
        )

        termination_condition = self.backend.solve(
            solver=self.run_config["solver"],
            solver_io=self.run_config.get("solver_io", None),
            solver_options=self.run_config.get("solver_options", None),
            save_logs=self.run_config.get("save_logs", None),
            warmstart=warmstart,
        )

        log_time(
            logger,
            self._timings,
            "solver_exit",
            time_since_solve_start=True,
            comment="Backend: solver finished running",
        )

        # Add additional post-processed result variables to results
        if termination_condition in ["optimal", "feasible"]:
            results = self.backend.load_results()
            results = postprocess_results.postprocess_model_results(
                results, self._model_data, self._timings
            )
        else:
            results = xarray.Dataset()

        self._model_data = self._model_data.drop_vars(to_drop)

        self._model_data.attrs.update(results.attrs)
        self._model_data.attrs["termination_condition"] = termination_condition
        self._model_data = xarray.merge(
            [results, self._model_data], compat="override", combine_attrs="no_conflicts"
        )
        self._add_model_data_methods()

    def run(self, force_rerun=False, **kwargs):
        """
        Run the model. If ``force_rerun`` is True, any existing results
        will be overwritten.

        Additional kwargs are passed to the backend.

        """
        warnings.warn(
            "`run()` is deprecated and will be removed in a "
            "future version. Use `model.build()` followed by `model.solve()`.",
            DeprecationWarning,
        )
        self.build(force=force_rerun)
        self.solve(force=force_rerun)

    def get_formatted_array(self, var):
        """
        Return an xarray.DataArray with nodes, techs, and carriers as
        separate dimensions.

        Parameters
        ----------
        var : str
            Decision variable for which to return a DataArray.

        """
        warnings.warn(
            "get_formatted_array() is deprecated and will be removed in a "
            "future version. Use `model.results[var]` instead.",
            DeprecationWarning,
        )
        if var not in self._model_data.data_vars:
            raise KeyError("Variable {} not in Model data".format(var))

        return self._model_data[var]

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

    @copy_docstring(backends.BackendModel.to_lp)
    def to_lp(self, path: Union[str, Path]) -> None:
        """
        Raises:
            exceptions.ModelError: This method cannot be called prior to calling `build()`.
        """

        if not hasattr(self, "backend"):
            raise exceptions.ModelError(
                "Build the optimisation problem by calling `build()` before trying to generate an LP file."
            )
        self.backend.to_lp(path)

    def info(self) -> str:
        """Generate basic description of the model, combining its name and a rough indication of the model size.

        Returns:
            str: Basic description of the model.
        """
        info_strings = []
        model_name = self.model_config.get("name", "None")
        info_strings.append(f"Model name:   {model_name}")
        msize = dict(self._model_data.dims)
        msize_exists = (self._model_data.node_tech * self._model_data.carrier).sum()
        info_strings.append(
            f"Model size:   {msize} ({msize_exists.item()} valid node:tech:carrier:carrier_tier combinations)"
        )
        return "\n".join(info_strings)

    def validate_math_strings(self, math_dict: dict) -> None:
        """Validate that `expression` and `where` strings of a dictionary containing string mathematical formulations can be successfully parsed. This function can be used to test custom math before attempting to build the optimisation problem.

        NOTE: strings are not checked for evaluation validity. Evaluation issues will be raised only on calling `Model.build()`.

        Args:
            math_dict (dict): Math formulation dictionary to validate. Top level keys must be one or more of ["variables", "global_expressions", "constraints", "objectives"], e.g.:
            {
                "constraints": {
                    "my_constraint_name":
                        {
                            "foreach": ["nodes"],
                            "where": "inheritance(supply)",
                            "equation": "sum(energy_cap, over=techs) >= 10"
                        }
                }

            }
        Returns:
            If all components of the dictionary are parsed successfully, this function will log a success message to the INFO logging level and return None.
            Otherwise, a calliope.ModelError will be raised with parsing issues listed.
        """
        valid_math_element_names = [
            *self.math["variables"].keys(),
            *self.math["global_expressions"].keys(),
            *math_dict.get("variables", {}).keys(),
            *math_dict.get("global_expressions", {}).keys(),
            *self.inputs.data_vars.keys(),
            *self.defaults.keys(),
            # FIXME: these should not be hardcoded, but rather end up in model data keys
            "bigM",
            *["objective_" + k for k in self.run_config["objective_options"].keys()],
        ]
        collected_errors: dict = dict()
        for component_group, component_dicts in math_dict.items():
            for name, component_dict in component_dicts.items():
                parsed = parsing.ParsedBackendComponent(
                    component_group, name, component_dict
                )
                parsed.parse_top_level_where(errors="ignore")
                parsed.parse_equations(set(valid_math_element_names), errors="ignore")
                if not parsed._is_valid:
                    collected_errors[f"({component_group}, {name})"] = parsed._errors

        if collected_errors:
            exceptions.print_warnings_and_raise_errors(
                during="math string parsing (marker indicates where parsing stopped, not strictly the equation term that caused the failure)",
                errors=collected_errors,
            )

        logger.info("Model: validated math strings")
