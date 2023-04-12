from __future__ import annotations

import re
from abc import ABC, abstractmethod
import typing
from typing import (
    Any,
    Callable,
    Optional,
    Literal,
    TypeVar,
    Generic,
    Union,
    Iterator,
    Iterable,
)

import os
from contextlib import redirect_stdout, redirect_stderr, contextmanager
import logging

import xarray as xr
import pandas as pd
import pyomo.environ as pe
import pyomo.kernel as pmo
from pyomo.opt import SolverFactory
from pyomo.common.tempfiles import TempfileManager
import numpy as np

from calliope.exceptions import BackendError, BackendWarning
from calliope.exceptions import warn as model_warn
from calliope.core.util.logging import LogWriter
from calliope.backend import parsing, equation_parser


T = TypeVar("T")
_COMPONENTS_T = Literal[
    "variables", "constraints", "objectives", "parameters", "expressions"
]

logger = logging.getLogger(__name__)


class BackendModel(ABC, Generic[T]):
    _VALID_COMPONENTS: tuple[_COMPONENTS_T, ...] = typing.get_args(_COMPONENTS_T)

    def __init__(self, instance: T):
        """Abstract base class for interfaces to solvers.

        Args:
            instance (T): Interface model instance.
        """

        self._instance = instance
        self._dataset = xr.Dataset()
        self.valid_math_element_names: set = set()

    @abstractmethod
    def add_parameter(
        self,
        parameter_name: str,
        parameter_values: xr.DataArray,
        default: Any = np.nan,
        use_inf_as_na: bool = False,
    ) -> None:
        """
        Add input parameter to backend model in-place.
        If the backend interface allows for mutable parameter objects, they will be
        generated, otherwise a copy of the model input dataset will be used.
        In either case, NaN values are filled with the given parameter default value.

        Args:
            parameter_name (str): Name of parameter.
            parameter_values (xr.DataArray): Array of parameter values.
            default (Any, optional):
                Default value to fill NaN entries in parameter values array.
                Defaults to np.nan.
            use_inf_as_na (bool, optional):
                If True, will consider np.inf parameter value entries as np.nan and
                consequently try to fill those entries with the parameter default value.
                Defaults to False.
        """

    @abstractmethod
    def add_constraint(
        self,
        model_data: xr.Dataset,
        name: str,
        constraint_dict: parsing.UnparsedConstraintDict,
    ) -> None:
        """
        Add constraint equation to backend model in-place.
        Resulting backend dataset entries will be constraint objects.

        Args:
            model_data (xr.Dataset):
                Calliope model data with which to create an array mask - only those
                dataset entries in the mask will be generated.
            name (str):
                Name of the constraint
            constraint_dict (parsing.UnparsedConstraintDict):
                Constraint configuration dictionary, ready to be parsed and then evaluated.
        """

    @abstractmethod
    def add_expression(
        self,
        model_data: xr.Dataset,
        name: str,
        expression_dict: parsing.UnparsedConstraintDict,
    ) -> None:
        """
        Add expression (arithmetic combination of parameters and/or decision variables)
        to backend model in-place.
        Resulting backend dataset entries will be linear expression objects.

        Args:
            model_data (xr.Dataset):
                Calliope model data with which to create an array mask - only those
                dataset entries in the mask will be generated.
            name (str):
                Name of the expression
            expression_dict (parsing.UnparsedConstraintDict):
                Expression configuration dictionary, ready to be parsed and then evaluated.
        """

    @abstractmethod
    def add_variable(
        self,
        model_data: xr.Dataset,
        name: str,
        variable_dict: parsing.UnparsedVariableDict,
    ) -> None:
        """
        Add decision variable to backend model in-place.
        Resulting backend dataset entries will be decision variable objects.

        Args:
            model_data (xr.Dataset):
                Calliope model data with which to create an array mask - only those
                dataset entries in the mask will be generated.
            name (str):
                Name of the variable.
            variable_dict (parsing.UnparsedVariableDict):
                Variable configuration dictionary, ready to be parsed and then evaluated.
        """

    @abstractmethod
    def add_objective(
        self,
        model_data: xr.Dataset,
        name: str,
        objective_dict: parsing.UnparsedObjectiveDict,
    ) -> None:
        """
        Add objective arithmetic to backend model in-place.
        Resulting backend dataset entry will be a single, unindexed objective object.

        Args:
            model_data (xr.Dataset):
                Calliope model data with which to create a constraint mask - only those
                dataset entries in the mask will be generated.
            name (str):
                Name of the objective.
            objective_dict (parsing.UnparsedObjectiveDict):
                Objective configuration dictionary, ready to be parsed and then evaluated.
        """

    @abstractmethod
    def get_parameter(
        self, parameter_name: str, as_backend_objs: bool = True
    ) -> Optional[xr.DataArray]:
        """
        Extract parameter from backend dataset.

        Args:
            parameter_name (str): Name of parameter.
            TODO: hide this and create a method to edit parameter values
                  (to handle interfaces with non-mutable params)
            as_backend_objs (bool, optional):
                If True, will keep the array entries as backend interface objects,
                which can be updated to update the underlying model.
                Otherwise, parameter values are given directly, with default values in place of NaNs.
                Defaults to True.

        Returns:
            Optional[xr.DataArray]: If parameter is not in backend dataset, will return None.
        """

    @abstractmethod
    def get_constraint(
        self,
        constraint_name: str,
        as_backend_objs: bool = True,
        eval_body: bool = False,
    ) -> Optional[Union[xr.DataArray, xr.Dataset]]:
        """
        Get constraint data as either a table of details or as an array of backend interface objects.
        Can be used to inspect and debug built constraints.

        Args:
            constraint_name (str): Name of constraint, as given in YAML constraint key.
            TODO: hide this and create a method to edit constraints that handles differences in interface APIs.
            as_backend_objs (bool, optional):
                If True, will keep the array entries as backend interface objects,
                which can be updated to change the underlying model.
                Otherwise, constraint body, and lower and upper bounds are given in a table.
                Defaults to True.
            eval_body (bool, optional):
                If True and as_backend_objs is False, will attempt to evaluate the constraint body.
                If the model has been optimised, this attempt will produce a numeric value to see where the constraint sits between the lower or upper bound.
                If the model has not yet been optimised, this attempt will fall back on the same as
                if `eval_body` was set to False, i.e. a string representation of the linear expression in the constraint body.
                Defaults to False.

        Returns:
            Optional[Union[xr.DataArray, xr.Dataset]]:
                If constraint is not in backend dataset, will return None.
                If as_backend_objs is True, will return an xr.DataArray.
                Otherwise, a xr.Dataset will be given, indexed over the same dimensions as the xr.DataArray, with variables for the constraint body, and upper (`ub`) and lower (`lb`) bounds.
        """

    @abstractmethod
    def get_variable(
        self, variable_name: str, as_backend_objs: bool = True
    ) -> Optional[xr.DataArray]:
        """Extract decision variable array from backend dataset

        Args:
            variable_name (str): Name of variable.
            TODO: hide this and create a method to edit variables that handles differences in interface APIs.
            as_backend_objs (bool, optional):
                If True, will keep the array entries as backend interface objects,
                which can be updated to update the underlying model.
                Otherwise, variable values are given directly.
                If the model has not been successfully optimised, variable values will all be None.
                Defaults to True.

        Returns:
            Optional[xr.DataArray]: If decision variable is not in backend dataset, will return None.
        """

    @abstractmethod
    def get_expression(
        self, expression_name: str, as_backend_objs: bool = True, eval_body: bool = True
    ) -> Optional[xr.DataArray]:
        """Exrtact expression array from backend dataset

        Args:
            expression_name (str): Name of expression
            TODO: hide this and create a method to edit expressions that handles differences in interface APIs.
            as_backend_objs (bool, optional):
                If True, will keep the array entries as backend interface objects,
                which can be updated to update the underlying model.
                Otherwise, expression values are given directly.
                If the model has not been successfully optimised, expression values will all be provided as strings.
                Defaults to True.
            eval_body (bool, optional):
                If True and as_backend_objs is False, will attempt to evaluate the expression.
                If the model has been optimised, this attempt will produce a numeric value.
                If the model has not yet been optimised, this attempt will fall back on the same as
                if `eval_body` was set to False, i.e. a string representation of the linear expression.
                Defaults to True.

        Returns:
            Optional[xr.DataArray]: If expression is not in backend dataset, will return None.
        """

    @abstractmethod
    def solve(
        self,
        solver: str,
        solver_io: Optional[str] = None,
        solver_options: Optional[dict] = None,
        save_logs: Optional[str] = None,
        warmstart: bool = False,
        **solve_kwargs,
    ):
        """
        Optimise built model. If solution is optimal, interface objects
        (decision variables, expressions, constraints, objective) can be successfully
        evaluated for their values at optimality.

        Args:
            solver (str): Name of solver to optimise with.
            solver_io (Optional[str], optional):
                If chosen solver has a python interface, set to "python" for potential
                performance gains, otherwise should be left as None. Defaults to None.
            solver_options (Optional[dict], optional):
                Solver options/parameters to pass directly to solver.
                See solver documentation for available parameters that can be influenced.
                Defaults to None.
            save_logs (Optional[str], optional):
                If given, solver logs and built LP file will be saved to this filepath.
                Defaults to None.
            warmstart (bool, optional):
                If True, and the chosen solver is capable of implementing it, an existing
                optimal solution will be used to warmstart the next solve run.
                Defaults to False.
        """

    def load_results(self) -> xr.Dataset:
        """
        Evaluate backend decision variables, expressions, and parameters (if not in inputs)
        after a successful model run.

        Returns:
            xr.Dataset: Dataset of optimal solution results (all numeric data).
        """
        all_variables = {
            name_: self.get_variable(name_, as_backend_objs=False)
            for name_, var in self.variables.items()
            if var.notnull().any()
        }
        all_expressions = {
            name_: self.get_expression(name_, as_backend_objs=False, eval_body=True)
            for name_, expr in self.expressions.items()
            if expr.notnull().any()
        }

        results = xr.Dataset({**all_variables, **all_expressions})

        return results

    def add_all_parameters(self, model_data: xr.Dataset, run_config: dict) -> None:
        """
        Add all parameters to backend dataset in-place, including those in the run configuration.
        If model data does not include a parameter, their default values will be added here
        as unindexed backend dataset parameters.

        TODO: Move the decision on which run config params to generate as backend params
              earlier in the process.
        Parameters in "objective_options" and the bigM parameter will be added from
        run configuration.

        Args:
            model_data (xr.Dataset): Input model data.
            defaults (dict): Parameter defaults.
            run_config (UpdateObserverDict): Run configuration dictionary.
        """

        for param_name, param_data in model_data.filter_by_attrs(
            is_result=0
        ).data_vars.items():
            default_val = model_data.attrs["defaults"].get(param_name, np.nan)
            self.add_parameter(param_name, param_data, default_val)
        for param_name, default_val in model_data.attrs["defaults"].items():
            if param_name in self.parameters.keys():
                continue
            self.add_parameter(
                param_name, xr.DataArray(default_val), use_inf_as_na=False
            )

        for option_name, option_val in run_config["objective_options"].items():
            if option_name == "cost_class":
                objective_cost_class = {
                    k: v for k, v in option_val.items() if k in model_data.costs
                }
                self.add_parameter(
                    "objective_cost_class",
                    xr.DataArray.from_series(
                        pd.Series(objective_cost_class).rename_axis(index="costs")
                    ),
                )
            else:
                self.add_parameter("objective_" + option_name, xr.DataArray(option_val))
        self.add_parameter("bigM", xr.DataArray(run_config.get("bigM", 1e10)))

    def apply_func(
        self, func: Callable, *args, output_core_dims: tuple = ((),), **kwargs
    ) -> xr.DataArray:
        """
        Apply a function to every element of an arbitrary number of xarray DataArrays.

        Args:
            func (Callable):
                Un-vectorized function to call.
                Number of accepted args should equal len(args).
                Number of accepted kwargs should equal len(kwargs).
            args (xr.DataArray):
                xarray DataArrays which will be broadcast together and then iterated over
                to apply the function.
            output_core_dims (tuple):
                Additional dimensions which are expected to be passed back from `xr.apply_ufunc` after applying `func`.
                This is directly passed to `xr.apply_ufunc`; see their documentation for more details.
                Defaults to ((), )
            kwargs (dict[str, Any]):
                Additional keyword arguments to pass to `func`.

        Returns:
            xr.DataArray: Array with func applied to all elements.
        """
        return xr.apply_ufunc(
            func,
            *args,
            kwargs=kwargs,
            vectorize=True,
            keep_attrs=True,
            output_dtypes=[np.dtype("O")],
            output_core_dims=output_core_dims,
        )

    @abstractmethod
    def verbose_strings(self) -> None:
        """
        Update optimisation model object string representations to include the index coordinates of the object.

        E.g., `variables(carrier_prod)[0]` will become `variables(carrier_prod)[power, region1, ccgt, 2005-01-01 00:00]`

        This takes approximately 10% of the peak memory required to initially build the optimisation problem, so should only be invoked if inspecting the model in detail (e.g., debugging)

        Only string representations of model parameters and variables will be updated since global expressions automatically show the string representation of their contents.
        """

    def _raise_error_on_preexistence(self, key: str, obj_type: _COMPONENTS_T):
        f"""
        We do not allow any overlap of backend object names since they all have to
        co-exist in the backend dataset.
        I.e., users cannot overwrite any backend component with another
        (of the same type or otherwise).

        Args:
            key (str): Backend object name
            obj_type (str): Object type (one of {self._VALID_COMPONENTS})

        Raises:
            BackendError:
                Raised if `key` already exists in the backend model
                (either with the same or different type as `obj_type`).
        """
        if key in self._dataset.keys():
            if key in getattr(self, obj_type):
                raise BackendError(
                    f"Trying to add already existing `{key}` to backend model {obj_type}."
                )
            else:
                other_obj_type = [
                    k.removesuffix("s")
                    for k, v in self._dataset[key].attrs.items()
                    if k in self._VALID_COMPONENTS and v == 1
                ][0]
                raise BackendError(
                    f"Trying to add already existing *{other_obj_type}* `{key}` "
                    f"as a backend model *{obj_type.removesuffix('s')}*."
                )

    @staticmethod
    def _clean_arrays(*args) -> None:
        """
        Preemptively delete objects with large memory footprints that might otherwise
        stick around longer than necessary.
        """
        del args

    def _add_to_dataset(
        self,
        name: str,
        da: xr.DataArray,
        obj_type: _COMPONENTS_T,
        references: Optional[set] = None,
    ):
        """
        Add array of backend objects to backend dataset in-place.

        Args:
            name (str): Name of entry in dataset.
            da (xr.DataArray): Data to add.
            obj_type (str): Type of backend objects in the array.
            references (set):
                All other backend objects which are references in this backend object's linear expression(s).
                E.g. the constraint "carrier_prod / energy_eff <= energy_cap" references the variables ["carrier_prod", "energy_cap"]
                and the parameter ["energy_eff"].
                All referenced objects will have their "references" attribute updated with this object's name.
        """
        self._dataset[name] = da.assign_attrs(
            {obj_type: 1, "references": set(), "coords_in_name": False}
        )
        if references is not None:
            for reference in references:
                self._dataset[reference].attrs["references"].add(name)

    @property
    def constraints(self):
        "Slice of backend dataset to show only built constraints"
        return self._dataset.filter_by_attrs(constraints=1)

    @property
    def variables(self):
        "Slice of backend dataset to show only built variables"
        return self._dataset.filter_by_attrs(variables=1)

    @property
    def parameters(self):
        "Slice of backend dataset to show only built parameters"
        return self._dataset.filter_by_attrs(parameters=1)

    @property
    def expressions(self):
        "Slice of backend dataset to show only built expressions"
        return self._dataset.filter_by_attrs(expressions=1)

    @property
    def objectives(self):
        "Slice of backend dataset to show only built objectives"
        return self._dataset.filter_by_attrs(objectives=1)


class PyomoBackendModel(BackendModel):
    def __init__(self):
        BackendModel.__init__(
            self,
            instance=pmo.block(),
        )
        self._instance.parameters = pmo.parameter_dict()
        self._instance.variables = pmo.variable_dict()
        self._instance.expressions = pmo.expression_dict()
        self._instance.constraints = pmo.constraint_dict()
        self._instance.objectives = pmo.objective_list()

    def add_parameter(
        self,
        parameter_name: str,
        parameter_values: xr.DataArray,
        default: Any = np.nan,
        use_inf_as_na: bool = False,
    ) -> None:
        self._raise_error_on_preexistence(parameter_name, "parameters")

        self._create_pyomo_list(parameter_name, "parameters")

        parameter_da = self.apply_func(
            self._to_pyomo_param,
            parameter_values,
            name=parameter_name,
            default=default,
            use_inf_as_na=use_inf_as_na,
        )
        if parameter_da.isnull().all():
            self._delete_pyomo_list(parameter_name, "parameters")
            parameter_da = parameter_da.astype(float)

        self._add_to_dataset(parameter_name, parameter_da, "parameters")
        self.valid_math_element_names.add(parameter_name)

    def add_constraint(
        self,
        model_data: xr.Dataset,
        name: str,
        constraint_dict: parsing.UnparsedConstraintDict,
    ) -> None:
        def _constraint_setter(
            imask: xr.DataArray, expr: tuple[xr.DataArray, str, xr.DataArray]
        ) -> xr.DataArray:
            lhs, op, rhs = expr
            to_fill = self.apply_func(
                self._to_pyomo_constraint,
                imask,
                xr.DataArray(lhs).squeeze(drop=True),
                xr.DataArray(rhs).squeeze(drop=True),
                op=op,
                name=name,
            )
            self._clean_arrays(lhs, rhs)
            return to_fill

        self._add_constraint_or_expression(
            model_data,
            name,
            constraint_dict,
            _constraint_setter,
            "constraints",
        )

    def add_expression(
        self,
        model_data: xr.Dataset,
        name: str,
        expression_dict: parsing.UnparsedConstraintDict,
    ) -> None:
        def _expression_setter(imask: xr.DataArray, expr: xr.DataArray) -> xr.DataArray:
            to_fill = self.apply_func(
                self._to_pyomo_expression,
                imask,
                expr.squeeze(drop=True),
                name=name,
            )
            self._clean_arrays(expr)
            return to_fill

        self.valid_math_element_names.add(name)

        self._add_constraint_or_expression(
            model_data,
            name,
            expression_dict,
            _expression_setter,
            "expressions",
        )

    def add_variable(
        self,
        model_data: xr.Dataset,
        name: str,
        variable_dict: parsing.UnparsedVariableDict,
    ) -> None:
        self.valid_math_element_names.add(name)

        parsed_variable = parsing.ParsedBackendComponent(
            "variables", name, variable_dict
        )
        foreach_imask = parsed_variable.evaluate_foreach(model_data)
        if not foreach_imask.any():
            return None

        parsed_variable.parse_top_level_where()
        imask = parsed_variable.evaluate_where(model_data, foreach_imask)
        if not imask.any():
            return None

        imask = parsed_variable.align_imask_with_sets(imask)

        self._raise_error_on_preexistence(name, "variables")
        self._create_pyomo_list(name, "variables")

        domain = parsed_variable._unparsed.get("domain", "real")
        domain_type = getattr(pmo, f"{domain.title()}Set")

        ub, lb = self._get_capacity_bounds(variable_dict["bounds"], name=name)
        variable_da = self.apply_func(
            self._to_pyomo_variable,
            imask,
            ub,
            lb,
            name=name,
            domain_type=domain_type,
        )

        self._add_to_dataset(name, variable_da, "variables")

    def add_objective(
        self,
        model_data: xr.Dataset,
        name: str,
        objective_dict: parsing.UnparsedObjectiveDict,
    ) -> None:
        self._raise_error_on_preexistence(name, "objectives")
        sense_dict = {"minimize": 1, "minimise": 1, "maximize": -1, "maximise": -1}
        parsed_objective = parsing.ParsedBackendComponent(
            "objectives", name, objective_dict
        )
        equations = parsed_objective.parse_equations(self.valid_math_element_names)

        n_valid_exprs = 0
        for equation in equations:
            imask = equation.evaluate_where(model_data)
            if imask.any():
                expr = equation.evaluate_expression(model_data, self, imask).item()
                n_valid_exprs += 1

        if n_valid_exprs == 0:
            return None

        if n_valid_exprs > 1:
            raise BackendError(
                f"More than one {name} objective is valid for this "
                "optimisation problem; only one is allowed."
            )

        objective = pmo.objective(expr, sense=sense_dict[objective_dict["sense"]])

        if name == model_data.run_config["objective"]:
            objective.activate()
        else:
            objective.deactivate()

        self._instance.objectives.append(objective)

        self._add_to_dataset(name, xr.DataArray(objective), "objectives")

    def get_parameter(
        self, parameter_name: str, as_backend_objs: bool = True
    ) -> Optional[xr.DataArray]:
        parameter = self.parameters.get(parameter_name, None)
        if isinstance(parameter, xr.DataArray) and not as_backend_objs:
            return self.apply_func(self._from_pyomo_param, parameter)
        else:
            return parameter

    def get_constraint(
        self,
        constraint_name: str,
        as_backend_objs: bool = True,
        eval_body: bool = False,
    ) -> Optional[Union[xr.DataArray, xr.Dataset]]:
        constraint = self.constraints.get(constraint_name, None)
        if isinstance(constraint, xr.DataArray) and not as_backend_objs:
            constraint_attrs = self.apply_func(
                self._from_pyomo_constraint,
                constraint,
                eval_body=eval_body,
                output_core_dims=(["attributes"],),
            )
            constraint_attrs.coords["attributes"] = ["lb", "body", "ub"]
            constraint = constraint_attrs.to_dataset("attributes")
        return constraint

    def get_variable(
        self, variable_name: str, as_backend_objs: bool = True
    ) -> Optional[xr.DataArray]:
        variable = self.variables.get(variable_name, None)
        if isinstance(variable, xr.DataArray) and not as_backend_objs:
            return self.apply_func(self._from_pyomo_param, variable)
        else:
            return variable

    def get_expression(
        self,
        expression_name: str,
        as_backend_objs: bool = True,
        eval_body: bool = False,
    ) -> Optional[xr.DataArray]:
        expression = self.expressions.get(expression_name, None)
        if isinstance(expression, xr.DataArray) and not as_backend_objs:
            return self.apply_func(
                self._from_pyomo_expr, expression, eval_body=eval_body
            )
        else:
            return expression

    def solve(
        self,
        solver: str,
        solver_io: Optional[str] = None,
        solver_options: Optional[dict] = None,
        save_logs: Optional[str] = None,
        warmstart: bool = False,
        **solve_kwargs,
    ):
        opt = SolverFactory(solver, solver_io=solver_io)

        if solver_options:
            for k, v in solver_options.items():
                opt.options[k] = v

        if save_logs is not None:
            solve_kwargs.update({"symbolic_solver_labels": True, "keepfiles": True})
            os.makedirs(save_logs, exist_ok=True)
            TempfileManager.tempdir = save_logs  # Sets log output dir
        if warmstart and solver in ["glpk", "cbc"]:
            model_warn(
                "The chosen solver, {}, does not support warmstart, which may "
                "impact performance.".format(solver)
            )
            warmstart = False

        with redirect_stdout(LogWriter(logger, "debug", strip=True)):  # type: ignore
            with redirect_stderr(LogWriter(logger, "error", strip=True)):  # type: ignore
                # Ignore most of gurobipy's logging, as it's output is
                # already captured through STDOUT
                logging.getLogger("gurobipy").setLevel(logging.ERROR)
                results = opt.solve(self._instance, tee=True, **solve_kwargs)

        termination = results.solver[0].termination_condition

        if termination == pe.TerminationCondition.optimal:
            self._instance.load_solution(results.solution[0])

        else:
            logger.critical("Problem status:")
            for line in str(results.problem[0]).split("\n"):
                logger.critical(line)
            logger.critical("Solver status:")
            for line in str(results.solver[0]).split("\n"):
                logger.critical(line)

            model_warn("Model solution was non-optimal.", _class=BackendWarning)

        return str(termination)

    def verbose_strings(self) -> None:
        def __renamer(val, *idx):
            if pd.notnull(val):
                val.calliope_coords = idx

        with self._datetime_as_string(self._dataset):
            for component_group in ["parameters", "variables"]:
                for da in self._dataset.filter_by_attrs(
                    coords_in_name=False, **{component_group: 1}
                ).values():
                    self.apply_func(__renamer, da, *[da.coords[i] for i in da.dims])
                    da.attrs["coords_in_name"] = True

    def _create_pyomo_list(self, key: str, component_type: _COMPONENTS_T) -> None:
        """Attach an empty pyomo kernel list object to the pyomo model object.

        Args:
            key (str): Name of object
            component_type (str): Object type

        Raises:
            BackendError: Cannot overwrite object of same name and type.
        """
        component_dict = getattr(self._instance, component_type)
        if key in component_dict:
            raise BackendError(
                f"Trying to add already existing `{key}` to backend model {component_type}."
            )
        else:
            singular_component = component_type.removesuffix("s")
            component_dict[key] = getattr(pmo, f"{singular_component}_list")()

    def _delete_pyomo_list(self, key: str, component_type: _COMPONENTS_T) -> None:
        """Delete a pyomo kernel list object from the pyomo model object.

        Args:
            key (str): Name of object
            component_type (str): Object type
        """
        component_dict = getattr(self._instance, component_type)
        if key not in component_dict:
            return None
        else:
            del component_dict[key]

    def _add_constraint_or_expression(
        self,
        model_data: xr.Dataset,
        name: str,
        component_dict: parsing.UnparsedConstraintDict,
        component_setter: Callable,
        component_type: Literal["constraints", "expressions"],
    ) -> None:
        """Generalised function to add a constraint or expression array to the model.

        Args:
            model_data (xr.Dataset): Calliope model input data
            name: Name of the constraint or expression
            component_dict (parsing.UnparsedConstraintDict):
                Unparsed YAML dictionary configuration.
            component_setter (Callable):
                Function to combine evaluated xarray DataArrays into
                constraint/expression objects.
                Will receive outputs of `evaluate_where` and `evaluate_expression` as inputs.
            component_type (Literal[constraints, expressions])
            parser (Callable): Parsing rule to use for the component (differs between constraints and expressions)


        Raises:
            BackendError:
                The sub-equations of the parsed component cannot generate component
                objects on duplicate index entries.
        """
        references: set[str] = set()

        parsed_component = parsing.ParsedBackendComponent(
            component_type, name, component_dict
        )
        foreach_imask = parsed_component.evaluate_foreach(model_data)
        if not foreach_imask.any():
            return None

        parsed_component.parse_top_level_where()
        top_level_imask = parsed_component.evaluate_where(model_data, foreach_imask)
        if not top_level_imask.any():
            return None

        self._raise_error_on_preexistence(name, component_type)
        component_da = (
            xr.DataArray()
            .where(parsed_component.align_imask_with_sets(top_level_imask))
            .astype(np.dtype("O"))
        )
        self._create_pyomo_list(name, component_type)

        equations = parsed_component.parse_equations(self.valid_math_element_names)
        for element in equations:
            imask = element.evaluate_where(model_data, top_level_imask)
            if not imask.any():
                continue

            imask = parsed_component.align_imask_with_sets(imask)

            if component_da.where(imask).notnull().any():
                subset_overlap = component_da.where(imask).to_series().dropna().index

                raise BackendError(
                    "Trying to set two equations for the same index of "
                    f"{component_type.removesuffix('s')} `{name}`:\n{subset_overlap}"
                )

            expr = element.evaluate_expression(model_data, self, imask, references)
            to_fill = component_setter(imask, expr)
            component_da = component_da.fillna(to_fill)

        if component_da.isnull().all():
            self._delete_pyomo_list(name, component_type)
            return None

        self._add_to_dataset(name, component_da, component_type, references)

    def _get_capacity_bounds(
        self, bounds: parsing.UnparsedVariableBoundDict, name: str
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """
        Generate arrays corresponding to upper and lower bounds of a decision variable.
        If `equals` is given, then everywhere it is not None/np.nan it will be applied
        as the simultaneous upper and lower bound. Everywhere it is None/np.nan, it will
        be filled by `min` (for lower bound) and `max` (for upper bound).
        Upper and lower bounds will be scaled by `scale`, if `scale` is not None/np.nan.

        Args:
            bounds (dict): Dictionary of optional keys `min`, `max`, `equals`, and `scale`.
            name (str): Name of decision variable.

        Returns:
            tuple[xr.DataArray, xr.DataArray]:
                (upper bounds, lower bounds). Where unbounded, the array entry will be None.
        """

        def __get_bound(bound):
            this_bound = bounds.get(bound, None)
            if isinstance(this_bound, str):
                return self.get_parameter(this_bound)
            else:
                # TODO: decide if this parameter should be added to the backend dataset too
                name_ = f"TEMP_{name}_{bound}"
                self._create_pyomo_list(name_, "parameters")
                return xr.DataArray(self._to_pyomo_param(this_bound, name=name_))

        scale = __get_bound("scale")
        equals_ = __get_bound("equals")
        min_ = __get_bound("min")
        max_ = __get_bound("max")

        lb = equals_.fillna(min_)
        ub = equals_.fillna(max_)
        if scale.notnull().any():
            lb = lb * scale
            ub = ub * scale

        return ub.fillna(None), lb.fillna(None)

    def _to_pyomo_param(
        self, val: Any, *, name: str, default: Any = np.nan, use_inf_as_na: bool = True
    ) -> Union[type[ObjParameter], float]:
        """
        Utility function to generate a pyomo parameter for every element of an
        xarray DataArray.
        Output objects are of the type ObjParameter(pmo.parameter) since they need a
        "dtype" property to be handled by xarray.

        If not np.nan/None, output objects are also added to the backend model object in-place.


        Args:
            val (Any): Value to turn into a mutable pyomo parameter
            name (str): Name of parameter
            default (Any, optional): Default value if `val` is None/np.nan. Defaults to np.nan.
            use_inf_as_na (bool, optional): If True, see np.inf as np.nan. Defaults to True.

        Returns:
            Union[type[ObjParameter], float]:
                If both `val` and `default` are np.nan/None, return np.nan.
                Otherwise return ObjParameter(val/default).
        """
        with pd.option_context("mode.use_inf_as_na", use_inf_as_na):
            if pd.isnull(val):
                if pd.isnull(default):
                    param = np.nan
                else:
                    param = ObjParameter(default)
                    self._instance.parameters[name].append(param)
            else:
                param = ObjParameter(val)
                self._instance.parameters[name].append(param)
            return param

    def _to_pyomo_constraint(
        self,
        mask: Union[bool, np.bool_],
        lhs: Any,
        rhs: Any,
        *,
        op: Literal["==", ">=", "<="],
        name: str,
    ) -> Union[type[pmo.constraint], float]:
        """
        Utility function to generate a pyomo constraint for every element of an
        xarray DataArray.

        If not np.nan/None, output objects are also added to the backend model object in-place.

        Args:
            mask (Union[bool, np.bool_]): If True, add constraint, otherwise return np.nan
            lhs (Any): Equation left-hand-side expression
            rhs (Any): Equation right-hand-side expression

        Kwargs:
            op (Literal[, optional): Operator to compare `lhs` and `rhs`. Defaults to =", ">=", "<="].
            name (str): Name of constraint

        Returns:
            Union[type[pmo.constraint], float]:
                If mask is True, return np.nan.
                Otherwise return pmo_constraint(expr=lhs op rhs).
        """

        if not mask:
            return np.nan
        elif op == "==":
            constraint = pmo.constraint(expr=lhs == rhs)
        elif op == "<=":
            constraint = pmo.constraint(expr=lhs <= rhs)
        elif op == ">=":
            constraint = pmo.constraint(expr=lhs >= rhs)
        self._instance.constraints[name].append(constraint)
        return constraint

    def _to_pyomo_expression(
        self, mask: Union[bool, np.bool_], expr: Any, *, name: str
    ) -> Union[type[pmo.expression], float]:
        """
        Utility function to generate a pyomo expression for every element of an
        xarray DataArray.

        If not np.nan/None, output objects are also added to the backend model object in-place.


        Args:
            mask (Union[bool, np.bool_]): If True, add expression, otherwise return np.nan.
            expr (Any): Linear expression to add.
        Kwargs:
            name (str): Expression name.

        Returns:
            Union[type[pmo.expression], float]:
                If mask is True, return np.nan.
                Otherwise return pmo_expression(expr).
        """
        if mask:
            expr_obj = pmo.expression(expr)
            self._instance.expressions[name].append(expr_obj)
            return expr_obj
        else:
            return np.nan

    def _to_pyomo_variable(
        self,
        mask: Union[bool, np.bool_],
        ub: Any,
        lb: Any,
        *,
        name: str,
        domain_type: Literal["RealSet", "IntegerSet"],
    ) -> Union[type[ObjVariable], float]:
        """
        Utility function to generate a pyomo decision variable for every element of an
        xarray DataArray.

        If not np.nan/None, output objects are also added to the backend model object in-place.

        Args:
            mask (Union[bool, np.bool_]): If True, add variable, otherwise return np.nan.
            ub (Any): Upper bound to apply to the variable.
            lb (Any): Lower bound to apply to the variable.

        Kwargs:
            domain_type (Literal["RealSet", "IntegerSet"]):
                Domain over which variables are valid (real = continuous, integer = integer/binary)
            name (str): Name of variable.

        Returns:
            Union[type[ObjVariable], float]:
                If mask is True, return np.nan.
                Otherwise return pmo_variable(ub=ub, lb=lb, domain_type=domain_type).
        """
        if mask:
            var = ObjVariable(ub=ub, lb=lb, domain_type=domain_type)
            self._instance.variables[name].append(var)
            return var
        else:
            return np.nan

    @staticmethod
    def _from_pyomo_param(val: Union[ObjParameter, ObjVariable, float]) -> Any:
        """
        Evaluate value of Pyomo object.
        If the input object is a parameter, a numeric/string value will be given.
        If the input object is an expression or variable, a numeric value will be given
        only if the backend model has been successfully optimised, otherwise evaluation will return None.

        Args:
            val (Union[ObjParameter, pmo.expression, ObjVariable, np.nan]):
                Item to be evaluated.

        Returns:
            Any: If the input is nullable, return np.nan, otherwise evaluate the "value" method of the object.
        """
        if pd.isnull(val):
            return np.nan
        else:
            return val.value  # type: ignore

    @staticmethod
    def _from_pyomo_constraint(
        val: pmo.constraint, *, eval_body: bool = False
    ) -> pd.Series:
        """Evaluate Pyomo constraint object.

        Args:
            val (pmo.constraint): constraint object to be evaluated
        Kwargs:
            eval_body (bool, optional):
                If True, attempt to evaluate the constraint object `body`, which will evaluate the
                linear expression contained in the constraint body and produce a numeric value.
                This will only succeed if the backend model has been successfully optimised,
                otherwise a string representation of the linear expression will be returned
                (same as eval_body=False). Defaults to False.

        Returns:
            pd.Series: Array of upper bound (ub), body, and lower bound (lb).
        """
        if pd.isnull(val):
            vals = [np.nan, np.nan, np.nan]
        else:
            if eval_body:
                try:
                    body = val.body()
                except ValueError:
                    body = val.body.to_string()
            else:
                body = val.body.to_string()
            vals = [val.lb, body, val.ub]
        return pd.Series(data=vals, index=["lb", "body", "ub"])

    @staticmethod
    def _from_pyomo_expr(val: pmo.expression, *, eval_body: bool = False) -> Any:
        """Evaluate Pyomo expression object.

        Args:
            val (pmo.expression): expression object to be evaluated
            eval_body (bool, optional):
                If True, attempt to evaluate the expression object, which will produce a numeric value.
                This will only succeed if the backend model has been successfully optimised,
                otherwise a string representation of the linear expression will be returned
                (same as eval_body=False). Defaults to False.

        Returns:
            Any: If the input is nullable, return np.nan, otherwise a numeric value
            (eval_body=True and problem is optimised) or a string.
        """
        if pd.isnull(val):
            return np.nan
        else:
            if eval_body:
                expr = val()
                if expr is None:
                    return val.to_string()
                else:
                    return expr
            else:
                return val.to_string()

    @contextmanager
    def _datetime_as_string(self, data: Union[xr.DataArray, xr.Dataset]) -> Iterator:
        """Context manager to temporarily convert np.dtype("datetime64[ns]") coordinates (e.g. timesteps) to strings with a resolution of minutes.

        Args:
            data (Union[xr.DataArray, xr.Dataset]): xarray object on whose coordinates the conversion will take place.
        """
        datetime_coords = set()
        for name_, vals_ in data.coords.items():
            if vals_.dtype.kind == "M":
                data.coords[name_] = data.coords[name_].dt.strftime("%Y-%m-%d %H:%M")
                datetime_coords.add(name_)
        try:
            yield
        finally:
            for name_ in datetime_coords:
                data.coords[name_] = xr.apply_ufunc(
                    pd.to_datetime, data.coords[name_], keep_attrs=True
                )


class CoordObj(ABC):
    """Class with methods to update the `name` property of inheriting classes"""

    def __init__(self) -> None:
        self._calliope_coords: Optional[Iterable] = None

    def _update_name(self, old_name: str) -> str:
        """
        Update string of a list containing a single number with a string of a list containing any arbitrary number of elements

        Args:
            old_name (str): String representation of a list containing a single number

        Returns:
            str:
                If `self.calliope_coords` is None, returns `old_name`.
                Otherwise returns string representation of a list containing the contents of `self.calliope_coords`
        """
        if self._calliope_coords is None:
            return old_name

        if not self._calliope_coords:  # empty list = dimensionless component
            coord_list = ""
        else:
            coord_list = f"[{', '.join(str(i) for i in self._calliope_coords)}]"
        return re.sub(r"\[\d+\]", coord_list, old_name)

    @property
    def calliope_coords(self):
        return self._calliope_coords

    @calliope_coords.setter
    def calliope_coords(self, val):
        self._calliope_coords = val


class ObjParameter(pmo.parameter, CoordObj):
    """
    A pyomo parameter (`a object for storing a mutable, numeric value that can be used to build a symbolic expression`)
    with added `dtype` property and a `name` property setter (via the `pmo.parameter.getname` method) which replaces a list position as a name with a list of strings.
    """

    def __init__(self, value, **kwds):
        assert not pd.isnull(value)
        pmo.parameter.__init__(self, value, **kwds)
        CoordObj.__init__(self)

    @property
    def dtype(self):
        return "O"

    def getname(self, *args, **kwargs):
        return self._update_name(pmo.parameter.getname(self, *args, **kwargs))


class ObjVariable(pmo.variable, CoordObj):
    """
    A pyomo variable with a `name` property setter (via the `pmo.variable.getname` method) which replaces a list position as a name with a list of strings.

    """

    def __init__(self, **kwds):
        pmo.variable.__init__(self, **kwds)
        CoordObj.__init__(self)

    def getname(self, *args, **kwargs):
        return self._update_name(pmo.variable.getname(self, *args, **kwargs))
