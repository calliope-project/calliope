"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

run_checks.py
~~~~~~~~~~~~~

Checks for model consistency and possible errors when preparing run in the backend.

"""
import numpy as np
import pandas as pd
import xarray as xr
from calliope.core.attrdict import AttrDict
from calliope.core.util.observed_dict import UpdateObserverDict


def check_operate_params(model_data):
    """
    if model mode = `operate`, check for clashes in capacity constraints.
    In this mode, all capacity constraints are set to parameters in the backend,
    so can easily lead to model infeasibility if not checked.

    Returns
    -------
    comments : AttrDict
        debug output
    warnings : list
        possible problems that do not prevent the model run
        from continuing
    errors : list
        serious issues that should raise a ModelError

    """
    defaults = UpdateObserverDict(
        initial_yaml_string=model_data.attrs["defaults"],
        name="defaults",
        observer=model_data,
    )
    run_config = UpdateObserverDict(
        initial_yaml_string=model_data.attrs["run_config"],
        name="run_config",
        observer=model_data,
    )

    warnings, errors = [], []
    comments = AttrDict()

    def _get_param(loc_tech, var):
        if _is_in(loc_tech, var) and not model_data[var].loc[loc_tech].isnull().any():
            param = model_data[var].loc[loc_tech].values
        else:
            param = defaults.get(var, None)
        return param

    def _is_in(loc_tech, set_or_var):
        try:
            model_data[set_or_var].loc[loc_tech]
            return True
        except (KeyError, AttributeError):
            return False

    def _set_inf_and_warn(loc_tech, var, warnings, warning_text):
        # inf will only work on float dtypes
        # astype doesn't retain attributes: https://github.com/pydata/xarray/issues/2049
        _attrs = model_data[var].attrs
        model_data[var] = model_data[var].astype(float)
        model_data[var].attrs = _attrs
        if np.isinf(model_data[var].loc[loc_tech].item()):
            return (np.inf, warnings)
        elif model_data[var].loc[loc_tech].isnull().item():
            var_name = model_data[var].loc[loc_tech] = np.inf
            return (var_name, warnings)
        else:
            var_name = model_data[var].loc[loc_tech] = np.inf
            warnings.append(warning_text)
            return var_name, warnings

    # Storage initial is carried over between iterations, so must be defined along with storage
    if (
        "loc_techs_store" in model_data.dims.keys()
        and "storage_initial" not in model_data.data_vars.keys()
    ):
        model_data["storage_initial"] = xr.DataArray(
            [0.0 for loc_tech in model_data.loc_techs_store.values],
            dims="loc_techs_store",
        )
        model_data["storage_initial"].attrs["is_result"] = 0.0
        warnings.append(
            "Initial stored energy not defined, set to zero for all "
            "loc::techs in loc_techs_store, for use in iterative optimisation"
        )

    # Operated units is carried over between iterations, so must be defined in a milp model
    if (
        "loc_techs_milp" in model_data.dims.keys()
        and "operated_units" not in model_data.data_vars.keys()
    ):
        model_data["operated_units"] = xr.DataArray(
            [0 for loc_tech in model_data.loc_techs_milp.values], dims="loc_techs_milp"
        )
        model_data["operated_units"].attrs["is_result"] = 1
        model_data["operated_units"].attrs["operate_param"] = 1
        warnings.append(
            "daily operated units not defined, set to zero for all "
            "loc::techs in loc_techs_milp, for use in iterative optimisation"
        )

    for loc_tech in model_data.loc_techs.values:
        energy_cap = model_data.energy_cap.loc[loc_tech].item()
        # Must have energy_cap defined for all relevant techs in the model
        if (np.isinf(energy_cap) or np.isnan(energy_cap)) and (
            _get_param(loc_tech, "energy_cap_min_use")
            or (
                _get_param(loc_tech, "force_resource")
                and _get_param(loc_tech, "resource_unit") == "energy_per_cap"
            )
        ):
            errors.append(
                "Operate mode: User must define a finite energy_cap (via "
                "energy_cap_equals or energy_cap_max) for {}".format(loc_tech)
            )

        elif _is_in(loc_tech, "loc_techs_finite_resource"):
            # Cannot have infinite resource area if linking resource and area (resource_unit = energy_per_area)
            if (
                _is_in(loc_tech, "loc_techs_area")
                and model_data.resource_unit.loc[loc_tech].item() == "energy_per_area"
            ):
                if _is_in(loc_tech, "resource_area"):
                    area = model_data.resource_area.loc[loc_tech].item()
                else:
                    area = None
                if pd.isnull(area) or np.isinf(area):
                    errors.append(
                        "Operate mode: User must define a finite resource_area "
                        "(via resource_area_equals or resource_area_max) for {}, "
                        "as available resource is linked to resource_area "
                        "(resource_unit = `energy_per_area`)".format(loc_tech)
                    )

            # force resource overrides capacity constraints, so set capacity constraints to infinity
            if _get_param(loc_tech, "force_resource"):

                if not _is_in(loc_tech, "loc_techs_store"):
                    # set resource_area to inf if the resource is linked to energy_cap using energy_per_cap
                    if (
                        model_data.resource_unit.loc[loc_tech].item()
                        == "energy_per_cap"
                    ):
                        if _is_in(loc_tech, "resource_area"):
                            resource_area, warnings = _set_inf_and_warn(
                                loc_tech,
                                "resource_area",
                                warnings,
                                "Resource area constraint removed from {} as "
                                "force_resource is applied and resource is linked "
                                "to energy flow using `energy_per_cap`".format(
                                    loc_tech
                                ),
                            )
                    # set energy_cap to inf if the resource is linked to resource_area using energy_per_area
                    elif (
                        model_data.resource_unit.loc[loc_tech].item()
                        == "energy_per_area"
                    ):
                        energy_cap, warnings = _set_inf_and_warn(
                            loc_tech,
                            "energy_cap",
                            warnings,
                            "Energy capacity constraint removed from {} as "
                            "force_resource is applied and resource is linked "
                            "to energy flow using `energy_per_area`".format(loc_tech),
                        )
                    # set both energy_cap and resource_area to inf if the resource is not linked to anything
                    elif model_data.resource_unit.loc[loc_tech].item() == "energy":
                        if _is_in(loc_tech, "resource_area"):
                            resource_area, warnings = _set_inf_and_warn(
                                loc_tech,
                                "resource_area",
                                warnings,
                                "Resource area constraint removed from {} as "
                                "force_resource is applied and resource is not linked "
                                "to energy flow (resource_unit = `energy`)".format(
                                    loc_tech
                                ),
                            )
                        energy_cap, warnings = _set_inf_and_warn(
                            loc_tech,
                            "energy_cap",
                            warnings,
                            "Energy capacity constraint removed from {} as "
                            "force_resource is applied and resource is not linked "
                            "to energy flow (resource_unit = `energy`)".format(
                                loc_tech
                            ),
                        )

                if _is_in(loc_tech, "resource_cap"):
                    resource_cap, warnings = _set_inf_and_warn(
                        loc_tech,
                        "resource_cap",
                        warnings,
                        "Resource capacity constraint removed from {} as "
                        "force_resource is applied".format(loc_tech),
                    )

        if _is_in(loc_tech, "loc_techs_store"):
            storage_cap = model_data.storage_cap.loc[loc_tech].item()
            if not pd.isnull(storage_cap) and not pd.isnull(energy_cap):
                if _get_param(loc_tech, "charge_rate") is not None:
                    charge_rate = model_data["charge_rate"].loc[loc_tech].item()
                    if storage_cap * charge_rate < energy_cap:
                        errors.append(
                            "fixed storage capacity * charge_rate is not larger "
                            "than fixed energy capacity for loc::tech {}".format(
                                loc_tech
                            )
                        )
                if not np.isinf(_get_param(loc_tech, "energy_cap_per_storage_cap_max")):
                    energy_cap_per_storage_cap_max = (
                        model_data["energy_cap_per_storage_cap_max"]
                        .loc[loc_tech]
                        .item()
                    )
                    if storage_cap * energy_cap_per_storage_cap_max < energy_cap:
                        errors.append(
                            "fixed storage capacity * energy_cap_per_storage_cap_max is not larger "
                            "than fixed energy capacity for loc::tech {}".format(
                                loc_tech
                            )
                        )
                if _get_param(loc_tech, "energy_cap_per_storage_cap_min") != 0:
                    energy_cap_per_storage_cap_min = (
                        model_data["energy_cap_per_storage_cap_min"]
                        .loc[loc_tech]
                        .item()
                    )
                    if storage_cap * energy_cap_per_storage_cap_min > energy_cap:
                        errors.append(
                            "fixed storage capacity * energy_cap_per_storage_cap_min is not smaller "
                            "than fixed energy capacity for loc::tech {}".format(
                                loc_tech
                            )
                        )
    # Must define a resource capacity to ensure the Pyomo param is created
    # for it. But we just create an array of infs, so the capacity has no effect
    if (
        "resource_cap" not in model_data.data_vars.keys()
        and "loc_techs_supply_plus" in model_data.dims.keys()
    ):
        model_data["resource_cap"] = xr.DataArray(
            [np.inf for i in model_data.loc_techs_supply_plus.values],
            dims="loc_techs_supply_plus",
        )
        model_data["resource_cap"].attrs["is_result"] = 1
        model_data["resource_cap"].attrs["operate_param"] = 1
        warnings.append(
            "Resource capacity constraint defined and set to infinity "
            "for all supply_plus techs"
        )

    window = run_config.get("operation", {}).get("window", None)
    horizon = run_config.get("operation", {}).get("horizon", None)
    if not window or not horizon:
        errors.append(
            "Operational mode requires a timestep window and horizon to be "
            "defined under run.operation"
        )
    elif horizon < window:
        errors.append(
            "Iteration horizon must be larger than iteration window, "
            "for operational mode"
        )

    # Cyclic storage isn't really valid in operate mode, so we ignore it, using
    # initial_storage instead (allowing us to pass storage between operation windows)
    if run_config.get("cyclic_storage", True):
        warnings.append(
            "Storage cannot be cyclic in operate run mode, setting "
            "`run.cyclic_storage` to False for this run"
        )
        run_config["cyclic_storage"] = False

    if "group_demand_share_per_timestep_decision" in model_data.data_vars:
        warnings.append(
            "`demand_share_per_timestep_decision` group constraints cannot be "
            "used in operate mode, so will not be built."
        )
        del model_data["group_demand_share_per_timestep_decision"]

    return comments, warnings, errors
