"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

run_checks.py
~~~~~~~~~~~~~

Checks for model consistency and possible errors when preparing run in the backend.

"""
import numpy as np
from calliope.core.attrdict import AttrDict


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
    run_config = model_data.attrs["run_config"]

    warnings, errors = [], []
    comments = AttrDict()

    def _is_missing(var):
        if var not in model_data.data_vars.keys():
            return True
        else:
            return model_data[var].isnull()

    # Storage initial is carried over between iterations, so must be defined along with storage
    if (model_data.include_storage == 1).any():
        if "storage_initial" not in model_data.data_vars.keys():
            model_data["storage_initial"] = model_data.include_storage.astype(float)
        elif ((model_data.include_storage == 1) & _is_missing("storage_initial")).any():
            model_data.storage_initial = model_data.storage_initial.fillna(
                (model_data.include_storage == 1).astype(float)
            )
        model_data["storage_initial"].attrs["is_result"] = 0.0
        warnings.append(
            "Initial stored energy not defined, set to zero for all "
            "storage technologies, for use in iterative optimisation"
        )

        if ((~_is_missing("storage_cap")) & (~_is_missing("energy_cap"))).any():
            if (
                (
                    model_data.storage_cap
                    * model_data.get("energy_cap_per_storage_cap_max", np.inf)
                )
                < model_data.energy_cap
            ).any():
                errors.append(
                    "fixed storage capacity * energy_cap_per_storage_cap_max is not larger "
                    "than fixed energy capacity for some storage technologies"
                )
            if (
                (
                    model_data.storage_cap
                    * model_data.get("energy_cap_per_storage_cap_min", 0)
                )
                > model_data.energy_cap
            ).any():
                errors.append(
                    "fixed storage capacity * energy_cap_per_storage_cap_min is not smaller "
                    "than fixed energy capacity for some technologies"
                )

    # Operated units is carried over between iterations, so must be defined in a milp model
    if (model_data.cap_method == "integer").any():
        if "operated_units" not in model_data.data_vars.keys():
            model_data["operated_units"] = (model_data.cap_method == "integer").astype(
                float
            )
        elif (
            (model_data.cap_method == "integer") & _is_missing("operated_units")
        ).any():
            model_data.operated_units = model_data.operated_units.fillna(
                (model_data.cap_method == "integer").astype(float)
            )
        model_data["operated_units"].attrs["is_result"] = 1
        model_data["operated_units"].attrs["operate_param"] = 1
        warnings.append(
            "daily operated units not defined, set to zero for all technologies defining an integer capacity method, for use in iterative optimisation"
        )

    if (
        _is_missing("energy_cap")
        & (
            ~_is_missing("energy_cap_min_use")
            | (
                ~_is_missing("force_resource")
                & (model_data.resource_unit == "energy_per_cap")
            )
        )
    ).any():
        errors.append(
            "Operate mode: User must define a finite energy_cap (via energy_cap_equals "
            "or energy_cap_max) if using force_resource or energy_cap_min_use"
        )

    if (
        ~_is_missing("resource")
        & (
            _is_missing("resource_area")
            & (model_data.resource_unit == "energy_per_area")
        )
    ).any():
        errors.append(
            "Operate mode: User must define a finite resource_area "
            "(via resource_area_equals or resource_area_max) "
            "if available resource is linked to resource_area "
            "(resource_unit = `energy_per_area`)"
        )
    if (
        "resource_area" in model_data.data_vars
        and (model_data.resource_unit == "energy_per_cap").any()
    ):
        model_data["resource_area"] = model_data.resource_area.where(
            ~(
                (model_data.force_resource == 1)
                & (model_data.resource_unit == "energy_per_cap")
            )
        )
        warnings.append(
            "Resource area constraint removed from technologies with "
            "force_resource applied and resource linked "
            "to energy flow using `energy_per_cap`"
        )
    if (
        "energy_cap" in model_data.data_vars
        and (model_data.resource_unit == "energy_per_area").any()
    ):
        model_data["energy_cap"] = model_data.energy_cap.where(
            ~(
                (model_data.force_resource == 1)
                & (model_data.resource_unit == "energy_per_area")
            )
        )
        warnings.append(
            "Energy capacity constraint removed from technologies with "
            "force_resource applied and resource linked "
            "to energy flow using `energy_per_area`"
        )
    if (model_data.resource_unit == "energy").any():
        if "energy_cap" in model_data.data_vars:
            model_data["energy_cap"] = model_data.energy_cap.where(
                ~(
                    (model_data.force_resource == 1)
                    & (model_data.resource_unit == "energy")
                )
            )
            warnings.append(
                "Energy capacity constraint removed from technologies with "
                "force_resource applied and resource not linked "
                "to energy flow (resource_unit = `energy`)"
            )
        if "resource_area" in model_data.data_vars:
            model_data["resource_area"] = model_data.resource_area.where(
                ~(
                    (model_data.force_resource == 1)
                    & (model_data.resource_unit == "energy")
                )
            )
            warnings.append(
                "Energy capacity constraint removed from technologies with "
                "force_resource applied and resource not linked "
                "to energy flow (resource_unit = `energy`)"
            )
    if (
        "resource_cap" in model_data.data_vars
        and (model_data.force_resource == 1).any()
    ):
        model_data["resource_cap"] = model_data.resource_cap.where(
            model_data.force_resource == 0
        )
        warnings.append(
            "Resource capacity constraint removed from technologies with "
            "force_resource applied."
        )

    # Must define a resource capacity to ensure the Pyomo param is created
    # for it. But we just create an array of infs, so the capacity has no effect
    # TODO: implement this in the masks

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

    return comments, warnings, errors
