"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

generate_math.py
~~~~~~~~~~~~~~~~~

Generate LaTeX math to include in the documentation.

"""
from pathlib import Path

import calliope
import pandas as pd

BASEPATH = Path(__file__).resolve().parent
STATICPATH = BASEPATH / ".." / "_static"
NONDEMAND_TECHGROUPS = [
    "supply",
    "storage",
    "conversion",
    "transmission",
    "conversion_plus",
    "supply_plus",
]


def generate_base_math_model(model_config: dict) -> calliope.Model:
    """Generate RST file for the base math

    Args:
        model_config (dict): Calliope model config.

    Returns:
        calliope.Model: Base math model to use in generating custom math docs.
    """
    model = calliope.Model(
        model_definition=model_config, timeseries_dataframes=_ts_dfs()
    )
    model.math_documentation.build()
    model.math_documentation.write(STATICPATH / "math.rst")
    return model


def generate_custom_math_model(
    base_model: calliope.Model,
    model_config: dict,
    model_config_updates: dict,
    name: str,
) -> None:
    """Generate RST file for a built-in custom math file, showing only the changes made
    relative to the base math.

    Args:
        base_model (calliope.Model): Calliope model with only the base math applied.
        model_config (dict): Model config suitable for generating the base math.
        model_config_updates (dict): Changes to make to the model config to load the custom math.
        name (str): Name of the custom math to add to the file name.
    """
    model_config = calliope.AttrDict(model_config)
    model_config_updates = calliope.AttrDict(model_config_updates)
    model_config.union(model_config_updates)
    model = calliope.Model(
        model_definition=model_config, timeseries_dataframes=_ts_dfs()
    )
    _keep_only_changes(base_model, model)

    model.math_documentation.write(STATICPATH / f"math_{name}.rst")


def generate_model_config() -> dict[str, dict]:
    """To generate the written mathematical formulation of all possible base constraints, we first create a dummy model.

    This dummy has all the relevant technology groups defining all their allowed parameters.

    Parameters that can be defined over a timeseries are forced to be defined over a timeseries.
    Accordingly, the parameters will have "timesteps" in their dimensions in the formulation.
    """
    defaults = calliope.AttrDict.from_yaml(
        BASEPATH / ".." / ".." / "src" / "calliope" / "config" / "defaults.yaml"
    )

    allowed_: dict[str, dict] = {i: {"all": set()} for i in ["costs", "constraints"]}

    dummy_techs = {
        "demand_tech": {
            "essentials": {"parent": "demand", "carrier": "electricity"},
            "constraints": {"sink_equals": "df=ts"},
        }
    }

    for tech_group in NONDEMAND_TECHGROUPS:
        for config_ in ["costs", "constraints"]:
            tech_allowed_ = defaults.tech_groups[tech_group][f"allowed_{config_}"]
            # We keep parameter definitions to a bare minimum, so any that have been
            # defined for a previous tech group in the loop will not be defined for
            # later tech groups.
            allowed_[config_][tech_group] = set(tech_allowed_).difference(
                allowed_[config_]["all"]
            )
            # We keep lifetime and interest rate since all techs that define costs will
            # need them.
            allowed_[config_][tech_group].update(["lifetime", "interest_rate"])

            allowed_[config_]["all"].update(tech_allowed_)

        if "conversion" in tech_group:
            carriers = {"carrier_in": "electricity", "carrier_out": "heat"}
        else:
            carriers = {"carrier": "electricity"}

        dummy_techs[f"{tech_group}_tech"] = {
            "essentials": {"parent": tech_group, **carriers},
            "constraints": {
                k: _add_data(k, v)
                for k, v in defaults.techs.default_tech.constraints.items()
                if k in allowed_["constraints"][tech_group]
            },
            "costs": {
                "monetary": {
                    k: _add_data(k, v)
                    for k, v in defaults.techs.default_tech.costs.default_cost.items()
                    if k in allowed_["costs"][tech_group]
                }
            },
        }

    return {
        "nodes": {
            "A": {"techs": {k: None for k in dummy_techs.keys()}, "available_area": 1}
        },
        "techs": dummy_techs,
    }


def _add_data(name, default_val):
    "Some parameters need hardcoded values to be returned"
    if name == "carrier_ratios":
        return {"carrier_in.electricity": 1}
    elif name == "export_carrier":
        return "electricity"
    elif default_val is None or name == "interest_rate":
        return 1
    else:
        return default_val


def _keep_only_changes(base_model: calliope.Model, model: calliope.Model) -> None:
    """Compare custom math model with base model and keep only the math strings that are
    different between the two. Changes are made in-place in the custom math model docs

    Args:
        base_model (calliope.Model): Calliope model with base math applied.
        model (calliope.Model): Calliope model with custom math applied.
    """
    full_del = []
    expr_del = []
    for component_group, component_group_dict in model.math.items():
        for name, component_dict in component_group_dict.items():
            if name in base_model.math[component_group]:
                if not component_dict.get("active", True):
                    expr_del.append(name)
                    component_dict["description"] = ":red:`REMOVED`"
                    component_dict["active"] = True
                elif base_model.math[component_group].get(name, {}) != component_dict:
                    _add_to_description(component_dict, ":yellow:`UPDATED`")
                else:
                    full_del.append(name)
            else:
                _add_to_description(component_dict, ":green:`NEW`")
    model.math_documentation.build()
    for key in expr_del:
        model.math_documentation._instance._dataset[key].attrs["math_string"] = ""
    for key in full_del:
        del model.math_documentation._instance._dataset[key]


def _add_to_description(component_dict: dict, update_string: str) -> None:
    "Prepend the math component description"
    component_dict["description"] = f"{update_string}\n{component_dict['description']}"


def _ts_dfs() -> dict[str, pd.DataFrame]:
    "Generate dummy timeseries dataframes"
    ts = pd.DataFrame(
        [1, 1, 1],
        index=pd.date_range("2005-01-01 00:00", "2005-01-01 02:00", freq="H"),
        columns=["A"],
    )
    return {"ts": ts}


if __name__ == "__main__":
    base_model_config = generate_model_config()
    base_model = generate_base_math_model(base_model_config)

    generate_custom_math_model(
        base_model,
        base_model_config,
        {
            "config": {
                "init": {
                    "custom_math": ["storage_inter_cluster"],
                    "time": {
                        "function": "apply_clustering",
                        "function_options": {
                            "clustering_func": "kmeans",
                            "how": "mean",
                            "k": 1,
                        },
                    },
                }
            },
        },
        "storage_inter_cluster",
    )
    # FIXME: Operate mode replaces variables with parameters, so we cannot show that the
    # variable has been deleted in the doc because we cannot build a variable with the same
    # name as another model component.
    # generate_custom_math_model(
    #     base_model,
    #     base_model_config,
    #     {
    #         "model.custom_math": ["operate"],
    #         # FIXME: operate mode should have access to parameter defaults for capacity values
    #         "techs": {
    #             "supply_tech.constraints.energy_cap": 1,
    #             "supply_tech.constraints.purchased": 1,
    #             "supply_tech.constraints.units": 1,
    #             "storage_tech.constraints.storage_cap": 1,
    #             "supply_plus_tech.constraints.resource_cap": 1,
    #             "supply_plus_tech.constraints.resource_area": 1,
    #         },
    #     },
    #     "operate",
    # )

    # FIXME: need to generate the spores params for the spores mode math to build.
    # generate_custom_math_model(
    #    base_model, base_model_config.copy(), {"model.custom_math": ["spores"]}, "spores"
    # )
