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
from calliope.util import schema

BASEPATH = Path(__file__).resolve().parent

NONDEMAND_TECHGROUPS = ["supply", "storage", "conversion", "transmission"]


def generate_base_math_model(model_config: dict) -> calliope.Model:
    """Generate model with documentation for the base math

    Args:
        model_config (dict): Calliope model config.

    Returns:
        calliope.Model: Base math model to use in generating custom math docs.
    """
    model = calliope.Model(
        model_definition=model_config, timeseries_dataframes=_ts_dfs()
    )
    model.math_documentation.build()
    return model


def generate_custom_math_model(
    base_model: calliope.Model, model_config: dict, model_config_updates: dict
) -> calliope.Model:
    """Generate model with documentation for a built-in custom math file, showing only the changes made
    relative to the base math.

    Args:
        base_model (calliope.Model): Calliope model with only the base math applied.
        model_config (dict): Model config suitable for generating the base math.
        model_config_updates (dict): Changes to make to the model config to load the custom math.
    """
    model_config = calliope.AttrDict(model_config)
    model_config_updates = calliope.AttrDict(model_config_updates)
    model_config.union(model_config_updates)
    model = calliope.Model(
        model_definition=model_config,
        timeseries_dataframes=_ts_dfs(),
        time_subset=["2005-01-02", "2005-01-03"],
    )
    _keep_only_changes(base_model, model)

    return model


def generate_model_config() -> dict[str, dict]:
    """To generate the written mathematical formulation of all possible base constraints, we first create a dummy model.

    This dummy has all the relevant technology groups defining all their allowed parameters.

    Parameters that can be defined over a timeseries are forced to be defined over a timeseries.
    Accordingly, the parameters will have "timesteps" in their dimensions in the formulation.
    """
    defaults = schema.extract_from_schema(
        schema.MODEL_SCHEMA, "default", subset_top_level="techs"
    )
    dummy_techs = {
        "demand_tech": {
            "parent": "demand",
            "carrier_in": "electricity",
            "sink_equals": "df=ts",
        },
        "conversion_tech": {
            "parent": "conversion",
            "carrier_in": "gas",
            "carrier_out": ["electricity", "heat"],
        },
        "supply_tech": {"parent": "supply", "carrier_out": "gas"},
        "storage_tech": {
            "parent": "storage",
            "carrier_in": "electricity",
            "carrier_out": "electricity",
        },
        "transmission_tech": {
            "parent": "transmission",
            "carrier_in": "electricity",
            "carrier_out": "electricity",
            "from": "A",
            "to": "B",
        },
    }

    for tech_group in NONDEMAND_TECHGROUPS:
        for k, v in defaults.items():
            dummy_techs[f"{tech_group}_tech"][k] = _add_data(k, v)
    techs_at_nodes = {k: None for k in dummy_techs.keys() if k != "transmission_tech"}
    return {
        "nodes": {
            "A": {"techs": techs_at_nodes, "available_area": 1},
            "B": {"techs": techs_at_nodes},
        },
        "techs": dummy_techs,
    }


def _add_data(name, default_val):
    "Some parameters need hardcoded values to be returned"
    if name.startswith("cost_"):
        return {"data": 1, "index": "monetary", "dims": "costs"}
    elif name in ["export_carrier", "name", "color"]:
        return "foo"
    elif pd.isnull(default_val):
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
                    component_dict["description"] = "|REMOVED|"
                    component_dict["active"] = True
                elif base_model.math[component_group].get(name, {}) != component_dict:
                    _add_to_description(component_dict, "|UPDATED|")
                else:
                    full_del.append(name)
            else:
                _add_to_description(component_dict, "|NEW|")
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
        1, index=pd.date_range("2005-01-02", "2005-01-03", freq="H"), columns=["A"]
    )
    return {"ts": ts}


def generate_math_docs():
    base_model_config = generate_model_config()
    base_model = generate_base_math_model(base_model_config)
    base_model.math_documentation.write(BASEPATH / ".." / "_generated" / "math.md")

    custom_model = generate_custom_math_model(
        base_model,
        base_model_config,
        {
            "config": {
                "init": {
                    "custom_math": ["storage_inter_cluster"],
                    "time_cluster": (
                        BASEPATH
                        / ".."
                        / ".."
                        / "tests"
                        / "common"
                        / "test_model"
                        / "timeseries_data"
                        / "cluster_days.csv"
                    )
                    .absolute()
                    .as_posix(),
                }
            }
        },
    )

    custom_model.math_documentation.write(
        BASEPATH / ".." / "_generated" / "math_storage_inter_cluster.md"
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


if __name__ == "__main__":
    generate_math_docs()
