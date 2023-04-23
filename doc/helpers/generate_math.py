"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

generate_math.py
~~~~~~~~~~~~~~~~~

Generate LaTeX math to include in the documentation.

"""
from pathlib import Path

import pandas as pd

import calliope

BASEPATH = Path(__file__).resolve().parent
NONDEMAND_TECHGROUPS = [
    "supply",
    "storage",
    "conversion",
    "transmission",
    "conversion_plus",
    "supply_plus",
]


def generate_base_math_model(write: bool = True) -> calliope.Model:
    node_techs = generate_node_techs()
    model_config = {
        "model": {},
        "run": {"objective_options": {"cost_class": {"monetary": 1}}},
        **node_techs,
    }
    model = calliope.Model(config=model_config, timeseries_dataframes=_ts_dfs())
    model.build_math_documentation()
    if write:
        write_math(model, "math.rst")
    return model


def generate_storage_inter_cluster_math_model(write: bool = True):
    node_techs = generate_node_techs()
    model_config = {
        "model": {
            "custom_math": ["storage_inter_cluster"],
            "time": {
                "function": "apply_clustering",
                "function_options": {
                    "clustering_func": "kmeans",
                    "how": "mean",
                    "k": 1,
                },
            },
        },
        "run": {"objective_options": {"cost_class": {"monetary": 1}}},
        **node_techs,
    }
    base_model = generate_base_math_model(write=False)
    model = calliope.Model(config=model_config, timeseries_dataframes=_ts_dfs())
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
    model.build_math_documentation()
    for key in expr_del:
        model.math_documentation._dataset[key].attrs["math_string"] = ""
    for key in full_del:
        del model.math_documentation._dataset[key]

    if write:
        write_math(model, "math_storage_inter_cluster.rst")
    return model


def generate_node_techs() -> dict[str, dict]:
    """
    To generate the written mathematical formulation of all possible base constraints, we first create a dummy model that has all the relevant technology groups defining all their allowed parameters defined.

    Parameters that can be defined over a timeseries are forced to be defined over a timeseries. Accordingly, the parameters will have "timesteps" in their dimensions in the formulation.
    """
    defaults = calliope.AttrDict.from_yaml(
        BASEPATH / ".." / ".." / "calliope" / "config" / "defaults.yaml"
    )

    ts = pd.DataFrame(
        [1, 1, 1],
        index=pd.date_range("2005-01-01 00:00", "2005-01-01 02:00", freq="H"),
        columns=["A"],
    )

    allowed_: dict[str, dict] = {i: {"all": set()} for i in ["costs", "constraints"]}

    dummy_techs = {
        "demand_tech": {
            "essentials": {"parent": "demand", "carrier": "electricity"},
            "constraints": {"resource": "df=ts_neg"},
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
                k: _add_data(k, v, defaults)
                for k, v in defaults.techs.default_tech.constraints.items()
                if k in allowed_["constraints"][tech_group]
            },
            "costs": {
                "monetary": {
                    k: _add_data(k, v, defaults)
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


def write_math(model: calliope.Model, filename: str) -> None:
    model.write_math_documentation(filename=str(BASEPATH / ".." / "_static" / filename))


def _add_data(name, default_val, defaults):
    "If timeseries is allowed, we reference timeseries data. Some parameters need hardcoded values to be returned"
    if name in defaults["model"]["file_allowed"]:
        if name == "carrier_ratios":
            return {"carrier_in.electricity": "df=ts"}
        else:
            return "df=ts"
    elif name == "export_carrier":
        return "electricity"
    elif default_val is None or name == "interest_rate":
        return 1
    else:
        return default_val


def _add_to_description(component_dict: dict, update_string: str) -> None:
    component_dict["description"] = f"{update_string}\n{component_dict['description']}"


def _ts_dfs() -> dict[str, pd.DataFrame]:
    ts = pd.DataFrame(
        [1, 1, 1],
        index=pd.date_range("2005-01-01 00:00", "2005-01-01 02:00", freq="H"),
        columns=["A"],
    )
    return {"ts": ts, "ts_neg": -1 * ts}


if __name__ == "__main__":
    generate_base_math_model(write=True)
    generate_storage_inter_cluster_math_model(write=True)
