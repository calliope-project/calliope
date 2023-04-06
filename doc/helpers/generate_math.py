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


def generate_math():
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

    allowed_ = {i: {"all": set()} for i in ["costs", "constraints"]}

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

    model_config = {
        "model": {},
        "run": {"objective_options": {"cost_class": {"monetary": 1}}},
        # Hardcoding just one expected per-node parameter: available area
        "nodes": {
            "A": {"techs": {k: None for k in dummy_techs.keys()}, "available_area": 1}
        },
        "techs": dummy_techs,
    }
    m = calliope.Model(
        config=model_config, timeseries_dataframes={"ts": ts, "ts_neg": -1 * ts}
    )
    m.write_math_documentation(filename=BASEPATH / ".." / "_static" / "math.rst")


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


if __name__ == "__main__":
    generate_math()
