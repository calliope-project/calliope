"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

generate_tables.py
~~~~~~~~~~~~~~~~~~

Parses, extracts and formats default configuration data
for the documentation.

"""

import csv
import ruamel.yaml as yaml


def get_section(commented_map):
    """ Returns list of (setting, default, comment) tuples processed
    from a YAML section."""
    result = []
    for k, v in commented_map.items():
        comment = commented_map.ca.items[k][2].value.strip("#").strip()
        if "¦" in comment:
            comment = comment.split("¦")
            comment[0] = comment[0].replace("name:", "").strip()
            comment[1] = comment[1].replace("unit:", "").strip()
            comment[2] = comment[2].strip()
        else:
            comment = [comment]
        # Special case: empty dict gets turned into CommentedMap,
        # turn it back
        if isinstance(v, yaml.comments.CommentedMap):
            v = {}
        result.append((k, v, *comment))
    return result


def write_csv(filename, iterable):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(iterable)


def process():
    with open("../calliope/config/defaults.yaml", "r") as f:
        defaults = yaml.round_trip_load(f)

    write_csv(
        "./user/includes/default_essentials.csv",
        get_section(defaults["techs"]["default_tech"]["essentials"]),
    )
    write_csv(
        "./user/includes/default_constraints.csv",
        get_section(defaults["techs"]["default_tech"]["constraints"]),
    )
    write_csv(
        "./user/includes/default_costs.csv",
        get_section(defaults["techs"]["default_tech"]["costs"]["default_cost"]),
    )

    write_csv("./user/includes/model_settings.csv", get_section(defaults["model"]))
    write_csv("./user/includes/run_settings.csv", get_section(defaults["run"]))

    y = yaml.YAML()

    for tech_group in defaults["tech_groups"]:
        this_group_defaults = {
            "essentials": defaults["tech_groups"][tech_group].get("essentials", {}),
            "constraints": defaults["tech_groups"][tech_group].get("constraints", {}),
            "costs": defaults["tech_groups"][tech_group].get("costs", {}),
        }
        with open("./user/includes/basetech_{}.yaml".format(tech_group), "w") as f:
            f.write(yaml.dump(this_group_defaults, Dumper=yaml.RoundTripDumper))

        required_allowed = {
            "required_constraints": y.seq(
                defaults["tech_groups"][tech_group].get("required_constraints", [])
            ),
            "allowed_constraints": y.seq(
                defaults["tech_groups"][tech_group].get("allowed_constraints", [])
            ),
            "allowed_costs": y.seq(
                defaults["tech_groups"][tech_group].get("allowed_costs", [])
            ),
        }
        with open(
            "./user/includes/required_allowed_{}.yaml".format(tech_group), "w"
        ) as f:
            f.write(yaml.dump(required_allowed, indent=4, Dumper=yaml.RoundTripDumper))
