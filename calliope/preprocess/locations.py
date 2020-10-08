"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

preprocess_locations.py
~~~~~~~~~~~~~~~~~~~~~~~

Functions to deal with locations and their configuration.

"""

import math

from calliope.exceptions import ModelError, warn
from calliope.core.attrdict import AttrDict
from calliope.preprocess.util import vincenty
from calliope.preprocess.checks import DEFAULTS, POSSIBLE_COSTS


def process_locations(model_config, modelrun_techs):
    """
    Process locations by taking an AttrDict that may include compact keys
    such as ``1,2,3``, and returning an AttrDict with:

    * exactly one key per location with all of its settings
    * fully resolved installed technologies for each location
    * fully expanded transmission links for each location

    Parameters
    ----------
    model_config : AttrDict
    modelrun_techs : AttrDict

    Returns
    -------
    locations : AttrDict
    locations_comments : AttrDict

    """
    techs_in = model_config.techs.copy()
    tech_groups_in = model_config.tech_groups
    locations_in = model_config.locations
    links_in = model_config.get("links", AttrDict())

    allowed_from_file = DEFAULTS.model.file_allowed

    warnings = []
    errors = []
    locations_comments = AttrDict()

    ##
    # Expand compressed `loc1,loc2,loc3,loc4: ...` definitions
    ##
    locations = AttrDict()
    for key in locations_in:
        if ("--" in key) or ("," in key):
            key_locs = explode_locations(key)
            for subkey in key_locs:
                _set_loc_key(locations, subkey, locations_in[key])
        else:
            _set_loc_key(locations, key, locations_in[key])

    ##
    # Kill any locations that the modeller does not want to exist
    ##
    for loc in list(locations.keys()):
        if not locations[loc].get("exists", True):
            locations.del_key(loc)

    ##
    # Process technologies
    ##
    techs_to_delete = []
    for tech_name in techs_in:
        if not techs_in[tech_name].get("exists", True):
            techs_to_delete.append(tech_name)
            continue
        # Get inheritance chain generated in process_techs()
        inheritance_chain = modelrun_techs[tech_name].inheritance

        # Get and save list of required_constraints from base technology
        base_tech = inheritance_chain[-1]
        rq = model_config.tech_groups[base_tech].required_constraints
        # locations[loc_name].techs[tech_name].required_constraints = rq
        techs_in[tech_name].required_constraints = rq

    # Kill any techs that the modeller does not want to exist
    for tech_name in techs_to_delete:
        del techs_in[tech_name]

    ##
    # Fully expand all installed technologies for the location,
    # filling in any undefined parameters from defaults
    ##
    location_techs_to_delete = []

    for loc_name, loc in locations.items():

        if "techs" not in loc:
            # Mark this as a transmission-only node if it has not allowed
            # any technologies
            locations[loc_name].transmission_node = True
            locations_comments.set_key(
                "{}.transmission_node".format(loc_name),
                "Automatically inserted: specifies that this node is "
                "a transmission-only node.",
            )
            continue  # No need to process any technologies at this node

        for tech_name in loc.techs:
            if tech_name in techs_to_delete:
                # Techs that were removed need not be further considered
                continue

            if not isinstance(locations[loc_name].techs[tech_name], dict):
                locations[loc_name].techs[tech_name] = AttrDict()

            # Starting at top of the inheritance chain, for each level,
            # check if the level has location-specific group settings
            # and keep merging together the settings, overwriting as we
            # go along.
            tech_settings = AttrDict()
            for parent in reversed(modelrun_techs[tech_name].inheritance):
                # Does the parent group have model-wide settings?
                tech_settings.union(tech_groups_in[parent], allow_override=True)
                # Does the parent group have location-specific settings?
                if (
                    "tech_groups" in locations[loc_name]
                    and parent in locations[loc_name].tech_groups
                ):
                    tech_settings.union(
                        locations[loc_name].tech_groups[parent], allow_override=True
                    )
            # Now overwrite with the tech's own model-wide
            # and location-specific settings
            tech_settings.union(techs_in[tech_name], allow_override=True)
            if tech_name in locations[loc_name].techs:
                tech_settings.union(
                    locations[loc_name].techs[tech_name], allow_override=True
                )

            tech_settings = cleanup_undesired_keys(tech_settings)

            # Resolve columns in filename if necessary
            file_or_df_configs = [
                i
                for i in tech_settings.keys_nested()
                if (
                    isinstance(tech_settings.get_key(i), str)
                    and (
                        "file=" in tech_settings.get_key(i)
                        or "df=" in tech_settings.get_key(i)
                    )
                )
            ]
            for config_key in file_or_df_configs:
                config_value = tech_settings.get_key(config_key, "")
                if ":" not in config_value:
                    config_value = "{}:{}".format(config_value, loc_name)
                    tech_settings.set_key(config_key, config_value)

            tech_settings = check_costs_and_compute_depreciation_rates(
                tech_name, loc_name, tech_settings, warnings, errors
            )

            # Now merge the tech settings into the location-specific
            # tech dict -- but if a tech specifies ``exists: false``,
            # we kill it at this location
            if not tech_settings.get("exists", True):
                location_techs_to_delete.append(
                    "{}.techs.{}".format(loc_name, tech_name)
                )
            else:
                locations[loc_name].techs[tech_name].union(
                    tech_settings, allow_override=True
                )

    for k in location_techs_to_delete:
        locations.del_key(k)

    # Generate all transmission links
    processed_links = AttrDict()
    for link in links_in:
        loc_from, loc_to = [i.strip() for i in link.split(",")]
        # Skip this link entirely if it has been told not to exist
        if not links_in[link].get("exists", True):
            continue
        # Also skip this link - and warn about it - if it links to a
        # now-inexistant (because removed) location
        if loc_from not in locations.keys() or loc_to not in locations.keys():
            warnings.append(
                "Not building the link {},{} because one or both of its "
                "locations have been removed from the model by setting "
                "``exists: false``".format(loc_from, loc_to)
            )
            continue
        processed_transmission_techs = AttrDict()
        for tech_name in links_in[link].techs:
            # Skip techs that have been told not to exist
            # for this particular link
            if not links_in[link].get_key("techs.{}.exists".format(tech_name), True):
                continue
            if tech_name not in processed_transmission_techs:
                tech_settings = AttrDict()
                # Combine model-wide settings from all parent groups
                for parent in reversed(modelrun_techs[tech_name].inheritance):
                    tech_settings.union(tech_groups_in[parent], allow_override=True)
                # Now overwrite with the tech's own model-wide settings
                tech_settings.union(techs_in[tech_name], allow_override=True)

                # Add link-specific constraint overrides
                if links_in[link].techs[tech_name]:
                    tech_settings.union(
                        links_in[link].techs[tech_name], allow_override=True
                    )

                tech_settings = cleanup_undesired_keys(tech_settings)

                tech_settings = process_per_distance_constraints(
                    tech_name,
                    tech_settings,
                    locations,
                    locations_comments,
                    loc_from,
                    loc_to,
                )
                tech_settings = check_costs_and_compute_depreciation_rates(
                    tech_name, link, tech_settings, warnings, errors
                )
                processed_transmission_techs[tech_name] = tech_settings
            else:
                tech_settings = processed_transmission_techs[tech_name]

            processed_links.set_key(
                "{}.links.{}.techs.{}".format(loc_from, loc_to, tech_name),
                tech_settings.copy(),
            )

            processed_links.set_key(
                "{}.links.{}.techs.{}".format(loc_to, loc_from, tech_name),
                tech_settings.copy(),
            )

            # If this is a one-way link, we set the constraints for energy_prod
            # and energy_con accordingly on both parts of the link
            if tech_settings.get_key("constraints.one_way", False):
                processed_links.set_key(
                    "{}.links.{}.techs.{}.constraints.energy_prod".format(
                        loc_from, loc_to, tech_name
                    ),
                    False,
                )
                processed_links.set_key(
                    "{}.links.{}.techs.{}.constraints.energy_con".format(
                        loc_to, loc_from, tech_name
                    ),
                    False,
                )
    locations.union(processed_links, allow_override=True)

    return locations, locations_comments, list(set(warnings)), list(set(errors))


def explode_locations(k):
    """
    Expands the given key ``k``. ``k``s of the form ``'1--3'`` or
    ``'1,2,3'`` are both expanded into the list ``['1', '2', '3']``.

    Can deal with any combination, e.g. ``'1--3,6,9--11,a'`` results in::

        ['1', '2', '3', '6', '9', '10', '11', 'a']

    Always returns a list, even if ``k`` is just a simple key,
    i.e. ``explode_locations('1')`` returns ``['1']``.

    """
    # Ensure sure we don't pass in other things
    assert isinstance(k, str)
    finalkeys = []
    subkeys = k.split(",")
    for sk in subkeys:
        if "--" in sk:
            begin, end = sk.split("--")
            finalkeys += [str(i).strip() for i in range(int(begin), int(end) + 1)]
        else:
            finalkeys += [sk.strip()]
    if finalkeys == [] or finalkeys == [""]:
        raise KeyError("Empty key")
    return finalkeys


def _set_loc_key(d, k, value):
    """Set key ``k`` in ``d`` to ``value```."""
    if not value:
        return None
    if k in d:
        try:
            d[k].union(value.copy())
        except KeyError as e:
            raise KeyError("Problem at location {}: {}".format(k, str(e)))
    else:
        d[k] = value.copy()


def cleanup_undesired_keys(tech_settings):
    # These keys are removed from the constructed `locations` dict after
    # merging across the inheritance chain, as they are contained in the
    # `techs` dict
    # These are dealt with in process_techs(),
    # we do not want them here
    keys_to_kill = [
        "essentials",
        "allowed_constraints",
        "allowed_costs",
        "required_constraints",
        "allowed_group_constraints",
    ]
    for k in keys_to_kill:
        try:
            del tech_settings[k]
        except KeyError:
            pass

    # We also remove any system-wide constraints here,
    # as they should not be accidentally read from or
    # changed in per-location settings later
    # FIXME: Raise warning that these constraints are deleted?
    system_wide_keys = [
        k for k in tech_settings.constraints.keys() if k.endswith("_systemwide")
    ]
    for k in system_wide_keys:
        del tech_settings.constraints[k]

    return tech_settings


def process_per_distance_constraints(
    tech_name, tech_settings, locations, locations_comments, loc_from, loc_to
):
    # Process distance, if any per_distance constraints exist
    if any("per_distance" in i for i in tech_settings.keys_nested(subkeys_as="list")):
        # If no distance was given, we calculate it from coordinates
        if "distance" not in tech_settings:
            # Simple check - earlier sense-checking already ensures
            # that all locations have either lat/lon or x/y coords
            loc1 = locations[loc_from].coordinates
            loc2 = locations[loc_to].coordinates
            if "lat" in locations[loc_from].coordinates:
                distance = vincenty([loc1.lat, loc1.lon], [loc2.lat, loc2.lon])
            else:
                distance = math.sqrt((loc1.x - loc2.x) ** 2 + (loc1.y - loc2.y) ** 2)

            tech_settings.distance = distance
            locations_comments.set_key(
                "{}.links.{}.techs.{}.distance".format(loc_from, loc_to, tech_name),
                "Distance automatically computed from coordinates",
            )

        # Add per-distance values to their not-per-distance cousins
        # FIXME these are hardcoded for now
        if "energy_eff_per_distance" in tech_settings.constraints:
            distance_energy_eff = (
                tech_settings.constraints.energy_eff_per_distance
                ** tech_settings.distance
            )
            tech_settings.constraints.energy_eff = (
                tech_settings.constraints.get_key("energy_eff", 1.0)
                * distance_energy_eff
            )
            del tech_settings.constraints["energy_eff_per_distance"]
            locations_comments.set_key(
                "{}.links.{}.techs.{}.constraints.energy_eff".format(
                    loc_from, loc_to, tech_name
                ),
                "Includes value computed from energy_eff_per_distance",
            )

        for k in tech_settings.get("costs", AttrDict()).keys_nested(subkeys_as="list"):
            if "energy_cap_per_distance" in k:
                energy_cap_costs_per_distance = (
                    tech_settings.costs.get_key(k) * tech_settings.distance
                )
                tech_settings.costs[k.split(".")[0]].energy_cap = (
                    tech_settings.costs[k.split(".")[0]].get_key("energy_cap", 0)
                    + energy_cap_costs_per_distance
                )
                tech_settings.costs.del_key(k)
                locations_comments.set_key(
                    "{}.links.{}.techs.{}.costs.{}".format(
                        loc_from, loc_to, tech_name, k
                    ),
                    "Includes value computed from energy_cap_per_distance",
                )
            elif "purchase_per_distance" in k:
                purchase_costs_per_distance = (
                    tech_settings.costs.get_key(k) * tech_settings.distance
                )
                tech_settings.costs[k.split(".")[0]].purchase = (
                    tech_settings.costs[k.split(".")[0]].get_key("purchase", 0)
                    + purchase_costs_per_distance
                )
                tech_settings.costs.del_key(k)
                locations_comments.set_key(
                    "{}.links.{}.techs.{}.costs.{}".format(
                        loc_from, loc_to, tech_name, k
                    ),
                    "Includes value computed from purchase_per_distance",
                )

    return tech_settings


def check_costs_and_compute_depreciation_rates(
    tech_id, loc_or_link, tech_config, warnings, errors
):
    cost_classes = list(tech_config.get("costs", {}).keys())
    for cost in cost_classes:

        # Warning if a cost is defined without a cost class, which is probably a mistake
        if cost in POSSIBLE_COSTS:
            warnings.append(
                "`{}` at `{}` defines {} as a cost class. "
                "This is probably an indentation mistake.".format(
                    tech_id, loc_or_link, cost
                )
            )

        # Warning if a cost class is empty
        if not isinstance(tech_config.costs[cost], dict):
            warnings.append(
                "Deleting empty cost class `{}` for technology `{}` at `{}`.".format(
                    cost, tech_id, loc_or_link
                )
            )
            tech_config.costs.del_key(cost)
            # If the cost class is empty and deleted, the rest of the loop
            # is skipped entirely (as it expects the cost class to exist)
            continue

        plant_life = tech_config.constraints.get_key("lifetime", 0)
        interest = tech_config.costs[cost].get_key("interest_rate", None)

        # Only need depreciation rate for technologies that have investment costs
        if any(
            any(j in i for j in ["_area", "_cap", "purchase"])
            for i in tech_config.costs[cost].keys()
        ):
            # MUST define lifetime and interest_rate for these technologies
            if plant_life == 0 or interest is None:
                errors.append(
                    "Must specify constraints.lifetime and costs.{0}.interest_rate "
                    "when specifying fixed `{0}` costs for `{1}`. Set lifetime to 1 "
                    "and interest rate to 0 if you do not want them to have an effect".format(
                        cost, tech_id
                    )
                )
                continue
            # interest rate = 0 -> simple depreciation
            if interest == 0:
                warnings.append(
                    "`{}` interest rate of zero for technology {}, setting "
                    "depreciation rate as 1/lifetime.".format(cost, tech_id)
                )
                dep = 1 / plant_life
            # interest rate > 0 -> annualised depreciation
            else:
                dep = (interest * ((1 + interest) ** plant_life)) / (
                    ((1 + interest) ** plant_life) - 1
                )

            if math.isnan(dep) or dep == 0:
                warnings.append(
                    "No investment {0} cost will be incurred for `{1}` as "
                    "depreciation rate is 0 or NaN. Probably caused by "
                    "inifinte plant life".format(cost, tech_id)
                )
                dep = 0

            tech_config.costs[cost]["depreciation_rate"] = dep
        try:
            tech_config.costs[cost].del_key("interest_rate")
        except KeyError:
            pass
        # If, by deleting 'interest_rate', we end up with an empty dict, delete the cost class
        if len(tech_config.costs[cost].keys()) == 0:
            tech_config.costs.del_key(cost)
    # If, by deleting the cost class, we end up with an empty dict, delete the cost key
    if len(cost_classes) > 0 and len(tech_config.costs.keys()) == 0:
        tech_config.del_key("costs")

    return tech_config
