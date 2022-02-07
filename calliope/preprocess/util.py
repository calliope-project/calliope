"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

"""

import numpy as np

from calliope.core.attrdict import AttrDict


def concat_iterable(iterable, concatenators):
    """
    Take an iterable containing iterables of strings,
    return a list of strings concatenating inner iterables with '::'.

    E.g.:
    ``
    result = concat_iterable([('x', 'y', 'z'), ('1', '2', '3')], [':', ':'])
    result == ['x:y:z', '1:2:3']

    result = concat_iterable([('x', 'y', 'z'), ('1', '2', '3')], ['::', ':'])
    result == ['x::y:z', '1::2:3']
    ``

    """
    if len(iterable) == 0:
        return []
    concats = concatenators + [""]
    string_iter_len = len(iterable[0])
    assert all(len(i) == string_iter_len for i in iterable)
    assert string_iter_len == len(concats)

    return [
        "".join([string_iter[i] + concats[i] for i in range(string_iter_len)])
        for string_iter in iterable
    ]


def constraint_exists(model_run, loc_tech, search_entry):
    """
    Check if a cost/constraint exists for a loc_tech.

    E.g.:
    ``
    constraint_exists(model_run, 'X1::chp', 'constraints.energy_eff') will return True or False
    ``
    """
    if loc_tech in model_run.sets.loc_techs_transmission:
        loc_tech_dict = split_loc_techs_transmission(loc_tech)
        search_string = "locations.{loc_from}.links.{loc_to}.techs.{tech}.{}".format(
            search_entry, **loc_tech_dict
        )
    else:
        search_string = "locations.{0}.techs.{1}.{2}".format(
            *loc_tech.split("::"), search_entry
        )
    return model_run.get_key(search_string, None)


def get_all_carriers(config, direction="both"):
    if direction == "both":
        carrier_list = ["in", "out", "in_2", "out_2", "in_3", "out_3"]
    elif direction == "in":
        carrier_list = ["in", "in_2", "in_3"]
    elif direction == "out":
        carrier_list = ["out", "out_2", "out_3"]

    carriers = flatten_list(
        [config.get_key("carrier", "")]
        + [config.get_key("carrier_{}".format(k), "") for k in carrier_list]
    )

    return set(carriers) - set([""])


def flatten_list(unflattened_list):
    """
    Take list of iterables/non-iterables and outputs a list of non-iterables.
    """
    flattened_list = []
    for item in unflattened_list:
        if hasattr(item, "__iter__") and not isinstance(item, str):
            flattened_list.extend(item)
        else:
            flattened_list.append(item)

    return flattened_list


def split_loc_techs_transmission(transmission_string):
    """
    from loc::tech:link get out a dictionary of {loc_from:loc, loc_to:link, tech:tech}
    """
    loc, tech_link = transmission_string.split("::")
    tech, link = tech_link.split(":")

    return {"loc_from": loc, "loc_to": link, "tech": tech}


def get_systemwide_constraints(tech_config):
    if "constraints" in tech_config:
        constraints = AttrDict(
            {
                k: tech_config.constraints[k]
                for k in tech_config.constraints.keys()
                if k.endswith("_systemwide")
            }
        )
    else:
        constraints = AttrDict({})

    return constraints


def vincenty(coord1, coord2):
    """
    Vincenty's inverse method formula to calculate the distance in metres
    between two points on the surface of a spheroid (WGS84).
    modified from https://github.com/maurycyp/vincenty
    """

    a = 6378137  # equitorial radius in meters
    f = 1 / 298.257223563  # flattening from sphere to oblate spheroid
    b = a * (1 - f)  # polar radius in meters

    max_iter = 200
    thresh = 1e-12

    # short-circuit coincident points
    if coord1[0] == coord2[0] and coord1[1] == coord2[1]:
        return 0

    U1 = np.arctan((1 - f) * np.tan(np.radians(coord1[0])))
    U2 = np.arctan((1 - f) * np.tan(np.radians(coord2[0])))
    L = np.radians(coord2[1] - coord1[1])
    Lambda = L

    sinU1 = np.sin(U1)
    cosU1 = np.cos(U1)
    sinU2 = np.sin(U2)
    cosU2 = np.cos(U2)

    for iteration in range(max_iter):
        sinLambda = np.sin(Lambda)
        cosLambda = np.cos(Lambda)
        sinSigma = np.sqrt(
            (cosU2 * sinLambda) ** 2 + (cosU1 * sinU2 - sinU1 * cosU2 * cosLambda) ** 2
        )
        if sinSigma == 0:
            return 0.0  # coincident points
        cosSigma = sinU1 * sinU2 + cosU1 * cosU2 * cosLambda
        sigma = np.arctan2(sinSigma, cosSigma)
        sinAlpha = cosU1 * cosU2 * sinLambda / sinSigma
        cosSqAlpha = 1 - sinAlpha**2
        try:
            cos2SigmaM = cosSigma - 2 * sinU1 * sinU2 / cosSqAlpha
        except ZeroDivisionError:
            cos2SigmaM = 0
        C = f / 16 * cosSqAlpha * (4 + f * (4 - 3 * cosSqAlpha))
        LambdaPrev = Lambda
        Lambda = L + (1 - C) * f * sinAlpha * (
            sigma
            + C * sinSigma * (cos2SigmaM + C * cosSigma * (-1 + 2 * cos2SigmaM**2))
        )
        if abs(Lambda - LambdaPrev) < thresh:
            break  # successful convergence
    else:
        return None  # failure to converge

    uSq = cosSqAlpha * (a**2 - b**2) / (b**2)
    A = 1 + uSq / 16384 * (4096 + uSq * (-768 + uSq * (320 - 175 * uSq)))
    B = uSq / 1024 * (256 + uSq * (-128 + uSq * (74 - 47 * uSq)))
    deltaSigma = (
        B
        * sinSigma
        * (
            cos2SigmaM
            + B
            / 4
            * (
                cosSigma * (-1 + 2 * cos2SigmaM**2)
                - B
                / 6
                * cos2SigmaM
                * (-3 + 4 * sinSigma**2)
                * (-3 + 4 * cos2SigmaM**2)
            )
        )
    )
    D = b * A * (sigma - deltaSigma)

    return round(D)
