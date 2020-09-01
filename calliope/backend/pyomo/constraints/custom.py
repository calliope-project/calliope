"""
Copyright (C) 2013-2020 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

custom.py
~~~~~~~~

Custom constraints.

"""

import logging

import numpy as np
import pyomo.core as po  # pylint: disable=import-error

from calliope.backend.pyomo.util import loc_tech_is_in, get_param, check_value

logger = logging.getLogger(__name__)

ORDER = 20  # order in which to invoke constraints relative to other constraint files
ALL_PLACEHOLDER = "__ALL__"
SETS = "sets"
ITERATE_OVER = "iterate_over"


def return_noconstraint(*args):
    logger.debug("custom constraint returned NoConstraint: {}".format(",".join(args)))
    return po.Constraint.NoConstraint


def load_constraints(backend_model):
    return None
    custom_constraints = backend_model.__calliope_custom_constraints["constraints"]
    add_transmission_techs(backend_model)
    add_sets_and_params(backend_model)

    preprocess(custom_constraints, backend_model)
    for name, config in custom_constraints.items():
        add_custom_constraint(name, config, backend_model)


def preprocess(custom_constraints, backend_model):
    merge_loc_tech_carrier(custom_constraints, backend_model)
    create_iterators(custom_constraints, backend_model)
    # TODO do something with transmission (maybe here, maybe somewhere else)
    # To elaborate, there's two issues:
    # 1. with e.g. energy cap, we should divide the impact of each transmission link by 2
    # 2. with subsets of locations, should all transmission links be included, even those
    # to locs outside the specified subset?


def add_transmission_techs(backend_model):
    for set_name, set_values in backend_model.__calliope_custom_constraints[
        "sets"
    ].items():
        if "techs" in set_values.domain:
            to_remove = []
            for tech in set_values.elements:
                if tech in backend_model.techs_transmission_names:
                    set_values.elements.extend(
                        [i for i in backend_model.techs_transmission if tech in i]
                    )
                    to_remove += tech
            set_values.set_keys(
                "elements", [i for i in set_values if i not in to_remove]
            )


def merge_loc_tech_carrier(custom_constraints, backend_model):
    def _create_add_sets(set_tuple_name, set_tuple):
        if "locs" in set_tuple and "techs" in set_tuple:
            loc_techs = set(
                [
                    f"{loc}::{tech}"
                    for loc in set_tuple["locs"]
                    for tech in set_tuple["techs"]
                ]
            )
            loc_techs = list(loc_techs.intersection(backend_model.loc_techs))
            set_string = lambda x: f"{name}_{SETS}_{set_tuple_name}_{x}"
            add_set_to_config_and_backend(
                backend_model,
                set_tuple,
                "loc_techs",
                set_string("loc_techs"),
                loc_techs,
            )

            if "carriers" in set_tuple:
                loc_tech_carriers = [
                    backend_model.lookup_loc_techs[i]
                    for i in set_tuple["loc_techs"]
                    for j in set_tuple["carriers"]
                    if j in backend_model.lookup_loc_techs[i]
                ]
                add_set_to_config_and_backend(
                    backend_model,
                    set_tuple,
                    "loc_tech_carriers",
                    set_string("loc_tech_carriers"),
                    loc_tech_carriers,
                )

    for name, config in custom_constraints.items():
        for set_tuple_name, set_tuple in config[SETS].items():
            _create_add_sets(set_tuple_name, set_tuple)
        for set_tuple_name, set_tuple in config[ITERATE_OVER].items():
            _create_add_sets(set_tuple_name, set_tuple)


def add_set_to_config_and_backend(backend_model, config, config_key, backend_key, vals):
    setattr(backend_model, backend_key, po.Set(initialize=vals, ordered=True))
    config.set_key(config_key, getattr(backend_model, backend_key))


def create_iterators(custom_constraints, backend_model):
    for constraint_name, constraint_config in custom_constraints.items():
        for set_tuple_side, set_tuples in constraint_config[SETS].items():
            # constraint_config.set_key(set_tuple_name, set_tuple)
            for iterator_tuple_name, iterator_tuple in constraint_config[
                ITERATE_OVER
            ].items():
                continue
    pass


def add_custom_constraint(name, config, backend_model):
    lhs = parse_equation_part(config["eq"]["lhs"], backend_model)
    rhs = parse_equation_part(config["eq"]["rhs"], backend_model)
    operator = config["eq"]["operator"]
    iterate_over = config[ITERATE_OVER]
    setattr(
        backend_model,
        "custom_{}_constraint".format(name),
        po.Constraint(
            *iterate_over.values(),
            rule=constraint_rule(lhs, rhs, operator, *iterate_over),
        ),
    )


# [i.name for i in get_attr(backend_model, var_or_param).index_set().set_tuple]
# for carrier_prod: ['loc_tech_carriers_prod', 'timesteps']
def add_sets_and_params(backend_model):
    for obj in ["sets", "parameters"]:
        for _name, _value in backend_model.__calliope_custom_constraints[obj].items():
            if hasattr(backend_model, _name):
                raise KeyError(f"custom {obj[:-1]} `{_name}` already in backend model.")
            if obj == "sets":
                elements = _value["elements"]
                domain = getattr(backend_model, _value["within"])
                setattr(
                    backend_model, _name, po.Set(initialize=elements, domain=domain)
                )
            elif obj == "parameters":
                setattr(
                    backend_model, _name, po.Param(initialize=_value, domain=po.Any)
                )


def parse_equation_part(eq_string, backend_model):
    pass


# def constraint_rule(lhs, rhs, operator, *iterator_keys):
#    def constraint_rule(backend_model, *iterators):
#        # TODO: how do we know the order of the iterators, to apply them to lhs & rhs?
#        if operator == "<=":
#            return lhs(*iterators) <= rhs
#        elif operator == "==":
#            return lhs == rhs
#        elif operator == ">=":
#            return lhs >= rhs
#        else:
#            raise ValueError("Invalid operator: {}".format(operator))
#    return constraint_rule
#
# def lhs(*iterators):
#    return sum(
#        backend_model.carrier_prod[iterator1, iterator2]
#        for iterator1 in iterators1
#        for iterator2 in iterators2
#    )
