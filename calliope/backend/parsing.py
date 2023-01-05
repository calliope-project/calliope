import itertools

import pyparsing as pp

from calliope.backend import equation_parser


def process_constraint(constraints, constraint_name):
    constraint = constraints[constraint_name]
    constraint_equations = parse_constraint_equations(constraint, constraint_name)
    return {"name": constraint_name, "equations": constraint_equations}


def parse_constraint_equations(constraint, constraint_name):
    """
    Builds the equations for a given constraint.

    Returns
    -------
    expanded_equations : list of equation dicts

    """

    set_iterators = [i[0] for i in parse_foreach(constraint["foreach"])]
    parser = equation_parser.setup_parser(set_iterators=set_iterators)

    # Parse all equations and add their parsed_expression to the equations dict
    equations = constraint.get("equations", [])
    for i in range(len(equations)):
        equations[i]["expression"] = parser.parse_string(equations[i]["expression"])

    # Parse all components and add their parsed_expression to the components dict
    components = constraint.get("components", [])
    for k in components.keys():
        for i in range(len(components[k])):
            components[k][i]["expression"] = parser.parse_string(
                # Make sure it is a string for parsing,
                # in cases like a single int or float
                str(components[k][i]["expression"])
            )
            components[k][i]["name"] = k

    # For each equation, check what components it contains, build the product
    # of components, and collate the resulting component-free equations in a list
    expanded_equations = []
    for eq_index, equation in enumerate(equations):

        # Get all components in the equation
        eq_components = find_and_replace_components(
            equation["expression"], find_only=True
        )

        # Get variations of components
        eq_component_names = [i.replace("COMPONENT:", "") for i in eq_components]
        eq_component_variations = list(
            itertools.product(*[components[k] for k in eq_component_names])
        )

        # Generate a list of equations for all component variations,
        # replacing the components with their parsed expressions, and
        # combining the where lists of the components with that of the base equation
        equation_variants = get_variant_equations(
            equation,
            eq_component_variations,
            constraint["foreach"],
            base_name=constraint_name,
            name_top_index=eq_index,
        )

        expanded_equations.extend(equation_variants)

    # FIXME: also deal with index_items
    # index_items = constraint.get("index_items", [])

    return expanded_equations


def find_and_replace_components(parse_results, find_only=False):
    """
    Recursively find and replace components
    """
    components = []
    for item in parse_results:
        if isinstance(item, equation_parser.EvalComponent):
            components.append(str(item))
            if not find_only:
                pass
                # FIXME: do the replacement

        elif hasattr(item, "value") and hasattr(item.value, "len"):
            components.extend(find_and_replace_components(item.value))

    return set(components)


def get_variant_equations(
    equation, eq_component_variations, foreach, base_name, name_top_index
):

    new_equations = []
    eq_index = 0
    for component_var in eq_component_variations:
        eq_expression = equation["expression"].copy()[0].value

        component_wheres = []
        for component in component_var:
            component_name = component["name"]

            # Find list of indexes where the component shows up in the equation expression
            eq_expression_strings = [str(i) for i in eq_expression.as_list()]
            indexes = [
                i
                for i in range(len(eq_expression_strings))
                if eq_expression_strings[i] == f"COMPONENT:{component_name}"
            ]

            # Replace all components at their indexes with the component expression
            for i in indexes:
                eq_expression[i] = component["expression"][0]

            # Gather the per-component where lists
            component_wheres += component.get("where", [])

        # Build the combined "where" expression
        where = equation.get("where", []) + ["and"] + component_wheres

        new_equations.append(
            {
                "name": f"{base_name}_{name_top_index}_{eq_index}",
                "where": where,
                "foreach": foreach,
                "expression": eq_expression,
            }
        )

        eq_index += 1

    return new_equations


def parse_foreach(foreach):
    object_ = pp.Word(pp.alphas + "_", pp.alphanums + "_")
    dim_expr = object_ + pp.Suppress("in") + object_
    dim_expr.setParseAction(lambda t: [t[0], t[1]])
    return [dim_expr.parse_string(i) for i in foreach]
