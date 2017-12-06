import pyomo.core as po  # pylint: disable=import-error

from calliope.core.util.tools import load_function


def generate_model(model_data):
    """
    Generate a Pyomo model.

    """
    backend_model = po.ConcreteModel()

    # Sets
    for coord in list(model_data.coords):
        setattr(
            model_data, coord,
            po.Set(initialize=list(model_data.coords[coord].data), ordered=True)
        )

    # "Parameters"
    model_data_dict = {
        'data': {k: model_data[k].to_series().to_dict() for k in model_data.data_vars},
        'dims': {k: model_data[k].dims for k in model_data.data_vars},
        'sets': list(model_data.coords)
    }
    # FIXME must ensure here that dims are in the right order
    backend_model.__calliope_model_data__ = model_data_dict

    # Variables
    load_function(
        'calliope.backend.pyomo.variables.initialize_decision_variables'
    )(backend_model)

    # Constraints
    constraints_to_add = [
        'energy_balance.load_energy_balance_constraints',
        # 'base.unit_commitment',
        # 'base.node_energy_balance',
        # 'base.node_constraints_build',
        # 'base.node_constraints_operational',
        # 'base.node_constraints_transmission',
        # 'base.node_costs',
        # 'base.model_constraints',
        # 'planning.system_margin',
        # 'planning.node_constraints_build_total'
    ]

    for c in constraints_to_add:
        load_function(
            'calliope.backend.pyomo.constraints.' + c
        )(backend_model)

    # FIXME: Optional constraints
    # optional_constraints = model_data.attrs['constraints']
    # if optional_constraints:
    #     for c in optional_constraints:
    #         self.add_constraint(load_function(c))

    # Objective function
    objective_name = model_data.attrs['model.objective']
    objective_function = 'calliope.backend.pyomo.objective.' + objective_name
    load_function(objective_function)(backend_model)

    # delattr(backend_model, '__calliope_model_data__')

    return backend_model
