
import pyomo.core as po


def create_expressions(backend_model):
    if hasattr(backend_model, "system_balance_constraint_index"):
        backend_model.system_balance = po.Expression(
            backend_model.system_balance_constraint_index, initialize=0.0,
        )

    if hasattr(backend_model, "balance_demand_constraint_index"):
        backend_model.required_resource = po.Expression(
            backend_model.balance_demand_constraint_index,
            initialize=0.0,
        )

    if hasattr(backend_model, "cost_investment_index"):
        # Right-hand side expression can be updated by MILP investment costs
        backend_model.cost_investment_rhs = po.Expression(
            backend_model.cost_investment_index,
            initialize=0.0,
        )

    if hasattr(backend_model, "cost_var_index"):
        # Right-hand side expression can be updated by export costs/revenue
        backend_model.cost_var_rhs = po.Expression(
            backend_model.cost_var_index,
            initialize=0.0,
        )
