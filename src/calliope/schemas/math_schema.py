# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Schema for Calliope mathematical definition."""

from typing import Literal

from pydantic import Field, model_validator

from calliope.schemas.general import (
    AttrStr,
    CalliopeBaseModel,
    CalliopeDictModel,
    CalliopeListModel,
    NumericVal,
    UniqueList,
)


class ExpressionItem(CalliopeBaseModel):
    """Schema for equations, subexpressions and slices."""

    where: str = "True"
    """Condition to determine whether the accompanying expression is built."""
    expression: str
    """Expression for this component.
    - Equations: LHS OPERATOR RHS, where LHS and RHS are math expressions and OPERATOR is one of [==, <=, >=].
    - Subexpressions: be one term or a combination of terms using the operators [+, -, *, /, **].
    - Slices: a list of set items or a call to a helper function.
    """


class MathComponent(CalliopeBaseModel):
    """Generic math component class."""

    title: str = ""
    """The component long name, for use in visualisation."""
    description: str = ""
    """A verbose description of the component."""
    active: bool = True
    """If False, this component will be ignored during the build phase."""


class MathIndexedComponent(MathComponent):
    """Generic indexed component class."""

    foreach: UniqueList[AttrStr] = Field(default_factory=list)
    """Sets (a.k.a. dimensions) of the model over which the math formulation component
    will be built."""
    where: str = "True"
    """Top-level condition to determine whether the component exists in this
    optimisation problem. At all if `foreach` is not given, or for specific index items
    within the product of the sets given by `foreach`."""


class Equations(CalliopeListModel):
    """List of equations that can be updated when a parent pydantic model is updated."""

    root: list[ExpressionItem] = Field(default_factory=list)


class SubExpressions(CalliopeDictModel):
    """Dictionary of sub-expressions that can be updated when a parent pydantic model is updated."""

    root: dict[AttrStr, Equations] = Field(default_factory=dict)


class Constraint(MathIndexedComponent):
    """Schema for named constraints."""

    equations: Equations = Equations()
    """Constraint math equations."""
    sub_expressions: SubExpressions = SubExpressions()
    """Constraint named sub-expressions."""
    slices: SubExpressions = SubExpressions()
    """Constraint named index slices."""


class PiecewiseConstraint(MathIndexedComponent):
    """Schema for named piece-wise constraints.

    These link an `x`-axis decision variable with a `y`-axis decision variable with
    values at specified breakpoints.
    """

    x_expression: str
    """X variable name whose values are assigned at each breakpoint."""
    y_expression: str
    """Y variable name whose values are assigned at each breakpoint."""
    x_values: str
    """X parameter name containing data, indexed over the `breakpoints` dimension."""
    y_values: str
    """Y parameter name containing data, indexed over the `breakpoints` dimension."""


class GlobalExpression(MathIndexedComponent):
    """Schema for named global expressions.

    Can be used to combine parameters and variables and then used in one or more
    expressions elsewhere in the math formulation (i.e., in constraints, objectives,
    and other global expressions).

    NOTE: If expecting to use global expression `A` in global expression `B`, `A` must
    be defined above `B`.
    """

    unit: str = ""
    """Generalised unit of the component (e.g., length, time, quantity_per_hour, ...)."""
    default: NumericVal | None = None
    """If set, will be the default value for the expression."""
    equations: Equations = Equations()
    """Global expression math equations."""
    sub_expressions: SubExpressions = SubExpressions()
    """Global expression named sub-expressions."""
    slices: SubExpressions = SubExpressions()
    """Global expression named index slices."""
    order: int
    """Order in which to apply this global expression relative to all others, if different to its definition order."""


class Bounds(CalliopeBaseModel):
    """Bounds of decision variables.

    Either derived per-index item from a multi-dimensional input parameter, or given as
    a single value that is applied across all decision variable index items.
    """

    max: AttrStr | NumericVal = float("inf")
    """Decision variable upper bound, either as a reference to an input parameter or as a number."""
    min: AttrStr | NumericVal = float("-inf")
    """Decision variable lower bound, either as a reference to an input parameter or as a number."""


class Variable(MathIndexedComponent):
    """Schema for optimisation problem variables.

    A decision variable must be referenced in at least one constraint or in the
    objective for it to exist in the optimisation problem that is sent to the solver.
    """

    unit: str = ""
    """Generalised unit of the component (e.g., length, time, quantity_per_hour, ...)."""
    default: NumericVal | None = None
    """If set, will be the default value for the variable."""
    domain: Literal["real", "integer"] = "real"
    """Allowed values that the decision variable can take.
    Either real (a.k.a. continuous) or integer."""
    bounds: Bounds = Bounds()


class Objective(MathComponent):
    """Schema for optimisation problem objectives.

    Only one objective, the one referenced in model configuration `build.objective`
    will be activated for the optimisation problem.
    """

    equations: Equations
    """Objective math equations."""
    sub_expressions: SubExpressions = SubExpressions()
    """Objective named sub-expressions."""
    slices: SubExpressions = SubExpressions()
    """Objective named index slices."""
    sense: Literal["minimise", "maximise", "minimize", "maximize"]
    """Whether the objective function should be minimised or maximised in the
    optimisation."""
    foreach: UniqueList[AttrStr] = Field(default_factory=list, frozen=True)
    """Objectives are always adimensional."""


class Variables(CalliopeDictModel):
    """Calliope model variables dictionary."""

    root: dict[AttrStr, Variable] = Field(default_factory=dict)


class GlobalExpressions(CalliopeDictModel):
    """Calliope model global_expressions dictionary."""

    root: dict[AttrStr, GlobalExpression] = Field(default_factory=dict)


class Constraints(CalliopeDictModel):
    """Calliope model constraints dictionary."""

    root: dict[AttrStr, Constraint] = Field(default_factory=dict)


class PiecewiseConstraints(CalliopeDictModel):
    """Calliope model piecewise_constraints dictionary."""

    root: dict[AttrStr, PiecewiseConstraint] = Field(default_factory=dict)


class Objectives(CalliopeDictModel):
    """Calliope model objectives dictionary."""

    root: dict[AttrStr, Objective] = Field(default_factory=dict)


class MathSchema(CalliopeBaseModel):
    """Mathematical definition of Calliope math.

    Contains mathematical programming components available for optimising with Calliope.
    Can contain partial definitions if they are meant to be layered on top of another.
    E.g.: layering 'plan' and 'operate' math.
    """

    model_config = {"title": "Model math schema"}

    variables: Variables = Variables()
    """All decision variables to include in the optimisation problem."""
    global_expressions: GlobalExpressions = GlobalExpressions()
    """All global expressions that can be applied to the optimisation problem."""
    constraints: Constraints = Constraints()
    """All constraints to apply to the optimisation problem."""
    piecewise_constraints: PiecewiseConstraints = PiecewiseConstraints()
    """All _piecewise_ constraints to apply to the optimisation problem."""
    objectives: Objectives = Objectives()
    """Possible objectives to apply to the optimisation problem."""

    @model_validator(mode="before")
    @classmethod
    def set_expr_order(cls, data: dict) -> dict:
        """Set the position of the global expression in the order of application.

        Args:
            data (dict): Raw global expression data dictionary.

        Returns:
            dict: `data` with the `order` field set even if not given in the input.
        """
        grp = data.get("global_expressions", {})
        if not isinstance(grp, dict):
            return data
        for default_order, key in enumerate(grp.keys()):
            grp[key]["order"] = grp[key].get("order", default_order)
        return data


class CalliopeInputMath(CalliopeDictModel):
    """Calliope input math dictionary."""

    root: dict[AttrStr, CalliopeDictModel] = Field(default_factory=dict)


class CalliopeMath(CalliopeBaseModel):
    """Calliope math attribute container."""

    init: CalliopeInputMath = CalliopeInputMath()
    build: MathSchema = MathSchema()
