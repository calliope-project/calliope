# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Schema for Calliope mathematical definition."""

from typing import Literal

from pydantic import Field

from calliope.schemas.general import AttrStr, CalliopeBaseModel, NumericVal, UniqueList


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
    active: bool = Field(default=True)
    """If False, this component will be ignored during the build phase."""


class MathIndexedComponent(MathComponent):
    """Generic indexed component class."""

    foreach: UniqueList[AttrStr] = Field(default=[])
    """Sets (a.k.a. dimensions) of the model over which the math formulation component
    will be built."""
    where: str = "True"
    """Top-level condition to determine whether the component exists in this
    optimisation problem. At all if `foreach` is not given, or for specific index items
    within the product of the sets given by `foreach`."""


class Constraint(MathIndexedComponent):
    """Schema for named constraints."""

    equations: list[ExpressionItem]
    """Constraint math equations."""
    sub_expressions: dict[AttrStr, list[ExpressionItem]] = Field(default={})
    """Constraint named sub-expressions."""
    slices: dict[AttrStr, list[ExpressionItem]] = Field(default={})
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
    equations: list[ExpressionItem]
    """Global expression math equations."""
    sub_expressions: dict[AttrStr, list[ExpressionItem]] = Field(default={})
    """Global expression named sub-expressions."""
    slices: dict[AttrStr, list[ExpressionItem]] = Field(default={})
    """Global expression named index slices."""


class Bounds(CalliopeBaseModel):
    """Bounds of decision variables.

    Either derived per-index item from a multi-dimensional input parameter, or given as
    a single value that is applied across all decision variable index items.
    """

    max: AttrStr | NumericVal
    """Decision variable upper bound, either as a reference to an input parameter or as a number."""
    min: AttrStr | NumericVal
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
    domain: Literal["real", "integer"] = Field(default="real")
    """Allowed values that the decision variable can take.
    Either real (a.k.a. continuous) or integer."""
    bounds: Bounds


class Objective(MathComponent):
    """Schema for optimisation problem objectives.

    Only one objective, the one referenced in model configuration `build.objective`
    will be activated for the optimisation problem.
    """

    equations: list[ExpressionItem]
    """Objective math equations."""
    sub_expressions: dict[AttrStr, list[ExpressionItem]] = Field(default={})
    """Objective named sub-expressions."""
    slices: dict[AttrStr, list[ExpressionItem]] = Field(default={})
    """Objective named index slices."""
    sense: Literal["minimise", "maximise", "minimize", "maximize"]
    """Whether the objective function should be minimised or maximised in the
    optimisation."""


class MathSchema(CalliopeBaseModel):
    """Mathematical definition of Calliope math.

    Contains mathematical programming components available for optimising with Calliope.
    Can contain partial definitions if they are meant to be layered on top of another.
    E.g.: layering 'plan' and 'operate' math.
    """

    model_config = {"title": "Model math schema"}

    variables: dict[AttrStr, Variable] = Field(default_factory=dict)
    """All decision variables to include in the optimisation problem."""
    global_expressions: dict[AttrStr, GlobalExpression] = Field(default_factory=dict)
    """All global expressions that can be applied to the optimisation problem."""
    constraints: dict[AttrStr, Constraint] = Field(default_factory=dict)
    """All constraints to apply to the optimisation problem."""
    piecewise_constraints: dict[AttrStr, PiecewiseConstraint] = Field(
        default_factory=dict
    )
    """All _piecewise_ constraints to apply to the optimisation problem."""
    objectives: dict[AttrStr, Objective] = Field(default_factory=dict)
    """Possible objectives to apply to the optimisation problem."""
