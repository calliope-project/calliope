# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Schema for Calliope mathematical definition."""

from typing import Literal

from pydantic import Field

from calliope.schemas.general import AttrStr, CalliopeBaseModel, NumericVal, UniqueList


class EquationItem(CalliopeBaseModel):
    """Equation item schema.

    Equations may define conditions (`where`) defining on which index items in the
    product of sets (`foreach`) they will be applied. Conditions must be set up such
    that a maximum of one equation can be applied per index item.
    """

    where: str | None = None
    """Condition to determine whether the accompanying expression is built.
    At all if `foreach` is not given, or for specific index items within the product
    of the sets given by `foreach`."""
    expression: str
    """Equation expression valid for this component in the form LHS OPERATOR RHS, where
    LHS and RHS are math expressions and OPERATOR is one of [==, <=, >=]."""


class SubExpressionItem(CalliopeBaseModel):
    """Sub-expression item schema.

    Math sub-expressions which are used to replace any instances in which they are
    referenced in a component's equations. They must be referenced by their
    name preceded with the "$" symbol, e.g., `foo` in `$foo == 1` or `$foo + 1`.
    """

    where: str | None = None
    """Condition to determine whether the accompanying sub-expression is built."""
    expression: str
    """Math sub-expression which can be one term or a combination of terms using the
    operators [+, -, *, /, **]."""


class SliceItem(CalliopeBaseModel):
    """Slice item schema.

    Array index slices which are used to replace any instances in which they are
    referenced in a component's equations or sub-expressions. They must be referenced
    by their name preceded with the "$" symbol, e.g., `foo` in `flow_out_eff[techs=$foo]`.
    """

    where: str | None = None
    """Condition to determine whether the accompanying index slice is built."""
    expression: str
    """Index slice expression, such as a list of set items or a call to a helper
    function."""


class MathComponent(CalliopeBaseModel):
    """Generic math component class."""

    title: str | None = None
    """The component long name, for use in visualisation."""
    description: str | None = None
    """A verbose description of the component."""
    active: bool = Field(default=True)
    """If False, this component will be ignored entirely at the optimisation problem
    build phase."""


class MathIndexedComponent(MathComponent):
    """Generic indexed component class."""

    foreach: UniqueList[AttrStr] | None = None
    """Sets (a.k.a. dimensions) of the model over which the math formulation component
    will be built."""
    where: str | None = None
    """Top-level condition to determine whether the component exists in this
    optimisation problem. At all if `foreach` is not given, or for specific index items
    within the product of the sets given by `foreach`."""


class Constraint(MathIndexedComponent):
    """Schema for named constraints."""

    equations: list[EquationItem]
    """Constraint math equations."""
    sub_expressions: dict[AttrStr, list[SubExpressionItem]] | None = None
    """Constraint named sub-expressions."""
    slices: dict[AttrStr, list[SliceItem]] | None = None
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

    unit: str | None = None
    """Generalised unit of the component (e.g., length, time, quantity_per_hour, ...)."""
    default: NumericVal | None = None
    """If set, will be the default value for the expression."""
    equations: list[EquationItem]
    """Global expression math equations."""
    sub_expressions: dict[AttrStr, list[SubExpressionItem]] | None = None
    """Global expression named sub-expressions."""
    slices: dict[AttrStr, list[SliceItem]] | None = None
    """Global expression named index slices."""


class Bounds(CalliopeBaseModel):
    """Bounds of decision variables.

    Either derived per-index item from a multi-dimensional input parameter, or given as
    a single value that is applied across all decision variable index items.
    """

    max: AttrStr | NumericVal | None = None
    """Decision variable upper bound, either as a reference to an input parameter or as a number."""
    min: AttrStr | NumericVal | None = None
    """Decision variable lower bound, either as a reference to an input parameter or as a number."""


class Variable(MathIndexedComponent):
    """Schema for optimisation problem variables.

    A decision variable must be referenced in at least one constraint or in the
    objective for it to exist in the optimisation problem that is sent to the solver.
    """

    unit: str | None = None
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

    equations: list[EquationItem]
    """Objective math equations."""
    sub_expressions: dict[AttrStr, list[SubExpressionItem]] | None = None
    """Objective named sub-expressions."""
    slices: dict[AttrStr, list[SliceItem]] | None = None
    """Objective named index slices."""
    sense: Literal["minimise", "maximise", "minimize", "maximize"]
    """Whether the objective function should be minimised or maximised in the
    optimisation."""


class CalliopeMathDef(CalliopeBaseModel):
    """Calliope mathematical definition.

    All options available to formulate math to use in solving an optimisation problem
    with Calliope.
    """

    model_config = {"title": "Model math schema"}

    constraints: dict[AttrStr, Constraint]
    """All constraints to apply to the optimisation problem."""
    piecewise_constraints: dict[AttrStr, PiecewiseConstraint] | None = None
    """All _piecewise_ constraints to apply to the optimisation problem."""
    global_expressions: dict[AttrStr, GlobalExpression] | None = None
    """All global expressions that can be applied to the optimisation problem."""
    variables: dict[AttrStr, Variable]
    """All decision variables to include in the optimisation problem."""
    objectives: dict[AttrStr, Objective]
    """Possible objectives to apply to the optimisation problem."""
