# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Schema for Calliope mathematical definition."""

from collections.abc import Iterable
from functools import cached_property
from typing import ClassVar, Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from calliope.schemas.general import (
    AttrStr,
    CalliopeBaseModel,
    CalliopeDictModel,
    CalliopeListModel,
    NumericVal,
    UniqueList,
)

COMPONENTS_T = Literal[
    "dimensions",
    "parameters",
    "lookups",
    "variables",
    "global_expressions",
    "constraints",
    "piecewise_constraints",
    "objectives",
    "postprocessed",
]


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

    _group: ClassVar[COMPONENTS_T]
    """Return the component group this component belongs to."""


class Dimension(MathComponent):
    """Schema for named dimension."""

    dtype: Literal["string", "datetime", "date", "float", "integer"] = "string"
    """The data type of this dimension's items."""
    ordered: bool = False
    """If True, the order of the dimension items is meaningful (e.g. chronological time)."""
    iterator: str = "NEEDS_ITERATOR"
    """The name of the iterator to use in the LaTeX math formulation for this dimension."""

    _group: ClassVar[COMPONENTS_T] = "dimensions"

    @property
    def default(self) -> float:
        """Dummy variable to align with lookups and dims."""
        return float("nan")


class Parameter(MathComponent):
    """Schema for named parameter."""

    default: float | int = float("nan")
    """The default value for the parameter, if not set in the data."""
    resample_method: Literal["mean", "sum", "first"] = "first"
    """If resampling is applied over any of the parameter's dimensions, the method to use to aggregate the data."""
    unit: str = ""
    """The unit of the parameter, e.g. 'kW', 'm', 'kg', 'energy', 'power', ..."""

    @property
    def dtype(self) -> Literal["float"]:
        """Dummy variable to align with lookups and dims."""
        return "float"

    _group: ClassVar[COMPONENTS_T] = "parameters"


class Lookup(MathComponent):
    """Schema for named lookup arrays."""

    default: AttrStr | float | int | bool = float("nan")
    """The default value for the lookup, if not set in the data."""
    dtype: Literal["float", "string", "bool", "datetime", "date"] = "string"
    """The lookup data type."""
    resample_method: Literal["mean", "sum", "first"] = "first"
    """If resampling is applied over any of the lookup's dimensions, the method to use to aggregate the data."""
    one_of: list | None = None
    """If given, the lookup values must be one of these items."""
    pivot_values_to_dim: str | None = None
    """If given, the lookup will be pivoted such that its values become the index of a new dimension and its new values are boolean, True where the index values match the old values.
    For instance, if the lookup starts out indexed over `techs` with values of `[electricity, gas]` and `pivot_values_to_dim: carriers`,
    then the lookup will be converted to a boolean array with the dimensions ['techs', 'carriers'].
    """

    _group: ClassVar[COMPONENTS_T] = "lookups"


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


class MathEquationComponent(MathComponent):
    """Components necessary to generate math expressions."""

    equations: Equations = Equations()
    """Constraint math equations."""
    sub_expressions: SubExpressions = SubExpressions()
    """Named sub-expressions."""
    slices: SubExpressions = SubExpressions()
    """Named index slices."""

    @model_validator(mode="after")
    def must_have_equations_if_active(self) -> Self:
        """Ensure that equations are defined if the component is active."""
        if self.active and not self.equations.root:
            raise ValueError("Must have equations defined if component is active.")
        return self


class Constraint(MathIndexedComponent, MathEquationComponent):
    """Schema for named constraints."""

    _group: ClassVar[COMPONENTS_T] = "constraints"


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

    @property
    def equations(self) -> Equations:
        """Dummy property to satisfy type hinting."""
        return Equations()

    @property
    def sub_expressions(self) -> SubExpressions:
        """Dummy property to satisfy type hinting."""
        return SubExpressions()

    @property
    def slices(self) -> SubExpressions:
        """Dummy property to satisfy type hinting."""
        return SubExpressions()

    _group: ClassVar[COMPONENTS_T] = "piecewise_constraints"


class GlobalExpression(MathIndexedComponent, MathEquationComponent):
    """Schema for named global expressions.

    Can be used to combine parameters and variables and then used in one or more
    expressions elsewhere in the math formulation (i.e., in constraints, objectives,
    and other global expressions).

    NOTE: If expecting to use global expression `A` in global expression `B`, `A` must
    be defined above `B`.
    """

    unit: str = ""
    """Generalised unit of the component (e.g., length, time, quantity_per_hour, ...)."""
    default: NumericVal = float("nan")
    """If set, will be the default value for the expression."""
    equations: Equations = Equations()
    """Global expression math equations."""
    sub_expressions: SubExpressions = SubExpressions()
    """Global expression named sub-expressions."""
    slices: SubExpressions = SubExpressions()
    """Global expression named index slices."""
    order: int = 0
    """Order in which to apply this global expression relative to all others, if different to its definition order."""

    _group: ClassVar[COMPONENTS_T] = "global_expressions"


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
    default: NumericVal = float("nan")
    """If set, will be the default value for the variable."""
    domain: Literal["real", "integer"] = "real"
    """Allowed values that the decision variable can take.
    Either real (a.k.a. continuous) or integer."""
    bounds: Bounds = Bounds()

    @property
    def equations(self) -> Equations:
        """Dummy property to satisfy type hinting."""
        return Equations()

    @property
    def sub_expressions(self) -> SubExpressions:
        """Dummy property to satisfy type hinting."""
        return SubExpressions()

    @property
    def slices(self) -> SubExpressions:
        """Dummy property to satisfy type hinting."""
        return SubExpressions()

    _group: ClassVar[COMPONENTS_T] = "variables"


class Objective(MathEquationComponent):
    """Schema for optimisation problem objectives.

    Only one objective, the one referenced in model configuration `build.objective`
    will be activated for the optimisation problem.
    """

    sense: Literal["minimise", "maximise", "minimize", "maximize"]
    """Whether the objective function should be minimised or maximised in the
    optimisation."""

    @property
    def foreach(self) -> UniqueList[AttrStr]:
        """Objectives are always adimensional."""
        return []

    @property
    def where(self) -> str:
        """Dummy property to satisfy type hinting."""
        return "True"

    _group: ClassVar[COMPONENTS_T] = "objectives"


class PostprocessedExpression(GlobalExpression):
    """Schema for postprocessed expressions.

    Can be used to combine parameters, variables, and global expressions into a single expression solving the model.

    NOTE: If expecting to use postprocessed array `A` in postprocessed array `B`, `A` must
    be defined above `B`.
    """

    _group: ClassVar[COMPONENTS_T] = "postprocessed"


class Check(CalliopeBaseModel):
    """Schema for input data checks."""

    where: str
    """Top-level condition to check"""
    message: str
    """Message to display when the `where` array returns True, if raising or warning on error."""
    errors: Literal["raise", "warn"] = "raise"
    """How to respond to any instances in which the `where` array returns True."""
    active: bool = True
    """If False, this check will be ignored during the build phase."""


class MathDictModel(CalliopeDictModel):
    """Math dict model with computed field to return only active components."""

    @cached_property
    def _active(self) -> dict[str, MathComponent]:
        """Return only active components."""
        return {k: v for k, v in self.root.items() if v.active}


class Dimensions(MathDictModel):
    """Calliope model dimensions dictionary."""

    root: dict[AttrStr, Dimension] = Field(default_factory=dict)


class Parameters(MathDictModel):
    """Calliope model parameters dictionary."""

    root: dict[AttrStr, Parameter] = Field(default_factory=dict)


class Lookups(MathDictModel):
    """Calliope model lookup dictionary."""

    root: dict[AttrStr, Lookup] = Field(default_factory=dict)


class Variables(MathDictModel):
    """Calliope model variables dictionary."""

    root: dict[AttrStr, Variable] = Field(default_factory=dict)


class GlobalExpressions(MathDictModel):
    """Calliope model global_expressions dictionary."""

    root: dict[AttrStr, GlobalExpression] = Field(default_factory=dict)


class Constraints(MathDictModel):
    """Calliope model constraints dictionary."""

    root: dict[AttrStr, Constraint] = Field(default_factory=dict)


class PiecewiseConstraints(MathDictModel):
    """Calliope model piecewise_constraints dictionary."""

    root: dict[AttrStr, PiecewiseConstraint] = Field(default_factory=dict)


class Objectives(MathDictModel):
    """Calliope model objectives dictionary."""

    root: dict[AttrStr, Objective] = Field(default_factory=dict)


class PostprocessedExpressions(MathDictModel):
    """Calliope model postprocessed expressions dictionary."""

    root: dict[AttrStr, PostprocessedExpression] = Field(default_factory=dict)


class Checks(MathDictModel):
    """Calliope math checks dictionary."""

    root: dict[AttrStr, Check] = Field(default_factory=dict)


class CalliopeBuildMath(CalliopeBaseModel):
    """Mathematical definition of Calliope math.

    Contains mathematical programming components available for optimising with Calliope.
    Can contain partial definitions if they are meant to be layered on top of another.
    E.g.: layering 'base' and 'operate' math.
    """

    model_config = {"title": "Model math schema"}

    dimensions: Dimensions = Dimensions()
    """All dimensions to include in the optimisation problem."""
    parameters: Parameters = Parameters()
    """All parameters to include in the optimisation problem."""
    lookups: Lookups = Lookups()
    """All lookups to include in the optimisation problem."""
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
    postprocessed: PostprocessedExpressions = PostprocessedExpressions()
    """All postprocessed expressions generated after math has completed."""
    checks: Checks = Checks()
    """Checks to apply before building the optimisation problem."""

    @model_validator(mode="after")
    def unique_component_names(self):
        """Ensure all component names are unique."""
        groups = sorted(
            (
                {name for name in getattr(self, field)._active}
                for field in type(self).model_fields
            ),
            key=len,
        )
        seen = set()
        duplicates = set()
        for field_names in groups:
            duplicates |= field_names & seen
            seen |= field_names
        if duplicates:
            raise ValueError(
                f"Non-unique names in math components: {sorted(duplicates)}."
            )

        return self

    @cached_property
    def parsing_components(self) -> dict[str, dict[str, set[str]]]:
        """Return a set of valid component names in the model to use in `where` string parsing.

        Returns:
            dict[Literal["dimension_names", "input_names", "result_names"], set[str]]:
                Set of valid names grouped by location in the math in which they are defined.
        """
        parsing_components = {
            "dimensions": ["dimensions"],
            "inputs": ["lookups", "parameters"],
            "results": ["variables", "global_expressions"],
        }

        def _names():
            return {
                k: set().union(*[getattr(self, i)._active for i in v])
                for k, v in parsing_components.items()
            }

        where_names = _names()
        all_active = where_names["results"].union(where_names["inputs"])
        for component in ["inputs", "results"]:
            all_names = set().union(
                *(getattr(self, k).root for k in parsing_components[component])
            )
            where_names[component] |= all_names - all_active
        all_components = {"expression": _names(), "where": where_names}

        return all_components

    def find(
        self, component: str, subset: Iterable[COMPONENTS_T] | None = None
    ) -> MathComponent:
        """Find a component in the math schema."""
        fields: Iterable = subset or (set(type(self).model_fields) - {"checks"})

        found = {f for f in fields if component in getattr(self, f)._active}
        if not found:
            raise KeyError(f"Component name `{component}` not found in math schema.")
        if len(found) > 1:
            raise ValueError(
                f"Component name `{component}` found in multiple places: {found}."
            )
        return getattr(self, found.pop())[component]


class CalliopeInputMath(CalliopeDictModel):
    """Calliope input math dictionary."""

    root: dict[AttrStr, CalliopeDictModel] = Field(default_factory=dict)


class CalliopeMath(CalliopeBaseModel):
    """Calliope math attribute container."""

    init: CalliopeInputMath = CalliopeInputMath()
    build: CalliopeBuildMath = CalliopeBuildMath()
