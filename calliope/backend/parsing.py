from __future__ import annotations

import itertools
from typing import Optional, Union, Literal, Iterable, Callable, TypeVar, Generic
from typing_extensions import NotRequired, TypedDict, Required
import functools
import operator
from abc import ABC, abstractmethod

import pyparsing as pp
import xarray as xr
import pandas as pd

from calliope.backend import equation_parser, subset_parser
from calliope.backend.backends import BackendModel
from calliope.exceptions import BackendError
from calliope.backend import helper_functions

VALID_HELPER_FUNCTIONS = {
    "inheritance": helper_functions.inheritance,
}


class UnparsedEquationDict(TypedDict):
    where: NotRequired[str]
    expression: str


class UnparsedConstraintDict(TypedDict):
    foreach: Required[list]
    where: str
    equation: NotRequired[str]
    equations: NotRequired[list[UnparsedEquationDict]]
    components: NotRequired[dict[str, list[UnparsedEquationDict]]]
    index_items: NotRequired[dict[str, list[UnparsedEquationDict]]]


class ParsedEquationDict(TypedDict):
    id: tuple
    where: list[Optional[pp.ParseResults]]
    expression: Optional[pp.ParseResults]
    components: NotRequired[dict[str, pp.ParseResults]]
    index_items: NotRequired[dict[str, pp.ParseResults]]


class UnparsedVariableBoundDict(TypedDict):
    min: str
    max: str
    scale: NotRequired[str]


class UnparsedVariableDict(TypedDict):
    foreach: list[str]
    where: str
    domain: str
    bounds: NotRequired[UnparsedVariableBoundDict]


class UnparsedObjectiveDict(TypedDict):
    equation: NotRequired[str]
    equations: NotRequired[list[UnparsedEquationDict]]
    components: NotRequired[dict[str, list[UnparsedEquationDict]]]
    domain: str
    sense: str


T = TypeVar(
    "T",
    bound=Union[UnparsedConstraintDict, UnparsedVariableDict, UnparsedObjectiveDict],
)


class ParsedBackendComponent(ABC, Generic[T]):
    def __init__(self, unparsed_data: T, component_name: str) -> None:
        """
        Parse an optimisation problem configuration - defined in a dictionary of strings
        loaded from YAML - into a series of Python objects that can be passed onto a solver
        interface like Pyomo or Gurobipy.
        """
        self.name: str = component_name
        self._unparsed = unparsed_data

        # capture errors to dump after processing,
        # to make it easier for a user to fix the constraint YAML.
        self._errors: set = set()

        # Add objects that are used by shared functions
        self.sets: dict = dict()
        self.top_level_where: Optional[pp.ParseResults] = None
        self.index = pd.Series(dtype=str)
        self.equations: list[ParsedEquationDict] = []

        # Initialise switches
        self._is_valid: bool = True
        self._is_active: bool = True

    def evaluate_name(self, equation_id: Optional[tuple] = None) -> str:
        """
        Transform equation ID tuple into a string that could act as a valid python identifier.

        Args:
            equation_id (Optional[tuple]): If given, of the form (int, (str, int), ...).

        Returns:
            str: stringified equation ID tuple prepended with the component name.
        """
        id_ = "" if equation_id is None else "_" + self._name_component(equation_id)
        return self.name + id_

    def evaluate_subset(
        self,
        model_data: xr.Dataset,
        equation_where: list[Optional[pp.ParseResults]],
        equation_name: Optional[str] = None,
    ) -> Optional[pd.Index]:
        """
        Get subset of product of "foreach" sets that matches the "where" string conditions.

        Args:
            model_data (xr.Dataset): Calliope model data
            equation_where (list[Optional[pp.ParseResults]]): List of parsed "where" strings
            equation_name (Optional[str], optional): s
                String to name subset items in self.index.
                Defaults to None, in which case self.name is used.

        Returns:
            Optional[pd.Index]:
                If no valid index items are found in the set product, return None.
        """

        subset_ = self._create_subset_from_where(model_data, equation_where)

        if equation_name is None:
            equation_name = self.name
        self.index = pd.concat(
            [self.index, pd.Series(index=subset_, data=equation_name, dtype=str)]
        )

        return subset_

    def evaluate_rule(
        self,
        equation_name: str,
        equation_dict: ParsedEquationDict,
        backend_interface: BackendModel,
    ) -> Callable:
        """
        Create constraint rule function to be called by the backend interface in
        generating constraints

        Args:
            equation_name (str): For use in raising exeptions.
            equation_dict (ParsedEquationDict): Provides parsed strings to evaluate.
            backend_interface (BackendModel): Provides interface to the backend on evalution.

        Returns:
            Callable:
                Function that can be called with the constraint set iterators as arguments.
                Will return a constraint equation.
        """

        def _rule(_, *args):
            iterator_dict = {
                list(self.sets.keys())[idx]: arg for idx, arg in enumerate(args)
            }
            return equation_dict["expression"][0].eval(
                equation_name=equation_name,
                iterator_dict=iterator_dict,
                index_item_dict=equation_dict["index_items"],
                component_dict=equation_dict["components"],
                backend_interface=backend_interface,
                helper_func_dict=VALID_HELPER_FUNCTIONS,
                sets=self.sets,
                as_dict=False,
            )

        return _rule

    @abstractmethod
    def parse_strings(self) -> None:
        pass

    def _add_parse_error(
        self, instring: str, expression_group: str, error_message: str
    ) -> None:
        """
        Add error message to the list self._errors following a predefined structure of
        `(expression_group, instring): error`, e.g. `(foreach, a in A): Found duplicate set iterator`.

        Also set self._is_valid flag to False since at least one error has been caught.

        Args:
            instring (str): String being parsed where the error was caught.
            expression_group (str):
                Location in the constraint definition where the string was defined,
                e.g., "foreach", "equations", "components".
            error_message (str): Description of error.
        """
        self._is_valid = False
        self._errors.add(f"({expression_group}, {instring}): {error_message}")

    def _create_subset_from_where(
        self, model_data: xr.Dataset, where_list: list[Optional[pp.ParseResults]]
    ) -> Optional[pd.Index]:
        """
        Returns the subset of combined constraint set items (given by "foreach")
        valid on evaluation of equation "where" strings.

        Args:
            model_data (xr.Dataset): Calliope model dataset.
            where_list (list[Optional[pp.ParseResults]]): List of parsed "where" strings.

        Returns:
            Optional[pd.Index]: If no valid subset of set product, returns None.
        """

        # Start with a mask that is True where the tech exists at a node
        # (across all timesteps and for a each carrier and cost, where appropriate)
        imask_foreach: xr.DataArray = self._imask_foreach(model_data)

        evaluated_wheres = [
            where[0].eval(  # type: ignore
                model_data=model_data,
                helper_func_dict=VALID_HELPER_FUNCTIONS,
                errors=self._errors,
                imask=imask_foreach,
                defaults=model_data.attrs["defaults"],
            )
            for where in [self.top_level_where, *where_list]
        ]

        imask: xr.DataArray = functools.reduce(
            operator.and_, [imask_foreach, *evaluated_wheres]
        )

        if imask.any():
            # Squeeze out any unwanted dimensions
            unwanted_dims = set(imask.dims).difference(self.sets.values())
            imask = (imask.sum(unwanted_dims) > 0).astype(bool)
            subset_index = self._get_subset_as_index(imask)
        else:
            self._is_valid = False
            subset_index = None

        return subset_index

    def _get_subset_as_index(self, imask: xr.DataArray) -> pd.Index:
        """
        Dump index items from a boolean imasking array for all points in the array that are set to True.

        Args:
            imask (xr.DataArray):
                Boolean imasking array with same number of dimensions as the length of `self.sets`.

        Returns:
            pd.Index: Index or MultiIndex listing all valid combinations of set items.
        """
        if len(imask.dims) == 1:
            return imask[imask].coords.to_index()
        else:
            mask_stacked = imask.stack(dim_0=imask.dims)
            return (
                mask_stacked[mask_stacked]  # type: ignore
                .coords.to_index()
                .reorder_levels(self.sets.values())
            )

    def _imask_foreach(self, model_data: xr.Dataset) -> xr.DataArray:
        """
        Generate a multi-dimensional imasking array based on the sets
        over which the constraint is to be built (defined by "foreach").
        Irrespective of the sets defined by "foreach", this array will always include ["nodes", "techs", "carriers", "carrier_tiers"] to ensure only valid combinations of technologies consuming/producing specific carriers at specific nodes are included in later imasking.

        Args:
            model_data (xr.Dataset): Calliope model dataset.

        Returns:
            xr.DataArray: imasking boolean array.
        """
        # Start with (carriers, carrier_tiers, nodes, techs) and go from there
        initial_imask = model_data.carrier.notnull() * model_data.node_tech.notnull()
        # Add other dimensions (costs, timesteps, etc.)
        add_dims = set(self.sets.values()).difference(initial_imask.dims)
        if add_dims.difference(model_data.dims):
            raise BackendError(
                "Unidentified model set name(s) defined: "
                f"`{add_dims.difference(model_data.dims)}`."
            )
        all_imasks = [initial_imask, *[model_data[i].notnull() for i in add_dims]]

        return functools.reduce(operator.and_, all_imasks)

    def _parse_string(
        self,
        parser: pp.ParserElement,
        parse_string: str,
        expression_group: Literal[
            "foreach", "where", "equations", "components", "index_items"
        ],
    ) -> Optional[pp.ParseResults]:
        """
        Parse equation string according to predefined string parsing grammar
        given by `self.parser`

        Args:
            parser (pp.ParserElement): Parsing grammar.
            parse_string (str): String to parse according to parser grammar.
            expression_group (str): For error reporting, the constraint dict key corresponding to the parse_string.

        Returns:
            Optional[pp.ParseResults]:
                Parsed string. If any parsing errors are caught,
                they will be logged to `self._errors` to raise later.
        """
        try:
            parsed = parser.parse_string(parse_string, parse_all=True)
        except (pp.ParseException, KeyError) as excinfo:
            parsed = None
            self._add_parse_error(parse_string, expression_group, str(excinfo))

        return parsed

    def _parse_where_string(
        self,
        equation_dict: Union[
            UnparsedEquationDict, UnparsedConstraintDict, UnparsedVariableDict
        ],
    ) -> Optional[pp.ParseResults]:
        """Parse a "where" string of the form "CONDITION OPERATOR CONDITION", where the
        operator can be "and"/"or"/"not and"/"not or".

        Args:
            equation_dict (Union[UnparsedEquationDict, UnparsedConstraintDict]):
                Dictionary with optional "where" key.
                If not found, the where string will default to "True", to have no effect
                on the subsequent subsetting.

        Returns:
            pp.ParseResults: Parsed string. If any parsing errors are caught,
                they will be logged to `self._errors` to raise later.
        """
        parser = subset_parser.generate_where_string_parser()
        where_string = equation_dict.get("where", "True")
        return self._parse_string(parser, where_string, "where")

    def _name_component(self, equation_id: Union[tuple, int]) -> str:
        final_string_list: list[str] = []

        if isinstance(equation_id, int):
            constraint_name = str(equation_id)
        else:
            for i in equation_id:
                if isinstance(i, tuple):
                    final_string_list.append(self._name_component(i))
                else:
                    final_string_list.append(str(i))
            constraint_name = "_".join(final_string_list)
        return constraint_name

    def _get_sets_from_foreach(self, foreach_list: list[str]) -> dict[str, str]:
        """
        Process "foreach" key in constraint to access the set iterators and the
        identifier for set items in the constraint equation expessions.
        """
        foreach_parser = equation_parser.foreach_parser()
        sets: dict = dict()
        for string_ in foreach_list:
            parsed_ = self._parse_string(foreach_parser, string_, "foreach")
            if parsed_ is not None:
                set_iterator, set_name = parsed_[0].as_list()
                if set_iterator in sets.keys():
                    self._add_parse_error(
                        string_,
                        "foreach",
                        f"Found duplicate set iterator `{set_iterator}`.",
                    )

                sets[set_iterator] = set_name

        return sets

    def _parse_where_expression(
        self,
        expression_parser: pp.ParserElement,
        expression_list: list[UnparsedEquationDict],
        expression_group: Literal["foreach", "equations", "components", "index_items"],
        id_prefix: Optional[str] = None,
    ) -> list[ParsedEquationDict]:
        """
        Align user-defined constraint equations/components by parsing expressions,
        specifying a default "where" string if not defined,
        and providing an ID to enable returning to the initial dictionary.

        Args:
            expression_list (list[dict]):
                list of constraint equations or components with arithmetic expression
                string and optional where string.
            expression_group (str):
                For error reporting, the constraint dict key corresponding to the parse_string.
            id_prefix (Optional[str]):
                If provided, will extend the ID from a number corresponding to the
                expression_list position `idx` to a tuple of the form (id_prefix, idx).

        Returns:
            list[UnparsedConstraintDict]:
                Aligned expression dictionaries with parsed expression strings.
        """

        return [
            {
                "id": tuple([idx]) if id_prefix is None else tuple([id_prefix, idx]),
                "where": [self._parse_where_string(expression_data)],
                "expression": self._parse_string(
                    expression_parser, expression_data["expression"], expression_group
                ),
            }
            for idx, expression_data in enumerate(expression_list)
        ]

    def _find_items_in_expression(
        self,
        parser_elements: Union[list, pp.ParseResults],
        to_find: type[equation_parser.EvalString],
        valid_eval_classes: tuple[type[equation_parser.EvalString], ...],
    ) -> set[str]:
        """
        Recursively find components / index items defined in an equation expression.

        Args:
            parser_elements (pp.ParseResults): list of parser elements to check.
            to_find (type[equation_parser.EvalString]): type of equation element to search for
            valid_eval_classes (tuple[type(equation_parser.EvalString)]):
                Other expression elements that can be recursively searched

        Returns:
            set[str]: All unique component / index item names.
        """
        items: list = []
        recursive_func = functools.partial(
            self._find_items_in_expression,
            to_find=to_find,
            valid_eval_classes=valid_eval_classes,
        )
        for parser_element in parser_elements:
            if isinstance(parser_element, to_find):
                items.append(parser_element.name)

            elif isinstance(parser_element, pp.ParseResults):
                items.extend(recursive_func(parser_elements=parser_element))

            elif isinstance(parser_element, valid_eval_classes):
                items.extend(recursive_func(parser_elements=parser_element.values))
        return set(items)

    def _get_expression_group_product(
        self,
        equation_data: ParsedEquationDict,
        parsed_items: dict[str, list[ParsedEquationDict]],
        expression_group: Literal["components", "index_items"],
    ) -> itertools.product:
        """
        Find all components referenced in an equation expression and return a
        product of the component data.

        Args:
            equation_data (UnparsedConstraintDict): Equation data dictionary.
            parsed_items (dict[list[UnparsedConstraintDict]]):
                Dictionary of expressions to replace within the equation data dictionary.
            expression_group (Literal["components", "index_items"]):
                Name of expression group that the parsed_items dict is referencing.

        Returns:
            list[list[UnparsedConstraintDict]]:
                Each nested list contains a unique product of parsed_item dictionaries.
        """
        eq_expr: pp.ParseResults = equation_data["expression"]  # type: ignore
        valid_eval_classes: tuple = (
            equation_parser.EvalOperatorOperand,
            equation_parser.EvalFunction,
        )
        elements: list = [eq_expr[0].values]
        if expression_group == "components":
            to_find = equation_parser.EvalComponent

        elif expression_group == "index_items":
            elements += list(equation_data.get("components", {}).values())
            to_find = equation_parser.EvalIndexItems  # type: ignore
            valid_eval_classes += (equation_parser.EvalIndexedParameterOrVariable,)

        eq_items = self._find_items_in_expression(elements, to_find, valid_eval_classes)

        invalid_items = eq_items.difference(parsed_items.keys())
        if invalid_items:
            self._add_parse_error(
                eq_expr.__repr__(),
                "equation",
                f"Undefined {expression_group} found in equation: {invalid_items}",
            )

        eq_items.difference_update(invalid_items)

        return itertools.product(*[parsed_items[k] for k in eq_items])

    def _add_exprs_to_equation_data(
        self,
        equation_data: ParsedEquationDict,
        expression_combination: Iterable[ParsedEquationDict],
        expression_group: Literal["components", "index_items"],
    ) -> ParsedEquationDict:
        """
        Create new equation dictionaries with evaluatable expressions for components or index items.
        The new equation dict has an updated ID, `where` list, and an additional key
        with the expression in.

        Args:
            equation_data (UnparsedConstraintDict):
                Original equation data dictionary with reference to the expression group.
            expression_combination (list[UnparsedConstraintDict]):
                list of data dictionaries to use in updating the equation data.
            expression_group (str): Name of the source of replacement expressions.

        Returns:
            UnparsedConstraintDict:
                Updated equation dictionary with unique ID and equation
                expressions attached under the key given by expression_group.
        """
        new_equation_data = equation_data.copy()

        return {
            "id": (
                *new_equation_data.pop("id"),
                *[expr["id"] for expr in expression_combination],
            ),
            "where": [
                *new_equation_data.pop("where"),
                *[expr["where"][0] for expr in expression_combination],
            ],
            expression_group: {  # type: ignore
                expr["id"][0]: expr["expression"] for expr in expression_combination
            },
            **new_equation_data,
        }

    def _add_sub_exprs_per_equation_expr(
        self,
        equations: list[ParsedEquationDict],
        expression_dict: dict[str, list[ParsedEquationDict]],
        expression_group: Literal["components", "index_items"],
    ) -> list[ParsedEquationDict]:
        """
        Build new list of equation dictionaries with nested expression group information.

        Args:
            equations (list[UnparsedConstraintDict]):
                List of original equation data dictionaries with reference to the expression group.
            expression_dict (dict[str, list[UnparsedConstraintDict]]):

            expression_group (Literal[&quot;components&quot;, &quot;index_items&quot;]): _description_

        Returns:
            List[UnparsedConstraintDict]: _description_
        """
        updated_equations = []
        for equation_dict in equations:
            component_product = self._get_expression_group_product(
                equation_dict, expression_dict, expression_group
            )
            for component_combination in component_product:
                updated_equation_dict = self._add_exprs_to_equation_data(
                    equation_dict, component_combination, expression_group
                )
                updated_equations.append(updated_equation_dict)
        return updated_equations


class ParsedConstraint(ParsedBackendComponent):
    def __init__(
        self, constraint: UnparsedConstraintDict, constraint_name: str
    ) -> None:
        """Parse a constraint defined in a dictionary of strings loaded from YAML into a series of Python objects that can be passed onto a solver interface like Pyomo or Gurobipy.

        Args:
            constraint (UnparsedConstraintDict): Dictionary of the form:
                foreach: list[str]  <- sets over which to iterate the constraint, e.g. ["timestep in timesteps"], where "timesteps" is the set and "timestep" is the reference to the set iterator in the constraint equation(s)
                where: list[str]  <- conditions defining which items in the product of all sets to apply the constraint to  FIXME: not implemented.
                equation: String  <- if no other conditions, single constraint equation of the form LHS OPERATOR RHS, e.g. "energy_cap[node, tech] >= my_parameter[tech] * 2"  FIXME: not implemented.
                equations: list[dict]  <- if different equations for different conditions, list them here with an additional "where" statement and an associated expression: {"where": [...], "expression": "..."}  FIXME: not implemented.
                components: dict[list[dict]]  <- if applying the same expression to multiple equations in the constraint, store them here with optional conditions on their exact composition, e.g. "$energy_prod" in an equation would refer to a component {"energy_prod": [{"where": [...], "expression": "..."}, ...]}  FIXME: not implemented.
                index_items: dict[list[dict]]  <- if indexing a parameter/variable separately to the set iterators given in "foreach", define them here. FIXME: not implemented.
            constraint_name (str): Name of constraint.
        """
        ParsedBackendComponent.__init__(self, constraint, constraint_name)

    def parse_strings(self) -> None:
        """
        Parse all elements of the constraint: "foreach", "equation(s)", "component".

        Args:
            model_data (xr.Dataset):
                Calliope processed model dataset with all necessary parameters, sets,
                and run configuration options defined.

        """
        sets = self._get_sets_from_foreach(self._unparsed["foreach"])
        top_level_where = self._parse_where_string(self._unparsed)
        equation_expression_list: list[UnparsedEquationDict]
        if "equation" in self._unparsed.keys():
            equation_expression_list = [{"expression": self._unparsed["equation"]}]
        else:
            equation_expression_list = self._unparsed["equations"]

        equations = self._parse_where_expression(
            expression_parser=equation_parser.generate_equation_parser(),
            expression_list=equation_expression_list,
            expression_group="equations",
        )

        component_dict = {
            c_name: self._parse_where_expression(
                expression_parser=equation_parser.generate_arithmetic_parser(),
                expression_list=c_list,
                expression_group="components",
                id_prefix=c_name,
            )
            for c_name, c_list in self._unparsed.get("components", {}).items()
        }
        index_item_dict = {
            idx_name: self._parse_where_expression(
                expression_parser=equation_parser.generate_index_item_parser(),
                expression_list=idx_list,
                expression_group="index_items",
                id_prefix=idx_name,
            )
            for idx_name, idx_list in self._unparsed.get("index_items", {}).items()
        }

        if self._is_valid:
            equations_with_components = self._add_sub_exprs_per_equation_expr(
                equations, component_dict, "components"
            )
            equations_with_components_and_index_items = (
                self._add_sub_exprs_per_equation_expr(
                    equations_with_components, index_item_dict, "index_items"
                )
            )

        if self._is_valid:
            self.sets = sets
            self.equations = equations_with_components_and_index_items
            self.top_level_where = top_level_where

        return None


class ParsedVariable(ParsedBackendComponent):
    def __init__(self, variable: UnparsedVariableDict, variable_name: str) -> None:
        """Parse a variable configuration dictionary.

        Args:
            variable (UnparsedVariableDict): Dictionary of the form:
                foreach: list[str]  <- sets over which to iterate the constraint, e.g. ["techs", "timesteps"],
                where: list[str]  <- conditions defining which items in the product of all sets build the variable for
                domain: str <- limit on types of numeric values that can be assigned to the variable
                bounds: UnparsedBoundsDict <- link to parameters with which to apply explicit numeric bounds on each item in the variable
                    min: str
                    max: str
                    scale: str
            variable_name (str): Name of variable.
        """
        ParsedBackendComponent.__init__(self, variable, variable_name)

    def parse_strings(self) -> None:
        """
        Parse all elements of the variable: "foreach", "where".

        Args:
            model_data (xr.Dataset):
                Calliope processed model dataset with all necessary parameters, sets,
                and run configuration options defined.

        """

        sets = self._get_sets_from_foreach(self._unparsed["foreach"])
        top_level_where = self._parse_where_string(self._unparsed)

        if self._is_valid:
            self.sets = sets
            self.top_level_where = top_level_where

        return None


class ParsedObjective(ParsedBackendComponent):
    def __init__(self, objective: UnparsedObjectiveDict, objective_name: str) -> None:
        """Parse an objective configuration dictionary.

        Args:
            variable (UnparsedObjectiveDict): Dictionary of the form:
                equations: list[str]
            objective_name (str): Name of objective.
        """
        ParsedBackendComponent.__init__(self, objective, objective_name)

    def parse_strings(self) -> None:
        """
        Parse all elements of the objective: "equation(s)", "component".

        """
        equation_expression_list: list[UnparsedEquationDict]
        base_where_dict: UnparsedEquationDict = {"where": "True", "expression": ""}
        top_level_where = self._parse_where_string(base_where_dict)

        if "equation" in self._unparsed.keys():
            equation_expression_list = [{"expression": self._unparsed["equation"]}]
        else:
            equation_expression_list = self._unparsed["equations"]

        equations = self._parse_where_expression(
            expression_parser=equation_parser.generate_arithmetic_parser(),
            expression_list=equation_expression_list,
            expression_group="equations",
        )

        component_dict = {
            c_name: self._parse_where_expression(
                expression_parser=equation_parser.generate_arithmetic_parser(),
                expression_list=c_list,
                expression_group="components",
                id_prefix=c_name,
            )
            for c_name, c_list in self._unparsed.get("components", {}).items()
        }

        if self._is_valid:
            equations_with_components = self._add_sub_exprs_per_equation_expr(
                equations, component_dict, "components"
            )

        if self._is_valid:
            self.equations = equations_with_components
            self.top_level_where = top_level_where

        return None
