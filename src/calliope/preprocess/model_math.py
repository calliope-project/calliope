"""Calliope math handling with interfaces for pre-defined and user-defined files."""

import importlib.resources
import logging
import typing
from copy import deepcopy
from pathlib import Path

from calliope.attrdict import AttrDict
from calliope.exceptions import ModelError
from calliope.util.schema import MATH_SCHEMA, validate_dict
from calliope.util.tools import relative_path

LOGGER = logging.getLogger(__name__)
ORDERED_COMPONENTS_T = typing.Literal[
    "variables",
    "global_expressions",
    "constraints",
    "piecewise_constraints",
    "objectives",
]


class CalliopeMath:
    """Calliope math handling."""

    ATTRS_TO_SAVE = ("history", "data")
    ATTRS_TO_LOAD = ("history",)

    def __init__(
        self, math_to_add: list[str | dict], model_def_path: str | Path | None = None
    ):
        """Calliope YAML math handler.

        Args:
            math_to_add (list[str | dict]):
                List of Calliope math to load.
                If a string, it can be a reference to pre-/user-defined math files.
                If a dictionary, it is equivalent in structure to a YAML math file.
            model_def_path (str | Path | None, optional): Model definition path, needed when using relative paths. Defaults to None.
        """
        self.history: list[str] = []
        self.data: AttrDict = AttrDict(
            {name: {} for name in typing.get_args(ORDERED_COMPONENTS_T)}
        )

        for math in math_to_add:
            if isinstance(math, dict):
                self.add(AttrDict(math))
            else:
                self._init_from_string(math, model_def_path)

    def __eq__(self, other):
        """Compare between two model math instantiations."""
        if not isinstance(other, CalliopeMath):
            return NotImplemented
        return self.history == other.history and self.data == other.data

    def __iter__(self):
        """Enable dictionary conversion."""
        for key in self.ATTRS_TO_SAVE:
            yield key, deepcopy(getattr(self, key))

    def __repr__(self) -> str:
        """Custom string representation of class."""
        return f"""Calliope math definition dictionary with:
    {len(self.data["variables"])} decision variable(s)
    {len(self.data["global_expressions"])} global expression(s)
    {len(self.data["constraints"])} constraint(s)
    {len(self.data["piecewise_constraints"])} piecewise constraint(s)
    {len(self.data["objectives"])} objective(s)
        """

    def add(self, math: AttrDict):
        """Add math into the model.

        Args:
            math (AttrDict): Valid math dictionary.
        """
        self.data.union(math, allow_override=True)

    @classmethod
    def from_dict(cls, math_dict: dict) -> "CalliopeMath":
        """Load a CalliopeMath object from a dictionary representation, recuperating relevant attributes.

        Args:
            math_dict (dict): Dictionary representation of a CalliopeMath object.

        Returns:
            CalliopeMath: Loaded from supplied dictionary representation.
        """
        new_self = cls([math_dict["data"]])
        for attr in cls.ATTRS_TO_LOAD:
            setattr(new_self, attr, math_dict[attr])
        return new_self

    def in_history(self, math_name: str) -> bool:
        """Evaluate if math has already been applied.

        Args:
            math_name (str): Math file to check.

        Returns:
            bool: `True` if found in history. `False` otherwise.
        """
        return math_name in self.history

    def validate(self) -> None:
        """Test current math and optional external math against the MATH schema."""
        validate_dict(self.data, MATH_SCHEMA, "math")
        LOGGER.info("Math preprocessing | validated math against schema.")

    def _add_pre_defined_file(self, filename: str) -> None:
        """Add pre-defined Calliope math.

        Args:
            filename (str): name of Calliope internal math (no suffix).

        Raises:
            ModelError: If math has already been applied.
        """
        if self.in_history(filename):
            raise ModelError(
                f"Math preprocessing | Overwriting with previously applied pre-defined math: '{filename}'."
            )
        with importlib.resources.as_file(
            importlib.resources.files("calliope") / "math"
        ) as f:
            self._add_file(f / f"{filename}.yaml", filename)

    def _add_user_defined_file(
        self, relative_filepath: str | Path, model_def_path: str | Path | None
    ) -> None:
        """Add user-defined Calliope math, relative to the model definition path.

        Args:
            relative_filepath (str | Path): Path to user math, relative to model definition.
            model_def_path (str | Path): Model definition path.

        Raises:
            ModelError: If file has already been applied.
        """
        math_name = str(relative_filepath)
        if self.in_history(math_name):
            raise ModelError(
                f"Math preprocessing | Overwriting with previously applied user-defined math: '{relative_filepath}'."
            )
        self._add_file(relative_path(model_def_path, relative_filepath), math_name)

    def _init_from_string(
        self, math_to_add: str, model_def_path: str | Path | None = None
    ):
        """Load math definition from a list of files.

        Args:
            math_to_add (str): Calliope math file to load. Suffix implies user-math.
            model_def_path (str | Path | None, optional): Model definition path. Defaults to None.

        Raises:
            ModelError: User-math requested without providing `model_def_path`.
        """
        if not math_to_add.endswith((".yaml", ".yml")):
            self._add_pre_defined_file(math_to_add)
        else:
            self._add_user_defined_file(math_to_add, model_def_path)

    def _add_file(self, yaml_filepath: Path, name: str) -> None:
        try:
            math = AttrDict.from_yaml(yaml_filepath, allow_override=True)
        except FileNotFoundError:
            raise ModelError(
                f"Math preprocessing | File does not exist: {yaml_filepath}"
            )
        self.add(math)
        self.history.append(name)
        LOGGER.info(f"Math preprocessing | added file '{name}'.")
