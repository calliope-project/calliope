"""Calliope math handling."""

import logging
from copy import deepcopy
from importlib.resources import files
from pathlib import Path

from calliope.attrdict import AttrDict
from calliope.exceptions import ModelError
from calliope.util.schema import MATH_SCHEMA, validate_dict
from calliope.util.tools import relative_path

LOGGER = logging.getLogger(__name__)
MATH_DIR = files("calliope") / "math"


class ModelMath:
    """Calliope math preprocessing."""

    ATTRS_TO_SAVE = ("_history",)

    def __init__(
        self,
        math_to_add: list | dict | None = None,
        model_def_path: str | Path | None = None,
    ):
        """Calliope YAML math handler.

        Can be initialised in the following ways:
        - default: base model math is loaded.
        - list of math files: pre-defined or user-defined math files.
        - dictionary: fully defined math dictionary with configuration saved as keys (see `ATTRS_TO_SAVE`).

        Args:
            math_to_add (list | dict | None, optional): Calliope math to load. Defaults to None (only base math).
            model_def_path (str | Path | None, optional): Model definition path, needed for user math. Defaults to None.
        """
        self._history: list[str] = []
        self._data: AttrDict = AttrDict()
        if math_to_add is None:
            math_to_add = []

        if isinstance(math_to_add, list):
            self._init_from_list(math_to_add, model_def_path)
        else:
            self._init_from_dict(math_to_add)

    def __eq__(self, other):
        """Compare between two model math instantiations."""
        if not isinstance(other, ModelMath):
            return NotImplemented
        return self._history == other._history and self._data == other._data

    def _init_from_list(
        self, math_to_add: list[str], model_def_path: str | Path | None = None
    ):
        """Load math definition from a list of files.

        Args:
            math_to_add (list[str]): Calliope math files to load. Suffix implies user-math.
            model_def_path (str | Path | None, optional): Model definition path. Defaults to None.

        Raises:
            ModelError: user-math requested without providing `model_def_path`.
        """
        for math_name in ["base"] + math_to_add:
            if not math_name.endswith((".yaml", ".yml")):
                self.add_pre_defined_math(math_name)
            elif model_def_path is not None:
                self.add_user_defined_math(math_name, model_def_path)
            else:
                raise ModelError(
                    "Must declare `model_def_path` when requesting user math."
                )

    def _init_from_dict(self, math_dict: dict) -> None:
        """Load math from a dictionary definition, recuperating relevant attributes."""
        self._data = AttrDict(math_dict)
        for attr in self.ATTRS_TO_SAVE:
            setattr(self, attr, self._data[attr])
            del self._data[attr]

    def to_dict(self) -> dict:
        """Translate into a dictionary."""
        math = deepcopy(self._data)
        for attr in self.ATTRS_TO_SAVE:
            math[attr] = getattr(self, attr)
        return math

    def check_in_history(self, math_name: str) -> bool:
        """Evaluate if math has already been applied."""
        return math_name in self._history

    def _add_math(self, math: AttrDict):
        """Add math into the model."""
        self._data.union(math, allow_override=True)

    def _add_math_from_file(self, yaml_filepath: Path, name: str) -> None:
        try:
            math = AttrDict.from_yaml(yaml_filepath)
        except FileNotFoundError:
            raise ModelError(
                f"Attempted to load math file that does not exist: {yaml_filepath}"
            )
        self._add_math(math)
        self._history.append(name)

    def add_pre_defined_math(self, math_name: str) -> None:
        """Add pre-defined Calliope math (no suffix)."""
        if self.check_in_history(math_name):
            raise ModelError(
                f"Attempted to override math with pre-defined math file '{math_name}'."
            )
        self._add_math_from_file(MATH_DIR / f"{math_name}.yaml", math_name)

    def add_user_defined_math(
        self, math_relative_path: str | Path, model_def_path: str | Path
    ) -> None:
        """Add user-defined Calliope math, relative to the model definition path."""
        math_name = str(math_relative_path)
        if self.check_in_history(math_name):
            raise ModelError(
                f"Attempted to override math with user-defined math file '{math_name}'"
            )
        self._add_math_from_file(
            relative_path(model_def_path, math_relative_path), math_name
        )

    def validate(self) -> None:
        """Test that the model math is correctly defined."""
        validate_dict(self._data, MATH_SCHEMA, "math")
