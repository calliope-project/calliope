"""Calliope math handling with interfaces for pre-defined and user-defined files."""

import logging
from copy import deepcopy
from importlib.resources import files
from pathlib import Path

from calliope import exceptions
from calliope.attrdict import AttrDict
from calliope.util.schema import MATH_SCHEMA, validate_dict
from calliope.util.tools import relative_path

LOGGER = logging.getLogger(__name__)
MATH_DIR = files("calliope") / "math"


class ModelMath:
    """Calliope math preprocessing."""

    ATTRS_TO_SAVE = ("history", "data")

    def __init__(
        self,
        math_to_add: list | dict | None = None,
        model_def_path: str | Path | None = None,
    ):
        """Calliope YAML math handler.

        Can be initialised in the following ways:
        - default: 'plan' model math is loaded.
        - list of math files: pre-defined or user-defined math files.
        - dictionary: fully defined math dictionary with configuration saved as keys (see `ATTRS_TO_SAVE`).

        Args:
            math_to_add (list | dict | None, optional): Calliope math to load. Defaults to None (only base math).
            model_def_path (str | Path | None, optional): Model definition path, needed for user math. Defaults to None.
        """
        self.history: list[str] = []
        self.data: AttrDict = AttrDict()
        if math_to_add is None:
            math_to_add = []

        if isinstance(math_to_add, list):
            self._init_from_list(["plan"] + math_to_add, model_def_path)
        else:
            self._init_from_dict(math_to_add)

    def __eq__(self, other):
        """Compare between two model math instantiations."""
        if not isinstance(other, ModelMath):
            return NotImplemented
        return self.history == other.history and self.data == other.data

    def __iter__(self):
        """Enable dictionary conversion."""
        for key in self.ATTRS_TO_SAVE:
            yield key, deepcopy(getattr(self, key))

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
        for math_name in math_to_add:
            if not math_name.endswith((".yaml", ".yml")):
                self.add_pre_defined_file(math_name)
            elif model_def_path is not None:
                self.add_user_defined_file(math_name, model_def_path)
            else:
                raise exceptions.ModelError(
                    "Must declare `model_def_path` when requesting user math."
                )

    def _init_from_dict(self, math_dict: dict) -> None:
        """Load math from a dictionary definition, recuperating relevant attributes."""
        for attr in self.ATTRS_TO_SAVE:
            setattr(self, attr, math_dict[attr])

    def check_in_history(self, math_name: str) -> bool:
        """Evaluate if math has already been applied."""
        return math_name in self.history

    def _add_math(self, math: AttrDict):
        """Add math into the model."""
        self.data.union(math, allow_override=True)

    def _add_file(self, yaml_filepath: Path, name: str) -> None:
        try:
            math = AttrDict.from_yaml(yaml_filepath)
        except FileNotFoundError:
            raise exceptions.ModelError(
                f"Attempted to load math file that does not exist: {yaml_filepath}"
            )
        self._add_math(math)
        self.history.append(name)
        LOGGER.info(f"ModelMath: added file '{name}'.")

    def add_pre_defined_file(self, filename: str) -> None:
        """Add pre-defined Calliope math (no suffix)."""
        if self.check_in_history(filename):
            raise exceptions.ModelError(
                f"Attempted to override math with pre-defined math file '{filename}'."
            )
        self._add_file(MATH_DIR / f"{filename}.yaml", filename)

    def add_user_defined_file(
        self, relative_filepath: str | Path, model_def_path: str | Path
    ) -> None:
        """Add user-defined Calliope math, relative to the model definition path."""
        math_name = str(relative_filepath)
        if self.check_in_history(math_name):
            raise exceptions.ModelError(
                f"Attempted to override math with user-defined math file '{math_name}'"
            )
        self._add_file(relative_path(model_def_path, relative_filepath), math_name)

    def validate(self, extra_math: dict | None = None):
        """Test current math and optional external math against the MATH schema.

        Args:
            extra_math (dict | None, optional): Temporary math to merge into the check. Defaults to None.
        """
        math_to_validate = deepcopy(self.data)
        if extra_math is not None:
            math_to_validate.union(AttrDict(extra_math), allow_override=True)
        validate_dict(math_to_validate, MATH_SCHEMA, "math")
        LOGGER.info("ModelMath: validated math against schema.")
