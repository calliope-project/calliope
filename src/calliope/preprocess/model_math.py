"""Calliope math handling."""

import logging
from copy import deepcopy
from pathlib import Path

from calliope.attrdict import AttrDict
from calliope.exceptions import ModelError
from calliope.util.schema import MATH_SCHEMA, validate_dict
from calliope.util.tools import relative_path

MATH_DIR = Path(__file__).parent / "math"
LOGGER = logging.getLogger(__name__)


class ModelMath:
    """Calliope math preprocessing."""

    ATTRS_TO_SAVE = ("applied_files",)

    def __init__(self, model_def_path: str | Path | None, math_to_add: list | dict):
        """Contains and handles Calliope YAML math definitions.

        Args:
            model_def_path (str): path to model definition.
            math_to_add (list | dict): Either a list of math file paths or a saved dictionary.
        """
        self.applied_files: set[str]
        self.math: AttrDict = AttrDict()
        self.def_path: str | Path | None = model_def_path

        if isinstance(math_to_add, list):
            self._init_from_list(math_to_add)
        elif isinstance(math_to_add, dict):
            self._init_from_dict(math_to_add)

    def _init_from_list(self, math_to_add: list[str]) -> None:
        """Load the base math and optionally merge additional math.

        Internal math has no suffix. User defined math must be relative to the model definition file.

        Args: math_to_add (list): References to math files to merge.

        Returns:
            AttrDict: Dictionary of math (constraints, variables, objectives, and global expressions).
        """
        math_files = ["base"] + math_to_add
        for filename in math_files:
            self.add_math(filename)

    def _init_from_dict(self, math_dict: dict) -> None:
        """Load math from a dictionary definition.

        Args:
            math_dict (dict): dictionary with model math.
        """
        self.math = AttrDict(math_dict)
        for attr in self.ATTRS_TO_SAVE:
            setattr(self, attr, self.math[attr])
            del self.math[attr]

    def to_dict(self) -> dict:
        """Translate into a dictionary."""
        math = deepcopy(self.math)
        for attr in self.ATTRS_TO_SAVE:
            math[attr] = getattr(self, attr)
        return math

    def add_math(self, math_file: str | Path, override=False) -> None:
        """If not given in the add_math list, override model math with run mode math."""
        file = str(math_file)

        if file in self.applied_files and not override:
            raise ModelError(f"Attempted to override existing math definition {file}.")

        if f"{file}".endswith((".yaml", ".yml")):
            yaml_filepath = relative_path(self.def_path, file)
        else:
            yaml_filepath = MATH_DIR / f"{file}.yaml"

        try:
            math = AttrDict.from_yaml(yaml_filepath)
        except FileNotFoundError:
            raise ModelError(f"Failed to load math from {yaml_filepath}")

        self.math.union(math, allow_override=True)
        self.applied_files.add(file)
        LOGGER.debug(f"Adding {file} math formulation.")

    def validate(self) -> None:
        """Test that the model math is correct."""
        validate_dict(self.math, MATH_SCHEMA, "math")
