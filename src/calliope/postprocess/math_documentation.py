"""Post-processing functions to create math documentation."""

import logging
import typing
from copy import deepcopy
from pathlib import Path
from typing import Literal, overload

from calliope import AttrDict, Model
from calliope.backend import ALLOWED_MATH_FILE_FORMATS, BackendSetup, LatexBackendModel

LOGGER = logging.getLogger(__name__)


class MathDocumentation:
    """For math documentation."""

    def __init__(
        self, model: Model, include: Literal["all", "valid"] = "all", **kwargs
    ) -> None:
        """Math documentation builder/writer.

        Backend is always built by default.

        Args:
            model (Model): initialised Callipe model instance.
            include (Literal["all", "valid"], optional):
                Either include all possible math equations ("all") or only those for
                which at least one "where" case is valid ("valid"). Defaults to "all".
            **kwargs: kwargs for the LaTeX backend.
        """
        # TODO-Ivan: should be model.name?
        self.name: str = model.inputs.attrs["name"] + " math"
        self.math: AttrDict = deepcopy(model.math)

        setup = BackendSetup(inputs=model._model_data, math=model.math)
        self.backend: LatexBackendModel = LatexBackendModel(setup, include, **kwargs)
        self.backend.add_all_math()

    # Expecting string if not giving filename.
    @overload
    def write(
        self,
        filename: Literal[None] = None,
        mkdocs_tabbed: bool = False,
        format: ALLOWED_MATH_FILE_FORMATS | None = None,
    ) -> str: ...

    # Expecting None (and format arg is not needed) if giving filename.
    @overload
    def write(self, filename: str | Path, mkdocs_tabbed: bool = False) -> None: ...

    def write(
        self,
        filename: str | Path | None = None,
        mkdocs_tabbed: bool = False,
        format: ALLOWED_MATH_FILE_FORMATS | None = None,
    ) -> str | None:
        """Write model documentation.

        Args:
            filename (str | Path | None, optional):
                If given, will write the built mathematical formulation to a file with
                the given extension as the file format. Defaults to None.
            mkdocs_tabbed (bool, optional):
                If True and Markdown docs are being generated, the equations will be on
                a tab and the original YAML math definition will be on another tab.
                Defaults to False.
            format (_ALLOWED_MATH_FILE_FORMATS | None, optional):
                Not required if filename is given (as the format will be automatically inferred).
                Required if expecting a string return from calling this function. The LaTeX math will be embedded in a document of the given format (tex=LaTeX, rst=reStructuredText, md=Markdown).
                Defaults to None.

        Raises:
            ValueError: The file format (inferred automatically from `filename` or given by `format`) must be one of ["tex", "rst", "md"].

        Returns:
            str | None:
                If `filename` is None, the built mathematical formulation documentation will be returned as a string.
        """
        if format is None and filename is not None:
            format = Path(filename).suffix.removeprefix(".")  # type: ignore
            LOGGER.info(
                f"Inferring math documentation format from filename as `{format}`."
            )

        allowed_formats = typing.get_args(ALLOWED_MATH_FILE_FORMATS)
        if format is None or format not in allowed_formats:
            raise ValueError(
                f"Math documentation format must be one of {allowed_formats}, received `{format}`"
            )
        populated_doc = self.backend.generate_math_doc(format, mkdocs_tabbed)

        if filename is None:
            return populated_doc
        else:
            Path(filename).write_text(populated_doc)
            return None
