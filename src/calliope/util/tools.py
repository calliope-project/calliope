# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).


import importlib.resources
import textwrap
from pathlib import Path
from typing import Any, Optional, TypeVar

from IPython.display import Code
from typing_extensions import ParamSpec

P = ParamSpec("P")
T = TypeVar("T")


def relative_path(base_path_file, path) -> Path:
    """
    If ``path`` is not absolute, it is interpreted as relative to the
    path of the given ``base_path_file``.

    """
    # Check if base_path_file is a string because it might be an AttrDict
    path = Path(path)
    if base_path_file is not None:
        base_path_file = Path(base_path_file)
        if base_path_file.is_file():
            base_path_file = base_path_file.parent
        if not path.is_absolute():
            path = base_path_file.absolute() / path
    return path


def listify(var: Any) -> list:
    """Transform the input into a list.

    If it is a non-string iterable, it will be coerced to a list.
    If it is a string or non-iterable (numeric, boolean, ...), then it will be a single item list.

    Args:
        var (Any): Value to transform.

    Returns:
        list: List containing `var` or elements of `var` (if input was a non-string iterable).
    """
    if not isinstance(var, str) and hasattr(var, "__iter__"):
        var = list(var)
    else:
        var = [var]
    return var


def yaml_snippet(filepath_relative: str | Path, fence_id: Optional[str] = None) -> Code:
    """View YAML file contents with the option to display only a snippet of the file contents.

    Snippets are created based on the commented fences in the YAML file.


    Args:
        filepath_relative (str | Path): _description_
        fence_id (Optional[str], optional): _description_. Defaults to None.

    Returns:
        Code: _description_

    Examples:
        model_directory/input_file.yaml:
        `yaml
        foo:
        # my-fence-start
        bar: 1
        # my-fence-end
        `

        Calling:
        `python
        yaml_snippet("input_file.yaml", "my-fence")
        `

        Will output:
        `yaml
        bar: 1
        `
    """
    file = importlib.resources.files("calliope") / filepath_relative
    snippet = file.read_text()
    if fence_id is not None:
        snippet = textwrap.dedent(
            snippet.split(f"# {fence_id}-start")[1].split(f"# {fence_id}-end")[0]
        )
    return Code(snippet, language="yaml")
