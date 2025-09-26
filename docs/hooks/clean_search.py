# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Clean up rendered notebook entries in mkdocs search index."""

import json
import logging
from pathlib import Path

logger = logging.getLogger("mkdocs")


def on_post_build(config: dict):
    """Hook to run after the build process."""
    search_file = Path(config["site_dir"]) / "search" / "search_index.json"

    search_data = json.loads(search_file.read_text(encoding="utf-8"))
    # 1. remove code blocks from search results
    cleaned_search_data = [
        i for i in search_data["docs"] if not i["text"].startswith("In\u00a0")
    ]
    updated_examples = []
    for i in cleaned_search_data:
        # 2. remove header icon (¶) from end of title text in search
        i["title"] = i["title"].removesuffix("\u00b6")
        # 3. remove cross-ref (#i-am-a-cross-ref) from the first search result of each rendered notebook page so that it acts the same as top-level headers in .md files in search results.
        if (
            i["location"].startswith("examples/")
            and (example := i["location"].split("/")[1]) not in updated_examples
        ):
            i["location"] = f"examples/{example}/"
            updated_examples.append(example)
    search_data["docs"] = cleaned_search_data
    search_file.write_text(json.dumps(search_data), encoding="utf-8")
