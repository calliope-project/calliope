#!/usr/bin/env python3
"""Sort ``.all-contributorsrc`` by contribution count, then first-contribution date.

Contributors are ordered by:

1. number of contributions/emojis (``len(contributions)``), descending;
2. ``firstContribution`` date (``YYYY-MM-DD``), ascending — earliest first;
3. ``login`` (case-insensitive), as a deterministic final tiebreak.

Every contributor entry must carry a ``firstContribution`` field. This is stored
directly in ``.all-contributorsrc`` (the all-contributors tooling preserves the
extra key), so no separate cache is needed. When the all-contributors bot adds a
new contributor it lands without this field, so add it by hand (the date of their
first commit/contribution) before re-running this script.

This is the only script needed to maintain the ordering going forward; the
``git-contrib-*.sh`` helpers were one-time tooling used to seed the dates.

Usage (run from anywhere):
    python .github/scripts/git-contrib-sort.py          # sort the file in place
    python .github/scripts/git-contrib-sort.py --check  # verify only, non-zero if unsorted
    python .github/scripts/git-contrib-sort.py --path X # operate on another config file
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# This script lives in .github/scripts/; the config is at the repository root.
DEFAULT_PATH = Path(__file__).resolve().parents[2] / ".all-contributorsrc"
DATE_FIELD = "firstContribution"


def sort_key(contributor: dict) -> tuple[int, str, str]:
    """Return the ordering key: most contributions first, then earliest date."""
    return (
        -len(contributor["contributions"]),
        contributor[DATE_FIELD],
        contributor["login"].lower(),
    )


def render(data: dict) -> str:
    """Serialise the config exactly as the all-contributors tooling does."""
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Sort (or check) the contributors list. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_PATH,
        help="path to .all-contributorsrc (default: repo root .all-contributorsrc)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the file is not already sorted; do not write",
    )
    args = parser.parse_args(argv)

    original = args.path.read_text(encoding="utf-8")
    data = json.loads(original)
    contributors = data["contributors"]

    missing = [c["login"] for c in contributors if not c.get(DATE_FIELD)]
    if missing:
        print(
            f"error: contributors missing a '{DATE_FIELD}' (YYYY-MM-DD) date "
            f"in {args.path.name}:",
            file=sys.stderr,
        )
        for login in missing:
            print(
                f"  - {login}: add a '{DATE_FIELD}' date to this entry", file=sys.stderr
            )
        return 1

    data["contributors"] = sorted(contributors, key=sort_key)
    updated = render(data)

    if updated == original:
        return 0

    if args.check:
        print(
            f"error: {args.path.name} is not sorted; run this script without --check to fix.",
            file=sys.stderr,
        )
        return 1

    args.path.write_text(updated, encoding="utf-8")
    print(f"sorted {len(contributors)} contributors in {args.path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
