"""Sort ``.all-contributorsrc`` by contribution count, then first-contribution date.

Contributors are ordered by:

1. number of contributions/emojis (``len(contributions)``), descending;
2. ``firstContribution`` date (``YYYY-MM-DD``), ascending — earliest first;
3. ``login`` (case-insensitive), as a deterministic final tiebreak.
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[2] / ".all-contributorsrc"
DATE_FIELD = "firstContribution"


def is_valid_date(value: object) -> bool:
    """Check that a value is a valid date in YYYY-MM-DD format."""
    valid = False
    if isinstance(value, str):
        try:
            valid = date.fromisoformat(value).isoformat() == value
        except ValueError:
            pass
    return valid


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

    invalid = [c["login"] for c in contributors if not is_valid_date(c.get(DATE_FIELD))]
    if invalid:
        print(
            f"error: contributors missing a valid '{DATE_FIELD}' (YYYY-MM-DD) date "
            f"in {args.path.name}:",
            file=sys.stderr,
        )
        for login in invalid:
            print(
                f"  - {login}: set '{DATE_FIELD}' to a valid YYYY-MM-DD date",
                file=sys.stderr,
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
