"""Private traced child for a consumed zero-update discovery run."""
from __future__ import annotations

import sys
from pathlib import Path

from .profile_discovery import _worker


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 4:
        raise SystemExit("private worker requires consumed binding, result, source and phase paths")
    return _worker(*(Path(value) for value in args))


if __name__ == "__main__":
    raise SystemExit(main())
