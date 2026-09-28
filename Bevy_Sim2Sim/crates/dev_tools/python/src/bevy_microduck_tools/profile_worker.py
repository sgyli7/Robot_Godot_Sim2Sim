"""Retired private source worker entry; kept only to reject old invocations."""
from __future__ import annotations

from .authorization import Rejection


def main(argv=None) -> int:
    raise Rejection("Private v2 source worker disabled until a new live scope and budget protocol exists")


if __name__ == "__main__":
    raise SystemExit(main())
