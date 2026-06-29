"""Helpers for local developer tooling checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def is_uv_managed_executable(executable: Path) -> bool:
    executable_text = str(executable)
    return (
        ".venv" in executable.parts
        or ".local/share/uv" in executable_text
        or "/opt/uv/python/" in executable_text
    )


def uv_runtime_summary() -> dict[str, str | bool]:
    executable = Path(sys.executable).resolve()
    return {
        "executable": str(executable),
        "uv_managed": is_uv_managed_executable(executable),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = uv_runtime_summary()
    if not summary["uv_managed"]:
        message = f"Expected a uv-managed Python executable, got {summary['executable']}"
        if args.json:
            print(json.dumps({"error": message, **summary}, indent=2))
        else:
            print(message)
        return 1

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"uv-managed runtime detected: {summary['executable']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
