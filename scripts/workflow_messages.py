"""Emit Telegram-Markdown messages separated by NUL bytes.

Used by .github/workflows/*.yml; consumed by a bash while-read loop
that splits on NUL and sends each to Telegram.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from shared.telegram_formatter import emit_messages_for_workflow


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tickets-json", type=Path, required=True)
    p.add_argument("--date", type=str, required=True)
    p.add_argument("--format", choices=["picks"], default="picks")
    args = p.parse_args()

    msgs = emit_messages_for_workflow(args.tickets_json, args.date)
    sys.stdout.write("\x00".join(msgs))
    sys.stdout.write("\x00")
    return 0


if __name__ == "__main__":
    sys.exit(main())
