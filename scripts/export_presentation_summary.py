from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.presentation import (
    build_presentation_rows,
    write_presentation_csv,
    write_presentation_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a presentation-friendly CSV/JSON summary from a compare_runs CSV."
    )
    parser.add_argument("--input-csv", required=True, help="Path to a compare_runs CSV.")
    parser.add_argument("--output-csv", default=None, help="Optional path for the slim presentation CSV.")
    parser.add_argument("--output-json", default=None, help="Optional path for the slim presentation JSON.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_csv = Path(args.input_csv)
    rows = build_presentation_rows(input_csv)

    default_stem = input_csv.stem.replace("-compare", "-presentation-summary")
    output_csv = Path(args.output_csv) if args.output_csv else input_csv.with_name(f"{default_stem}.csv")
    output_json = Path(args.output_json) if args.output_json else input_csv.with_name(f"{default_stem}.json")

    write_presentation_csv(output_csv, rows)
    write_presentation_json(output_json, rows)

    print(
        json.dumps(
            {
                "input_csv": str(input_csv),
                "output_csv": str(output_csv),
                "output_json": str(output_json),
                "row_count": len(rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
