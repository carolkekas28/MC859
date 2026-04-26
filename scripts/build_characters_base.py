#!/usr/bin/env python3
"""Build base character table from Unihan_Readings + IDS coverage."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def codepoint_to_char(codepoint: str) -> str:
    return chr(int(codepoint[2:], 16))


def parse_definitions(readings_path: Path) -> dict[str, str]:
    definitions: dict[str, str] = {}
    with readings_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            codepoint, field, value = parts
            if field == "kDefinition":
                definitions[codepoint] = value.strip()
    return definitions


def parse_ids_codepoints(ids_path: Path) -> set[str]:
    codepoints: set[str] = set()
    with ids_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and parts[0].startswith("U+"):
                codepoints.add(parts[0])
    return codepoints


def write_characters_table(
    definitions: dict[str, str],
    ids_codepoints: set[str],
    output_csv: Path,
) -> None:
    all_codepoints = sorted(set(definitions.keys()) | ids_codepoints)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["codepoint", "char", "definition", "has_definition", "in_ids"],
        )
        writer.writeheader()
        for cp in all_codepoints:
            definition = definitions.get(cp, "")
            writer.writerow(
                {
                    "codepoint": cp,
                    "char": codepoint_to_char(cp),
                    "definition": definition,
                    "has_definition": "true" if definition else "false",
                    "in_ids": "true" if cp in ids_codepoints else "false",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--readings",
        type=Path,
        default=Path("Unihan/Unihan_Readings.txt"),
        help="Path to Unihan_Readings.txt",
    )
    parser.add_argument(
        "--ids",
        type=Path,
        default=Path("Unihan/ids.txt"),
        help="Path to ids.txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/characters.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()

    definitions = parse_definitions(args.readings)
    ids_codepoints = parse_ids_codepoints(args.ids)
    write_characters_table(definitions, ids_codepoints, args.output)

    print(f"Wrote characters table to: {args.output}")
    print(f"Rows: {len(set(definitions) | ids_codepoints)}")
    print(f"With definition: {len(definitions)}")
    print(f"In IDS: {len(ids_codepoints)}")


if __name__ == "__main__":
    main()
