#!/usr/bin/env python3
"""Build selected IDS decomposition table (one row per character)."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

TAG_RE = re.compile(r"\[([A-Z]+)\]")
PRECEDENCE = {"G": 0, "T": 1, "J": 2, "K": 3, "V": 4, "X": 5}
# Circled/enclosed numerals used as graphic placeholders in IDS.
CIRCLED_NUMERAL_RE = re.compile(r"[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]")


def in_target_range(codepoint: str, min_cp: int, max_cp: int) -> bool:
    if not codepoint.startswith("U+"):
        return False
    cp = int(codepoint[2:], 16)
    return min_cp <= cp <= max_cp


def clean_expr(expr: str) -> str:
    """Remove tag blocks like [G], [TJK], [X]."""
    return TAG_RE.sub("", expr).strip()


def normalize_expr_for_graph(expr_clean: str) -> str:
    """Remove low-semantic graphic placeholders before graph edges."""
    return CIRCLED_NUMERAL_RE.sub("", expr_clean).strip()


def extract_tags(expr: str) -> list[str]:
    """Extract individual tags from all tag blocks in an expression."""
    tags: list[str] = []
    for match in TAG_RE.finditer(expr):
        tags.extend(list(match.group(1)))
    return tags


def best_tag(expr: str) -> tuple[int, str]:
    """Return (rank, tag) according to precedence."""
    tags = extract_tags(expr)
    if not tags:
        return (len(PRECEDENCE), "NONE")
    ranked = sorted((PRECEDENCE.get(tag, 999), tag) for tag in tags)
    return ranked[0]


def parse_ids_selected(ids_path: Path, min_cp: int, max_cp: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with ids_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue

            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue

            codepoint, char = parts[0], parts[1]
            if not in_target_range(codepoint, min_cp, max_cp):
                continue
            exprs = parts[2:]

            chosen_idx: int | None = None
            chosen_key: tuple[int, int] | None = None
            chosen_tag = "NONE"

            for idx, expr in enumerate(exprs):
                rank, tag = best_tag(expr)
                key = (rank, idx)  # rank first, then keep first occurrence
                if chosen_key is None or key < chosen_key:
                    chosen_key = key
                    chosen_idx = idx
                    chosen_tag = tag

            if chosen_idx is None:
                continue

            selected_raw = exprs[chosen_idx]
            selected_clean = clean_expr(selected_raw)
            selected_graph = normalize_expr_for_graph(selected_clean)
            rows.append(
                {
                    "codepoint": codepoint,
                    "char": char,
                    "ids_expr_raw_selected": selected_raw,
                    "ids_expr_clean_selected": selected_clean,
                    "ids_expr_graph_selected": selected_graph,
                    "selected_tag": chosen_tag,
                    "selected_by_precedence": "true" if len(exprs) > 1 else "false",
                    "variant_count": str(len(exprs)),
                    "selection_reason": (
                        f"precedence={chosen_tag};rank={chosen_key[0]};variant_index={chosen_idx}"
                    ),
                }
            )
    return rows


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "codepoint",
                "char",
                "ids_expr_raw_selected",
                "ids_expr_clean_selected",
                "ids_expr_graph_selected",
                "selected_tag",
                "selected_by_precedence",
                "variant_count",
                "selection_reason",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ids",
        type=Path,
        default=Path("Unihan/ids.txt"),
        help="Path to ids.txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/ids_selected.csv"),
        help="Output CSV path",
    )
    parser.add_argument("--min-codepoint", type=str, default="U+3400", help="Minimum codepoint to include")
    parser.add_argument("--max-codepoint", type=str, default="U+9FFF", help="Maximum codepoint to include")
    args = parser.parse_args()

    min_cp = int(args.min_codepoint[2:], 16)
    max_cp = int(args.max_codepoint[2:], 16)
    rows = parse_ids_selected(args.ids, min_cp, max_cp)
    write_csv(rows, args.output)

    print(f"Wrote ids_selected table to: {args.output}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
