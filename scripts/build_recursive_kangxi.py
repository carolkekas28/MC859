#!/usr/bin/env python3
"""Recursively decompose IDS expressions until Kangxi radical leaves."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

IDS_OPERATORS = set("⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻")


def codepoint_from_char(ch: str) -> str:
    return f"U+{ord(ch):04X}"


def terminal_components(expr: str) -> list[str]:
    return [ch for ch in expr if ch not in IDS_OPERATORS and not ch.isspace()]


def load_kangxi_reference(path: Path) -> tuple[set[str], dict[str, dict[str, str]]]:
    kangxi_chars: set[str] = set()
    by_symbol: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            symbol = row["symbol"]
            kangxi_chars.add(symbol)
            by_symbol[symbol] = row
    return kangxi_chars, by_symbol


def load_ids_selected(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_expr_maps(ids_rows: Iterable[dict[str, str]]) -> tuple[dict[str, str], dict[str, str]]:
    cp_to_expr: dict[str, str] = {}
    cp_to_tag: dict[str, str] = {}
    for row in ids_rows:
        expr = row.get("ids_expr_graph_selected") or row.get("ids_expr_clean_selected") or ""
        cp_to_expr[row["codepoint"]] = expr
        cp_to_tag[row["codepoint"]] = row.get("selected_tag", "NONE")
    return cp_to_expr, cp_to_tag


def recursive_expand(
    root_cp: str,
    root_char: str,
    root_expr: str,
    selected_tag: str,
    cp_to_expr: dict[str, str],
    kangxi_chars: set[str],
    max_depth: int,
) -> tuple[list[dict[str, str]], list[tuple[str, int]]]:
    rows: list[dict[str, str]] = []
    leaves: list[tuple[str, int]] = []

    def visit(component: str, depth: int, visited: set[str], order_hint: int) -> None:
        if component in kangxi_chars:
            rows.append(
                {
                    "character_codepoint": root_cp,
                    "character": root_char,
                    "selected_ids_expr": root_expr,
                    "source_component": component,
                    "expanded_expr": "",
                    "kangxi_leaf": component,
                    "recursion_depth": str(depth),
                    "expansion_status": "already_kangxi",
                    "selected_tag": selected_tag,
                }
            )
            leaves.append((component, order_hint))
            return

        cp = codepoint_from_char(component)
        if depth >= max_depth:
            rows.append(
                {
                    "character_codepoint": root_cp,
                    "character": root_char,
                    "selected_ids_expr": root_expr,
                    "source_component": component,
                    "expanded_expr": "",
                    "kangxi_leaf": "",
                    "recursion_depth": str(depth),
                    "expansion_status": "max_depth",
                    "selected_tag": selected_tag,
                }
            )
            return

        if cp in visited:
            rows.append(
                {
                    "character_codepoint": root_cp,
                    "character": root_char,
                    "selected_ids_expr": root_expr,
                    "source_component": component,
                    "expanded_expr": "",
                    "kangxi_leaf": "",
                    "recursion_depth": str(depth),
                    "expansion_status": "cycle_detected",
                    "selected_tag": selected_tag,
                }
            )
            return

        expr = cp_to_expr.get(cp, "")
        if not expr:
            rows.append(
                {
                    "character_codepoint": root_cp,
                    "character": root_char,
                    "selected_ids_expr": root_expr,
                    "source_component": component,
                    "expanded_expr": "",
                    "kangxi_leaf": "",
                    "recursion_depth": str(depth),
                    "expansion_status": "missing_ids",
                    "selected_tag": selected_tag,
                }
            )
            return

        children = terminal_components(expr)
        rows.append(
            {
                "character_codepoint": root_cp,
                "character": root_char,
                "selected_ids_expr": root_expr,
                "source_component": component,
                "expanded_expr": expr,
                "kangxi_leaf": "",
                "recursion_depth": str(depth),
                "expansion_status": "expanded",
                "selected_tag": selected_tag,
            }
        )

        next_visited = set(visited)
        next_visited.add(cp)
        for idx, child in enumerate(children, start=1):
            visit(child, depth + 1, next_visited, order_hint * 1000 + idx)

    for idx, component in enumerate(terminal_components(root_expr), start=1):
        visit(component, 0, {root_cp}, idx)

    return rows, leaves


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ids-selected",
        type=Path,
        default=Path("data/processed/ids_selected.csv"),
        help="Input ids_selected.csv path",
    )
    parser.add_argument(
        "--kangxi-reference",
        type=Path,
        default=Path("data/reference/kangxi_radicals.csv"),
        help="Kangxi reference CSV path",
    )
    parser.add_argument(
        "--recursive-output",
        type=Path,
        default=Path("data/processed/ids_recursive_kangxi.csv"),
        help="Recursive decomposition output path",
    )
    parser.add_argument(
        "--edges-output",
        type=Path,
        default=Path("data/processed/radical_character_edges.csv"),
        help="Bipartite edges output path",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=8,
        help="Max recursive depth before fallback",
    )
    args = parser.parse_args()

    ids_rows = load_ids_selected(args.ids_selected)
    kangxi_chars, kangxi_meta = load_kangxi_reference(args.kangxi_reference)
    cp_to_expr, cp_to_tag = build_expr_maps(ids_rows)

    recursive_rows: list[dict[str, str]] = []
    edge_rows: list[dict[str, str]] = []
    seen_edges: set[tuple[str, str]] = set()

    for row in ids_rows:
        root_cp = row["codepoint"]
        root_char = row["char"]
        root_expr = cp_to_expr.get(root_cp, "")
        selected_tag = cp_to_tag.get(root_cp, "NONE")
        if not root_expr:
            continue

        rec_rows, leaves = recursive_expand(
            root_cp=root_cp,
            root_char=root_char,
            root_expr=root_expr,
            selected_tag=selected_tag,
            cp_to_expr=cp_to_expr,
            kangxi_chars=kangxi_chars,
            max_depth=args.max_depth,
        )
        recursive_rows.extend(rec_rows)

        leaves_sorted = sorted(leaves, key=lambda item: item[1])
        for component_order, (leaf, _) in enumerate(leaves_sorted, start=1):
            edge_key = (root_cp, leaf)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            meta = kangxi_meta[leaf]
            edge_rows.append(
                {
                    "character_codepoint": root_cp,
                    "character": root_char,
                    "kangxi_radical": leaf,
                    "kangxi_radical_number": meta["radical_number"],
                    "kangxi_index_codepoint": meta["index_codepoint"],
                    "kangxi_real_codepoint": meta["real_codepoint"],
                    "component_order": str(component_order),
                    "ids_expr_clean_selected": root_expr,
                    "selected_tag": selected_tag,
                    "edge_type": "kangxi_radical_character",
                    "weight": "1.0",
                }
            )

    write_csv(
        args.recursive_output,
        recursive_rows,
        [
            "character_codepoint",
            "character",
            "selected_ids_expr",
            "source_component",
            "expanded_expr",
            "kangxi_leaf",
            "recursion_depth",
            "expansion_status",
            "selected_tag",
        ],
    )
    write_csv(
        args.edges_output,
        edge_rows,
        [
            "character_codepoint",
            "character",
            "kangxi_radical",
            "kangxi_radical_number",
            "kangxi_index_codepoint",
            "kangxi_real_codepoint",
            "component_order",
            "ids_expr_clean_selected",
            "selected_tag",
            "edge_type",
            "weight",
        ],
    )

    print(f"Wrote recursive rows: {len(recursive_rows)} -> {args.recursive_output}")
    print(f"Wrote edge rows: {len(edge_rows)} -> {args.edges_output}")


if __name__ == "__main__":
    main()
