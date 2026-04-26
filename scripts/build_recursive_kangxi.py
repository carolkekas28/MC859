#!/usr/bin/env python3
"""Recursively decompose IDS expressions until Kangxi radical leaves."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable

IDS_OPERATORS = set("⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻")
IDS_BINARY_OPERATORS = set("⿰⿱⿴⿵⿶⿷⿸⿹⿺⿻")
IDS_TERNARY_OPERATORS = set("⿲⿳")
# Common component variants mapped to Kangxi canonical forms.
COMPONENT_TO_KANGXI = {
    "王": "玉",
    "亻": "人",
    "刂": "刀",
    "扌": "手",
    "氵": "水",
    "忄": "心",
    "攵": "攴",
    "灬": "火",
    "罒": "网",
    "讠": "言",
    "糹": "糸",
    "饣": "食",
    "飠": "食",
    "钅": "金",
    "纟": "糸",
    "犭": "犬",
    "礻": "示",
    "衤": "衣",
    "艹": "艸",
    "爫": "爪",
    "丷": "八",
    "⺊": "卜",
    "𧾷": "足",
    "辶": "辵",
    "门": "門",
    "页": "頁",
    "车": "車",
    "风": "風",
    "飞": "飛",
    "马": "馬",
    "鱼": "魚",
    "鸟": "鳥",
    "卤": "鹵",
    "麦": "麥",
    "黄": "黃",
    "黾": "黽",
    "齐": "齊",
    "齿": "齒",
    "龙": "龍",
    "龟": "龜",
    "长": "長",
    "贝": "貝",
}


def codepoint_from_char(ch: str) -> str:
    return f"U+{ord(ch):04X}"


def terminal_components(expr: str) -> list[str]:
    return [ch for ch in expr if ch not in IDS_OPERATORS and not ch.isspace()]


def parse_ids_node(expr: str, index: int = 0) -> tuple[object, int]:
    while index < len(expr) and expr[index].isspace():
        index += 1
    if index >= len(expr):
        return ("",), index

    ch = expr[index]
    if ch in IDS_BINARY_OPERATORS:
        left, next_index = parse_ids_node(expr, index + 1)
        right, next_index = parse_ids_node(expr, next_index)
        return (ch, [left, right]), next_index
    if ch in IDS_TERNARY_OPERATORS:
        c1, next_index = parse_ids_node(expr, index + 1)
        c2, next_index = parse_ids_node(expr, next_index)
        c3, next_index = parse_ids_node(expr, next_index)
        return (ch, [c1, c2, c3]), next_index
    return ch, index + 1


def flatten_leaves_with_context(
    node: object,
    parent_op: str | None = None,
    child_side: str | None = None,
) -> list[tuple[str, str | None, str | None]]:
    if isinstance(node, str):
        if not node:
            return []
        return [(node, parent_op, child_side)]
    if not isinstance(node, tuple) or len(node) != 2:
        return []

    op, children = node
    if len(children) == 2:
        sides = ["left", "right"]
    else:
        sides = ["left", "middle", "right"]

    leaves: list[tuple[str, str | None, str | None]] = []
    for side, child in zip(sides, children):
        leaves.extend(flatten_leaves_with_context(child, op, side))
    return leaves


def resolve_component_to_kangxi(
    component: str,
    parent_op: str | None,
    child_side: str | None,
) -> str:
    # Contextual disambiguation for 阜/邑 form.
    if component == "阝" and parent_op == "⿰":
        if child_side == "left":
            return "阜"
        if child_side == "right":
            return "邑"
    # Contextual disambiguation for 月 vs 肉 form.
    if component == "月":
        if parent_op == "⿰" and child_side == "left":
            return "肉"
        return "月"
    return COMPONENT_TO_KANGXI.get(component, component)


def load_kangxi_reference(path: Path) -> tuple[set[str], dict[str, dict[str, str]]]:
    kangxi_chars: set[str] = set()
    by_symbol: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            symbol = row["symbol"]
            kangxi_chars.add(symbol)
            by_symbol[symbol] = row
    return kangxi_chars, by_symbol


def load_krsunicode_main_radical(path: Path, radical_to_symbol: dict[str, str]) -> dict[str, str]:
    """Map codepoint -> Kangxi symbol using Unihan kRSUnicode as fallback."""
    cp_to_symbol: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            cp, field, value = parts
            if field != "kRSUnicode":
                continue
            # Examples: 149.5, 149'.5, 149.5 170.4
            first = value.strip().split(" ")[0]
            m = re.match(r"(\d+)", first)
            if not m:
                continue
            radical_number = m.group(1)
            symbol = radical_to_symbol.get(radical_number)
            if symbol:
                cp_to_symbol[cp] = symbol
    return cp_to_symbol


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
    cp_to_fallback_symbol: dict[str, str],
    max_depth: int,
) -> tuple[list[dict[str, str]], list[tuple[str, int]]]:
    rows: list[dict[str, str]] = []
    leaves: list[tuple[str, int]] = []

    def visit(
        component: str,
        depth: int,
        visited: set[str],
        order_hint: int,
        parent_op: str | None = None,
        child_side: str | None = None,
    ) -> None:
        canonical = resolve_component_to_kangxi(component, parent_op, child_side)
        if canonical in kangxi_chars:
            rows.append(
                {
                    "character_codepoint": root_cp,
                    "character": root_char,
                    "selected_ids_expr": root_expr,
                    "source_component": component,
                    "expanded_expr": "",
                    "kangxi_leaf": canonical,
                    "recursion_depth": str(depth),
                    "expansion_status": "already_kangxi",
                    "selected_tag": selected_tag,
                }
            )
            leaves.append((canonical, order_hint))
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

        parsed, _ = parse_ids_node(expr, 0)
        children_with_context = flatten_leaves_with_context(parsed)
        if expr == component:
            fallback_symbol = cp_to_fallback_symbol.get(cp, "")
            if fallback_symbol:
                rows.append(
                    {
                        "character_codepoint": root_cp,
                        "character": root_char,
                        "selected_ids_expr": root_expr,
                        "source_component": component,
                        "expanded_expr": expr,
                        "kangxi_leaf": fallback_symbol,
                        "recursion_depth": str(depth),
                        "expansion_status": "fallback_krsunicode",
                        "selected_tag": selected_tag,
                    }
                )
                leaves.append((fallback_symbol, order_hint))
                return
            rows.append(
                {
                    "character_codepoint": root_cp,
                    "character": root_char,
                    "selected_ids_expr": root_expr,
                    "source_component": component,
                    "expanded_expr": expr,
                    "kangxi_leaf": "",
                    "recursion_depth": str(depth),
                    "expansion_status": "self_decomposition",
                    "selected_tag": selected_tag,
                }
            )
            return

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
        for idx, (child, next_parent_op, next_child_side) in enumerate(children_with_context, start=1):
            visit(
                child,
                depth + 1,
                next_visited,
                order_hint * 1000 + idx,
                next_parent_op,
                next_child_side,
            )

    root_parsed, _ = parse_ids_node(root_expr, 0)
    root_components = flatten_leaves_with_context(root_parsed)
    for idx, (component, parent_op, child_side) in enumerate(root_components, start=1):
        visit(component, 0, {root_cp}, idx, parent_op, child_side)

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
    parser.add_argument(
        "--irg-sources",
        type=Path,
        default=Path("Unihan/Unihan_IRGSources.txt"),
        help="Path to Unihan_IRGSources.txt for kRSUnicode fallback",
    )
    args = parser.parse_args()

    ids_rows = load_ids_selected(args.ids_selected)
    kangxi_chars, kangxi_meta = load_kangxi_reference(args.kangxi_reference)
    radical_to_symbol = {
        row["radical_number"]: symbol
        for symbol, row in kangxi_meta.items()
        if "SIMPLIFIED" not in row["name_en"]
    }
    cp_to_fallback_symbol = load_krsunicode_main_radical(args.irg_sources, radical_to_symbol)
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
            cp_to_fallback_symbol=cp_to_fallback_symbol,
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
