#!/usr/bin/env python3
"""Export structural and weighted bipartite graphs to GraphML."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import networkx as nx


def add_nodes_from_edges(
    graph: nx.Graph,
    row: dict[str, str],
) -> tuple[str, str]:
    char_cp = row["character_codepoint"]
    char_symbol = row["character"]
    radical = row["kangxi_radical"]
    radical_num = row.get("kangxi_radical_number", "")

    char_node = f"char:{char_cp}"
    radical_node = f"rad:{radical_num}:{radical}"

    if char_node not in graph:
        graph.add_node(
            char_node,
            node_type="character",
            codepoint=char_cp,
            symbol=char_symbol,
            bipartite="character",
        )
    if radical_node not in graph:
        graph.add_node(
            radical_node,
            node_type="radical",
            radical_symbol=radical,
            radical_number=radical_num,
            bipartite="radical",
        )
    return radical_node, char_node


def export_structural_graphml(edges_csv: Path, output_graphml: Path) -> tuple[int, int]:
    g = nx.Graph(name="kangxi_character_structural")
    with edges_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            radical_node, char_node = add_nodes_from_edges(g, row)
            g.add_edge(
                radical_node,
                char_node,
                edge_type=row.get("edge_type", "kangxi_radical_character"),
                weight=float(row.get("weight", "1.0")),
                selected_tag=row.get("selected_tag", "NONE"),
                ids_expr=row.get("ids_expr_clean_selected", ""),
            )

    output_graphml.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(g, output_graphml)
    return g.number_of_nodes(), g.number_of_edges()


def export_weighted_graphml(weighted_edges_csv: Path, output_graphml: Path) -> tuple[int, int]:
    g = nx.Graph(name="kangxi_character_semantic_weighted")
    with weighted_edges_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            radical_node, char_node = add_nodes_from_edges(g, row)
            g.add_edge(
                radical_node,
                char_node,
                edge_type=row.get("edge_type", "kangxi_radical_character"),
                weight_struct=float(row.get("weight", "1.0")),
                weight_cosine=float(row.get("weight_cosine", "0.0")),
                weight_norm_01=float(row.get("weight_norm_01", "0.0")),
                selected_tag=row.get("selected_tag", "NONE"),
                ids_expr=row.get("ids_expr_clean_selected", ""),
                embedding_model=row.get("embedding_model", ""),
            )

    output_graphml.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(g, output_graphml)
    return g.number_of_nodes(), g.number_of_edges()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--edges-csv",
        type=Path,
        default=Path("data/processed/radical_character_edges.csv"),
        help="Structural edge CSV path",
    )
    parser.add_argument(
        "--weighted-edges-csv",
        type=Path,
        default=Path("data/processed/radical_character_edges_weighted.csv"),
        help="Weighted edge CSV path",
    )
    parser.add_argument(
        "--structural-output",
        type=Path,
        default=Path("data/graphs/graph_structural.graphml"),
        help="Output GraphML path for structural graph",
    )
    parser.add_argument(
        "--weighted-output",
        type=Path,
        default=Path("data/graphs/graph_semantic_weighted.graphml"),
        help="Output GraphML path for semantic weighted graph",
    )
    args = parser.parse_args()

    s_nodes, s_edges = export_structural_graphml(args.edges_csv, args.structural_output)
    print(f"Structural GraphML: {args.structural_output}")
    print(f"  nodes={s_nodes}, edges={s_edges}")

    w_nodes, w_edges = export_weighted_graphml(args.weighted_edges_csv, args.weighted_output)
    print(f"Weighted GraphML: {args.weighted_output}")
    print(f"  nodes={w_nodes}, edges={w_edges}")


if __name__ == "__main__":
    main()
