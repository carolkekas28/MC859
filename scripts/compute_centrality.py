#!/usr/bin/env python3
"""Compute graph centrality metrics for radicals and characters.

Calculates Degree, Betweenness, Closeness, Eigenvector and PageRank on the
structural bipartite graph. Metrics that require connectivity (betweenness,
closeness, eigenvector) are computed on the largest connected component;
nodes outside it receive score 0 and are flagged in the output.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import networkx as nx

_START = time.monotonic()


def log(message: str) -> None:
    elapsed = time.monotonic() - _START
    print(f"[{elapsed:7.1f}s] {message}", flush=True)


def build_structural_graph(edges_csv: Path) -> nx.Graph:
    graph = nx.Graph(name="kangxi_character_structural")
    with edges_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
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
                )
            if radical_node not in graph:
                graph.add_node(
                    radical_node,
                    node_type="radical",
                    radical_symbol=radical,
                    radical_number=radical_num,
                    symbol=radical,
                )
            graph.add_edge(radical_node, char_node, weight=1.0)
    return graph


def largest_component_nodes(graph: nx.Graph) -> set[str]:
    components = list(nx.connected_components(graph))
    largest = max(components, key=len)
    log(
        f"Connected components: {len(components)} "
        f"(largest={len(largest)} nodes, {len(components[0]) if len(components) == 1 else '...'})"
    )
    return set(largest)


def compute_metrics(
    graph: nx.Graph,
    gcc: nx.Graph,
    gcc_nodes: set[str],
    betweenness_k: int | None,
) -> dict[str, dict[str, float]]:
    log("Computing degree...")
    degree = dict(graph.degree())
    degree_centrality = nx.degree_centrality(graph)

    if betweenness_k is not None and betweenness_k > 0 and betweenness_k < gcc.number_of_nodes():
        log(
            f"Computing approximate betweenness on largest component "
            f"(k={betweenness_k} sources)..."
        )
        betweenness_gcc = nx.betweenness_centrality(gcc, k=betweenness_k, normalized=True, seed=42)
        betweenness_note = f"approximate (k={betweenness_k})"
    else:
        log(f"Computing exact betweenness on largest component ({gcc.number_of_nodes()} nodes)...")
        betweenness_gcc = nx.betweenness_centrality(gcc, normalized=True)
        betweenness_note = "exact"
    betweenness = {node: 0.0 for node in graph.nodes}
    betweenness.update(betweenness_gcc)

    log("Computing closeness on largest component...")
    closeness_gcc = nx.closeness_centrality(gcc)
    closeness = {node: 0.0 for node in graph.nodes}
    closeness.update(closeness_gcc)

    log("Computing eigenvector centrality on largest component...")
    try:
        eigenvector_gcc = nx.eigenvector_centrality(gcc, max_iter=1000, tol=1e-6)
    except nx.PowerIterationFailedConvergence:
        log("Eigenvector did not converge; retrying with more iterations...")
        eigenvector_gcc = nx.eigenvector_centrality(gcc, max_iter=5000, tol=1e-5)
    eigenvector = {node: 0.0 for node in graph.nodes}
    eigenvector.update(eigenvector_gcc)

    log("Computing PageRank on full graph...")
    pagerank = nx.pagerank(graph, alpha=0.85)

    return {
        "degree": {n: float(degree[n]) for n in graph.nodes},
        "degree_centrality": degree_centrality,
        "betweenness_centrality": betweenness,
        "closeness_centrality": closeness,
        "eigenvector_centrality": eigenvector,
        "pagerank": pagerank,
        "in_largest_component": {n: float(n in gcc_nodes) for n in graph.nodes},
        "betweenness_note": betweenness_note,
    }


def node_rows(graph: nx.Graph, metrics: dict[str, dict[str, float]]) -> list[dict[str, str | float | int]]:
    rows: list[dict[str, str | float | int]] = []
    for node in graph.nodes:
        attrs = graph.nodes[node]
        node_type = attrs.get("node_type", "")
        row: dict[str, str | float | int] = {
            "node_id": node,
            "node_type": node_type,
            "in_largest_component": int(metrics["in_largest_component"][node]),
            "degree": int(metrics["degree"][node]),
            "degree_centrality": metrics["degree_centrality"][node],
            "betweenness_centrality": metrics["betweenness_centrality"][node],
            "closeness_centrality": metrics["closeness_centrality"][node],
            "eigenvector_centrality": metrics["eigenvector_centrality"][node],
            "pagerank": metrics["pagerank"][node],
        }
        if node_type == "radical":
            row["kangxi_radical"] = attrs.get("radical_symbol", "")
            row["kangxi_radical_number"] = attrs.get("radical_number", "")
            row["symbol"] = attrs.get("radical_symbol", "")
            row["codepoint"] = ""
        else:
            row["codepoint"] = attrs.get("codepoint", "")
            row["symbol"] = attrs.get("symbol", "")
            row["kangxi_radical"] = ""
            row["kangxi_radical_number"] = ""
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def top_examples(rows: list[dict], metric: str, n: int = 10) -> list[dict]:
    return sorted(rows, key=lambda r: float(r[metric]), reverse=True)[:n]


def write_summary(path: Path, radical_rows: list[dict], character_rows: list[dict]) -> None:
    metrics = [
        "degree",
        "degree_centrality",
        "betweenness_centrality",
        "closeness_centrality",
        "eigenvector_centrality",
        "pagerank",
    ]
    summary_rows: list[dict[str, str | float | int]] = []
    for node_type, rows in [("radical", radical_rows), ("character", character_rows)]:
        for metric in metrics:
            for rank, row in enumerate(top_examples(rows, metric, n=10), start=1):
                summary_rows.append(
                    {
                        "node_type": node_type,
                        "metric": metric,
                        "rank": rank,
                        "symbol": row["symbol"],
                        "codepoint": row.get("codepoint", ""),
                        "kangxi_radical_number": row.get("kangxi_radical_number", ""),
                        "value": row[metric],
                    }
                )
    write_csv(
        path,
        summary_rows,
        fieldnames=["node_type", "metric", "rank", "symbol", "codepoint", "kangxi_radical_number", "value"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--edges-input",
        type=Path,
        default=Path("data/processed/radical_character_edges.csv"),
        help="Structural bipartite edge CSV",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("entrega_final/centralidade"),
        help="Output directory for centrality tables",
    )
    parser.add_argument(
        "--betweenness-k",
        type=int,
        default=1000,
        help="Sample size for approximate betweenness; use 0 for exact (slow on large graphs)",
    )
    args = parser.parse_args()

    log(f"Building graph from {args.edges_input}...")
    graph = build_structural_graph(args.edges_input)
    log(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    gcc_nodes = largest_component_nodes(graph)
    gcc = graph.subgraph(gcc_nodes).copy()

    metrics = compute_metrics(graph, gcc, gcc_nodes, betweenness_k=args.betweenness_k)
    betweenness_note = metrics.pop("betweenness_note")
    all_rows = node_rows(graph, metrics)
    radical_rows = [r for r in all_rows if r["node_type"] == "radical"]
    character_rows = [r for r in all_rows if r["node_type"] == "character"]

    fieldnames = [
        "node_id",
        "node_type",
        "symbol",
        "codepoint",
        "kangxi_radical",
        "kangxi_radical_number",
        "in_largest_component",
        "degree",
        "degree_centrality",
        "betweenness_centrality",
        "closeness_centrality",
        "eigenvector_centrality",
        "pagerank",
    ]

    write_csv(args.out_dir / "centrality_all.csv", all_rows, fieldnames)
    write_csv(args.out_dir / "centrality_radicals.csv", radical_rows, fieldnames)
    write_csv(args.out_dir / "centrality_characters.csv", character_rows, fieldnames)
    write_summary(args.out_dir / "centrality_top10.csv", radical_rows, character_rows)

    meta = {
        "graph": "structural bipartite (unit weights)",
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "radicals": len(radical_rows),
        "characters": len(character_rows),
        "largest_component_nodes": len(gcc_nodes),
        "metrics": [
            "degree",
            "degree_centrality",
            "betweenness_centrality",
            "closeness_centrality",
            "eigenvector_centrality",
            "pagerank",
        ],
        "notes": {
            "betweenness": betweenness_note,
            "betweenness_closeness_eigenvector": "computed on largest connected component",
            "pagerank_degree": "computed on full graph",
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "centrality_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    log(f"All nodes -> {args.out_dir / 'centrality_all.csv'}")
    log(f"Radicals -> {args.out_dir / 'centrality_radicals.csv'}")
    log(f"Characters -> {args.out_dir / 'centrality_characters.csv'}")
    log(f"Top-10 summary -> {args.out_dir / 'centrality_top10.csv'}")
    log("Done.")


if __name__ == "__main__":
    main()
