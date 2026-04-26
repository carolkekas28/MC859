#!/usr/bin/env python3
"""Plot degree distribution from a GraphML file.

X-axis: vertex degree (number of connections)
Y-axis: number of vertices with that degree
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import networkx as nx
import plotly.express as px


def build_degree_distribution(graph_path: Path) -> list[tuple[int, int]]:
    g = nx.read_graphml(graph_path)
    degree_counter = Counter(dict(g.degree()).values())
    return sorted((int(k), int(v)) for k, v in degree_counter.items())


def group_degree_distribution(distribution: list[tuple[int, int]], cutoff: int = 7) -> list[tuple[str, int]]:
    grouped: list[tuple[str, int]] = []
    high_bucket = 0
    for degree, count in distribution:
        if degree >= cutoff:
            high_bucket += count
        else:
            grouped.append((str(degree), count))
    grouped.append((f"{cutoff}+", high_bucket))
    return grouped


def save_distribution_csv(distribution: list[tuple[str, int]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["degree_group", "vertex_count"])
        writer.writeheader()
        for degree_group, count in distribution:
            writer.writerow({"degree_group": degree_group, "vertex_count": count})


def save_distribution_png(distribution: list[tuple[str, int]], output_png: Path, title: str) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    x_vals = [d for d, _ in distribution]
    y_vals = [c for _, c in distribution]
    fig = px.bar(
        x=x_vals,
        y=y_vals,
        labels={"x": "Número de conexões do vértice (grau)", "y": "Quantidade de vértices"},
        title=title,
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_type="category",
        bargap=0.05,
        xaxis_tickangle=-90,
    )
    fig.write_image(output_png, width=1400, height=800, scale=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graphml",
        type=Path,
        default=Path("data/graphs/graph_structural.graphml"),
        help="Input GraphML file (e.g. copy under entrega_parcial/ if you only keep that)",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("entrega_parcial/csvs/degree_distribution.csv"),
        help="Output CSV with degree distribution",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=Path("entrega_parcial/plots/degree_distribution.png"),
        help="Output PNG chart",
    )
    args = parser.parse_args()

    distribution = build_degree_distribution(args.graphml)
    grouped_distribution = group_degree_distribution(distribution, cutoff=7)
    save_distribution_csv(grouped_distribution, args.out_csv)
    save_distribution_png(
        grouped_distribution,
        args.out_png,
        title=f"Distribuição de grau - {args.graphml.name}",
    )

    print(f"GraphML: {args.graphml}")
    print(f"Saved CSV: {args.out_csv}")
    print(f"Saved PNG: {args.out_png}")
    print(f"Distinct degree values (original): {len(distribution)}")
    print(f"Groups after aggregation: {len(grouped_distribution)}")


if __name__ == "__main__":
    main()
