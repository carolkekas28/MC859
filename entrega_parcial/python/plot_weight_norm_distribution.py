#!/usr/bin/env python3
"""Plot normalized edge weight distribution for weighted graph CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import plotly.express as px


def read_weight_norm_values(weighted_edges_csv: Path) -> np.ndarray:
    values = []
    with weighted_edges_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                values.append(float(row["weight_norm_01"]))
            except (ValueError, KeyError):
                continue
    return np.array(values, dtype=float)


def save_histogram_bins(values: np.ndarray, bins: int, output_csv: Path) -> None:
    counts, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["bin_left", "bin_right", "bin_center", "count"],
        )
        writer.writeheader()
        for i, count in enumerate(counts):
            left = float(edges[i])
            right = float(edges[i + 1])
            center = (left + right) / 2.0
            writer.writerow(
                {
                    "bin_left": f"{left:.6f}",
                    "bin_right": f"{right:.6f}",
                    "bin_center": f"{center:.6f}",
                    "count": int(count),
                }
            )


def save_summary(values: np.ndarray, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "pct_equal_1_0": float(np.mean(values == 1.0) * 100.0),
    }

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for metric, value in summary.items():
            writer.writerow({"metric": metric, "value": value})


def save_histogram_png(values: np.ndarray, bins: int, output_png: Path, title: str) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig = px.histogram(
        x=values,
        nbins=bins,
        labels={"x": "Peso normalizado da aresta (weight_norm_01)", "y": "Quantidade de arestas"},
        title=title,
    )
    fig.update_layout(template="plotly_white", bargap=0.05)
    fig.write_image(output_png, width=1400, height=800, scale=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weighted-edges-csv",
        type=Path,
        default=Path("data/processed/radical_character_edges_weighted.csv"),
        help="Input weighted edge CSV",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=Path("entrega_parcial/plots/weight_norm_distribution.png"),
        help="Output PNG plot path",
    )
    parser.add_argument(
        "--out-bins-csv",
        type=Path,
        default=Path("entrega_parcial/csvs/weight_norm_distribution_bins.csv"),
        help="Output histogram bins CSV",
    )
    parser.add_argument(
        "--out-summary-csv",
        type=Path,
        default=Path("entrega_parcial/csvs/weight_norm_summary.csv"),
        help="Output summary statistics CSV",
    )
    args = parser.parse_args()

    values = read_weight_norm_values(args.weighted_edges_csv)
    if values.size == 0:
        raise RuntimeError("No valid weight_norm_01 values found in input CSV.")

    save_histogram_png(
        values,
        bins=args.bins,
        output_png=args.out_png,
        title=f"Distribuição de weight_norm_01 - {args.weighted_edges_csv.name}",
    )
    save_histogram_bins(values, bins=args.bins, output_csv=args.out_bins_csv)
    save_summary(values, output_csv=args.out_summary_csv)

    print(f"Input: {args.weighted_edges_csv}")
    print(f"Saved PNG: {args.out_png}")
    print(f"Saved bins CSV: {args.out_bins_csv}")
    print(f"Saved summary CSV: {args.out_summary_csv}")
    print(f"Count: {values.size}")


if __name__ == "__main__":
    main()
