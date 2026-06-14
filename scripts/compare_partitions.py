#!/usr/bin/env python3
"""Compare structural and semantic partitions (step 5).

For radicals and characters separately, this script measures how well the
structural KMeans partition aligns with the semantic KMeans partition.

At the primary K (saved in step 4) it reports full metrics and contingency
matrices. Across the K grid it re-clusters both spaces at each K and records
how alignment changes with the number of clusters.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import plotly.express as px
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    fowlkes_mallows_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)
from sklearn.metrics.cluster import contingency_matrix

_START = time.monotonic()


def log(message: str) -> None:
    elapsed = time.monotonic() - _START
    print(f"[{elapsed:7.1f}s] {message}", flush=True)


def read_partition(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_labels_from_partitions(
    structural_csv: Path,
    semantic_csv: Path,
    id_field: str,
) -> tuple[list[str], np.ndarray, np.ndarray, int]:
    struct_rows = read_partition(structural_csv)
    sem_rows = read_partition(semantic_csv)
    sem_by_id = {row[id_field]: row for row in sem_rows}

    ids: list[str] = []
    struct_labels: list[int] = []
    sem_labels: list[int] = []
    k_values: set[int] = set()

    for row in struct_rows:
        obj_id = row[id_field]
        if obj_id not in sem_by_id:
            continue
        sem_row = sem_by_id[obj_id]
        ids.append(obj_id)
        struct_labels.append(int(row["cluster_kmeans"]))
        sem_labels.append(int(sem_row["cluster_kmeans"]))
        k_values.add(int(row["k"]))

    if len(k_values) != 1:
        raise ValueError(f"Expected one K in {structural_csv}, found {sorted(k_values)}")
    return ids, np.array(struct_labels), np.array(sem_labels), next(iter(k_values))


def compute_metrics(labels_a: np.ndarray, labels_b: np.ndarray) -> dict[str, float]:
    return {
        "ari": float(adjusted_rand_score(labels_a, labels_b)),
        "ami": float(adjusted_mutual_info_score(labels_a, labels_b)),
        "nmi": float(normalized_mutual_info_score(labels_a, labels_b)),
        "v_measure": float(v_measure_score(labels_a, labels_b)),
        "homogeneity": float(homogeneity_score(labels_a, labels_b)),
        "completeness": float(completeness_score(labels_a, labels_b)),
        "fowlkes_mallows": float(fowlkes_mallows_score(labels_a, labels_b)),
    }


def write_contingency_csv(
    path: Path,
    labels_struct: np.ndarray,
    labels_sem: np.ndarray,
    row_prefix: str = "struct_cluster",
    col_prefix: str = "sem_cluster",
) -> np.ndarray:
    matrix = contingency_matrix(labels_struct, labels_sem)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header = [""] + [f"{col_prefix}_{j}" for j in range(matrix.shape[1])]
        writer.writerow(header)
        for i, row in enumerate(matrix):
            writer.writerow([f"{row_prefix}_{i}", *row.tolist()])
    return matrix


def save_contingency_heatmap(
    matrix: np.ndarray,
    path: Path,
    title: str,
    row_label: str,
    col_label: str,
) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig = px.imshow(
            matrix,
            labels={"x": col_label, "y": row_label, "color": "Contagem"},
            x=[f"sem_{j}" for j in range(matrix.shape[1])],
            y=[f"struct_{i}" for i in range(matrix.shape[0])],
            title=title,
            color_continuous_scale="Blues",
            aspect="auto",
        )
        fig.update_layout(template="plotly_white")
        fig.write_image(str(path), scale=2)
        log(f"Saved heatmap -> {path}")
    except Exception as exc:
        log(f"Skipped heatmap {path.name}: {exc}")


def save_metrics_plot(rows: list[dict], path: Path, title: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        metrics = ["ari", "nmi", "fowlkes_mallows"]
        long_rows = []
        for row in rows:
            for metric in metrics:
                long_rows.append(
                    {
                        "k": int(row["k"]),
                        "metric": metric.upper(),
                        "value": float(row[metric]),
                        "is_primary_k": bool(row["is_primary_k"]),
                    }
                )
        fig = px.line(
            long_rows,
            x="k",
            y="value",
            color="metric",
            markers=True,
            title=title,
            labels={"k": "Número de clusters (K)", "value": "Similaridade entre partições"},
        )
        primary_k = next((int(r["k"]) for r in rows if r["is_primary_k"]), None)
        if primary_k is not None:
            fig.add_vline(x=primary_k, line_dash="dash", line_color="gray", annotation_text="K primário")
        fig.update_layout(template="plotly_white")
        fig.write_image(str(path), scale=2)
        log(f"Saved metrics plot -> {path}")
    except Exception as exc:
        log(f"Skipped metrics plot {path.name}: {exc}")


def to_dense_if_sparse(matrix: np.ndarray | sparse.spmatrix) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix.toarray().astype(np.float32)
    return np.ascontiguousarray(matrix, dtype=np.float32)


def load_character_semantic_embeddings(
    embeddings_npy: Path,
    embeddings_index: Path,
    character_index: list[dict[str, str]],
) -> np.ndarray:
    embeddings = np.load(embeddings_npy)
    codepoint_to_row = {}
    with embeddings_index.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            codepoint_to_row[row["codepoint"]] = int(row["row_id"])
    rows = [codepoint_to_row[row["codepoint"]] for row in character_index]
    return np.ascontiguousarray(embeddings[rows], dtype=np.float32)


def fit_kmeans(features: np.ndarray, k: int, random_state: int) -> np.ndarray:
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    return model.fit_predict(features)


def compare_at_k(
    struct_features: np.ndarray,
    sem_features: np.ndarray,
    k: int,
    random_state: int,
) -> dict[str, float]:
    struct_labels = fit_kmeans(struct_features, k, random_state)
    sem_labels = fit_kmeans(sem_features, k, random_state)
    return compute_metrics(struct_labels, sem_labels)


def compare_object_class(
    name: str,
    partitions_dir: Path,
    out_dir: Path,
    primary_k: int,
    k_grid: list[int],
    struct_features: np.ndarray,
    sem_features: np.ndarray,
    id_field: str,
    random_state: int,
    skip_k_grid: bool,
) -> list[dict]:
    log(f"{name}: loading primary partitions (K={primary_k})...")
    _, struct_labels, sem_labels, _ = load_labels_from_partitions(
        partitions_dir / f"{name}_structural.csv",
        partitions_dir / f"{name}_semantic.csv",
        id_field=id_field,
    )
    n_objects = len(struct_labels)
    log(f"{name}: computing metrics at primary K...")
    primary_metrics = compute_metrics(struct_labels, sem_labels)
    log(
        f"{name}: ARI={primary_metrics['ari']:.4f}, "
        f"NMI={primary_metrics['nmi']:.4f}, "
        f"FMI={primary_metrics['fowlkes_mallows']:.4f}"
    )

    log(f"{name}: writing contingency matrix...")
    matrix = write_contingency_csv(
        out_dir / f"contingency_{name}.csv",
        struct_labels,
        sem_labels,
    )
    save_contingency_heatmap(
        matrix,
        out_dir / f"contingency_{name}_heatmap.png",
        title=f"Contingência estrutural × semântica ({name}, K={primary_k})",
        row_label="Cluster estrutural",
        col_label="Cluster semântico",
    )

    summary_rows: list[dict] = []
    summary_rows.append(
        {
            "object_class": name,
            "k": primary_k,
            "is_primary_k": True,
            "n_objects": n_objects,
            **primary_metrics,
        }
    )

    if skip_k_grid:
        return summary_rows

    log(f"{name}: sweeping alignment across K grid ({len(k_grid)} values)...")
    grid_rows = []
    for k in k_grid:
        if k == primary_k or k >= n_objects:
            continue
        metrics = compare_at_k(struct_features, sem_features, k, random_state)
        grid_rows.append(
            {
                "object_class": name,
                "k": k,
                "is_primary_k": k == primary_k,
                "n_objects": n_objects,
                **metrics,
            }
        )
        log(f"{name}: K={k} -> ARI={metrics['ari']:.4f}, NMI={metrics['nmi']:.4f}")

    save_metrics_plot(
        summary_rows,
        out_dir / f"metrics_vs_k_{name}.png",
        title=f"Alinhamento estrutural × semântico ({name})",
    )
    summary_rows.extend(grid_rows)
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partitions-dir", type=Path, default=Path("data/processed/partitions"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out-dir", type=Path, default=Path("entrega_final/comparacao"))
    parser.add_argument("--character-embeddings", type=Path, default=Path("data/processed/semantic_embeddings.npy"))
    parser.add_argument(
        "--character-embeddings-index",
        type=Path,
        default=Path("data/processed/semantic_embeddings_index.csv"),
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--skip-k-grid",
        action="store_true",
        help="Only compute metrics at the primary K (faster)",
    )
    args = parser.parse_args()

    meta = json.loads((args.partitions_dir / "clustering_meta.json").read_text(encoding="utf-8"))
    radical_k_grid = meta["radicals_k_grid"]
    character_k_grid = meta["characters_k_grid"]
    radicals_primary_k = int(meta["radicals_primary_k"])
    characters_primary_k = int(meta["characters_primary_k"])

    log("Loading feature matrices for K-grid comparison...")
    radical_index = read_partition(args.data_dir / "struct_index_radicals.csv")
    character_index = read_partition(args.data_dir / "struct_index_characters.csv")
    radical_struct = to_dense_if_sparse(np.load(args.data_dir / "struct_features_radicals.npy"))
    radical_sem = to_dense_if_sparse(np.load(args.data_dir / "radical_embeddings.npy"))
    character_struct = to_dense_if_sparse(sparse.load_npz(args.data_dir / "struct_features_characters.npz"))
    character_sem = to_dense_if_sparse(
        load_character_semantic_embeddings(
            args.character_embeddings,
            args.character_embeddings_index,
            character_index,
        )
    )

    all_rows: list[dict] = []
    all_rows.extend(
        compare_object_class(
            name="radicals",
            partitions_dir=args.partitions_dir,
            out_dir=args.out_dir,
            primary_k=radicals_primary_k,
            k_grid=radical_k_grid,
            struct_features=radical_struct,
            sem_features=radical_sem,
            id_field="kangxi_radical",
            random_state=args.random_state,
            skip_k_grid=args.skip_k_grid,
        )
    )
    all_rows.extend(
        compare_object_class(
            name="characters",
            partitions_dir=args.partitions_dir,
            out_dir=args.out_dir,
            primary_k=characters_primary_k,
            k_grid=character_k_grid,
            struct_features=character_struct,
            sem_features=character_sem,
            id_field="codepoint",
            random_state=args.random_state,
            skip_k_grid=args.skip_k_grid,
        )
    )

    summary_path = args.out_dir / "metrics_summary.csv"
    fieldnames = [
        "object_class",
        "k",
        "is_primary_k",
        "n_objects",
        "ari",
        "ami",
        "nmi",
        "v_measure",
        "homogeneity",
        "completeness",
        "fowlkes_mallows",
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            out = dict(row)
            out["is_primary_k"] = str(bool(out["is_primary_k"])).lower()
            writer.writerow(out)

    log(f"Metrics summary -> {summary_path}")
    log(f"Contingency tables and plots -> {args.out_dir}")
    log("Done.")


if __name__ == "__main__":
    main()
