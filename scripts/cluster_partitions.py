#!/usr/bin/env python3
"""Cluster radicals and characters in structural and semantic spaces.

For each object class (radicals, characters) this script:

1. loads structural and semantic feature matrices;
2. sweeps K and picks a primary K by average silhouette across both spaces;
3. fits KMeans (primary) and AgglomerativeClustering (cross-check) at that K;
4. writes four partition label files plus a silhouette sweep table.

Character silhouette / agglomerative steps use a subsample for efficiency.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

_START = time.monotonic()


def log(message: str) -> None:
    elapsed = time.monotonic() - _START
    print(f"[{elapsed:7.1f}s] {message}", flush=True)


def read_index(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_character_semantic_embeddings(
    embeddings_npy: Path,
    embeddings_index: Path,
    character_index: list[dict[str, str]],
) -> np.ndarray:
    if not embeddings_npy.exists():
        raise FileNotFoundError(
            f"Character embeddings not found: {embeddings_npy}. "
            "Run scripts/build_semantic_embeddings.py or fix the symlink."
        )

    embeddings = np.load(embeddings_npy)
    codepoint_to_row: dict[str, int] = {}
    with embeddings_index.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            codepoint_to_row[row["codepoint"]] = int(row["row_id"])

    rows = []
    missing = 0
    for row in character_index:
        cp = row["codepoint"]
        if cp not in codepoint_to_row:
            missing += 1
            continue
        rows.append(codepoint_to_row[cp])

    if missing:
        raise ValueError(f"{missing} graph characters are missing from {embeddings_index}")

    return np.ascontiguousarray(embeddings[rows], dtype=np.float32)


def to_dense_if_sparse(matrix: np.ndarray | sparse.spmatrix) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix.toarray().astype(np.float32)
    return np.ascontiguousarray(matrix, dtype=np.float32)


def subsample_indices(n_samples: int, max_samples: int, random_state: int) -> np.ndarray:
    if n_samples <= max_samples:
        return np.arange(n_samples)
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(n_samples, size=max_samples, replace=False))


def sweep_kmeans_silhouette(
    features: np.ndarray,
    k_values: list[int],
    sample_idx: np.ndarray | None,
    random_state: int,
) -> list[dict[str, float | int]]:
    x = features if sample_idx is None else features[sample_idx]
    results: list[dict[str, float | int]] = []
    for k in k_values:
        if k >= x.shape[0]:
            continue
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(x)
        if len(set(labels)) < 2:
            continue
        score = float(silhouette_score(x, labels, metric="cosine"))
        results.append({"k": k, "silhouette": score})
    return results


def choose_primary_k(
    structural_sweep: list[dict[str, float | int]],
    semantic_sweep: list[dict[str, float | int]],
) -> int:
    sem_by_k = {int(r["k"]): float(r["silhouette"]) for r in semantic_sweep}
    best_k = None
    best_score = -1.0
    for row in structural_sweep:
        k = int(row["k"])
        if k not in sem_by_k:
            continue
        avg = (float(row["silhouette"]) + sem_by_k[k]) / 2.0
        if avg > best_score:
            best_score = avg
            best_k = k
    if best_k is None:
        raise ValueError("Could not choose a common K from silhouette sweeps")
    return best_k


def fit_kmeans(features: np.ndarray, k: int, random_state: int) -> np.ndarray:
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    return model.fit_predict(features)


def fit_agglomerative(features: np.ndarray, k: int, sample_idx: np.ndarray | None) -> np.ndarray:
    x = features if sample_idx is None else features[sample_idx]
    model = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
    labels = model.fit_predict(x)
    if sample_idx is None:
        return labels
    # Cross-check labels exist only on the subsample; full labels stay KMeans.
    return labels


def write_partition(
    path: Path,
    index_rows: list[dict[str, str]],
    labels_kmeans: np.ndarray,
    labels_agg: np.ndarray | None,
    id_fields: list[str],
    k: int,
    space: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [*id_fields, "cluster_kmeans", "cluster_agglomerative", "k", "space"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(index_rows):
            out = {field: row[field] for field in id_fields}
            out["cluster_kmeans"] = int(labels_kmeans[i])
            out["cluster_agglomerative"] = (
                int(labels_agg[i]) if labels_agg is not None and len(labels_agg) == len(index_rows) else ""
            )
            out["k"] = k
            out["space"] = space
            writer.writerow(out)


def cluster_object_class(
    name: str,
    index_rows: list[dict[str, str]],
    structural_features: np.ndarray,
    semantic_features: np.ndarray,
    k_values: list[int],
    id_fields: list[str],
    out_dir: Path,
    random_state: int,
    character_subsample: int,
) -> tuple[int, list[dict], list[dict]]:
    log(f"{name}: preparing features ({structural_features.shape[0]} objects)")
    struct_dense = to_dense_if_sparse(structural_features)
    sem_dense = to_dense_if_sparse(semantic_features)

    sample_idx = None
    agg_sample_idx = None
    if name == "characters":
        sample_idx = subsample_indices(struct_dense.shape[0], character_subsample, random_state)
        agg_sample_idx = sample_idx
        log(f"{name}: silhouette sweep uses subsample of {len(sample_idx)} objects")

    log(f"{name}: sweeping K on structural space...")
    struct_sweep = sweep_kmeans_silhouette(struct_dense, k_values, sample_idx, random_state)
    log(f"{name}: sweeping K on semantic space...")
    sem_sweep = sweep_kmeans_silhouette(sem_dense, k_values, sample_idx, random_state)
    primary_k = choose_primary_k(struct_sweep, sem_sweep)
    log(f"{name}: primary K = {primary_k}")

    log(f"{name}: fitting KMeans at K={primary_k}...")
    struct_kmeans = fit_kmeans(struct_dense, primary_k, random_state)
    sem_kmeans = fit_kmeans(sem_dense, primary_k, random_state)

    if name == "radicals":
        log(f"{name}: fitting AgglomerativeClustering at K={primary_k}...")
        struct_agg = fit_agglomerative(struct_dense, primary_k, None)
        sem_agg = fit_agglomerative(sem_dense, primary_k, None)
    else:
        log(f"{name}: Agglomerative cross-check on subsample only")
        struct_agg = None
        sem_agg = None

    write_partition(
        out_dir / f"{name}_structural.csv",
        index_rows,
        struct_kmeans,
        struct_agg,
        id_fields,
        primary_k,
        "structural",
    )
    write_partition(
        out_dir / f"{name}_semantic.csv",
        index_rows,
        sem_kmeans,
        sem_agg,
        id_fields,
        primary_k,
        "semantic",
    )

    sweep_rows = []
    sem_by_k = {int(r["k"]): float(r["silhouette"]) for r in sem_sweep}
    for row in struct_sweep:
        k = int(row["k"])
        if k not in sem_by_k:
            continue
        sweep_rows.append(
            {
                "object_class": name,
                "k": k,
                "silhouette_structural": float(row["silhouette"]),
                "silhouette_semantic": sem_by_k[k],
                "silhouette_average": (float(row["silhouette"]) + sem_by_k[k]) / 2.0,
                "is_primary_k": k == primary_k,
            }
        )
    return primary_k, sweep_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/partitions"))
    parser.add_argument("--character-embeddings", type=Path, default=Path("data/processed/semantic_embeddings.npy"))
    parser.add_argument("--character-embeddings-index", type=Path, default=Path("data/processed/semantic_embeddings_index.csv"))
    parser.add_argument("--radical-k-min", type=int, default=3)
    parser.add_argument("--radical-k-max", type=int, default=15)
    parser.add_argument(
        "--character-k-values",
        type=str,
        default="5,10,15,20,30,40,60",
        help="Comma-separated K values for character silhouette sweep",
    )
    parser.add_argument("--character-subsample", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    radical_k_values = list(range(args.radical_k_min, args.radical_k_max + 1))
    character_k_values = [int(v.strip()) for v in args.character_k_values.split(",") if v.strip()]

    log("Loading structural features and indices...")
    radical_index = read_index(args.data_dir / "struct_index_radicals.csv")
    character_index = read_index(args.data_dir / "struct_index_characters.csv")
    radical_struct = np.load(args.data_dir / "struct_features_radicals.npy")
    character_struct = sparse.load_npz(args.data_dir / "struct_features_characters.npz")

    log("Loading semantic features...")
    radical_sem = np.load(args.data_dir / "radical_embeddings.npy")
    character_sem = load_character_semantic_embeddings(
        args.character_embeddings,
        args.character_embeddings_index,
        character_index,
    )

    all_sweep_rows: list[dict] = []
    summary: dict[str, int] = {}

    _, radical_sweep = cluster_object_class(
        name="radicals",
        index_rows=radical_index,
        structural_features=radical_struct,
        semantic_features=radical_sem,
        k_values=radical_k_values,
        id_fields=["row_id", "kangxi_radical", "kangxi_radical_number"],
        out_dir=args.out_dir,
        random_state=args.random_state,
        character_subsample=args.character_subsample,
    )
    all_sweep_rows.extend(radical_sweep)
    summary["radicals_primary_k"] = int(next(r["k"] for r in radical_sweep if r["is_primary_k"]))

    _, character_sweep = cluster_object_class(
        name="characters",
        index_rows=character_index,
        structural_features=character_struct,
        semantic_features=character_sem,
        k_values=character_k_values,
        id_fields=["row_id", "codepoint", "char"],
        out_dir=args.out_dir,
        random_state=args.random_state,
        character_subsample=args.character_subsample,
    )
    all_sweep_rows.extend(character_sweep)
    summary["characters_primary_k"] = int(next(r["k"] for r in character_sweep if r["is_primary_k"]))

    sweep_path = args.out_dir / "kmeans_sweep.csv"
    with sweep_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "object_class",
                "k",
                "silhouette_structural",
                "silhouette_semantic",
                "silhouette_average",
                "is_primary_k",
            ],
        )
        writer.writeheader()
        writer.writerows(all_sweep_rows)

    meta_path = args.out_dir / "clustering_meta.json"
    meta = {
        "radicals_k_grid": radical_k_values,
        "characters_k_grid": character_k_values,
        "character_subsample": args.character_subsample,
        "random_state": args.random_state,
        **summary,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"Sweep table -> {sweep_path}")
    log(f"Metadata -> {meta_path}")
    log(f"Partitions -> {args.out_dir}")
    log("Done.")


if __name__ == "__main__":
    main()
