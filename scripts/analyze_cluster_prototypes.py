#!/usr/bin/env python3
"""Build interpretable cluster profiles via centroids and nearest prototypes.

For each partition (radicals/characters x structural/semantic), computes the
cluster centroid, ranks members by cosine similarity to it, and records top
prototypes with readable definitions. Character clusters also get the most
frequent constituent radicals.

Embeddings are not decoded to text; prototypes serve as representative tokens.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse

_START = time.monotonic()


def log(message: str) -> None:
    elapsed = time.monotonic() - _START
    print(f"[{elapsed:7.1f}s] {message}", flush=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def to_dense(matrix: np.ndarray | sparse.spmatrix) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix.toarray().astype(np.float32)
    return np.ascontiguousarray(matrix, dtype=np.float32)


def load_character_semantic_embeddings(
    embeddings_npy: Path,
    embeddings_index: Path,
    character_index: list[dict[str, str]],
) -> np.ndarray:
    embeddings = np.load(embeddings_npy)
    codepoint_to_row = {
        row["codepoint"]: int(row["row_id"])
        for row in read_csv(embeddings_index)
    }
    rows = [codepoint_to_row[row["codepoint"]] for row in character_index]
    return np.ascontiguousarray(embeddings[rows], dtype=np.float32)


def load_character_metadata(characters_csv: Path) -> dict[str, dict[str, str]]:
    meta: dict[str, dict[str, str]] = {}
    with characters_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            meta[row["codepoint"]] = {
                "label": row["char"],
                "definition": (row.get("definition") or "").strip(),
                "semantic_text": (row.get("semantic_text") or "").strip(),
            }
    return meta


def load_radical_metadata(radical_index_csv: Path) -> dict[str, dict[str, str]]:
    meta: dict[str, dict[str, str]] = {}
    for row in read_csv(radical_index_csv):
        symbol = row["kangxi_radical"]
        meta[symbol] = {
            "label": symbol,
            "definition": (row.get("name_en") or "").strip(),
            "semantic_text": (row.get("semantic_text") or "").strip(),
        }
    return meta


def load_character_radicals(edges_csv: Path) -> dict[str, list[str]]:
    by_char: dict[str, list[str]] = defaultdict(list)
    with edges_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            by_char[row["character_codepoint"]].append(row["kangxi_radical"])
    return dict(by_char)


def compute_centroid(vectors: np.ndarray, normalize: bool) -> np.ndarray:
    centroid = vectors.mean(axis=0)
    if normalize:
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
    return np.ascontiguousarray(centroid, dtype=np.float32)


def cosine_to_centroid(vectors: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return np.array([], dtype=np.float32)
    v_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    v_norms[v_norms == 0] = 1.0
    v_unit = vectors / v_norms
    c_norm = np.linalg.norm(centroid)
    if c_norm == 0:
        return np.zeros(len(vectors), dtype=np.float32)
    c_unit = centroid / c_norm
    return (v_unit @ c_unit).astype(np.float32)


def frequent_radicals(
    member_ids: list[str],
    char_radicals: dict[str, list[str]],
    top_n: int = 8,
) -> str:
    counts: Counter[str] = Counter()
    for cp in member_ids:
        for radical in char_radicals.get(cp, []):
            counts[radical] += 1
    if not counts:
        return ""
    total = len(member_ids)
    parts = []
    for radical, count in counts.most_common(top_n):
        parts.append(f"{radical}({count / total:.0%})")
    return ", ".join(parts)


def load_high_overlap_clusters(convergence_csv: Path | None) -> set[tuple[str, str, int]]:
    """Return (object_class, space, cluster_id) keys with strong struct/sem overlap."""
    if convergence_csv is None or not convergence_csv.exists():
        return set()
    keys: set[tuple[str, str, int]] = set()
    for row in read_csv(convergence_csv):
        if row.get("pattern") != "convergence":
            continue
        obj = row["object_class"]
        overlap_frac = float(row.get("overlap_fraction_smaller_cluster") or 0)
        if overlap_frac < 0.5:
            continue
        keys.add((obj, "structural", int(row["struct_cluster"])))
        keys.add((obj, "semantic", int(row["sem_cluster"])))
    return keys


def analyze_partition(
    object_class: str,
    space: str,
    partition_csv: Path,
    features: np.ndarray,
    id_field: str,
    label_field: str,
    metadata: dict[str, dict[str, str]],
    normalize_centroid: bool,
    top_prototypes: int,
    char_radicals: dict[str, list[str]] | None,
    high_overlap: set[tuple[str, str, int]],
) -> tuple[list[dict], list[dict], dict[str, np.ndarray]]:
    rows = read_csv(partition_csv)
    if len(rows) != features.shape[0]:
        raise ValueError(
            f"{partition_csv}: {len(rows)} labels but {features.shape[0]} feature rows"
        )

    labels = np.array([int(r["cluster_kmeans"]) for r in rows], dtype=np.int32)
    ids = [r[id_field] for r in rows]
    unique_clusters = sorted(set(labels.tolist()))

    prototype_rows: list[dict] = []
    profile_rows: list[dict] = []
    centroids: dict[str, np.ndarray] = {}

    for cluster_id in unique_clusters:
        mask = labels == cluster_id
        idxs = np.where(mask)[0]
        cluster_features = features[idxs]
        member_ids = [ids[i] for i in idxs]

        centroid = compute_centroid(cluster_features, normalize=normalize_centroid)
        centroids[f"{object_class}_{space}_{cluster_id}"] = centroid
        scores = cosine_to_centroid(cluster_features, centroid)
        order = np.argsort(-scores)

        proto_labels: list[str] = []
        proto_defs: list[str] = []
        for rank, local_idx in enumerate(order[:top_prototypes], start=1):
            global_idx = idxs[local_idx]
            obj_id = ids[global_idx]
            meta = metadata.get(obj_id, {})
            label = meta.get("label") or rows[global_idx].get(label_field, "")
            definition = meta.get("definition") or meta.get("semantic_text") or ""
            proto_labels.append(label)
            if definition:
                proto_defs.append(definition[:120])
            prototype_rows.append(
                {
                    "object_class": object_class,
                    "space": space,
                    "cluster_id": cluster_id,
                    "cluster_size": int(mask.sum()),
                    "prototype_rank": rank,
                    "object_id": obj_id,
                    "object_label": label,
                    "definition": definition,
                    "cosine_to_centroid": round(float(scores[local_idx]), 6),
                }
            )

        top_radicals = ""
        if object_class == "characters" and char_radicals is not None:
            top_radicals = frequent_radicals(member_ids, char_radicals)

        profile_rows.append(
            {
                "object_class": object_class,
                "space": space,
                "cluster_id": cluster_id,
                "cluster_size": int(mask.sum()),
                "high_overlap_flag": str(
                    (object_class, space, cluster_id) in high_overlap
                ).lower(),
                "top_radicals": top_radicals,
                "prototype_labels": " | ".join(proto_labels),
                "prototype_definitions_sample": " | ".join(proto_defs[:5]),
                "mean_cosine_to_centroid": round(float(scores.mean()), 6),
                "max_cosine_to_centroid": round(float(scores.max()), 6),
            }
        )

    return prototype_rows, profile_rows, centroids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--partitions-dir", type=Path, default=Path("data/processed/partitions"))
    parser.add_argument("--out-dir", type=Path, default=Path("entrega_final/exemplos"))
    parser.add_argument(
        "--centroids-dir",
        type=Path,
        default=Path("data/processed/partitions/centroids"),
        help="Optional directory for centroid .npz archive",
    )
    parser.add_argument(
        "--convergence-csv",
        type=Path,
        default=Path("entrega_final/exemplos/convergence.csv"),
    )
    parser.add_argument("--top-prototypes", type=int, default=10)
    parser.add_argument("--character-embeddings", type=Path, default=Path("data/processed/semantic_embeddings.npy"))
    parser.add_argument(
        "--character-embeddings-index",
        type=Path,
        default=Path("data/processed/semantic_embeddings_index.csv"),
    )
    args = parser.parse_args()

    log("Loading indices, features, and metadata...")
    radical_index = read_csv(args.data_dir / "struct_index_radicals.csv")
    character_index = read_csv(args.data_dir / "struct_index_characters.csv")

    radical_struct = to_dense(np.load(args.data_dir / "struct_features_radicals.npy"))
    character_struct = to_dense(sparse.load_npz(args.data_dir / "struct_features_characters.npz"))
    radical_sem = to_dense(np.load(args.data_dir / "radical_embeddings.npy"))
    character_sem = load_character_semantic_embeddings(
        args.character_embeddings,
        args.character_embeddings_index,
        character_index,
    )

    char_meta = load_character_metadata(args.data_dir / "characters.csv")
    rad_meta = load_radical_metadata(args.data_dir / "radical_embeddings_index.csv")
    char_radicals = load_character_radicals(args.data_dir / "radical_character_edges.csv")
    high_overlap = load_high_overlap_clusters(args.convergence_csv)

    configs = [
        ("radicals", "structural", "radicals_structural.csv", radical_struct, "kangxi_radical", "kangxi_radical", rad_meta, False),
        ("radicals", "semantic", "radicals_semantic.csv", radical_sem, "kangxi_radical", "kangxi_radical", rad_meta, True),
        ("characters", "structural", "characters_structural.csv", character_struct, "codepoint", "char", char_meta, False),
        ("characters", "semantic", "characters_semantic.csv", character_sem, "codepoint", "char", char_meta, True),
    ]

    all_prototypes: list[dict] = []
    all_profiles: list[dict] = []
    all_centroids: dict[str, np.ndarray] = {}

    for object_class, space, filename, features, id_field, label_field, metadata, normalize in configs:
        log(f"Analyzing {object_class} / {space}...")
        protos, profiles, centroids = analyze_partition(
            object_class=object_class,
            space=space,
            partition_csv=args.partitions_dir / filename,
            features=features,
            id_field=id_field,
            label_field=label_field,
            metadata=metadata,
            normalize_centroid=normalize,
            top_prototypes=args.top_prototypes,
            char_radicals=char_radicals if object_class == "characters" else None,
            high_overlap=high_overlap,
        )
        all_prototypes.extend(protos)
        all_profiles.extend(profiles)
        all_centroids.update(centroids)
        log(f"  {len(profiles)} clusters, {len(protos)} prototype rows")

    prototype_fields = [
        "object_class", "space", "cluster_id", "cluster_size", "prototype_rank",
        "object_id", "object_label", "definition", "cosine_to_centroid",
    ]
    profile_fields = [
        "object_class", "space", "cluster_id", "cluster_size", "high_overlap_flag",
        "top_radicals", "prototype_labels", "prototype_definitions_sample",
        "mean_cosine_to_centroid", "max_cosine_to_centroid",
    ]

    write_csv(args.out_dir / "cluster_prototypes.csv", all_prototypes, prototype_fields)
    write_csv(args.out_dir / "cluster_profiles.csv", all_profiles, profile_fields)

    args.centroids_dir.mkdir(parents=True, exist_ok=True)
    centroids_path = args.centroids_dir / "cluster_centroids.npz"
    np.savez_compressed(centroids_path, **all_centroids)

    meta = {
        "top_prototypes": args.top_prototypes,
        "partitions_analyzed": len(configs),
        "total_clusters": len(all_profiles),
        "centroids_file": str(centroids_path),
        "centroid_keys": sorted(all_centroids.keys()),
        "notes": {
            "semantic_centroids": "L2-normalized after mean",
            "structural_centroids": "arithmetic mean without normalization",
            "interpretation": "use cluster_prototypes.csv for readable tokens, not centroid decode",
        },
    }
    (args.out_dir / "cluster_centroids_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    log(f"Prototypes -> {args.out_dir / 'cluster_prototypes.csv'}")
    log(f"Profiles -> {args.out_dir / 'cluster_profiles.csv'}")
    log(f"Centroids -> {centroids_path}")
    log(f"Meta -> {args.out_dir / 'cluster_centroids_meta.json'}")
    log("Done.")


if __name__ == "__main__":
    main()
