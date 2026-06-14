#!/usr/bin/env python3
"""Interpretive analysis of partition alignment (step 6).

Identifies representative cases of convergence (structural and semantic clusters
that strongly overlap) and divergence (objects or clusters that disagree across
the two views).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

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


def load_merged_partitions(
    structural_csv: Path,
    semantic_csv: Path,
    id_field: str,
    label_field: str,
) -> list[dict[str, str]]:
    struct_rows = read_csv(structural_csv)
    sem_by_id = {row[id_field]: row for row in read_csv(semantic_csv)}
    merged = []
    for row in struct_rows:
        obj_id = row[id_field]
        sem_row = sem_by_id.get(obj_id)
        if sem_row is None:
            continue
        merged.append(
            {
                "id": obj_id,
                "label": row[label_field],
                "struct_cluster": int(row["cluster_kmeans"]),
                "sem_cluster": int(sem_row["cluster_kmeans"]),
                "k": int(row["k"]),
            }
        )
    return merged


def load_character_definitions(characters_csv: Path) -> dict[str, dict[str, str]]:
    defs: dict[str, dict[str, str]] = {}
    with characters_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            defs[row["codepoint"]] = {
                "char": row["char"],
                "definition": (row.get("definition") or "").strip(),
            }
    return defs


def load_character_edge_stats(edges_csv: Path) -> dict[str, dict[str, float | str]]:
    """Per-character stats over weighted radical edges."""
    by_char: dict[str, list[float]] = defaultdict(list)
    radicals_by_char: dict[str, set[str]] = defaultdict(set)
    with edges_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            cp = row["character_codepoint"]
            by_char[cp].append(float(row["weight_cosine"]))
            radicals_by_char[cp].add(row["kangxi_radical"])
    stats: dict[str, dict[str, float | str]] = {}
    for cp, weights in by_char.items():
        stats[cp] = {
            "mean_weight_cosine": float(np.mean(weights)),
            "min_weight_cosine": float(np.min(weights)),
            "max_weight_cosine": float(np.max(weights)),
            "radicals": ", ".join(sorted(radicals_by_char[cp])),
        }
    return stats


def build_contingency(objects: list[dict[str, str]]) -> np.ndarray:
    struct_labels = [obj["struct_cluster"] for obj in objects]
    sem_labels = [obj["sem_cluster"] for obj in objects]
    n_struct = max(struct_labels) + 1
    n_sem = max(sem_labels) + 1
    matrix = np.zeros((n_struct, n_sem), dtype=int)
    for s, t in zip(struct_labels, sem_labels):
        matrix[s, t] += 1
    return matrix


def cluster_sizes(labels: list[int]) -> Counter[int]:
    return Counter(labels)


def find_convergence(
    object_class: str,
    objects: list[dict[str, str]],
    definitions: dict[str, dict[str, str]] | None,
    top_pairs: int,
    examples_per_pair: int,
) -> list[dict]:
    matrix = build_contingency(objects)
    struct_sizes = cluster_sizes([o["struct_cluster"] for o in objects])
    sem_sizes = cluster_sizes([o["sem_cluster"] for o in objects])

    pair_scores: list[tuple[int, int, int, float]] = []
    for s in range(matrix.shape[0]):
        for t in range(matrix.shape[1]):
            overlap = int(matrix[s, t])
            if overlap == 0:
                continue
            purity = overlap / min(struct_sizes[s], sem_sizes[t])
            pair_scores.append((s, t, overlap, purity))

    pair_scores.sort(key=lambda x: (x[3], x[2]), reverse=True)
    rows: list[dict] = []
    by_pair_objects: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for obj in objects:
        by_pair_objects[(obj["struct_cluster"], obj["sem_cluster"])].append(obj)

    for s, t, overlap, purity in pair_scores[:top_pairs]:
        examples = by_pair_objects[(s, t)][:examples_per_pair]
        for idx, obj in enumerate(examples):
            definition = ""
            if definitions and obj["id"] in definitions:
                definition = definitions[obj["id"]].get("definition", "")
            rows.append(
                {
                    "object_class": object_class,
                    "struct_cluster": s,
                    "sem_cluster": t,
                    "overlap_count": overlap,
                    "struct_cluster_size": struct_sizes[s],
                    "sem_cluster_size": sem_sizes[t],
                    "overlap_fraction_smaller_cluster": round(purity, 4),
                    "example_rank": idx + 1,
                    "object_id": obj["id"],
                    "object_label": obj["label"],
                    "definition": definition,
                    "pattern": "convergence",
                }
            )
    return rows


def find_divergence(
    object_class: str,
    objects: list[dict[str, str]],
    definitions: dict[str, dict[str, str]] | None,
    edge_stats: dict[str, dict[str, float | str]] | None,
    min_cluster_size: int,
    examples_per_case: int,
) -> list[dict]:
    rows: list[dict] = []

    by_struct: dict[int, list[dict]] = defaultdict(list)
    by_sem: dict[int, list[dict]] = defaultdict(list)
    for obj in objects:
        by_struct[obj["struct_cluster"]].append(obj)
        by_sem[obj["sem_cluster"]].append(obj)

    # Structural cluster split across multiple semantic clusters.
    for struct_cluster, members in sorted(by_struct.items()):
        if len(members) < min_cluster_size:
            continue
        sem_counts = Counter(m["sem_cluster"] for m in members)
        if len(sem_counts) <= 1:
            continue
        dominant_sem, dominant_count = sem_counts.most_common(1)[0]
        purity = dominant_count / len(members)
        minorities = [m for m in members if m["sem_cluster"] != dominant_sem]
        minorities.sort(key=lambda m: m["label"])
        for idx, obj in enumerate(minorities[:examples_per_case]):
            rows.append(_divergence_row(
                object_class=object_class,
                pattern="struct_together_sem_split",
                obj=obj,
                struct_cluster=struct_cluster,
                sem_cluster=obj["sem_cluster"],
                cluster_size=len(members),
                minority_size=len(minorities),
                dominant_sem_cluster=dominant_sem,
                purity=round(purity, 4),
                example_rank=idx + 1,
                definitions=definitions,
                edge_stats=edge_stats,
            ))

    # Semantic cluster split across multiple structural clusters.
    for sem_cluster, members in sorted(by_sem.items()):
        if len(members) < min_cluster_size:
            continue
        struct_counts = Counter(m["struct_cluster"] for m in members)
        if len(struct_counts) <= 1:
            continue
        dominant_struct, dominant_count = struct_counts.most_common(1)[0]
        purity = dominant_count / len(members)
        minorities = [m for m in members if m["struct_cluster"] != dominant_struct]
        minorities.sort(key=lambda m: m["label"])
        for idx, obj in enumerate(minorities[:examples_per_case]):
            rows.append(_divergence_row(
                object_class=object_class,
                pattern="sem_together_struct_split",
                obj=obj,
                struct_cluster=obj["struct_cluster"],
                sem_cluster=sem_cluster,
                cluster_size=len(members),
                minority_size=len(minorities),
                dominant_struct_cluster=dominant_struct,
                purity=round(purity, 4),
                example_rank=idx + 1,
                definitions=definitions,
                edge_stats=edge_stats,
            ))

    return rows


def _divergence_row(
    object_class: str,
    pattern: str,
    obj: dict[str, str],
    struct_cluster: int,
    sem_cluster: int,
    cluster_size: int,
    minority_size: int,
    purity: float,
    example_rank: int,
    definitions: dict[str, dict[str, str]] | None,
    edge_stats: dict[str, dict[str, float | str]] | None,
    dominant_sem_cluster: int | None = None,
    dominant_struct_cluster: int | None = None,
) -> dict:
    definition = ""
    mean_weight = ""
    radicals = ""
    if definitions and obj["id"] in definitions:
        definition = definitions[obj["id"]].get("definition", "")
    if edge_stats and obj["id"] in edge_stats:
        mean_weight = edge_stats[obj["id"]]["mean_weight_cosine"]
        radicals = edge_stats[obj["id"]]["radicals"]
    return {
        "object_class": object_class,
        "pattern": pattern,
        "struct_cluster": struct_cluster,
        "sem_cluster": sem_cluster,
        "cluster_size": cluster_size,
        "minority_size": minority_size,
        "dominant_sem_cluster": dominant_sem_cluster if dominant_sem_cluster is not None else "",
        "dominant_struct_cluster": dominant_struct_cluster if dominant_struct_cluster is not None else "",
        "cluster_purity": purity,
        "example_rank": example_rank,
        "object_id": obj["id"],
        "object_label": obj["label"],
        "definition": definition,
        "mean_weight_cosine": mean_weight,
        "radicals": radicals,
    }


def write_digest(
    path: Path,
    meta: dict,
    metrics_rows: list[dict[str, str]],
    convergence_rows: list[dict],
    divergence_rows: list[dict],
) -> None:
    def primary_metric(obj_class: str) -> dict[str, str] | None:
        for row in metrics_rows:
            if row["object_class"] == obj_class and row["is_primary_k"] == "true":
                return row
        return None

    rad_metrics = primary_metric("radicals")
    ch_metrics = primary_metric("characters")

    lines = [
        "# Análise interpretativa (etapa 6)",
        "",
        "Resumo automático dos casos de convergência e divergência entre agrupamentos",
        "estruturais e semânticos.",
        "",
        "## Contexto",
        "",
        f"- K primário (radicais): {meta.get('radicals_primary_k')}",
        f"- K primário (caracteres): {meta.get('characters_primary_k')}",
        "",
    ]
    if rad_metrics:
        lines.extend(
            [
                "## Métricas globais (K primário)",
                "",
                f"- Radicais: ARI={float(rad_metrics['ari']):.4f}, NMI={float(rad_metrics['nmi']):.4f}",
                f"- Caracteres: ARI={float(ch_metrics['ari']):.4f}, NMI={float(ch_metrics['nmi']):.4f}",
                "",
                "Valores baixos de ARI/NMI indicam pouco alinhamento global entre as duas visões.",
                "A análise abaixo destaca onde ainda há sobreposição local (convergência) e onde",
                "estrutura e semântica se separam (divergência).",
                "",
            ]
        )

    lines.extend(["## Convergência (amostra)", ""])
    for row in convergence_rows[:8]:
        lines.append(
            f"- **{row['object_class']}** {row['object_label']} "
            f"(struct={row['struct_cluster']}, sem={row['sem_cluster']}, "
            f"overlap={row['overlap_count']}): {row.get('definition', '')[:120]}"
        )

    lines.extend(["", "## Divergência (amostra)", ""])
    for row in divergence_rows[:10]:
        note = row["pattern"]
        if row.get("definition"):
            note += f" — {row['definition'][:100]}"
        lines.append(
            f"- **{row['object_class']}** {row['object_label']} "
            f"({row['pattern']}, struct={row['struct_cluster']}, sem={row['sem_cluster']}): {note}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partitions-dir", type=Path, default=Path("data/processed/partitions"))
    parser.add_argument("--comparison-dir", type=Path, default=Path("entrega_final/comparacao"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out-dir", type=Path, default=Path("entrega_final/exemplos"))
    parser.add_argument("--top-convergence-pairs", type=int, default=10)
    parser.add_argument("--examples-per-pair", type=int, default=5)
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--divergence-examples-per-case", type=int, default=3)
    args = parser.parse_args()

    meta = json.loads((args.partitions_dir / "clustering_meta.json").read_text(encoding="utf-8"))
    metrics_rows = read_csv(args.comparison_dir / "metrics_summary.csv")
    char_defs = load_character_definitions(args.data_dir / "characters.csv")
    edge_stats = load_character_edge_stats(args.data_dir / "radical_character_edges_weighted.csv")

    log("Loading partitions...")
    radicals = load_merged_partitions(
        args.partitions_dir / "radicals_structural.csv",
        args.partitions_dir / "radicals_semantic.csv",
        id_field="kangxi_radical",
        label_field="kangxi_radical",
    )
    characters = load_merged_partitions(
        args.partitions_dir / "characters_structural.csv",
        args.partitions_dir / "characters_semantic.csv",
        id_field="codepoint",
        label_field="char",
    )

    log("Finding convergence examples...")
    convergence_rows = []
    convergence_rows.extend(
        find_convergence(
            "radicals", radicals, definitions=None,
            top_pairs=args.top_convergence_pairs,
            examples_per_pair=args.examples_per_pair,
        )
    )
    convergence_rows.extend(
        find_convergence(
            "characters", characters, definitions=char_defs,
            top_pairs=args.top_convergence_pairs,
            examples_per_pair=args.examples_per_pair,
        )
    )

    log("Finding divergence examples...")
    divergence_rows = []
    divergence_rows.extend(
        find_divergence(
            "radicals", radicals, definitions=None, edge_stats=None,
            min_cluster_size=3,
            examples_per_case=args.divergence_examples_per_case,
        )
    )
    divergence_rows.extend(
        find_divergence(
            "characters", characters, definitions=char_defs, edge_stats=edge_stats,
            min_cluster_size=args.min_cluster_size,
            examples_per_case=args.divergence_examples_per_case,
        )
    )

    convergence_fields = [
        "object_class", "pattern", "struct_cluster", "sem_cluster",
        "overlap_count", "struct_cluster_size", "sem_cluster_size",
        "overlap_fraction_smaller_cluster", "example_rank",
        "object_id", "object_label", "definition",
    ]
    divergence_fields = [
        "object_class", "pattern", "struct_cluster", "sem_cluster",
        "cluster_size", "minority_size", "dominant_sem_cluster",
        "dominant_struct_cluster", "cluster_purity", "example_rank",
        "object_id", "object_label", "definition",
        "mean_weight_cosine", "radicals",
    ]
    write_csv(args.out_dir / "convergence.csv", convergence_rows, convergence_fields)
    write_csv(args.out_dir / "divergence.csv", divergence_rows, divergence_fields)
    write_digest(
        args.out_dir / "digest.md",
        meta,
        metrics_rows,
        convergence_rows,
        divergence_rows,
    )

    log(f"Convergence examples: {len(convergence_rows)} -> {args.out_dir / 'convergence.csv'}")
    log(f"Divergence examples: {len(divergence_rows)} -> {args.out_dir / 'divergence.csv'}")
    log(f"Digest -> {args.out_dir / 'digest.md'}")
    log("Done.")


if __name__ == "__main__":
    main()
