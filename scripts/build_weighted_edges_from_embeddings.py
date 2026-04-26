#!/usr/bin/env python3
"""Build weighted radical-character edges using pure embeddings.

This script creates a weighted version of `radical_character_edges.csv` where:
- character embeddings are read from precomputed embedding files
- radical embeddings are computed directly from radical symbol text
- no neighborhood averaging is used for radicals
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from tqdm import tqdm


def load_character_embeddings(
    embeddings_npy: Path,
    index_csv: Path,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    embeddings = np.load(embeddings_npy)
    codepoint_to_row: dict[str, int] = {}
    codepoint_to_char: dict[str, str] = {}

    with index_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cp = row["codepoint"]
            row_id = int(row["row_id"])
            codepoint_to_row[cp] = row_id
            codepoint_to_char[cp] = row["char"]

    codepoint_to_embedding = {
        cp: embeddings[row_id] for cp, row_id in codepoint_to_row.items() if row_id < len(embeddings)
    }
    return codepoint_to_embedding, codepoint_to_char


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def build_radical_embeddings(radicals: list[str], model_name: str, batch_size: int) -> dict[str, np.ndarray]:
    from sentence_transformers import SentenceTransformer

    texts = [f"character: {r}." for r in radicals]
    model = SentenceTransformer(model_name)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    emb = normalize_rows(emb)
    return {radical: emb[i] for i, radical in enumerate(radicals)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--edges-input",
        type=Path,
        default=Path("data/processed/radical_character_edges.csv"),
        help="Input structural edge CSV",
    )
    parser.add_argument(
        "--embeddings-npy",
        type=Path,
        default=Path("data/processed/semantic_embeddings.npy"),
        help="Character embeddings .npy file",
    )
    parser.add_argument(
        "--embeddings-index",
        type=Path,
        default=Path("data/processed/semantic_embeddings_index.csv"),
        help="Character embeddings index CSV (row_id, codepoint, ...)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Embedding model used for radical pure embeddings",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for radical embedding generation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/radical_character_edges_weighted.csv"),
        help="Output weighted edge CSV",
    )
    args = parser.parse_args()

    # Load structural edges
    with args.edges_input.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        edge_rows = list(reader)
        fieldnames = reader.fieldnames or []

    # Load character embeddings
    cp_to_emb, _ = load_character_embeddings(args.embeddings_npy, args.embeddings_index)

    # Build pure radical embeddings from radical symbols only
    unique_radicals = sorted({row["kangxi_radical"] for row in edge_rows})
    radical_to_emb = build_radical_embeddings(unique_radicals, args.model, args.batch_size)

    # Prepare output
    extra_cols = ["weight_cosine", "weight_norm_01", "embedding_model", "radical_semantic_text"]
    output_fields = [*fieldnames, *[c for c in extra_cols if c not in fieldnames]]
    missing_chars = 0
    written = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()

        for row in tqdm(edge_rows, desc="Scoring edge weights", unit="edge"):
            cp = row["character_codepoint"]
            radical = row["kangxi_radical"]
            char_emb = cp_to_emb.get(cp)
            rad_emb = radical_to_emb.get(radical)
            if char_emb is None or rad_emb is None:
                missing_chars += 1
                continue

            cosine = float(np.dot(char_emb, rad_emb))
            norm_01 = (cosine + 1.0) / 2.0

            out = dict(row)
            out["weight_cosine"] = f"{cosine:.8f}"
            out["weight_norm_01"] = f"{norm_01:.8f}"
            out["embedding_model"] = args.model
            out["radical_semantic_text"] = f"character: {radical}."
            writer.writerow(out)
            written += 1

    print(f"Read edges: {len(edge_rows)}")
    print(f"Wrote weighted edges: {written} -> {args.output}")
    print(f"Skipped edges due to missing embeddings: {missing_chars}")
    print(f"Unique radicals embedded: {len(unique_radicals)}")


if __name__ == "__main__":
    main()
