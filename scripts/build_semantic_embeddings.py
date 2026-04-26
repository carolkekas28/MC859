#!/usr/bin/env python3
"""Overwrite characters.csv with semantic_text and generate embeddings."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def build_semantic_text(char: str, definition: str) -> str:
    definition = (definition or "").strip()
    if definition:
        return f"character: {char}. definition: {definition}."
    return f"character: {char}."


def overwrite_characters_with_semantic_text(characters_csv: Path) -> tuple[list[dict[str, str]], list[str]]:
    with characters_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if "semantic_text" not in fieldnames:
        fieldnames = [*fieldnames, "semantic_text"]

    for row in rows:
        row["semantic_text"] = build_semantic_text(row.get("char", ""), row.get("definition", ""))

    tmp_path = characters_csv.with_suffix(".tmp.csv")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(characters_csv)
    return rows, fieldnames


def generate_embeddings(
    rows: list[dict[str, str]],
    model_name: str,
    batch_size: int,
    output_npy: Path,
    output_meta_csv: Path,
    output_meta_json: Path,
) -> None:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    texts = [row["semantic_text"] for row in rows]
    codepoints = [row.get("codepoint", "") for row in rows]
    chars = [row.get("char", "") for row in rows]
    has_definition = [row.get("has_definition", "").lower() == "true" for row in rows]

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_npy, embeddings)

    output_meta_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_meta_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row_id", "codepoint", "char", "has_definition", "semantic_text"],
        )
        writer.writeheader()
        for idx, (cp, ch, has_def, text) in enumerate(zip(codepoints, chars, has_definition, texts)):
            writer.writerow(
                {
                    "row_id": idx,
                    "codepoint": cp,
                    "char": ch,
                    "has_definition": str(has_def).lower(),
                    "semantic_text": text,
                }
            )

    meta = {
        "model_name": model_name,
        "row_count": len(rows),
        "embedding_dim": int(embeddings.shape[1]) if len(rows) else 0,
        "batch_size": batch_size,
        "files": {
            "embeddings_npy": str(output_npy),
            "index_csv": str(output_meta_csv),
        },
    }
    output_meta_json.parent.mkdir(parents=True, exist_ok=True)
    output_meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--characters-csv",
        type=Path,
        default=Path("data/processed/characters.csv"),
        help="Characters CSV to overwrite with semantic_text",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model name",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for embedding generation",
    )
    parser.add_argument(
        "--embeddings-output",
        type=Path,
        default=Path("data/processed/semantic_embeddings.npy"),
        help="Output .npy path for embeddings",
    )
    parser.add_argument(
        "--index-output",
        type=Path,
        default=Path("data/processed/semantic_embeddings_index.csv"),
        help="Output CSV path for embedding index/metadata",
    )
    parser.add_argument(
        "--meta-output",
        type=Path,
        default=Path("data/processed/semantic_embeddings_meta.json"),
        help="Output JSON path for embedding metadata",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Only overwrite characters.csv with semantic_text",
    )
    args = parser.parse_args()

    rows, _ = overwrite_characters_with_semantic_text(args.characters_csv)
    print(f"Updated semantic_text in: {args.characters_csv}")
    print(f"Rows updated: {len(rows)}")

    if args.skip_embeddings:
        print("Skipped embedding generation (--skip-embeddings).")
        return

    generate_embeddings(
        rows=rows,
        model_name=args.model,
        batch_size=args.batch_size,
        output_npy=args.embeddings_output,
        output_meta_csv=args.index_output,
        output_meta_json=args.meta_output,
    )
    print(f"Embeddings saved to: {args.embeddings_output}")
    print(f"Index saved to: {args.index_output}")
    print(f"Metadata saved to: {args.meta_output}")


if __name__ == "__main__":
    main()
