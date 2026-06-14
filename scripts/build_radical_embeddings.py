#!/usr/bin/env python3
"""Build semantic embeddings for the Kangxi radicals.

Each radical present in the structural graph is encoded once with the same
SentenceTransformer model used for characters. The text combines the radical
symbol with its English meaning, e.g. ``character: 水. meaning: water.``.

Output embeddings are L2-normalized so cosine similarity equals the dot product.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

_START = time.monotonic()


def log(message: str) -> None:
    """Print a timestamped progress message that is flushed immediately."""
    elapsed = time.monotonic() - _START
    print(f"[{elapsed:7.1f}s] {message}", flush=True)


def load_graph_radicals(edges_csv: Path) -> dict[str, str]:
    """Return {radical_symbol: kangxi_radical_number} for radicals in the graph."""
    radical_to_number: dict[str, str] = {}
    with edges_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            radical_to_number.setdefault(row["kangxi_radical"], row.get("kangxi_radical_number", ""))
    return radical_to_number


def load_radical_meanings(reference_csv: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Return (by_number, by_symbol) maps from the Kangxi reference to English names."""
    by_number: dict[str, str] = {}
    by_symbol: dict[str, str] = {}
    with reference_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = (row.get("name_en", "") or "").strip()
            number = (row.get("radical_number", "") or "").strip()
            symbol = (row.get("symbol", "") or "").strip()
            if number:
                by_number.setdefault(number, name)
            if symbol:
                by_symbol.setdefault(symbol, name)
    return by_number, by_symbol


def build_text(symbol: str, meaning: str) -> str:
    meaning = (meaning or "").strip().lower()
    if meaning:
        return f"character: {symbol}. meaning: {meaning}."
    return f"character: {symbol}."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--edges-input",
        type=Path,
        default=Path("data/processed/radical_character_edges.csv"),
        help="Structural edge CSV (source of the radical set used in the graph)",
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=Path("data/reference/kangxi_radicals.csv"),
        help="Kangxi radicals reference with English names",
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
        default=Path("data/processed/radical_embeddings.npy"),
        help="Output .npy path for radical embeddings",
    )
    parser.add_argument(
        "--index-output",
        type=Path,
        default=Path("data/processed/radical_embeddings_index.csv"),
        help="Output CSV index for radical embeddings",
    )
    args = parser.parse_args()

    log("Importing numpy...")
    import numpy as np

    log("Reading radical set and meanings...")
    radical_to_number = load_graph_radicals(args.edges_input)
    by_number, by_symbol = load_radical_meanings(args.reference_csv)

    radicals = sorted(radical_to_number)
    rows = []
    for symbol in radicals:
        number = radical_to_number.get(symbol, "")
        meaning = by_number.get(number) or by_symbol.get(symbol, "")
        rows.append((symbol, number, meaning, build_text(symbol, meaning)))

    missing = sum(1 for _, _, meaning, _ in rows if not meaning)
    log(f"Radicals in graph: {len(rows)} (without English meaning: {missing})")

    texts = [text for _, _, _, text in rows]

    log("Importing sentence-transformers (first import can be slow)...")
    from sentence_transformers import SentenceTransformer

    log(f"Loading model '{args.model}' (uses local cache; set HF_HUB_OFFLINE=1 to skip network checks)...")
    model = SentenceTransformer(args.model)
    log("Model loaded. Encoding radicals...")

    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    log(f"Encoding done: {embeddings.shape}. Saving outputs...")

    args.embeddings_output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.embeddings_output, embeddings)

    args.index_output.parent.mkdir(parents=True, exist_ok=True)
    with args.index_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "kangxi_radical", "kangxi_radical_number", "name_en", "semantic_text"])
        for idx, (symbol, number, meaning, text) in enumerate(rows):
            writer.writerow([idx, symbol, number, meaning, text])

    log(f"Embeddings: {embeddings.shape} -> {args.embeddings_output}")
    log(f"Index saved to: {args.index_output}")
    log("Done.")


if __name__ == "__main__":
    main()
