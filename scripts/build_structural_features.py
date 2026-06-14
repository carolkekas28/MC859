#!/usr/bin/env python3
"""Build structural feature spaces for radicals and characters.

From the bipartite radical-character edge list this script materializes a single
sparse incidence matrix ``B`` (radicals x characters) and derives:

- character structural features: rows of ``B.T`` (each character described by which
  of the 214 Kangxi radicals compose it), kept sparse;
- radical structural features: ``B`` reduced with TruncatedSVD (each radical
  described by its low-dimensional connectivity profile over characters).

"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load_edges(edges_csv: Path) -> tuple[list[tuple[str, str]], dict[str, str], dict[str, str]]:
    """Return (radical, character) pairs plus readable label maps.

    - ``radical_label[radical]`` -> kangxi_radical_number
    - ``character_label[codepoint]`` -> character glyph
    """
    pairs: list[tuple[str, str]] = []
    radical_label: dict[str, str] = {}
    character_label: dict[str, str] = {}
    with edges_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            radical = row["kangxi_radical"]
            cp = row["character_codepoint"]
            pairs.append((radical, cp))
            radical_label.setdefault(radical, row.get("kangxi_radical_number", ""))
            character_label.setdefault(cp, row.get("character", ""))
    return pairs, radical_label, character_label


def build_incidence(
    pairs: list[tuple[str, str]],
):
    """Build a binary sparse incidence matrix B (radicals x characters).

    Returns (B_csr, radicals, characters) with deterministic ordering.
    """
    import numpy as np
    from scipy import sparse

    radicals = sorted({r for r, _ in pairs})
    characters = sorted({c for _, c in pairs})
    radical_to_row = {r: i for i, r in enumerate(radicals)}
    character_to_col = {c: i for i, c in enumerate(characters)}

    rows = np.fromiter((radical_to_row[r] for r, _ in pairs), dtype=np.int32, count=len(pairs))
    cols = np.fromiter((character_to_col[c] for _, c in pairs), dtype=np.int32, count=len(pairs))
    data = np.ones(len(pairs), dtype=np.float32)

    incidence = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(len(radicals), len(characters)),
    ).tocsr()
    # Collapse any duplicate (radical, character) pairs to binary presence.
    incidence.data[:] = 1.0
    incidence.sum_duplicates()
    incidence.data[:] = 1.0
    return incidence, radicals, characters


def write_index(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--edges-input",
        type=Path,
        default=Path("data/processed/radical_character_edges.csv"),
        help="Structural edge CSV (radical-character pairs)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for output feature/index files",
    )
    parser.add_argument(
        "--svd-components",
        type=int,
        default=50,
        help="Number of TruncatedSVD components for radical structural features",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for TruncatedSVD",
    )
    args = parser.parse_args()

    import numpy as np
    from scipy import sparse
    from sklearn.decomposition import TruncatedSVD

    pairs, radical_label, character_label = load_edges(args.edges_input)
    incidence, radicals, characters = build_incidence(pairs)
    print(f"Edges read: {len(pairs)}")
    print(f"Incidence matrix B: {incidence.shape[0]} radicals x {incidence.shape[1]} characters")
    print(f"Non-zeros (unique radical-character pairs): {incidence.nnz}")

    # Character structural features = B.T (characters x radicals), kept sparse.
    char_features = incidence.T.tocsr()
    char_features_path = args.out_dir / "struct_features_characters.npz"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(char_features_path, char_features)
    print(f"Character structural features: {char_features.shape} -> {char_features_path}")

    # Radical structural features = TruncatedSVD(B).
    n_components = min(args.svd_components, min(incidence.shape) - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=args.random_state)
    radical_features = svd.fit_transform(incidence)
    radical_features = np.ascontiguousarray(radical_features, dtype=np.float32)
    radical_features_path = args.out_dir / "struct_features_radicals.npy"
    np.save(radical_features_path, radical_features)
    explained = float(svd.explained_variance_ratio_.sum())
    print(
        f"Radical structural features: {radical_features.shape} "
        f"(SVD {n_components} comps, explained variance {explained:.3f}) -> {radical_features_path}"
    )

    # Index files (row order matches the feature matrices).
    write_index(
        args.out_dir / "struct_index_characters.csv",
        ["row_id", "codepoint", "char"],
        [[str(i), cp, character_label.get(cp, "")] for i, cp in enumerate(characters)],
    )
    write_index(
        args.out_dir / "struct_index_radicals.csv",
        ["row_id", "kangxi_radical", "kangxi_radical_number"],
        [[str(i), r, radical_label.get(r, "")] for i, r in enumerate(radicals)],
    )
    print(f"Indices saved to: {args.out_dir}/struct_index_characters.csv, struct_index_radicals.csv")


if __name__ == "__main__":
    main()
