"""Compares two embeddings CSVs file by file: the front-end parity check (Phase 3, Task 5).

Usage: python -I training/compare_embeddings.py A.csv B.csv

Typically A comes from `:harness:embedFiles` (Kotlin front end) and B from embed_onnx.py (Python
front end), with the same model on the same files. Rows pair up by file stem. Reports the cosine
distribution and the files below PARITY, the bar the plan sets for "fed the same features".
"""
import csv
import sys
from pathlib import Path

import numpy as np

PARITY = 0.999


def load(path):
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        return {Path(row[0]).stem: np.array(row[1:], dtype=np.float64) for row in reader}


a, b = load(sys.argv[1]), load(sys.argv[2])
shared = sorted(a.keys() & b.keys())
cosines = np.array([a[k] @ b[k] / (np.linalg.norm(a[k]) * np.linalg.norm(b[k])) for k in shared])
print(f"{len(shared)} files in both ({len(a) - len(shared)} only in A, {len(b) - len(shared)} only in B)")
print(f"cosine min {cosines.min():.6f}, 1st percentile {np.percentile(cosines, 1):.6f}, "
      f"median {np.median(cosines):.6f}")
below = [(c, k) for c, k in zip(cosines, shared) if c < PARITY]
print(f"{len(below)} files below {PARITY}" + "".join(f"\n  {k}: {c:.6f}" for c, k in sorted(below)[:10]))
sys.exit(1 if below else 0)
