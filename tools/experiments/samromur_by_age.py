"""Samrómur verification EER and minDCF split by speaker age band.

Usage:
    python -I tools/experiments/samromur_by_age.py SPLIT EMBEDDINGS_CSV [EMBEDDINGS_CSV ...]

SPLIT is dev or test; trials and ages come from data/samromur/trials_SPLIT.txt and
metadata_SPLIT.tsv. A trial counts towards a band only when both utterances' speakers are in it.
Bands are aggregates over speakers; nothing here tries to tell who a speaker is.
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training"))
from score_trials import eer, load_embeddings, min_dcf  # noqa: E402

BANDS = {"9-12": range(9, 13), "13-15": range(13, 16)}


def main():
    split, paths = sys.argv[1], sys.argv[2:]
    with open(f"data/samromur/metadata_{split}.tsv", encoding="utf-8") as f:
        age = {r["audio_id"]: int(r["age"]) for r in csv.DictReader(f, delimiter="\t")}
    with open(f"data/samromur/trials_{split}.txt", encoding="utf-8") as f:
        trials = [line.split() for line in f]
    print("| Embeddings | Ages | Target trials | Non-target trials | EER | minDCF |")
    print("|---|---|---:|---:|---:|---:|")
    for path in paths:
        emb = {k: v / np.linalg.norm(v) for k, v in load_embeddings(path).items()}
        for band, ages in BANDS.items():
            targets, nontargets = [], []
            for a, b, label, _ in trials:
                if age[a] in ages and age[b] in ages and a in emb and b in emb:
                    (targets if label == "target" else nontargets).append(float(emb[a] @ emb[b]))
            t, n = np.array(targets), np.array(nontargets)
            print(f"| {path} | {band} | {len(t)} | {len(n)} | {eer(t, n):.2%} | {min_dcf(t, n):.3f} |")


if __name__ == "__main__":
    main()
