"""Scores verification trials from an embeddings CSV: EER and minDCF.

Usage: python -I training/score_trials.py EMBEDDINGS_CSV TRIALS_TXT

The embeddings come from `:harness:embedFiles`, which runs the app's own Kotlin front end and
model, so these numbers describe what ships rather than a Python re-implementation. Trials come
from make_trials.py. Utterances are matched by file stem (the audio id).

Reported separately for `hard` non-targets (same gender, at most a year apart) and `random`
ones. minDCF uses p_target = 0.01 with unit costs, the common NIST setting.
"""
import csv
import sys
from pathlib import Path

import numpy as np

P_TARGET = 0.01


def load_embeddings(path):
    """Embeddings CSV (embedFiles or embed_onnx.py layout), keyed by file stem."""
    embeddings = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            embeddings[Path(row[0]).stem] = np.array(row[1:], dtype=np.float64)
    return embeddings


def trial_scores(embeddings, trials_path):
    """Cosine scores by kind ("target", "hard", "random"), and how many trials lacked an embedding."""
    scores = {"target": [], "hard": [], "random": []}
    missing = 0
    for line in open(trials_path, encoding="utf-8"):
        a, b, label, kind = line.split()
        if a not in embeddings or b not in embeddings:
            missing += 1
            continue
        ea, eb = embeddings[a], embeddings[b]
        score = float(ea @ eb / (np.linalg.norm(ea) * np.linalg.norm(eb)))
        scores["target" if label == "target" else kind].append(score)
    return {kind: np.array(values) for kind, values in scores.items()}, missing


def error_rates(targets, nontargets):
    """Miss and false-alarm rates at every threshold, scanning from high scores to low."""
    labels = np.concatenate([np.ones(len(targets)), np.zeros(len(nontargets))])
    order = np.argsort(-np.concatenate([targets, nontargets]), kind="stable")
    labels = labels[order]
    false_alarms = np.concatenate([[0], np.cumsum(1 - labels)]) / len(nontargets)
    misses = 1 - np.concatenate([[0], np.cumsum(labels)]) / len(targets)
    return misses, false_alarms


def eer(targets, nontargets):
    misses, false_alarms = error_rates(targets, nontargets)
    i = np.argmin(np.abs(misses - false_alarms))
    return (misses[i] + false_alarms[i]) / 2


def min_dcf(targets, nontargets):
    misses, false_alarms = error_rates(targets, nontargets)
    dcf = P_TARGET * misses + (1 - P_TARGET) * false_alarms
    return float(dcf.min() / min(P_TARGET, 1 - P_TARGET))


def main():
    scores, missing = trial_scores(load_embeddings(sys.argv[1]), sys.argv[2])
    t = scores["target"]
    print(f"{len(t)} target trials" + (f" ({missing} trials skipped: utterance not embedded)" if missing else ""))
    for kind in ("hard", "random"):
        n = scores[kind]
        if len(n) == 0:
            continue
        print(f"  {kind:6} non-targets ({len(n):5d}): EER {eer(t, n):6.2%}   minDCF(0.01) {min_dcf(t, n):.3f}   "
              f"mean cosine target {t.mean():.3f} / non-target {n.mean():.3f}")


if __name__ == "__main__":
    main()
