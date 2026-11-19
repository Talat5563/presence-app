"""How well the engine's score separates the target child from each other role, per corpus.

Usage:
    ./gradlew :harness:run --args="... --embeddings reports/x/embeddings.csv"
    python -I tools/experiments/household_auc.py reports/x/embeddings.csv [reports/y/embeddings.csv ...]
    python -I tools/experiments/household_auc.py --households h1,h2 ...   # score only these households

For each household with an enrolled target child, every observation is scored as
HouseholdMatcher does (centroid per enrolled speaker, the raw log-odds before any gate), and the
target child's observations are compared with each other role's, pooled over households. Runs
on several dumps of the same manifest side by side, so a model change reads as one table.
"""
import csv
import sys
from collections import defaultdict

import numpy as np

TEMP = 0.1  # MatcherConfig.temperature


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def auc(pos, neg):
    pos, neg = np.sort(pos), np.sort(neg)
    lo, hi = np.searchsorted(neg, pos, "left"), np.searchsorted(neg, pos, "right")
    return float((lo + 0.5 * (hi - lo)).sum() / (len(pos) * len(neg)))


def scores(path, households=None):
    enroll, obs = defaultdict(list), []
    target = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            kind, _, hh, spk, role = row[:5]
            if households is not None and hh not in households:
                continue
            vec = np.array(row[7:], dtype=np.float64)
            if kind == "enroll":
                enroll[(hh, spk)].append(vec)
                if role == "target_child":
                    target[hh] = spk
            else:
                obs.append((hh, spk, role, vec))
    by_role, closest_self = defaultdict(list), []
    for hh, child in target.items():
        ids = [child] + sorted(s for h, s in enroll if h == hh and s != child)
        centroids = np.stack([unit(np.mean(enroll[(hh, s)], axis=0)) for s in ids])
        rows = [o for o in obs if o[0] == hh]
        if not rows:
            continue
        sims = unit(np.stack([o[3] for o in rows])) @ centroids.T
        e = np.exp((sims - sims.max(axis=1, keepdims=True)) / TEMP)
        post = np.clip(e[:, 0] / e.sum(axis=1), 1e-6, 1 - 1e-6)
        raw = np.log(post / (1 - post))
        for (_, spk, role, _), value, best in zip(rows, raw, sims.argmax(axis=1)):
            by_role[role].append(value)
            if spk == child:
                closest_self.append(best == 0)
    return {r: np.array(v) for r, v in by_role.items()}, float(np.mean(closest_self))


def main():
    args, households = sys.argv[1:], None
    if args[:1] == ["--households"]:
        households, args = set(args[1].split(",")), args[2:]
    results = [(p, *scores(p, households)) for p in args]
    roles = sorted({r for _, by_role, _ in results for r in by_role if r != "target_child"})
    print("| Embeddings | Child obs | Child closest to own centroid | "
          + " | ".join(f"AUC vs {r}" for r in roles) + " | AUC vs all others |")
    print("|---|---:|---:|" + "---:|" * (len(roles) + 1))
    for path, by_role, closest in results:
        child = by_role["target_child"]
        others = np.concatenate([v for r, v in by_role.items() if r != "target_child"])
        cells = [f"{auc(child, by_role[r]):.3f} (n={len(by_role[r])})" if r in by_role else "—" for r in roles]
        print(f"| {path} | {len(child)} | {closest:.0%} | " + " | ".join(cells) + f" | {auc(child, others):.3f} |")


if __name__ == "__main__":
    main()
