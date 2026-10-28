"""Replays the engine's scoring on dumped embeddings and tries normalization variants.

Usage:
    ./gradlew :harness:run --args="... --embeddings reports/x/embeddings.csv"
    python -I tools/experiments/normalization.py reports/x/embeddings.csv MANIFEST_DIR
    MIN_TARGET=-9 python -I ...   # disable the absolute gate, for scale-changing variants

Only enrollment data is used to fit any normalization, so every variant is deployable. The
constants mirror MatcherConfig, AccumulatorConfig and Metrics defaults; the baseline must
reproduce the harness report exactly before any variant's numbers mean anything. Written for a
single-household manifest. Results: docs/superpowers/plans/2026-10-28-childes-rehearsal.md.
"""
import csv
import os
import sys
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np

TEMP, MIN_Z, FALLBACK_MEAN, FALLBACK_SD, MAX_LLR, MIN_SD = 0.1, 0.5, 0.15, 0.10, 4.0, 0.02
MIN_TARGET = float(os.environ.get("MIN_TARGET", "0.35"))
WINDOW_MS, COOLDOWN_MS, BUDGET_MS = 20_000, 45_000, 10_000

emb_path, manifest = sys.argv[1], Path(sys.argv[2])
enroll, obs = defaultdict(list), []
with open(emb_path, encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        kind, path, hh, spk, role, lang, ts = row[:7]
        vec = np.array(row[7:], dtype=np.float64)
        if kind == "enroll":
            enroll[spk].append(vec)
        else:
            obs.append((path, spk, role, int(ts), vec))

roles = {}
with open(manifest / "recordings.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        roles[r["speaker_id"]] = r["role"]
target_id = next(s for s, r in roles.items() if r == "target_child")
cohort_ids = [s for s in enroll if s != target_id]
events = set()
with open(manifest / "events.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        events.add(r["recording_path"])
durations = {}
for path in {o[0] for o in obs} | events:
    with wave.open(str(manifest / path)) as w:
        durations[path] = w.getnframes() / w.getframerate()
# Recordings with no observations still count as exposure (the harness counts every trace).
with open(manifest / "recordings.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["content_type"] != "enrollment" and r["path"] not in durations:
            with wave.open(str(manifest / r["path"])) as w:
                durations[r["path"]] = w.getnframes() / w.getframerate()

O = np.stack([o[4] for o in obs])
obs_speaker = [o[1] for o in obs]


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def llr_from_sims(target_sim, cohort_sims):
    """HouseholdMatcher.score, vectorised over observations."""
    all_sims = np.concatenate([target_sim[:, None], cohort_sims], axis=1)
    m = all_sims.max(axis=1, keepdims=True)
    e = np.exp((all_sims - m) / TEMP)
    post = np.clip(e[:, 0] / e.sum(axis=1), 1e-6, 1 - 1e-6)
    raw = np.log(post / (1 - post))
    if cohort_sims.shape[1] < 2:
        z = (target_sim - FALLBACK_MEAN) / FALLBACK_SD
    else:
        z = (target_sim - cohort_sims.mean(axis=1)) / np.maximum(cohort_sims.std(axis=1), MIN_SD)
    llr = raw.copy()
    llr[z < MIN_Z] = np.minimum(llr[z < MIN_Z], 0)
    llr[target_sim < MIN_TARGET] = np.minimum(llr[target_sim < MIN_TARGET], 0)
    return raw, np.clip(llr, -MAX_LLR, MAX_LLR)


def replay(llr, threshold):
    """EvidenceAccumulator + Metrics.scoreEvents over every recording."""
    by_rec = defaultdict(list)
    for (path, _, _, ts, _), value in zip(obs, llr):
        by_rec[path].append((ts, value))
    detected = false_alarms = 0
    for path in durations:
        window, fired_at, hit = [], None, False
        for ts, value in sorted(by_rec.get(path, [])):
            if fired_at is not None and ts - fired_at < COOLDOWN_MS:
                continue
            window.append((ts, value))
            window = [(t, v) for t, v in window if t >= ts - WINDOW_MS]
            if sum(v for _, v in window) >= threshold:
                fired_at, window = ts, []
                if path in events and not hit and ts <= BUDGET_MS:
                    hit = True
                    detected += 1
                else:
                    false_alarms += 1
    hours = sum(durations.values()) / 3600
    return detected / len(events), false_alarms / hours


def auc(pos, neg):
    pos, neg = np.sort(pos), np.sort(neg)
    lo, hi = np.searchsorted(neg, pos, "left"), np.searchsorted(neg, pos, "right")
    return float((lo + 0.5 * (hi - lo)).sum() / (len(pos) * len(neg)))


def evaluate(name, transform_obs, centroids, score_fn=None):
    ids = [target_id] + cohort_ids
    X = transform_obs(O)
    C = np.stack([centroids[s] for s in ids])
    sims = X @ C.T if score_fn is None else score_fn(X, C, ids)
    raw, llr = llr_from_sims(sims[:, 0], sims[:, 1:])
    is_target = np.array([s == target_id for s in obs_speaker])
    top1 = {}
    for s in ids:
        mask = np.array([o == s for o in obs_speaker])
        if mask.any():
            top1[s.split("-")[-1]] = float(np.mean(sims[mask].argmax(axis=1) == ids.index(s)))
    sib = [s for s in cohort_ids if roles.get(s) == "sibling"]
    sib_mask = np.array([o in sib for o in obs_speaker])
    print(f"\n== {name}")
    print("  closest-is-self: " + "  ".join(f"{k} {v:.0%}" for k, v in top1.items()))
    print(f"  AUC raw LLR target vs all {auc(raw[is_target], raw[~is_target]):.3f}"
          + (f", vs sibling {auc(raw[is_target], raw[sib_mask]):.3f}" if sib_mask.any() else ""))
    for t in (2.0, 4.0, 6.0, 8.0, 12.0):
        d, fa = replay(llr, t)
        print(f"  threshold {t:>3}: detection {d:6.1%}  FA/hour {fa:6.2f}")


ids = [target_id] + cohort_ids
E = {s: np.stack(v) for s, v in enroll.items()}

# Baseline: exactly what the engine does.
evaluate("baseline (engine as is)", lambda X: X, {s: unit(E[s].mean(axis=0)) for s in ids})

# Centering on the household's enrollment mean, balanced so a talkative speaker does not dominate.
mean = np.mean([E[s].mean(axis=0) for s in ids], axis=0)
evaluate("centered on household mean", lambda X: unit(X - mean), {s: unit(unit(E[s] - mean).mean(axis=0)) for s in ids})

# Per-centroid score normalization (Z-norm): each centroid's scores against the *other* enrolled
# speakers' clips set its impostor mean and spread.
def znorm(X, C, ids):
    raw = X @ C.T
    out = np.empty_like(raw)
    for j, s in enumerate(ids):
        impostors = np.concatenate([E[o] for o in ids if o != s]) @ C[j]
        out[:, j] = (raw[:, j] - impostors.mean()) / impostors.std()
    # Back to a cosine-like scale so the matcher's temperature and absolute gate still mean something.
    return 0.5 + 0.1 * out
evaluate("Z-norm per centroid", lambda X: X, {s: unit(E[s].mean(axis=0)) for s in ids}, znorm)

# WCCN: whiten by the within-speaker covariance of the enrollment clips, after centering.
for reg in (0.1, 0.3, 1.0):
    clips = np.concatenate([unit(E[s] - mean) for s in ids])
    within = sum(np.cov(unit(E[s] - mean).T, bias=True) * len(E[s]) for s in ids) / len(clips)
    within += reg * np.trace(within) / within.shape[0] * np.eye(within.shape[0])
    w = np.linalg.cholesky(np.linalg.inv(within))
    project = lambda X, w=w: unit(unit(X - mean) @ w)
    evaluate(f"centered + WCCN (reg {reg})", project, {s: unit(project(E[s]).mean(axis=0)) for s in ids})

print(f"\n{len(obs)} observations, {len(events)} events, {sum(durations.values())/3600:.2f} h total; "
      f"enroll clips: " + ", ".join(f"{s.split('-')[-1]} {len(E[s])}" for s in ids))
