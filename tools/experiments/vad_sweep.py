"""Sweeps VAD gate settings offline over per-frame Silero probabilities.

Usage: python -I tools/experiments/vad_sweep.py PROBES_DIR [PROBES_DIR ...]

PROBES_DIR is the output of `:harness:vadProbes`. Replays VadGate and DetectionEngine's
windowing exactly, frame by frame, and reports for each setting how much of each role's speech
would reach the embedder:

- target episodes yielding at least one / at least two observations (two are needed to reach the
  default fire threshold of 6, since each observation's LLR is capped at 4)
- seconds of enrollable target-child speech per household (segments that pass minEmbed)
- observations per hour of other speakers' audio, the embedding cost the setting adds

Every CHILDES file is one speaker's speech stitched with at most 1 s gaps, so this cannot show
what a setting does to silence, TV or two speakers talking over each other. A setting that bridges
longer pauses looks free here and is not in a real room.
"""
import csv
import itertools
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

FRAME_MS = 32
WINDOW_FRAMES = 3_000 / FRAME_MS  # embedWindowMs, in frames (a segment this long or shorter is one window)
HOP_FRAMES = 1_500 / FRAME_MS


def segments(probs, start, end, hangover, min_segment_ms, max_segment_ms=3_000):
    """VadGate.accept/flush: (first frame, frame count) of every emitted segment."""
    out = []
    in_speech = False
    silent = 0
    first = 0
    length = 0

    def close():
        kept = length - silent
        if kept * FRAME_MS >= min_segment_ms:
            out.append((first, kept))

    for i, p in enumerate(probs):
        if not in_speech:
            if p >= start:
                in_speech, silent, first, length = True, 0, i, 1
        else:
            length += 1
            silent = silent + 1 if p < end else 0
            if silent >= hangover or length * FRAME_MS >= max_segment_ms:
                close()
                in_speech = False
    if in_speech:
        close()
    return out


def observations(segment_frames, min_embed_ms):
    """DetectionEngine.windowsOf: how many embeddings one segment produces."""
    if segment_frames * FRAME_MS < min_embed_ms:
        return 0
    if segment_frames <= WINDOW_FRAMES:
        return 1
    count = int((segment_frames - WINDOW_FRAMES) // HOP_FRAMES) + 1
    uncovered = segment_frames - ((count - 1) * HOP_FRAMES + WINDOW_FRAMES)
    return count + (1 if uncovered >= HOP_FRAMES else 0)


def load(probes_dir):
    probes_dir = Path(probes_dir)
    probs = np.fromfile(probes_dir / "probs.f32", dtype=">f4")
    with open(probes_dir / "index.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        first, count = int(row["first_frame"]), int(row["frames"])
        row["probs"] = probs[first:first + count].tolist()  # plain floats: the replay loop is per frame
    return rows


def summarise(rows, start, end, hangover, min_segment_ms, min_embed_ms):
    episodes = one = two = 0
    enroll_seconds = defaultdict(float)
    other_obs = other_frames = 0
    for row in rows:
        segs = segments(row["probs"], start, end, hangover, min_segment_ms)
        obs = sum(observations(n, min_embed_ms) for _, n in segs)
        child = row["role"] == "target_child"
        if row["content_type"] == "enrollment":
            if child:
                enroll_seconds[row["household_id"]] += sum(
                    n * FRAME_MS / 1000 for _, n in segs if n * FRAME_MS >= min_embed_ms)
        elif child:
            episodes += 1
            one += obs >= 1
            two += obs >= 2
        else:
            other_obs += obs
            other_frames += len(row["probs"])
    households = sorted({r["household_id"] for r in rows if r["role"] == "target_child"})
    enrolled = [enroll_seconds[h] for h in households]
    return {
        "episodes": episodes,
        ">=1 obs": one / max(episodes, 1),
        ">=2 obs": two / max(episodes, 1),
        "enroll s (median)": float(np.median(enrolled)) if enrolled else 0.0,
        "children >=20 s": sum(s >= 20 for s in enrolled),
        "children": len(households),
        "other obs/h": other_obs / max(other_frames * FRAME_MS / 3_600_000, 1e-9),
    }


SETTINGS = [
    # (start, end, hangover frames, minSegmentMs, minEmbedDurationMs); the first is today's default
    (0.5, 0.35, 8, 400, 800),
] + [
    (start, round(start * 0.7, 3), hangover, min_segment, min_embed)
    for start, hangover, min_segment, min_embed in itertools.product(
        (0.5, 0.35, 0.25, 0.15), (8, 16), (400, 250), (800, 600, 400))
]


def main():
    corpora = {Path(d).name: load(d) for d in sys.argv[1:]}
    header = f"{'start':>5} {'end':>5} {'hang':>4} {'minSeg':>6} {'minEmb':>6}"
    for name, rows in corpora.items():
        print(f"\n== {name}")
        print(f"{header}  {'>=1 obs':>7} {'>=2 obs':>7} {'enroll s':>8} {'kids>=20s':>9} {'other obs/h':>11}")
        for setting in dict.fromkeys(SETTINGS):
            s = summarise(rows, *setting)
            start, end, hangover, min_segment, min_embed = setting
            print(f"{start:5.2f} {end:5.2f} {hangover:4d} {min_segment:6d} {min_embed:6d}  "
                  f"{s['>=1 obs']:7.1%} {s['>=2 obs']:7.1%} {s['enroll s (median)']:8.1f} "
                  f"{s['children >=20 s']:4d}/{s['children']:<4d} {s['other obs/h']:11.0f}")


if __name__ == "__main__":
    main()
