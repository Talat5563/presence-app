"""Builds speaker-verification trials for a Samrómur Children split.

Usage: python -I training/make_trials.py data/samromur test

Writes DATA_DIR/trials_<split>.txt with lines `utt_a utt_b label kind`, where label is
`target` or `nontarget`, and kind says which non-target pool a pair came from:

- `hard`: same gender (when known) and at most a year apart in age. The closest the corpus gets
  to telling siblings apart, which is what the product actually faces.
- `random`: any other speaker. Comparable with published EERs, and much easier.

Target pairs are every pair of a speaker's utterances, capped per speaker so a talkative child
cannot dominate. Seeded, so the same trials come out every time.
"""
import csv
import itertools
import random
import sys
from collections import defaultdict
from pathlib import Path

MAX_TARGET_PAIRS_PER_SPEAKER = 20
NONTARGET_PER_TARGET = 3

data_dir, split = Path(sys.argv[1]), sys.argv[2]
rng = random.Random(20261028)

rows = list(csv.DictReader(open(data_dir / f"metadata_{split}.tsv", encoding="utf-8"), delimiter="\t"))
by_speaker = defaultdict(list)
info = {}
for row in rows:
    by_speaker[row["speaker_id"]].append(row["audio_id"])
    gender = row["gender"] if row["gender"] in ("female", "male") else None
    age = int(row["age"]) if row["age"].isdigit() else None
    info[row["speaker_id"]] = (gender, age)

targets = []
for speaker, utts in sorted(by_speaker.items()):
    pairs = list(itertools.combinations(sorted(utts), 2))
    rng.shuffle(pairs)
    targets += pairs[:MAX_TARGET_PAIRS_PER_SPEAKER]


def similar(a, b):
    (ga, aa), (gb, ab) = info[a], info[b]
    same_gender = ga is None or gb is None or ga == gb
    close_age = aa is not None and ab is not None and abs(aa - ab) <= 1
    return same_gender and close_age


speakers = sorted(by_speaker)
utt_speaker = {u: s for s, us in by_speaker.items() for u in us}
nontargets = {"hard": [], "random": []}
wanted = len(targets) * NONTARGET_PER_TARGET
for kind in ("hard", "random"):
    attempts = 0
    while len(nontargets[kind]) < wanted and attempts < wanted * 200:
        attempts += 1
        a, b = rng.sample(speakers, 2)
        if kind == "hard" and not similar(a, b):
            continue
        nontargets[kind].append((rng.choice(by_speaker[a]), rng.choice(by_speaker[b])))

out = data_dir / f"trials_{split}.txt"
with open(out, "w", encoding="utf-8") as f:
    for a, b in targets:
        f.write(f"{a} {b} target both\n")
    for kind, pairs in nontargets.items():
        for a, b in pairs:
            f.write(f"{a} {b} nontarget {kind}\n")

multi = sum(1 for u in by_speaker.values() if len(u) > 1)
print(f"{split}: {len(targets)} target pairs from {multi} speakers with 2+ utterances; "
      f"{len(nontargets['hard'])} hard and {len(nontargets['random'])} random non-target pairs -> {out}")
