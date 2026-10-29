"""Scores WeSpeaker checkpoints on verification trials, on the GPU (Phase 3, Task 4).

Usage: python training/score_epochs.py --wespeaker DIR --config CONFIG --list LIST --trials TRIALS
                                       --out CSV [--device cuda] CHECKPOINT [CHECKPOINT ...]

Runs on Kaggle, beside the training run, to pick the epoch: every checkpoint embeds every file in
LIST (a file list or wav.scp) and is scored on TRIALS. Appends one row per checkpoint to CSV, and
writes each checkpoint's embeddings beside it as <checkpoint>-embeddings.csv, in the layout
score_trials.py and compare_embeddings.py read.

The features are embed_onnx.py's, which match the app's Kotlin front end exactly (see
training/README.md), and are computed once for all checkpoints. Weights load strictly: every
backbone tensor must be present, so a checkpoint that silently loads nothing cannot score as a
model.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed_onnx import features, read_listing  # noqa: E402
from score_trials import eer, min_dcf, trial_scores  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    for flag in ("--wespeaker", "--config", "--list", "--trials", "--out"):
        parser.add_argument(flag, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("checkpoints", nargs="+")
    args = parser.parse_args()

    sys.path.insert(0, args.wespeaker)
    from wespeaker.models.speaker_model import get_speaker_model

    config = yaml.safe_load(open(args.config, encoding="utf-8"))
    listing = Path(args.list)
    entries = read_listing(listing)
    print(f"computing features for {len(entries)} files ...", flush=True)
    feats = {}
    for entry in entries:
        path = Path(entry) if Path(entry).is_absolute() else listing.parent / entry
        feats[entry] = torch.from_numpy(features(path))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out.exists()
    with open(out, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        if new_file:
            writer.writerow(["checkpoint", "eer_hard", "eer_random", "min_dcf_hard", "target_trials", "missing"])
        for checkpoint in map(Path, args.checkpoints):
            model = get_speaker_model(config["model"])(**config["model_args"])
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict({k: v for k, v in state.items() if not k.startswith("projection.")}, strict=True)
            model.eval().to(args.device)

            embeddings = {}
            embeddings_csv = out.parent / f"{checkpoint.stem}-embeddings.csv"
            with torch.no_grad(), open(embeddings_csv, "w", encoding="utf-8", newline="") as e:
                rows = csv.writer(e, lineterminator="\n")
                rows.writerow(["path"] + [f"e{i}" for i in range(config["model_args"]["embed_dim"])])
                for entry, feat in feats.items():
                    output = model(feat.unsqueeze(0).to(args.device))
                    embedding = (output[-1] if isinstance(output, tuple) else output)[0].cpu().numpy()
                    embeddings[Path(entry).stem] = embedding.astype(np.float64)
                    rows.writerow([entry] + [f"{v:.6g}" for v in embedding])

            scores, missing = trial_scores(embeddings, args.trials)
            t = scores["target"]
            row = [checkpoint.name, eer(t, scores["hard"]), eer(t, scores["random"]),
                   min_dcf(t, scores["hard"]), len(t), missing]
            writer.writerow([row[0]] + [f"{v:.5f}" for v in row[1:4]] + row[4:])
            f.flush()
            print(f"{checkpoint.name}: EER hard {row[1]:.2%}, random {row[2]:.2%}, minDCF hard {row[3]:.3f}"
                  + (f" ({missing} trials missing an embedding)" if missing else ""), flush=True)


if __name__ == "__main__":
    main()
