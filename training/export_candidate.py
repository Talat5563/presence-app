"""Exports a blend of two WeSpeaker checkpoints as an app-ready ONNX speaker model (Phase 3 sweep).

Usage: python -I training/export_candidate.py --wespeaker DIR --config CONFIG --base BASE
                                               --tuned TUNED --alpha A --out ONNX

Every backbone tensor becomes (1 - A) * BASE + A * TUNED. The classification head is TUNED's, and
it is never exported. A = 1 exports TUNED unchanged, which is how an early epoch is exported.
Blending a fine-tuned model back towards its starting point (WiSE-FT, Wortsman et al. 2022) often
keeps the starting model's robustness outside the fine-tuning data while keeping part of the gain
on it.

The export is WeSpeaker's own export_onnx.py, run unchanged, so the file has the shipped model's
tensor contract. It runs without WeSpeaker's optional dependencies: the package __init__ imports
its CLI (silero-vad, umap, hdbscan) and the model registry imports every model family (s3prl for
the SSL ones). An empty package and a ResNet-only registry are registered in their place.
"""
import argparse
import runpy
import sys
import tempfile
import types
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    for flag in ("--wespeaker", "--config", "--base", "--tuned", "--out"):
        parser.add_argument(flag, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        sys.exit("--alpha must be between 0 and 1")

    base = torch.load(args.base, map_location="cpu", weights_only=True)
    tuned = torch.load(args.tuned, map_location="cpu", weights_only=True)
    backbone = [k for k in tuned if not k.startswith("projection.")]
    missing = [k for k in backbone if k not in base or base[k].shape != tuned[k].shape]
    if missing:
        sys.exit(f"BASE lacks or reshapes {len(missing)} backbone tensors, e.g. {missing[:3]}")
    blended = dict(tuned)
    for k in backbone:
        blended[k] = (1 - args.alpha) * base[k] + args.alpha * tuned[k]

    src = Path(args.wespeaker).resolve()
    package = types.ModuleType("wespeaker")
    package.__path__ = [str(src / "wespeaker")]
    sys.modules["wespeaker"] = package
    sys.path.insert(0, str(src))
    import wespeaker.models.resnet as resnet

    def get_speaker_model(name):
        if not name.startswith("ResNet"):
            sys.exit(f"only ResNet models are supported here, not {name}")
        return getattr(resnet, name)

    registry = types.ModuleType("wespeaker.models.speaker_model")
    registry.get_speaker_model = get_speaker_model
    sys.modules[registry.__name__] = registry

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp) / "blended.pt"
        torch.save(blended, checkpoint)
        sys.argv = ["export_onnx.py", "--config", args.config, "--checkpoint", str(checkpoint),
                    "--output_model", args.out]
        runpy.run_path(str(src / "wespeaker" / "bin" / "export_onnx.py"), run_name="__main__")
    print(f"alpha {args.alpha}: {args.out}")


if __name__ == "__main__":
    main()
