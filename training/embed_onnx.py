"""Embeds audio files with an ONNX speaker model, through WeSpeaker's inference front end.

Usage: python -I training/embed_onnx.py MODEL.onnx LIST OUT.csv [--root DIR]

LIST is either a file list (one path per line, as `:harness:embedFiles` takes) or a WeSpeaker
wav.scp (`key path`). Relative paths resolve against --root (default: LIST's directory). OUT.csv
has the same layout as embedFiles writes, so score_trials.py reads either.

This is for Kaggle, where the Kotlin harness does not run: it scores each fine-tuning epoch on
the holdout trials. The front end is WeSpeaker's at inference: Kaldi fbank with 80 mel bins,
25 ms Hamming window, 10 ms shift, int16-scaled samples, no dither, then per-utterance mean
normalization. The Kotlin FbankExtractor implements the same thing. compare_embeddings.py
measures how closely the two agree on real audio.
"""
import csv
import sys
from pathlib import Path

import onnxruntime
import soundfile
import torch
import torchaudio.compliance.kaldi as kaldi


def features(path):
    samples, rate = soundfile.read(str(path), dtype="float32")
    if rate != 16000 or samples.ndim != 1:
        raise ValueError(f"{path}: expected 16 kHz mono, got {rate} Hz with shape {samples.shape}")
    waveform = torch.from_numpy(samples).unsqueeze(0) * (1 << 15)
    feats = kaldi.fbank(waveform, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0,
                        sample_frequency=16000, window_type="hamming", use_energy=False)
    return (feats - feats.mean(dim=0)).numpy()


def read_listing(listing):
    """Audio paths from a file list or a wav.scp, in order."""
    return [line.split()[-1] for line in listing.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    args = sys.argv[1:]
    root = None
    if "--root" in args:
        i = args.index("--root")
        root = Path(args[i + 1])
        del args[i:i + 2]
    model, listing, out = Path(args[0]), Path(args[1]), Path(args[2])
    entries = read_listing(listing)
    root = root or listing.parent

    session = onnxruntime.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["path"] + [f"e{i}" for i in range(256)])
        for n, entry in enumerate(entries, 1):
            path = Path(entry) if Path(entry).is_absolute() else root / entry
            embedding = session.run(["embs"], {"feats": features(path)[None]})[0][0]
            writer.writerow([entry] + [f"{v:.6g}" for v in embedding])
            if n % 500 == 0:
                print(f"  ... {n}/{len(entries)} embedded", flush=True)
    print(f"{len(entries)} embeddings -> {out}")


if __name__ == "__main__":
    main()
