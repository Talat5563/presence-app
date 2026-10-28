"""Converts a Samrómur Children split's FLAC to 16 kHz mono 16-bit WAV and writes a file list.

Usage: python -I training/prepare_samromur_eval.py data/samromur dev

Expects the split unpacked as DATA_DIR/wav/<split>/<speaker>/<audio_id>.flac, beside
DATA_DIR/metadata_<split>.tsv, as the corpus's dev.tar.gz / test.tar.gz and metadata unpack.
Writes DATA_DIR/<split>.list, one WAV path per line relative to DATA_DIR, for
`:harness:embedFiles`. Conversion goes through ffmpeg, so the harness's strict WAV reader never
has to resample.
"""
import csv
import subprocess
import sys
from pathlib import Path

data_dir, split = Path(sys.argv[1]), sys.argv[2]
rows = list(csv.DictReader(open(data_dir / f"metadata_{split}.tsv", encoding="utf-8"), delimiter="\t"))

listed = []
for i, row in enumerate(rows, 1):
    flac = data_dir / "wav" / split / row["speaker_id"] / f"{row['audio_id']}.flac"
    wav = flac.with_suffix(".wav")
    if not wav.exists():
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(flac), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav)],
            check=True,
        )
    listed.append(wav.relative_to(data_dir).as_posix())
    if i % 500 == 0:
        print(f"  ... {i}/{len(rows)} converted")

(data_dir / f"{split}.list").write_text("\n".join(listed) + "\n", encoding="utf-8")
print(f"{split}: {len(listed)} files listed in {data_dir / f'{split}.list'}")
