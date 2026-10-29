"""Prepares Samrómur Children's training split for WeSpeaker fine-tuning (Phase 3, Task 2).

Usage: python -I training/prepare_samromur.py DATA_DIR [--fetch]

  --fetch        download metadata_train.tsv and the three train tars (6.6 GB) at REVISION,
                 check their SHA-256, and unpack them to DATA_DIR/wav/train/<speaker>/<id>.flac.
                 Without it the script uses whatever is already in DATA_DIR, and with no audio
                 there writes the lists anyway; with it, any listed file missing is an error.
  --fetch-test   likewise for the test split (96 MB), and write DATA_DIR/test.scp. Only for
                 checking a scorer against the published baseline; never for selection.

Writes:
  DATA_DIR/wespeaker/{train,holdout}/wav.scp and utt2spk   WeSpeaker's list format, absolute paths
  DATA_DIR/metadata_holdout.tsv                            for make_trials.py DATA_DIR holdout

The audio is already 16 kHz mono 16-bit FLAC, which WeSpeaker reads directly, so nothing is
converted: the lists point at the FLACs, and every one is checked for format when soundfile is
installed. Without the audio (metadata only) the lists and counts are still written.

Speaker IDs are recording-session IDs (dataset card), and nothing in the metadata links two
sessions of one child. Linking them by voice would mean identifying speakers, which the
corpus's terms forbid, so a child who recorded twice stays two classes in training and can form
a "non-target" trial with itself. That pessimises EER slightly; it is reported, not corrected.

Speakers with fewer than 2 utterances are dropped from training: AAM-softmax needs more than
one example per class. HOLDOUT_SIZE speakers are held out from training for model selection,
equally per gender and age band (Samrómur's own dev split is one speaker in two-thirds).
"""
import csv
import hashlib
import random
import sys
import tarfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

REPO = "https://huggingface.co/datasets/language-and-voice-lab/samromur_children/resolve"
REVISION = "6bb44ee982da842051e42e1bbe8f81c3ffd04daf"
TRAIN_DOWNLOADS = {
    "corpus/files/metadata_train.tsv": "fb0462f85ab82adf2408dd1030e0cfde122e2219506c4d00e32c7cd220b986f5",
    "corpus/speech/train/train_part_01.tar.gz": "a34abbaa2bbe125bac1389673089f26ce8c4821c0e7e8042c1e360ce932e75cd",
    "corpus/speech/train/train_part_02.tar.gz": "4a1d3a45c6f58999ebf04ad2a41a63e95257bac47908c2a53d9abf769ef31916",
    "corpus/speech/train/train_part_03.tar.gz": "0776b4c4bf25e0e46eada3dd50623ec330e15a8f86d22f399acb7301cd52bdf7",
}
TEST_DOWNLOADS = {
    "corpus/files/metadata_test.tsv": "f8e17de89bb6456151634576bf6858dce9bfef7120cff622db423297ff36a355",
    "corpus/speech/test.tar.gz": "8c25ba87b02ed93834f140597c087f9457fee304b96eca39af2f49ab0b5350a5",
}

HOLDOUT_SIZE = 152  # 19 per stratum: 2 genders x 4 age bands
AGE_BANDS = ((4, 8), (9, 11), (12, 14), (15, 17))
# Holdout candidates need enough utterances for target trials, and are capped so that holding
# one out does not take a large share of the training audio with it.
HOLDOUT_UTTERANCES = (5, 100)
SEED = 20261029


def unpack(archive, root, split):
    """Unpacks archive's speaker folders into root/split/.

    Each archive has one top-level folder (train_part_01/ ... train_part_03/, test/), which is
    replaced by the split's name, so every split lands at wav/<split>/<speaker>/<id>.flac.
    """
    root = root.resolve()
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        for member in members:
            member.name = Path(split, *Path(member.name).parts[1:]).as_posix()
            if not (root / member.name).resolve().is_relative_to(root) or not (member.isfile() or member.isdir()):
                sys.exit(f"{archive}: refusing unsafe member {member.name}")
        tar.extractall(root, members=members)


def fetch(data_dir, downloads, split):
    for path, sha256 in downloads.items():
        target = data_dir / Path(path).name
        if not target.exists():
            print(f"downloading {path} ...", flush=True)
            urllib.request.urlretrieve(f"{REPO}/{REVISION}/{path}", target)
        digest = hashlib.sha256()
        with open(target, "rb") as f:
            for block in iter(lambda: f.read(1 << 22), b""):
                digest.update(block)
        if digest.hexdigest() != sha256:
            sys.exit(f"{target}: SHA-256 {digest.hexdigest()} does not match the pinned {sha256}")
        if target.suffixes[-2:] == [".tar", ".gz"]:
            print(f"unpacking {target.name} ...", flush=True)
            unpack(target, data_dir / "wav", split)


def age_band(age):
    if not age.isdigit():  # "13-17" spans bands
        return None
    return next((band for band in AGE_BANDS if band[0] <= int(age) <= band[1]), None)


def summarise(name, rows):
    per_speaker = Counter(r["speaker_id"] for r in rows)
    counts = sorted(per_speaker.values())
    hours = sum(float(r["duration"]) for r in rows) / 3600
    print(f"{name}: {len(per_speaker)} speakers, {len(rows)} utterances, {hours:.1f} h; "
          f"utterances per speaker min {counts[0]}, median {counts[len(counts) // 2]}, max {counts[-1]}")


def main():
    data_dir = Path(sys.argv[1])
    fetched = "--fetch" in sys.argv[2:]
    if fetched:
        fetch(data_dir, TRAIN_DOWNLOADS, "train")
    if "--fetch-test" in sys.argv[2:]:
        fetch(data_dir, TEST_DOWNLOADS, "test")
        with open(data_dir / "metadata_test.tsv", encoding="utf-8") as f, \
                open(data_dir / "test.scp", "w", encoding="utf-8", newline="\n") as scp:
            for r in csv.DictReader(f, delimiter="\t"):
                flac = (data_dir / "wav" / "test" / r["speaker_id"] / f"{r['audio_id']}.flac").resolve()
                scp.write(f"{r['audio_id']} {flac.as_posix()}\n")

    splits = {}
    for split in ("train", "dev", "test"):
        path = data_dir / f"metadata_{split}.tsv"
        if path.exists():
            splits[split] = list(csv.DictReader(open(path, encoding="utf-8"), delimiter="\t"))
    rows = splits["train"]
    speaker_info = {r["speaker_id"]: (r["gender"], r["age"]) for r in rows}

    print("== Counts")
    for split, split_rows in splits.items():
        summarise(split, split_rows)
    seconds_by_age = Counter()
    speakers_by_age = defaultdict(set)
    for r in rows:
        seconds_by_age[r["age"]] += float(r["duration"])
        speakers_by_age[r["age"]].add(r["speaker_id"])
    print("train by age: " + ", ".join(
        f"{age} {len(speakers_by_age[age])} spk/{seconds_by_age[age] / 3600:.1f} h"
        for age in sorted(seconds_by_age, key=lambda a: (len(a), a))))
    print("train by gender: " + ", ".join(
        f"{g} {n}" for g, n in Counter(g for g, _ in speaker_info.values()).most_common()))

    print("== Session IDs")
    ids = {split: {r["speaker_id"] for r in split_rows} for split, split_rows in splits.items()}
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        if a in ids and b in ids:
            print(f"{a} and {b} share {len(ids[a] & ids[b])} session IDs {sorted(ids[a] & ids[b])[:5]}")

    by_speaker = defaultdict(list)
    for r in rows:
        by_speaker[r["speaker_id"]].append(r)
    dropped = sorted(s for s, utts in by_speaker.items() if len(utts) < 2)

    rng = random.Random(SEED)
    strata = defaultdict(list)
    for speaker, utts in sorted(by_speaker.items()):
        gender, age = speaker_info[speaker]
        band = age_band(age)
        if gender in ("female", "male") and band and HOLDOUT_UTTERANCES[0] <= len(utts) <= HOLDOUT_UTTERANCES[1]:
            strata[(gender, band)].append(speaker)
    per_stratum = HOLDOUT_SIZE // (2 * len(AGE_BANDS))
    holdout = set()
    print("== Holdout")
    for key in sorted(strata):
        chosen = rng.sample(strata[key], min(per_stratum, len(strata[key])))
        holdout.update(chosen)
        print(f"{key[0]:6} ages {key[1][0]:2}-{key[1][1]:2}: {len(chosen)} of {len(strata[key])} candidates")
    train = [s for s in sorted(by_speaker) if s not in holdout and s not in dropped]

    out = data_dir / "wespeaker"
    missing, bad_format = [], []
    try:
        import soundfile
    except ImportError:
        soundfile = None
    for name, speakers in (("train", train), ("holdout", sorted(holdout))):
        (out / name).mkdir(parents=True, exist_ok=True)
        with open(out / name / "wav.scp", "w", encoding="utf-8", newline="\n") as scp, \
                open(out / name / "utt2spk", "w", encoding="utf-8", newline="\n") as utt2spk:
            for speaker in speakers:
                for r in sorted(by_speaker[speaker], key=lambda r: r["audio_id"]):
                    flac = (data_dir / "wav" / "train" / speaker / f"{r['audio_id']}.flac").resolve()
                    scp.write(f"{r['audio_id']} {flac.as_posix()}\n")
                    utt2spk.write(f"{r['audio_id']} {speaker}\n")
                    if not flac.exists():
                        missing.append(flac)
                    elif soundfile:
                        info = soundfile.info(str(flac))
                        if (info.samplerate, info.channels) != (16000, 1):
                            bad_format.append(f"{flac}: {info.samplerate} Hz, {info.channels} channels")
        summarise(name, [r for s in speakers for r in by_speaker[s]])
    print(f"dropped {len(dropped)} single-utterance speakers from training")

    with open(data_dir / "metadata_holdout.tsv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(r for r in rows if r["speaker_id"] in holdout)

    listed = sum(len(by_speaker[s]) for s in train) + sum(len(by_speaker[s]) for s in holdout)
    if len(missing) == listed and not fetched:
        print(f"audio not present under {data_dir / 'wav' / 'train'}: lists written for the paths it will have")
    elif missing:
        sys.exit(f"{len(missing)} of {listed} listed files are missing, e.g. {missing[0]}")
    elif soundfile is None:
        print(f"all {listed} files present (install soundfile to check their format)")
    elif bad_format:
        sys.exit(f"{len(bad_format)} files are not 16 kHz mono, e.g. {bad_format[0]}")
    else:
        print(f"all {listed} files present, 16 kHz mono")


if __name__ == "__main__":
    main()
