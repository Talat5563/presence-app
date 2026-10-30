# Phase 3, stage 1 — results of fine-tuning on Samrómur Children

**Date:** 2026-10-30
**Plan:** [2026-10-28-phase3-fine-tuning.md](2026-10-28-phase3-fine-tuning.md), Tasks 5 and 6.
**Model:** `speaker_embedding_ft.onnx`, sha256 `e6cea18c8cd4377a7360a9b0ecdaad5030b519dfc5ad05a1b58fd450612c2812`.
WeSpeaker ResNet34-LM fine-tuned on Samrómur Children (Kaggle, 2 × T4, 15 epochs, 2.7 h, commit
6660e69). Epoch 7 was chosen on the 152-speaker training holdout. Same tensor contract as the
shipped model, so no Kotlin changed.
**Compared against:** the shipped WeSpeaker voxceleb_resnet34_LM, re-run through today's harness
on the same VAD probes. That control reproduces every Task 0 baseline exactly, so each difference
below comes from the model alone.

## The verdict

**Do not ship it.** Every CHILDES run is still FALL_BACK, in English and in Hebrew. The model
learned Samrómur and nothing that carries over to the CHILDES homes. This is the plan's second
outcome: "the model learned the benchmark, not the room."

## Samrómur Children: a large gain

Utterance against utterance; test scored once.

| Split | Model | EER (hard) | EER (random) | minDCF 0.01 (hard) |
|---|---|---:|---:|---:|
| dev (34 speakers) | shipped | 8.97% | 8.33% | 0.666 |
| | fine-tuned | **4.22%** | **4.00%** | **0.482** |
| test (625 speakers) | shipped | 11.26% | 9.05% | 0.876 |
| | fine-tuned | **4.23%** | **3.97%** | 0.847 |
| training holdout (152 speakers) | shipped | 13.61% | 10.24% | 0.794 |
| | fine-tuned | **6.70%** | **4.98%** | 0.808 |

EER more than halves everywhere. But minDCF at p_target = 0.01, which weights the low-false-alarm
region the product runs in, barely moves on test and gets slightly worse on the holdout. The
gain is in the middle of the score range, not where a parent would feel it. Holdout EER is within
0.3 points across epochs 7–15, so a later epoch would not change the picture.

## Task 5: the model runs correctly in our pipeline

- **Signature:** `feats [B, T, 80]` → `embs [B, 256]`, opset 14, the same as the shipped file.
- **Integration test:** `:shared:integrationTest` with the fine-tuned file in place of
  `models/speaker_embedding.onnx` passes 17 of 17 (OnnxModelSession 3, OnnxSpeakerEmbedder 7,
  SileroVad 7), with no change to the test.
- **Front-end parity:** the Kotlin harness and `training/embed_onnx.py` embed all 1,489 Samrómur
  dev files at cosine 1.000000 (minimum, 1st percentile and median). This was checked on every
  dev file, not just 20 clips.
- **Export:** the ONNX file and the PyTorch checkpoint agree at cosine 1.000000 on all 3,891
  holdout files.

So every number in this doc is the model's, not a front-end mismatch.

## CHILDES, through the product harness

Default engine settings, with VAD probes replayed (full MacWhinney runs live; replay and live runs
are identical). Shipped → fine-tuned.

| Corpus | Language | Detection / FA/h at threshold 2 | Best detection at ≤ 2 FA/h | Verdict |
|---|---|---|---|---|
| MacWhinney slice, Ross 4;00–4;05 | English | 5.2% / 1.14 → 3.3% / 0.98 | 5.2% → 3.3% | FALL_BACK |
| MacWhinney full, Ross 4;00–6;11 | English | 2.9% / 7.8 → 5.7% / 11.4 | 0% → 0% | FALL_BACK |
| Gleason, 22 families | English | 8.1% / 0.35 → 6.5% / 4.57 | 8.1% → 1.6% | FALL_BACK |
| BermanLong, 4 children | **Hebrew** | 7.0% / 6.8 → 1.4% / 0.75 | 0.9% → 1.4% | FALL_BACK |

The ITERATE bar is at least 50% detection at no more than 2 FA/h. No run comes within an order of
magnitude of it, with either model.

### Why: the embedding separates no better

Pooled over households with an enrolled target, using HouseholdMatcher's raw log-odds
(`tools/experiments/household_auc.py`). AUC 0.5 is chance.

| Corpus | Model | Child closest to own centroid | vs parent | vs sibling | vs stranger | vs all others |
|---|---|---:|---:|---:|---:|---:|
| MacWhinney slice | shipped | 27% | 0.568 | 0.533 | — | 0.566 |
| | fine-tuned | 27% | 0.576 | **0.498** | — | 0.571 |
| MacWhinney full | shipped | 47% | 0.701 | 0.413 | 0.516 | 0.595 |
| | fine-tuned | 49% | 0.665 | 0.395 | 0.381 | 0.565 |
| Gleason | shipped | 67% | 0.891 | — | 0.758 | 0.886 |
| | fine-tuned | 58% | **0.795** | — | 0.636 | 0.790 |
| BermanLong | shipped | 69% | 0.781 | — | 0.508 | 0.718 |
| | fine-tuned | 63% | 0.779 | — | 0.555 | 0.727 |

(Slice strangers: n = 3, omitted. BermanLong has 77 observations from other households and 3 from
a sibling; MacWhinney full has 3 from another household. All are too few to read.)

- **Child against sibling is still at chance** in the one household that has a sibling. This was
  the failure fine-tuning was meant to fix.
- **Gleason got worse.** Child against parent was the one thing the shipped model did well, and
  it falls from 0.89 to 0.80. False alarms rise from 3 to 39 in 8.5 h. The scores move toward
  each other: the share of the child's observations that count as evidence for the child (raw
  log-odds above 0) falls from 58% to 44%, while everyone else's rises from 6% to 10%.
- **BermanLong's lower false-alarm rate is not better separation.** Its AUCs are unchanged. Every
  score moved down together: the share above 0 falls from 50% to 35% for the child and from 17%
  to 7% for everyone else. Fewer events cross the threshold, for the child as much as for anyone.
- **Score normalization does not rescue it either.** `tools/experiments/normalization.py` on the
  slice reproduces the harness report exactly, then tries centering, Z-norm and WCCN. Ross
  against Mark stays between 0.50 and 0.53 under every variant, as it did with the shipped model.

## Reading this

Fine-tuning on Samrómur halved EER on Samrómur's own speakers. On CHILDES it made no difference,
or made things worse. The differences that matter:

- **Age:** Samrómur is ages 4–17 (its test split 6–16). CHILDES here is ages 1–6.
- **Speech style:** Samrómur is read sentences. CHILDES is spontaneous play.
- **Channel:** Samrómur was recorded on modern devices. CHILDES was recorded on 1980s home
  equipment.
- **Language:** Samrómur is Icelandic. CHILDES here is English and Hebrew.

CHILDES cannot say which of these matters. It also cannot say how a modern phone in the family's
home would behave. Our own recordings (Phase 1) can.

## Options from here (not yet decided)

1. **Choose a less specialised model from this run.** All 15 epoch checkpoints are kept locally.
   An early epoch, or a weight blend of the shipped and fine-tuned models, may keep the shipped
   model's robustness. To choose without testing on the same data, select on some Gleason
   families and report on the rest. Runs locally; about 25 minutes per candidate.
2. **Retrain with noise and reverberation augmentation.** This is the plan's prescribed next step.
   It needs one more Kaggle run, and a licence check on the noise and room-response data before
   any of it trains a shipped model.
3. **Weigh Phase 1 recordings as the deciding evidence**, especially if the real target child is
   school-age. Samrómur is closer to that case than CHILDES is.
4. Held back: ECAPA-TDNN, which the plan reserves for when neither benchmark moves, and the
   wake-phrase channel (Phase 7).

## Reproducing

The fine-tuned file goes into a models directory beside `silero_vad.onnx` and a `PROVENANCE.txt`,
named `speaker_embedding.onnx` (here `data/models-ft/`; the shipped set is copied to
`data/models-base/`).

```bash
./gradlew :harness:embedFiles --args="--list data/samromur/test.list --models models --model speaker_embedding_ft.onnx --out reports/phase3-ft/test-embeddings.csv"
python -I training/score_trials.py reports/phase3-ft/test-embeddings.csv data/samromur/trials_test.txt
python -I training/embed_onnx.py models/speaker_embedding_ft.onnx data/samromur/dev.list reports/phase3-ft/dev-embeddings-python.csv
python -I training/compare_embeddings.py reports/phase3-ft/dev-embeddings.csv reports/phase3-ft/dev-embeddings-python.csv
./gradlew :harness:run --args="--manifest data/childes-gleason --models data/models-ft --out reports/phase3-ft/childes-gleason \
  --embeddings reports/phase3-ft/childes-gleason/embeddings.csv --vad-probes reports/vad-probes/gleason"
python -I tools/experiments/household_auc.py reports/phase3-base/childes-gleason/embeddings.csv reports/phase3-ft/childes-gleason/embeddings.csv
python -I tools/experiments/normalization.py reports/phase3-ft/childes-slice/embeddings.csv data/childes-slice
```
