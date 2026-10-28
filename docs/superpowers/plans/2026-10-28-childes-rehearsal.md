# First real-audio rehearsal — CHILDES MacWhinney

**Date:** 2026-10-28
**Data:** CHILDES MacWhinney (CC BY-NC-SA), one household: Ross (target), his brother Mark, and
both parents, recorded at home on 1980s equipment. Imported with `:harness:importChildes` as
speaker episodes (see the README). **A rehearsal, not the Phase 2 number:** one household, and
audio far from a modern phone microphone.
**Models:** as in `models/PROVENANCE.txt` — Silero VAD v5, WeSpeaker ResNet34-LM (adult-trained,
not fine-tuned).

## The verdict

FALL_BACK at every threshold.

| Run | Episodes | Negative hours | Best detection | FA/hour there |
|---|---:|---:|---:|---:|
| Ross 4;00–6;11, 157 sessions | 1,997 | 44.8 | 2.9% (threshold 2) | 7.8 |
| Ross 4;00–4;05, 35 sessions | 367 | 6.1 | 5.2% (threshold 2) | 1.1 |

The second run sits next to the enrollment sessions in age, so age drift does not explain it.

## Where it fails

From `observations.csv`, the per-observation score dump the harness now writes.

**The pipeline works on adults, imperfectly.** Each speaker's speech, by the enrolled voice it
scored closest to (the target, Ross, is excluded from this column):

| Speaker | Closest: father | Closest: Mark | Closest: mother |
|---|---:|---:|---:|
| Father | **69%** | 27% | 4% |
| Mother | 21% | 14% | **65%** |
| Mark | 46% | **35%** | 18% |

Father and mother separate clearly (each rarely scores closest to the other), so there is no gross
front-end bug of the kind the VAD had. That adults top out near 65–69% among three candidates says
the recording channel itself costs a lot.

**The children do not separate at all.** Ross scores closest to his own centroid in only 27% of
observations; his father's centroid wins 50%. Separation of Ross from everyone else, as AUC on the
raw LLR, is 0.57; Ross against Mark is **0.53, chance**. Every cosine — anyone against anyone —
sits between about 0.27 and 0.53.

**The gates are not the cause.** The cohort z gate removed none of Ross's positive observations;
the absolute `minTargetSimilarity` gate removed 8%. Retuning them cannot help when the raw
evidence already points the wrong way.

**A secondary, structural limit.** 65 of 298 episodes yielded one embedding window, which caps
their evidence at `maxAbsLlr` (4) — below the default `fireThreshold` of 6, so they can never be
caught within 10 s at that setting. Small next to the above, but real for short calls.

## What it means

This is the failure spec §4.1 predicted for an adult-trained model on young children's speech,
measured on our own pipeline for the first time: speaker identity among children, and between a
child and a parent, is not in these embeddings. Threshold and gate tuning (Phase 4) cannot recover
it. The lever is the one the roadmap already names — **Phase 3, fine-tuning on children's speech**
— with the wake-phrase channel (Phase 7) as the fallback if that also fails.

Two cheap things are still worth trying before or alongside Phase 3, since both are engineering
rather than research:

- **Score normalization.** The father's centroid attracts most of the children's speech — a
  classic hub. Centering embeddings on the household mean, or adaptive score normalization, is
  standard practice and may recover some adult-vs-child separation. It will not fix child-vs-child.
- **Better audio.** Part of the loss is the 1980s channel. Gleason (no siblings in most families)
  and BermanLong (Hebrew) are still to be measured, and nothing substitutes for recordings made on
  a phone in a real home (Phase 1).
