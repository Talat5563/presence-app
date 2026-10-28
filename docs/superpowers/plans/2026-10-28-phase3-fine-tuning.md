# Phase 3 — Fine-tuning the speaker model on children's speech

**Date:** 2026-10-28
**Status:** Planned. Stage 1 is ready to start; stage 2 is gated on a licence check.
**Why now:** the first real-audio rehearsal ([2026-10-28-childes-rehearsal.md](2026-10-28-childes-rehearsal.md))
showed the adult-trained model cannot tell young children apart. Ross against his brother scores
AUC 0.53, which is chance. Neither gate tuning nor score normalization recovers it. Spec §4.1
names fine-tuning (rung L3) as the single largest lever, worth roughly an order of magnitude in EER.

**Goal:** produce a fine-tuned `speaker_embedding.onnx` that is a drop-in replacement for today's
model, with the same input, output and tensor names, so not one line of Kotlin changes. Then
measure it against the current model on the same data with the same harness.

**Approach:** offline, in Python, outside the Gradle build. Start from the WeSpeaker ResNet34-LM
checkpoint that today's ONNX file was exported from. Replace its speaker-classification head and
fine-tune on Samrómur Children with WeSpeaker's own large-margin recipe. Export ONNX, then prove
the Kotlin pipeline produces the same embeddings Python does before trusting any measurement.

**Tech stack:** Python 3.10+, WeSpeaker (pinned commit), PyTorch, torchaudio, onnxruntime, numpy;
Hugging Face `datasets` to fetch Samrómur. Training on a Kaggle notebook GPU. Measurement on this
machine with the existing Kotlin harness.

---

## Decisions taken (2026-10-28)

| Decision | Choice | Why |
|---|---|---|
| Where to train | **Kaggle free GPU** (T4 or P100) | This machine has no NVIDIA GPU. Kaggle is free, has internet access for Hugging Face, and a ResNet34 fine-tune on 127 h should fit within one or two sessions. |
| Training data, stage 1 | **Samrómur Children only** | CC BY 4.0 and ungated. The Hebrew path (ivrit.ai plus ChildAugment) waits until ivrit.ai's terms are confirmed. |
| Architecture | **Keep ResNet34-LM** | A true drop-in replacement with 80-bin input and 256-d output. Switching to ECAPA-TDNN (the published 2.9% EER) would change two variables at once. It stays an option if stage 1 falls short. |

## Global constraints

- **Licences decide what may train a shipped model.** The repository is Apache-2.0 and public.
  - **Allowed in training:** Samrómur Children (CC BY 4.0) and the WeSpeaker VoxCeleb checkpoint
    (CC BY 4.0, following VoxCeleb). Both require attribution, in the README and in the app's
    about screen.
  - **Evaluation only:** CHILDES and MyST (CC BY-NC-SA). They must **never** enter training, or the
    resulting weights inherit non-commercial share-alike terms.
  - **Your own family recordings:** only if the Phase 1 consent form covers training.
- **No audio and no weights in git.** `.gitignore` already blocks audio formats, `data/`,
  `reports/` and `*.onnx`. Trained models ship as GitHub Release assets, and `tools/fetch-models.sh`
  downloads them.
- **Samrómur's condition:** do not attempt to identify any speaker. Nothing in this plan does.
- **The front end is part of the model.** Training features must be the features the Kotlin
  `FbankExtractor` computes: 80 mel bins, 25 ms window, 10 ms shift, Kaldi-compatible, per-utterance
  mean normalization. Training may add dither; inference never does. The VAD bug showed what an
  unverified front-end assumption costs, so Task 5 proves parity with real audio rather than
  assuming it.
- **Tensor contract:** input `feats [B, T, 80]`, output `embs [B, 256]`. WeSpeaker's
  `export_onnx.py` uses exactly these names; Task 5 verifies the exported file anyway.
- **Choose on dev, measure on test once.** Pick the checkpoint and settings on Samrómur dev and
  on CHILDES dev households, then measure Samrómur test and any held-out households a single time.
- **Pin everything.** Record the WeSpeaker commit, every package version, the Samrómur dataset
  revision and the Kaggle accelerator type, so the result can be reproduced or explained.

## Layout

```
training/                         Python, outside the Gradle build
  README.md                       how to run each step, locally and on Kaggle
  requirements.txt                pinned versions
  prepare_samromur.py             HF dataset → 16 kHz wav + WeSpeaker lists (wav.scp, utt2spk)
  make_trials.py                  target/non-target verification trials for dev and test
  conf/samromur_resnet34_ft.yaml  fine-tuning config, derived from WeSpeaker's resnet_lm.yaml
  kaggle_finetune.ipynb           the GPU run: install, fetch, train, average, export
  score_trials.py                 EER and minDCF for any ONNX model on a trials list
  parity_check.py                 Python-side embeddings of the same files the Kotlin test embeds
```

---

## Tasks

### Task 0: Baselines first, on everything we will compare

> **Done 2026-10-28.** BermanLong (Hebrew) also falls back: it separates child from parents
> (AUC 0.78) but not from other children (0.51). The results are in
> [training/README.md](../../../training/README.md). Three findings change the rest of this plan:
>
> 1. **Samrómur test EER is 11.3% (hard) and 9.1% (random) already.** The adult model is not
>    hopeless on clean read speech from older children. Gains on Samrómur will overstate gains in
>    a home, so CHILDES and Phase 1 recordings stay the deciding evidence.
> 2. **Samrómur dev is unusable for selection:** 34 speakers, one of them holding 1,000 of 1,489
>    utterances. Task 4 instead holds out about 150 speakers from the training split as its dev set.
> 3. **Gleason fails before the embedding.** Silero rates only 13% of 2–5-year-olds' episode audio
>    as speech, against 36% for their parents. Seven children could not be enrolled at all, and
>    the 0.35 absolute gate removes a third of the remaining child evidence. Speaker separation
>    there is fine (AUC 0.89). **Fine-tuning the speaker model does not touch this.** It needs VAD
>    and gate calibration for young children's speech, which is Phase 4 work brought forward. It
>    can proceed in parallel with this plan, since it changes the engine and not the model.

Fine-tuning is only worth something relative to a number measured the same way.

- [ ] Score the **current** model on Samrómur dev and test trials (Task 3 builds them): EER, and
      minDCF at p_target = 0.01. This runs on CPU here; the two splits total about 3.7 h of audio.
- [ ] Record the CHILDES harness baselines with `--embeddings`, so later models can be compared
      offline:
  - MacWhinney slice (Ross 4;00–4;05): already measured. AUC 0.566 against everyone, 0.533
    against his brother; best point 5.7% detection at 1.1 FA/h.
  - Full MacWhinney: already measured, FALL_BACK.
  - **Gleason** (22 families, mostly without siblings): media complete, so import and measure it.
    This is the child-versus-parent case.
  - **BermanLong** (Hebrew, 4 children): measure once the download finishes. Today it is the only
    Hebrew evidence we have.
- [ ] Write the baselines into a results table in `training/README.md`.

### Task 1: Prove the starting point is what ships today

- [ ] Download the WeSpeaker `voxceleb_resnet34_LM` PyTorch checkpoint (`avg_model.pt` and
      `config.yaml`). Record its URL and SHA-256.
- [ ] Export it to ONNX with WeSpeaker's `export_onnx.py` and embed a few real CHILDES clips with
      both the exported file and `models/speaker_embedding.onnx`. Every cosine must be at least
      0.999. If not, the published ONNX was exported from a different checkpoint; stop and find
      the right one before fine-tuning anything.

### Task 2: Prepare Samrómur Children

- [ ] `prepare_samromur.py`: load the dataset from Hugging Face at a pinned revision. Write
      16-bit, 16 kHz mono WAV per utterance, plus WeSpeaker's `wav.scp` and `utt2spk` for each split.
- [ ] Print and record the counts: speakers, utterances and hours per split, the age histogram,
      and utterances per speaker.
- [ ] **Speaker IDs are session IDs** (dataset card), so a child who recorded twice appears as two
      "speakers". In training that is mild label noise. In trials it can make a "non-target" pair
      the same child. Count how often a session ID repeats across splits. If the card's metadata
      lets sessions be linked, deduplicate; otherwise record it as a known bias that pessimises EER.
- [ ] Drop speakers with fewer than 2 utterances from training, since AAM-softmax needs more than
      one example per class.

### Task 3: Verification trials

- [ ] `make_trials.py`: for dev and test separately, build balanced target and non-target pairs
      from held-out speakers only, seeded and written to `trials_{dev,test}.txt`. Non-target pairs
      are stratified by gender and age band, because a 5-year-old against a 15-year-old is too easy
      to tell us anything.
- [ ] `score_trials.py`: embed with a given ONNX file through the **same front end as Kotlin**
      (parity confirmed in Task 5), then compute cosine scores, EER and minDCF.

### Task 4: Fine-tune on Kaggle

- [ ] `conf/samromur_resnet34_ft.yaml`, derived from WeSpeaker's `resnet_lm.yaml` (which is itself
      a fine-tuning recipe):
  - `model_init` points at the Task 1 checkpoint.
  - The ArcMargin projection is resized to the number of Samrómur training speakers and starts
    fresh. **Verify** that WeSpeaker's checkpoint loading skips the mismatched projection weights
    rather than failing; if it fails, load the backbone weights explicitly.
  - Learning rate 1e-4 decaying to 2.5e-5, margin 0.5, SGD with momentum 0.9 — the recipe's own
    values as the starting point.
  - Chunk length: the recipe's 600 frames (6 s) is longer than most Samrómur read sentences.
    Start at 200 frames (2 s) and keep the utterance-length filter consistent with it.
  - Augmentation: speed perturbation on. Noise and reverberation (MUSAN, RIR) are on in the
    recipe, but those corpora must be fetched too, so start without them and add them as a
    variant if dev EER plateaus.
- [ ] `kaggle_finetune.ipynb`: install the pinned WeSpeaker, run `prepare_samromur.py`, train with
      a checkpoint every epoch to `/kaggle/working` (sessions are time-limited, so the run must
      resume), average the last checkpoints, and export ONNX.
- [ ] Select the epoch count and chunk length on the EER of **speakers held out from the training
      split** (about 150, stratified by age and gender), not on Samrómur's own dev split (too
      lopsided, see Task 0), not on test, and not on CHILDES.
- [ ] Download the exported `.onnx`, the config and the training log. Record the accelerator,
      wall-clock time and final dev EER.

### Task 5: Verify the exported model in our pipeline

- [ ] Check the signature: input `feats [B, T, 80]`, output `embs [B, 256]`. Run the existing
      `:shared:integrationTest` with the new file in place of `models/speaker_embedding.onnx`;
      the shape assertions must pass unchanged.
- [ ] **Front-end parity.** Embed the same 20 real clips through the Kotlin `OnnxSpeakerEmbedder`
      (a model-tagged test that writes its embeddings) and through `parity_check.py`'s Python
      features. Per-clip cosine must be at least 0.999. Anything lower means the model is being fed
      different features than it was trained on, and every number downstream would be wrong.

### Task 6: Measure

- [ ] Samrómur dev and test: EER and minDCF, fine-tuned against baseline. Test once only.
- [ ] Re-run the CHILDES harness with the new model on the same manifests as Task 0: MacWhinney
      slice and full, Gleason, BermanLong. Run the score analysis
      (`tools/experiments/normalization.py`, and the per-speaker confusion) on each.
- [ ] Write `docs/superpowers/plans/<date>-phase3-results.md` with the before/after table and the
      verdicts. Report English and Hebrew separately (spec §4.3).

**How to read the outcome.** The product decision is the harness verdict (spec §4.3), not EER.
The diagnostics say *why*:

- **Child-versus-sibling AUC moves well above chance, and the verdict reaches ITERATE or better:**
  proceed to Phase 4 calibration with the new model, then stage 2 for Hebrew.
- **Samrómur EER falls sharply but CHILDES barely moves:** the model learned the benchmark, not
  the room. The likely causes are domain (read Icelandic sentences by ages 4–17, against
  spontaneous toddler speech on 1980s tape) or channel. Add noise and reverberation augmentation,
  and weigh how much the CHILDES channel says about a modern phone before concluding anything.
  Our own recordings (Phase 1) become the deciding evidence.
- **Neither moves:** try ECAPA-TDNN, the architecture behind the published 2.9%, before
  concluding that text-independent detection is not viable and turning to the wake-phrase
  channel (Phase 7).

### Task 7: Ship the model

- [ ] Publish the `.onnx` as a GitHub Release asset on `Talat5563/presence-app`, with a model card:
      base checkpoint, training data, licences, recipe, the dev and test numbers, and the limits
      (Icelandic read speech, ages 4–17).
- [ ] Point `tools/fetch-models.sh` at the release and update `models/PROVENANCE.txt` in the
      repository's documentation.
- [ ] Add Samrómur Children and VoxCeleb/WeSpeaker attribution to the README's Attribution
      section. The Samrómur text is already drafted there. Add both to the about-screen strings
      for Phase 6.

---

## Stage 2 — Hebrew (gated)

Not broken into tasks until its gate clears:

1. **Gate: ivrit.ai's terms.** Accept the dataset terms on Hugging Face, read the licence of each
   subset that would be used, and confirm it permits training a model that is redistributed under
   open-source terms. If unclear, ask ivrit.ai. Spec §10 open question 8.
2. **ChildAugment** (arXiv 2402.15214): warp adult Hebrew speech toward child vocal-tract
   characteristics. Implement and test the warping on its own first, by listening and by checking
   formant shifts.
3. Fine-tune on Samrómur plus warped ivrit.ai, and compare against stage 1 on BermanLong and any
   Hebrew household recordings. Hebrew must clear the bar on its own (spec §4.3).

## Risks

| Risk | Consequence | Mitigation |
|---|---|---|
| Kaggle session or weekly GPU limits | A run is cut off midway | Checkpoint every epoch and resume; keep runs to one or two sessions |
| Front-end mismatch between training and Kotlin | Every measurement silently wrong | Task 5 parity check on real audio, ≥ 0.999 cosine |
| Pretrained checkpoint differs from the shipped ONNX | The comparison mixes two changes | Task 1 equivalence check |
| Samrómur session IDs are not child IDs | Noisy labels; pessimistic EER | Task 2 counts it; deduplicate if metadata allows |
| Domain gap: Icelandic read speech at ages 4–17, against spontaneous toddlers | Gains on Samrómur that do not transfer | Measure on CHILDES and Phase 1 recordings, not only Samrómur; add augmentation |
| Non-commercial data leaking into training | Weights that cannot ship | Training reads only Samrómur in stage 1; CHILDES lives under `data/` and no training script references it |
