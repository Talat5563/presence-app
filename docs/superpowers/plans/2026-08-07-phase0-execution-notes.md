# Phase 0 — execution notes

Written while implementing
[the Phase 0 plan](2026-08-07-phase0-accuracy-harness.md). Everything here is measured or decided,
not assumed. Read it before running an evaluation — several items will otherwise cost you a
confusing afternoon.

---

## Measured model behaviour

> **Correction, 2026-10-28: the VAD adapter was broken until the first real recordings.** Silero v5
> expects every 512-sample frame to be preceded by the previous frame's last 64 samples, as its
> own wrapper does. `SileroVad` sent bare frames, and on real speech that drove the probability to
> about 0.001: CHILDES audio produced **zero** speech segments, so no real voice could ever have
> reached the embedder. With the context, the same 333 s of a child's speech gives a mean
> probability of 0.64 and 106 segments. The synthetic table below was measured with the bug in
> place. It still shows that synthetic audio cannot exercise the VAD, which is exactly why the
> bug survived every test until real recordings arrived.

**Silero VAD, on synthetic signals** (averaged over 10 frames each; measured without the context,
see above):

| Signal | P(speech) |
|---|---|
| Digital silence | 5.42e-4 |
| Pure tone, 150 Hz | 5.26e-4 |
| Pure tone, 220 Hz | 5.26e-4 |
| Quiet noise | 5.35e-4 |
| Loud broadband noise | 8.22e-4 |

**Silero is not an energy detector.** A pure tone scores marginally *below* silence, and harmonic
stacks collapse into the same range — the model responds to speech specifically, not to sound.

**The consequence that will bite you:** these values sit three orders of magnitude below
`VadGateConfig`'s thresholds (start 0.5, end 0.35). Real speech drives Silero to 0.9+, so the
thresholds are right — but **synthetic audio will never open the VAD gate.** Any test or smoke run
that needs an actual speech segment must use a real recording. A synthetic corpus will report zero
detections and look like a broken pipeline.

**Speaker embedding (WeSpeaker resnet34_LM), on synthetic voices:**

- same "speaker", two durations: cosine **0.989**
- different synthetic voices: cosine **0.830**

Both sit high and the gap is only ~0.16 on signals sharing no vocal-tract structure at all, which
suggests plain cosine on this model may run hot. **Do not assume a wide separation band on real
audio** — measure it, and treat `MatcherConfig.minTargetSimilarity` (default 0.35) as unvalidated
until you have. It is the single most important open-set knob.

**Filterbank calibration:** a 300 Hz tone peaks at mel bin 10, a 3000 Hz tone at bin 52, both before
and after mean normalization.

---

## Decisions taken during implementation

Each of these changed the design. They are recorded because the reasoning is not obvious from the
code alone.

**The cohort z-gate could never fire.** For two cohort similarities, `max − mean` equals the
population standard deviation exactly, so a winning target always has `z > 1` — the 0.5 floor never
bound for a normal household, and the open-set protection was dead code. A stranger resembling the
target more than the family did would have passed unchallenged. Added `minTargetSimilarity`, an
absolute floor that does bind, with a test the z-gate provably cannot satisfy.

**The cooldown does not accumulate.** Evidence arriving while suppressed used to enter the window,
so a detection could fire the instant the cooldown lapsed with no new speech since — making the
cooldown a delay rather than a requirement for fresh reason to interrupt a parent. Observations
during cooldown are now dropped entirely.

**Overlapping embedding windows were double-counted.** A segment closing on the 3000 ms cap
overshoots by one frame, and the tail-window branch then emitted a second window overlapping the
first by 99.7% — summed by the accumulator as independent evidence. The tail is now emitted only
when the striding loop missed a full hop, which bounds its overlap at the same 50% every regular
pair already has.

**Every detection matching no labelled event is a false alarm, wherever it fired.** Target-child
recordings were previously exempt on both sides of the ratio, so the app firing while the child
played happily alone was invisible — and it still set the cooldown, so a real call 20 s later could
not be detected. The harness hid the false alarm and paid for it in miss rate.

**The go/no-go tiers are conjunctive.** Spec §4.3 originally read "2 false alarms per hour **or**
50% detection", which taken literally would call 5 alarms/hour with 60% detection "iterate". Both
conditions are required at each tier.

**The ×32768 int16 scaling is unobservable at realistic amplitudes.** Mean normalization cancels any
uniform gain exactly. It is load-bearing only where the log floor clips, which is why the test that
pins it uses an amplitude below one 16-bit LSB.

---

## Before you run an evaluation

1. **Settle consent scope first** (spec §10, open question 6). Whether the recordings may train a
   shipped model or only evaluate it cannot be added to a consent form afterwards.
2. **Collect per spec §9.1** — both languages, code-switching, varied distance and background,
   children's TV as negative material, at least three households with two held out.
3. **Label every call.** An unlabelled real call in a target-child recording now scores as a false
   alarm. That is the right default — an unlabelled corpus must not flatter the result — but it
   biases the verdict pessimistically. The report warns when any target recording has no events.
4. **Choose the operating point on dev, then measure once on holdout:**

   ```bash
   ./gradlew :harness:run --args="--manifest data --models models --out reports/dev --holdout h3,h4"
   # read reports/dev/report.md, pick a threshold, then:
   ./gradlew :harness:run --args="--manifest data --models models --out reports/final \
       --holdout h3,h4 --evaluate holdout --threshold 6"
   ```

   Sweeping the holdout and taking its best verdict is tuning on the data the holdout exists to
   protect. The CLI refuses to do it.
5. **Both languages must clear the bar independently.** A system passing in English and failing in
   Hebrew has not passed — it has one working language and one that will erode trust.

---

## Known limitations

Deliberately deferred. None blocks a dev-set measurement.

- **Matcher knobs are not swept.** The infrastructure exists — `MatchScore` is cached per
  observation and gating is centralised in `MatcherConfig.applyGates` — but only `fireThreshold`
  sweeps today. Calibrating `minTargetSimilarity`, `minCohortZ` or `softmaxTemperature` still means
  a full re-embed per value. Given the note above about separation running hot, this is the most
  likely thing you will want next.
- `MatchScore` does not carry `closestNegative`, so a replay cannot re-apply the negative veto.
  Inert in Phase 0, since enrollment produces no negatives.
- Holdout mode traces the whole corpus and then discards non-holdout traces — wasted inference.
- A recording with no language counts only in the `ALL` slice, never in the per-language ones, which
  quietly changes those denominators.
- A file skipped for a bad format also drops its events from `totalEvents`, shrinking the
  denominator with only a warning line.
- ~~A second `TARGET_CHILD` speaker in one household would land in the cohort and crush the
  target's score. Nothing rejects it.~~ Fixed 2026-10-28: `HouseholdMatcher` refuses it, and the
  harness skips such a household with a warning.
- `Csv` does not strip a UTF-8 BOM, so an Excel export fails with a confusing "missing required
  columns: path". Fails loudly, so it is safe — just unhelpful.

---

## Environment

JDK 17 and Gradle 8.11 are required. Neither ships with this repository; on the machine this was
built on they live in `~/.local/toolchains/` and are exported from `~/.bashrc`. `./gradlew check`
runs the 198 pure-logic tests; `./gradlew :shared:integrationTest` and `:harness:integrationTest`
run the tests that need the real ONNX models in `models/`.
