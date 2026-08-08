# Presence

An app that helps parents notice their kids.

A parent is on their phone. Their child says their name, asks a question, tries to show them
something — and it doesn't land. Presence listens for *that specific child's voice* and gently
redirects the parent's attention back to them: the screen dims, audio ducks, and the child's face
appears.

Everything happens on the device. No account, no server, no audio ever uploaded.

> **Working name.** "Presence" is a placeholder. It appears only in documentation at this stage, so
> renaming costs nothing.

---

## Status

**Building Phase 0.** The detection engine and its evaluation harness are under construction on the
`phase0-accuracy-harness` branch, following
[the implementation plan](docs/superpowers/plans/2026-08-07-phase0-accuracy-harness.md). No UI, no
Android code, and no iOS code yet — deliberately.

Phase 0 (spec §9) is a JVM harness that measures the detection *system* — not just the model —
against public child-speech benchmarks and a household simulation set, before any UI is built.

That ordering isn't caution for its own sake. Recognizing a specific child's voice is the whole
product and it is genuinely hard: adult-trained speaker models measure around **31% EER on children's
spontaneous speech**, which is unusable per-utterance. Fine-tuning on children's speech reaches
**2.9% EER**, and the design adds household-cohort scoring and cross-segment evidence accumulation on
top. Spec §4.1 lays out that accuracy ladder with sources, and §9 defines explicit go / no-go numbers.

## What's here

| Path | Contents |
|---|---|
| `docs/superpowers/specs/2026-08-06-presence-design.md` | The full design specification — architecture, detection pipeline, per-platform intervention design, permissions, failure modes, testing strategy |
| `docs/superpowers/plans/2026-08-07-phase0-accuracy-harness.md` | The Phase 0 implementation plan, task by task |
| `docs/design/UI_DESIGN_PROMPT.md` | A self-contained, ready-to-paste prompt for designing the complete UI. Hand it to a fresh Claude session |
| `shared/` | Kotlin Multiplatform detection engine — framing, VAD gating, embeddings, household scoring, evidence accumulation |
| `harness/` | JVM-only evaluation harness that drives the engine over a corpus |

**Collecting recordings?** Read **spec §9.1** first. It specifies consent scope, who to record, both
languages plus code-switching, acoustic conditions, negative material, labelling, and the held-out
split. Getting the metadata wrong means re-collecting, and consent scope in particular cannot be
widened retroactively.

## Planned shape

**Kotlin Multiplatform shared core, native UI per platform.** Android first, iOS second.
**English and Hebrew** at first release, which makes the UI bidirectional from the first screen and
makes bilingual voice enrollment mandatory — a child enrolled in one language and heard in the other
is close to a different-domain trial (spec §4.2).

The sensing engine is pure portable logic; everything touching the OS is a thin adapter. That seam
matters because the engine is the hard, valuable part while the OS integrations are what differ per
platform — and it means the whole recognition pipeline is unit-testable on the JVM against recorded
WAV fixtures, with no device and no emulator.

```
mic (16 kHz mono)
  └─> VAD gate — is anyone speaking?          ~1 MB, always on
        └─> speaker embedding                 ~20 MB, speech only (~5% duty cycle)
              └─> score vs every enrolled household member
                    └─> accumulate log-likelihood across segments
                          └─> intervention
```

The VAD gate keeps battery cost survivable. **The accumulator is what makes it accurate** — no single
0.9-second word has to carry the decision, and scoring against the whole household (siblings and the
parent included) turns an absolute threshold into a relative comparison. Enrolling the other people in
the house is therefore a requirement, not a nice-to-have.

**Platform reality:** Android can dim and draw over other apps. iOS cannot — there, the same product
is delivered through a Live Activity in the Dynamic Island, a Time-Sensitive notification, and
`AVAudioSession` ducking. See §6 of the spec for the full mapping and the two-tier iOS plan.

## Privacy commitments

Product requirements, not aspirations — each one testable:

- No audio leaves the device, ever
- No raw audio is persisted; enrollment clips become embeddings and the source is deleted
- Voice profiles live in platform-encrypted storage
- Event logs record timestamp, child, and confidence — never audio
- Real family audio never enters version control (enforced by `.gitignore`)
- Uninstalling deletes everything

## Running Phase 0

Requires JDK 17 and Gradle 8.11.

```bash
./tools/fetch-models.sh                         # download models into models/ (gitignored)
./gradlew check                                 # pure-logic tests — no models, no network
./gradlew integrationTest                       # tests that need the real model files
./gradlew :harness:run --args="--help"          # evaluation CLI
```

Model files are never committed. `models/PROVENANCE.txt` records which ones a given result used —
quote it whenever you report numbers.

## Attribution

If the shipped model is fine-tuned on **Samrómur Children** (spec §4.1), that corpus is licensed
**CC BY 4.0** and attribution is a license condition — it must appear both here and in the app's about
screen:

> Samrómur Children, Language and Voice Lab, Reykjavík University — CC BY 4.0.
> https://huggingface.co/datasets/language-and-voice-lab/samromur_children

Third-party model licenses (WeSpeaker, 3D-Speaker, NeMo, Silero VAD) must each be verified
individually before shipping — see spec §10, open question 2.

## License

Not yet chosen. Pick one before making this repository public.
